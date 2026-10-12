package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// refreshConfig selects explicitly configured sources for the refresh
// command. Workloads are paths to fetch-endpoint configuration files.
type refreshConfig struct {
	Interval    time.Duration      `json:"interval_nanos"`
	Timeout     time.Duration      `json:"timeout_nanos"`
	MaxAge      time.Duration      `json:"max_age_nanos"`
	Attribution *attributionConfig `json:"attribution,omitempty"`
	Workloads   []string           `json:"workloads,omitempty"`
}

// attributionConfig names Slurm attribution files below Root. See
// nvtelemetry.SlurmAttributionSource for what is opened.
type attributionConfig struct {
	Root          string         `json:"root"`
	Squeue        string         `json:"squeue"`
	Mappings      string         `json:"mappings"`
	Executions    string         `json:"executions"`
	ProcDir       string         `json:"proc_dir"`
	PIDs          []uint32       `json:"pids"`
	BootID        string         `json:"boot_id"`
	NodeID        string         `json:"node_id"`
	PIDNamespace  string         `json:"pid_namespace"`
	Snapshot      snapshotConfig `json:"snapshot"`
	TableCapacity int            `json:"table_capacity"`
	TableTTL      time.Duration  `json:"table_ttl_nanos"`
}

type snapshotConfig struct {
	SourceRevision string `json:"source_revision"`
	Authoritative  bool   `json:"authoritative"`
	NodeID         string `json:"node_id"`
	ClockDomain    string `json:"clock_domain"`
	ClockEpoch     string `json:"clock_epoch"`
	StartNanos     int64  `json:"start_nanos"`
	EndNanos       int64  `json:"end_nanos"`
}

func readRefreshConfig(path string) (refreshConfig, error) {
	var c refreshConfig
	raw, err := readEndpointFile(path, 64<<10)
	if err != nil {
		return c, err
	}
	if err := checkEndpointJSON(json.NewDecoder(bytes.NewReader(raw)), 0); err != nil {
		return c, errors.New("invalid refresh configuration JSON")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return c, errors.New("invalid refresh configuration fields")
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return c, errors.New("trailing refresh configuration JSON")
	}
	if c.Interval <= 0 || c.Timeout <= 0 || c.Timeout > 30*time.Second || c.MaxAge <= 0 {
		return c, errors.New("invalid refresh interval, timeout or max age")
	}
	if c.Attribution == nil && len(c.Workloads) == 0 || len(c.Workloads) > 32 {
		return c, errors.New("refresh requires an attribution source or 1 to 32 workloads")
	}
	return c, nil
}

func (a *attributionConfig) source() nvtelemetry.SlurmAttributionSource {
	s := a.Snapshot
	return nvtelemetry.SlurmAttributionSource{
		Squeue:       a.Squeue,
		Mappings:     a.Mappings,
		Executions:   a.Executions,
		ProcDir:      a.ProcDir,
		PIDs:         a.PIDs,
		BootID:       a.BootID,
		NodeID:       a.NodeID,
		PIDNamespace: a.PIDNamespace,
		Snapshot: nvtelemetry.EntitySnapshot{
			SourceRevision: s.SourceRevision,
			Authoritative:  s.Authoritative,
			NodeID:         s.NodeID,
			ClockDomain:    s.ClockDomain,
			ClockEpoch:     s.ClockEpoch,
			StartNanos:     s.StartNanos,
			EndNanos:       s.EndNanos,
		},
	}
}

// noFiles fails every open. ReadSlurmAttributionSnapshot validates its
// source before opening anything, so reading from noFiles reports either a
// configuration error or fs.ErrNotExist for a valid source.
type noFiles struct{}

func (noFiles) Open(string) (fs.File, error) { return nil, fs.ErrNotExist }

// refreshSource is one configured source driven by the refresh command.
type refreshSource struct {
	name   string
	step   func(context.Context) error
	ready  func() bool
	status func() nvtelemetry.RefreshStatus
	sha256 func() string
	close  func() error
}

type refreshLine struct {
	Cycle     int                       `json:"cycle"`
	Source    string                    `json:"source"`
	Ready     bool                      `json:"ready"`
	Error     string                    `json:"error,omitempty"`
	RawSHA256 string                    `json:"raw_sha256,omitempty"`
	Status    nvtelemetry.RefreshStatus `json:"status"`
}

func newAttributionSource(a *attributionConfig, clock nvtelemetry.Clock, cfg nvtelemetry.RefreshConfig) (refreshSource, error) {
	if a.Root == "" {
		return refreshSource{}, errors.New("attribution requires a root directory")
	}
	src := a.source()
	if _, err := nvtelemetry.ReadSlurmAttributionSnapshot(noFiles{}, src); !errors.Is(err, fs.ErrNotExist) {
		return refreshSource{}, err
	}
	table, err := nvtelemetry.NewJobAttributionTable(a.TableCapacity, int64(a.TableTTL))
	if err != nil {
		return refreshSource{}, err
	}
	fsys := os.DirFS(a.Root)
	read := func(context.Context) (nvtelemetry.AttributionRead, error) {
		return nvtelemetry.ReadSlurmAttributionSnapshot(fsys, src)
	}
	r, err := nvtelemetry.NewAttributionRefresher(table, read, clock, cfg)
	if err != nil {
		return refreshSource{}, err
	}
	return refreshSource{name: "attribution", step: r.Step, ready: r.Ready, status: r.Status, close: r.Close}, nil
}

func newWorkloadSource(i int, path string, clock nvtelemetry.Clock, cfg nvtelemetry.RefreshConfig) (refreshSource, error) {
	c, client, request, err := readEndpointConfig(path)
	if err != nil {
		return refreshSource{}, fmt.Errorf("workload %d: %w", i, err)
	}
	switch {
	case c.Managed != nil:
		err = errors.New("refresh does not accept managed endpoints")
	case c.Workload == nil:
		err = errors.New("endpoint has no workload source")
	case c.MaxPages != 0:
		err = errors.New("refresh does not page workload endpoints")
	}
	if err != nil {
		client.CloseIdleConnections()
		return refreshSource{}, fmt.Errorf("workload %d: %w", i, err)
	}
	src, limits := *c.Workload, nvtelemetry.WorkloadLimits{MaxBytes: c.MaxBytes}
	fetch := func(ctx context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
		req := request.Clone(ctx)
		if request.GetBody != nil {
			body, err := request.GetBody()
			if err != nil {
				return nvtelemetry.WorkloadHTTPResult{}, err
			}
			req.Body = body
		}
		return nvtelemetry.FetchWorkload(ctx, client, req, src, limits)
	}
	r, err := nvtelemetry.NewWorkloadRefresher(fetch, clock, nvtelemetry.RealClock{}.WallDomain(), "unix_epoch", cfg)
	if err != nil {
		client.CloseIdleConnections()
		return refreshSource{}, err
	}
	return refreshSource{
		name:   fmt.Sprintf("workload[%d]", i),
		step:   r.Step,
		ready:  r.Ready,
		status: r.Status,
		sha256: func() string {
			res, _ := r.Latest(clock.NowMonotonic())
			return res.Export.RawSHA256
		},
		close: func() error {
			client.CloseIdleConnections()
			return r.Close()
		},
	}, nil
}

func runRefresh(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("refresh", flag.ContinueOnError)
	flags.SetOutput(stderr)
	path := flags.String("config", "", "explicit refresh configuration JSON file")
	count := flags.Int("count", 0, "number of refresh cycles")
	duration := flags.Duration("duration", 0, "stop starting cycles after this long")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *path == "" || flags.NArg() != 0 {
		return errors.New("refresh requires a config file")
	}
	if *count < 0 || *count > 1<<20 || *duration < 0 || *count == 0 && *duration == 0 {
		return errors.New("refresh requires -count or -duration")
	}
	c, err := readRefreshConfig(*path)
	if err != nil {
		return err
	}
	clock := nvtelemetry.RealClock{}
	cfg := nvtelemetry.RefreshConfig{Interval: c.Interval, Timeout: c.Timeout, MaxAge: c.MaxAge}
	// Build every source and validate structural configuration before the first cycle.
	// FetchWorkload validates workload provider/version and identity on each Step.
	var sources []refreshSource
	defer func() {
		for _, s := range sources {
			s.close()
		}
	}()
	if c.Attribution != nil {
		s, err := newAttributionSource(c.Attribution, clock, cfg)
		if err != nil {
			return err
		}
		sources = append(sources, s)
	}
	for i, p := range c.Workloads {
		s, err := newWorkloadSource(i, p, clock, cfg)
		if err != nil {
			return err
		}
		sources = append(sources, s)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *duration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *duration)
		defer cancel()
	}
	enc := json.NewEncoder(stdout)
	for cycle := 1; ; cycle++ {
		if ctx.Err() != nil {
			return nil
		}
		for _, s := range sources {
			line := refreshLine{Cycle: cycle, Source: s.name}
			if err := s.step(ctx); err != nil {
				line.Error = err.Error()
			}
			line.Ready, line.Status = s.ready(), s.status()
			if s.sha256 != nil {
				line.RawSHA256 = s.sha256()
			}
			if err := enc.Encode(line); err != nil {
				return err
			}
		}
		if cycle == *count {
			return nil
		}
		t := time.NewTimer(c.Interval)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil
		case <-t.C:
		}
	}
}
