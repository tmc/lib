package nvtelemetry

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strconv"
	"sync"
	"time"
)

// RefreshConfig bounds a periodic source refresher. All fields must be
// positive. Interval separates Run cycles, Timeout bounds one read or fetch,
// and MaxAge is how long the last successful cycle is served before results
// are withheld as stale.
type RefreshConfig struct {
	Interval time.Duration
	Timeout  time.Duration
	MaxAge   time.Duration
}

func (c RefreshConfig) validate() error {
	if c.Interval <= 0 || c.Timeout <= 0 || c.MaxAge <= 0 {
		return errors.New("refresh: interval, timeout and max age must be positive")
	}
	return nil
}

// RefreshStatus reports a refresher's cycles. Times are the refresher clock's
// monotonic nanoseconds and are meaningful only when the matching Has field
// is true. Records and Withheld describe the last successful cycle.
type RefreshStatus struct {
	Cycles      int64  `json:"cycles"`
	Failures    int64  `json:"failures"`
	HasAttempt  bool   `json:"has_attempt"`
	LastAttempt int64  `json:"last_attempt_nanos"`
	HasSuccess  bool   `json:"has_success"`
	LastSuccess int64  `json:"last_success_nanos"`
	LastError   string `json:"last_error,omitempty"`
	Records     int    `json:"records"`
	Withheld    int    `json:"withheld"`
	Closed      bool   `json:"closed"`
}

// refreshState is the bookkeeping shared by the attribution and workload
// refreshers. Callers hold mu.
type refreshState struct {
	cfg      RefreshConfig
	clock    Clock
	ctx      context.Context // canceled by close
	cancel   context.CancelFunc
	stepMu   sync.Mutex // serializes Step
	mu       sync.Mutex
	inflight bool
	status   RefreshStatus
}

func newRefreshState(clock Clock, cfg RefreshConfig) (*refreshState, error) {
	if clock == nil {
		return nil, errors.New("refresh: nil clock")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &refreshState{cfg: cfg, clock: clock, ctx: ctx, cancel: cancel}, nil
}

// refreshDo runs fn in its own goroutine bounded by Timeout and close. A read that
// ignores cancellation is abandoned; until it returns, later cycles are
// refused rather than stacking goroutines.
func refreshDo[T any](ctx context.Context, s *refreshState, fn func(context.Context) (T, error)) (T, error) {
	var zero T
	s.mu.Lock()
	if s.status.Closed {
		s.mu.Unlock()
		return zero, errors.New("refresh: closed")
	}
	s.status.Cycles++
	s.status.HasAttempt = true
	s.status.LastAttempt = s.clock.NowMonotonic()
	if s.inflight {
		s.mu.Unlock()
		return zero, errors.New("previous read still running")
	}
	s.inflight = true
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(ctx, s.cfg.Timeout)
	defer cancel()
	stop := context.AfterFunc(s.ctx, cancel)
	defer stop()
	type result struct {
		v   T
		err error
	}
	ch := make(chan result, 1)
	go func() {
		v, err := fn(ctx)
		s.mu.Lock()
		s.inflight = false
		s.mu.Unlock()
		ch <- result{v, err}
	}()
	select {
	case r := <-ch:
		if r.err == nil {
			// A read that finished at its deadline is not trusted.
			r.err = ctx.Err()
		}
		return r.v, r.err
	case <-ctx.Done():
		return zero, ctx.Err()
	}
}

// fail records a failed cycle. The caller holds mu.
func (s *refreshState) fail(err error) {
	s.status.Failures++
	s.status.LastError = err.Error()
}

// succeed records a successful cycle. The caller holds mu.
func (s *refreshState) succeed(records, withheld int) {
	s.status.HasSuccess = true
	s.status.LastSuccess = s.clock.NowMonotonic()
	s.status.LastError = ""
	s.status.Records = records
	s.status.Withheld = withheld
}

// fresh reports whether the last success is within MaxAge of now, a
// monotonic time. The caller holds mu.
func (s *refreshState) fresh(now int64) bool {
	if s.status.Closed || !s.status.HasSuccess {
		return false
	}
	age := now - s.status.LastSuccess
	return age >= 0 && age <= int64(s.cfg.MaxAge)
}

func (s *refreshState) run(ctx context.Context, step func(context.Context) error) error {
	t := time.NewTimer(0)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.ctx.Done():
			return nil
		case <-t.C:
		}
		// Errors are recorded in Status; Run keeps the cadence.
		step(ctx)
		t.Reset(s.cfg.Interval)
	}
}

func (s *refreshState) close() {
	s.mu.Lock()
	s.status.Closed = true
	s.mu.Unlock()
	s.cancel()
}

// AttributionRead is one read of attribution sources. Withheld lists
// configured process IDs whose proc files were absent; they are omitted from
// Snapshot, so any earlier attribution for them is dropped on refresh.
type AttributionRead struct {
	Snapshot JobAttributionSnapshot
	Withheld []uint32
}

// SlurmAttributionSource names explicitly configured attribution inputs in an
// fs.FS. File names are fs.ValidPath names. The reader opens exactly Squeue,
// Mappings, Executions and, for each PID, ProcDir/PID/stat, status and cgroup.
// It never opens environ and never lists a directory.
//
// Squeue is squeue --json data_parser v0.0.43 output read by
// ReadSlurmSqueueJSON. Mappings is a version 1 document
// {"version":1,"mappings":[...]} of SlurmGPUMapping records. Executions is a
// version 1 EntityExport. Snapshot pins the scheduler revision, node, clock
// and validity interval; BootID, NodeID and PIDNamespace identify the host
// snapshot the proc files came from. The source does not authenticate its
// producers.
type SlurmAttributionSource struct {
	Squeue       string
	Mappings     string
	Executions   string
	ProcDir      string
	PIDs         []uint32
	BootID       string
	NodeID       string
	PIDNamespace string
	Snapshot     EntitySnapshot
}

func (s SlurmAttributionSource) validate() error {
	if err := s.Snapshot.assignment("slurm_squeue_v0.0.43").Validate(); err != nil {
		return fmt.Errorf("slurm attribution: unpinned snapshot: %w", err)
	}
	if s.BootID == "" || s.NodeID == "" || s.PIDNamespace == "" {
		return errors.New("slurm attribution: missing boot, node or pid namespace identity")
	}
	if s.Snapshot.NodeID != "" && s.Snapshot.NodeID != s.NodeID {
		return errors.New("slurm attribution: snapshot node differs from process node")
	}
	for _, name := range []string{s.Squeue, s.Mappings, s.Executions, s.ProcDir} {
		if !fs.ValidPath(name) {
			return fmt.Errorf("slurm attribution: invalid path %q", name)
		}
	}
	if len(s.PIDs) > entityMaxRecords {
		return errors.New("slurm attribution: too many pids")
	}
	seen := make(map[uint32]bool, len(s.PIDs))
	for _, pid := range s.PIDs {
		if pid == 0 || seen[pid] {
			return errors.New("slurm attribution: zero or duplicate pid")
		}
		seen[pid] = true
	}
	return nil
}

// ReadSlurmAttributionSnapshot reads the configured files from fsys into a
// JobAttributionSnapshot. The source is validated before any file is opened.
// A configured process is withheld if its stat, status or cgroup file is absent
// or inaccessible, or stat identifies a zombie or dead process. Other read or
// parse errors, including errors in the shared source files, fail the whole read.
func ReadSlurmAttributionSnapshot(fsys fs.FS, src SlurmAttributionSource) (AttributionRead, error) {
	var out AttributionRead
	if fsys == nil {
		return out, errors.New("slurm attribution: nil file system")
	}
	if err := src.validate(); err != nil {
		return out, err
	}
	err := withFile(fsys, src.Squeue, func(r io.Reader) (err error) {
		out.Snapshot.Allocations, err = ReadSlurmSqueueJSON(r, src.Snapshot)
		return err
	})
	if err != nil {
		return AttributionRead{}, err
	}
	err = withFile(fsys, src.Mappings, func(r io.Reader) (err error) {
		out.Snapshot.Mappings, err = readSlurmGPUMappings(r)
		return err
	})
	if err != nil {
		return AttributionRead{}, err
	}
	err = withFile(fsys, src.Executions, func(r io.Reader) error {
		e, err := ReadEntityExport(r)
		out.Snapshot.Executions = e.Assignments
		return err
	})
	if err != nil {
		return AttributionRead{}, err
	}
	for _, pid := range src.PIDs {
		p, err := readProcFiles(fsys, src, pid)
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) || errors.Is(err, ErrProcEntityNotLive) {
			out.Withheld = append(out.Withheld, pid)
			continue
		}
		if err != nil {
			return AttributionRead{}, err
		}
		out.Snapshot.Processes = append(out.Snapshot.Processes, p)
	}
	return out, nil
}

func withFile(fsys fs.FS, name string, fn func(io.Reader) error) error {
	f, err := fsys.Open(name)
	if err != nil {
		return fmt.Errorf("slurm attribution: %w", err)
	}
	defer f.Close()
	return fn(f)
}

func readProcFiles(fsys fs.FS, src SlurmAttributionSource, pid uint32) (ProcEntityFacts, error) {
	dir := path.Join(src.ProcDir, strconv.FormatUint(uint64(pid), 10))
	var files [3]fs.File
	defer func() {
		for _, f := range files {
			if f != nil {
				f.Close()
			}
		}
	}()
	for i, name := range []string{"stat", "status", "cgroup"} {
		f, err := fsys.Open(path.Join(dir, name))
		if err != nil {
			return ProcEntityFacts{}, fmt.Errorf("slurm attribution: %w", err)
		}
		files[i] = f
	}
	p, err := ReadProcEntity(files[0], files[1], files[2], src.BootID, src.NodeID, src.PIDNamespace)
	if err != nil {
		return ProcEntityFacts{}, fmt.Errorf("slurm attribution: pid %d: %w", pid, err)
	}
	if p.Process.Process.PID != pid {
		return ProcEntityFacts{}, fmt.Errorf("slurm attribution: pid %d: stat names pid %d", pid, p.Process.Process.PID)
	}
	return p, nil
}

func readSlurmGPUMappings(r io.Reader) ([]SlurmGPUMapping, error) {
	var doc struct {
		Version  int               `json:"version"`
		Mappings []SlurmGPUMapping `json:"mappings"`
	}
	if err := readEntityJSON(r, &doc, true); err != nil {
		return nil, fmt.Errorf("slurm gpu mapping: %w", err)
	}
	if doc.Version != 1 || doc.Mappings == nil || len(doc.Mappings) > entityMaxRecords {
		return nil, errors.New("slurm gpu mapping: unsupported version or mapping count")
	}
	for _, m := range doc.Mappings {
		if err := m.Validate(); err != nil {
			return nil, err
		}
	}
	return doc.Mappings, nil
}

// AttributionRefresher periodically reads attribution sources into a
// JobAttributionTable and withholds attribution when the last successful
// read is older than MaxAge. A failed read never refreshes or clears the
// table. It is exercised only against fixture sources; it does not qualify a
// live scheduler or procfs. The zero value is not usable; use
// NewAttributionRefresher.
type AttributionRefresher struct {
	s     *refreshState
	table *JobAttributionTable
	read  func(context.Context) (AttributionRead, error)
}

// NewAttributionRefresher returns a refresher that calls read once per cycle
// and refreshes table at the clock's Unix time, which must be the clock domain
// of the snapshot intervals. Freshness uses the clock's monotonic time.
func NewAttributionRefresher(table *JobAttributionTable, read func(context.Context) (AttributionRead, error), clock Clock, cfg RefreshConfig) (*AttributionRefresher, error) {
	if table == nil || read == nil {
		return nil, errors.New("attribution refresh: nil table or read")
	}
	s, err := newRefreshState(clock, cfg)
	if err != nil {
		return nil, err
	}
	return &AttributionRefresher{s: s, table: table, read: read}, nil
}

// Step runs one bounded read and, only if it succeeds, refreshes the table.
func (a *AttributionRefresher) Step(ctx context.Context) error {
	s := a.s
	s.stepMu.Lock()
	defer s.stepMu.Unlock()
	read, err := refreshDo(ctx, s, a.read)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.status.Closed {
		return errors.New("attribution refresh: closed")
	}
	if err != nil {
		s.fail(err)
		return fmt.Errorf("attribution refresh: %w", err)
	}
	results, err := a.table.Refresh(read.Snapshot, s.clock.NowUnix())
	if err != nil {
		s.fail(err)
		return fmt.Errorf("attribution refresh: %w", err)
	}
	s.succeed(len(results), len(read.Withheld))
	return nil
}

// Run calls Step immediately and then every Interval until ctx is done or the
// refresher is closed. Step errors are reported by Status. Run returns
// ctx.Err() on cancellation and nil after Close.
func (a *AttributionRefresher) Run(ctx context.Context) error {
	return a.s.run(ctx, a.Step)
}

// Resolve returns the table's attribution for proc on deviceUUID at at, or an
// unknown result with reason source_stale when no read has succeeded within
// MaxAge, or source_closed after Close.
func (a *AttributionRefresher) Resolve(proc EntityProcess, deviceUUID string, at int64) JobAttributionResult {
	s := a.s
	s.mu.Lock()
	closed := s.status.Closed
	fresh := s.fresh(s.clock.NowMonotonic())
	s.mu.Unlock()
	switch {
	case closed:
		return JobAttributionResult{Assignment: EntityAssignment{Process: proc, State: "unknown"}, Reason: "source_closed"}
	case !fresh:
		return JobAttributionResult{Assignment: EntityAssignment{Process: proc, State: "unknown"}, Reason: "source_stale"}
	}
	return a.table.Resolve(proc, deviceUUID, at)
}

// Ready reports whether a read has succeeded within MaxAge.
func (a *AttributionRefresher) Ready() bool {
	a.s.mu.Lock()
	defer a.s.mu.Unlock()
	return a.s.fresh(a.s.clock.NowMonotonic())
}

// Status returns a copy of the refresher's cycle accounting.
func (a *AttributionRefresher) Status() RefreshStatus {
	a.s.mu.Lock()
	defer a.s.mu.Unlock()
	return a.s.status
}

// Close stops Run, cancels an in-flight read and refuses later cycles. The
// table is left as it is. Close is safe to call more than once.
func (a *AttributionRefresher) Close() error {
	a.s.close()
	return nil
}
