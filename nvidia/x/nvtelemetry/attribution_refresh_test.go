package nvtelemetry_test

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/fstest"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// The b690-refresh fixtures are synthetic known-answer files, not captured
// scheduler, procfs or GPU data.
const refreshFixtureDir = "testdata/entity/b690-refresh"

const refreshDeviceUUID = "GPU-11111111-2222-3333-4444-555555555555"

func refreshFixtureFS(t testing.TB) fstest.MapFS {
	t.Helper()
	m := fstest.MapFS{}
	err := fs.WalkDir(os.DirFS(refreshFixtureDir), ".", func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := fs.ReadFile(os.DirFS(refreshFixtureDir), name)
		if err != nil {
			return err
		}
		m[name] = &fstest.MapFile{Data: b}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func refreshSource() nvtelemetry.SlurmAttributionSource {
	return nvtelemetry.SlurmAttributionSource{
		Squeue:       "squeue.json",
		Mappings:     "mappings.json",
		Executions:   "executions.json",
		ProcDir:      "proc",
		PIDs:         []uint32{1234},
		BootID:       "boot-1",
		NodeID:       "node-1",
		PIDNamespace: "pid:[4026531836]",
		Snapshot: nvtelemetry.EntitySnapshot{
			SourceRevision: "fixture-squeue-1",
			Authoritative:  true,
			NodeID:         "node-1",
			ClockDomain:    "synthetic_unix_utc",
			ClockEpoch:     "fixture-epoch-1",
			StartNanos:     100,
			EndNanos:       1_000_000_000_000,
		},
	}
}

func refreshQueryProcess(generation uint64) nvtelemetry.EntityProcess {
	return nvtelemetry.EntityProcess{
		Process:           cudaworkload.ProcessGenID{PID: 1234, Generation: generation},
		BootID:            "boot-1",
		NodeID:            "node-1",
		PIDNamespace:      "pid:[4026531836]",
		ExecGeneration:    1,
		HasExecGeneration: true,
	}
}

// recordingFS records every name passed to Open, including directories.
type recordingFS struct {
	fsys  fs.FS
	mu    sync.Mutex
	opens []string
}

func (r *recordingFS) Open(name string) (fs.File, error) {
	r.mu.Lock()
	r.opens = append(r.opens, name)
	r.mu.Unlock()
	return r.fsys.Open(name)
}

func (r *recordingFS) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.opens)
}

func TestReadSlurmAttributionSnapshot(t *testing.T) {
	rec := &recordingFS{fsys: refreshFixtureFS(t)}
	got, err := nvtelemetry.ReadSlurmAttributionSnapshot(rec, refreshSource())
	if err != nil {
		t.Fatal(err)
	}
	snap := got.Snapshot
	if len(snap.Allocations) != 1 || snap.Allocations[0].JobID != "42" {
		t.Fatalf("allocations = %+v, want running job 42 only", snap.Allocations)
	}
	if len(snap.Mappings) != 1 || snap.Mappings[0].Device.UUID != refreshDeviceUUID {
		t.Fatalf("mappings = %+v", snap.Mappings)
	}
	if len(snap.Executions) != 1 || len(snap.Processes) != 1 || len(got.Withheld) != 0 {
		t.Fatalf("executions %d processes %d withheld %v", len(snap.Executions), len(snap.Processes), got.Withheld)
	}
	p := snap.Processes[0]
	if p.Process.Process != (cudaworkload.ProcessGenID{PID: 1234, Generation: 1000}) || p.StartTimeTicks != 1000 {
		t.Fatalf("process = %+v", p)
	}
	want := []string{"squeue.json", "mappings.json", "executions.json", "proc/1234/stat", "proc/1234/status", "proc/1234/cgroup"}
	if opens := rec.names(); !slices.Equal(opens, want) {
		t.Fatalf("opened %q, want only %q", opens, want)
	}
}

func TestReadSlurmAttributionSnapshotRefusesBeforeOpen(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*nvtelemetry.SlurmAttributionSource)
	}{
		{"unpinned revision", func(s *nvtelemetry.SlurmAttributionSource) { s.Snapshot.SourceRevision = "" }},
		{"missing clock epoch", func(s *nvtelemetry.SlurmAttributionSource) { s.Snapshot.ClockEpoch = "" }},
		{"empty interval", func(s *nvtelemetry.SlurmAttributionSource) { s.Snapshot.EndNanos = s.Snapshot.StartNanos }},
		{"missing boot id", func(s *nvtelemetry.SlurmAttributionSource) { s.BootID = "" }},
		{"missing pid namespace", func(s *nvtelemetry.SlurmAttributionSource) { s.PIDNamespace = "" }},
		{"node differs from snapshot", func(s *nvtelemetry.SlurmAttributionSource) { s.NodeID = "node-2" }},
		{"missing squeue", func(s *nvtelemetry.SlurmAttributionSource) { s.Squeue = "" }},
		{"missing mappings", func(s *nvtelemetry.SlurmAttributionSource) { s.Mappings = "" }},
		{"missing executions", func(s *nvtelemetry.SlurmAttributionSource) { s.Executions = "" }},
		{"invalid path", func(s *nvtelemetry.SlurmAttributionSource) { s.Squeue = "../squeue.json" }},
		{"invalid proc dir", func(s *nvtelemetry.SlurmAttributionSource) { s.ProcDir = "/proc" }},
		{"zero pid", func(s *nvtelemetry.SlurmAttributionSource) { s.PIDs = []uint32{0} }},
		{"duplicate pid", func(s *nvtelemetry.SlurmAttributionSource) { s.PIDs = []uint32{1234, 1234} }},
		{"too many pids", func(s *nvtelemetry.SlurmAttributionSource) { s.PIDs = make([]uint32, 4097) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := refreshSource()
			tt.mutate(&src)
			rec := &recordingFS{fsys: refreshFixtureFS(t)}
			if _, err := nvtelemetry.ReadSlurmAttributionSnapshot(rec, src); err == nil {
				t.Fatal("read succeeded, want refusal")
			}
			if opens := rec.names(); len(opens) != 0 {
				t.Fatalf("refused source opened %q", opens)
			}
		})
	}
}

func TestReadSlurmAttributionSnapshotWithholdsMissingProcess(t *testing.T) {
	m := refreshFixtureFS(t)
	m["proc/77/stat"] = m["proc/1234/stat"]
	m["proc/77/status"] = m["proc/1234/status"]
	// proc/77/cgroup is absent: the process exited between reads.
	src := refreshSource()
	src.PIDs = []uint32{1234, 5678, 77}
	got, err := nvtelemetry.ReadSlurmAttributionSnapshot(m, src)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Snapshot.Processes) != 1 || got.Snapshot.Processes[0].Process.Process.PID != 1234 {
		t.Fatalf("processes = %+v", got.Snapshot.Processes)
	}
	if !slices.Equal(got.Withheld, []uint32{5678, 77}) {
		t.Fatalf("withheld = %v, want [5678 77]", got.Withheld)
	}
}

func TestReadSlurmAttributionSnapshotErrors(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(fstest.MapFS)
	}{
		{"missing squeue", func(m fstest.MapFS) { delete(m, "squeue.json") }},
		{"missing mappings", func(m fstest.MapFS) { delete(m, "mappings.json") }},
		{"invalid stat", func(m fstest.MapFS) { m["proc/1234/stat"] = &fstest.MapFile{Data: []byte("1234 python\n")} }},
		{"stat pid mismatch", func(m fstest.MapFS) {
			m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(strings.Replace(string(m["proc/1234/stat"].Data), "1234 (", "999 (", 1))}
		}},
		{"unknown mapping field", func(m fstest.MapFS) {
			m["mappings.json"] = &fstest.MapFile{Data: []byte(`{"version":1,"mappings":[],"extra":1}`)}
		}},
		{"unsupported mapping version", func(m fstest.MapFS) {
			m["mappings.json"] = &fstest.MapFile{Data: []byte(`{"version":2,"mappings":[]}`)}
		}},
		{"invalid mapping", func(m fstest.MapFS) {
			m["mappings.json"] = &fstest.MapFile{Data: []byte(`{"version":1,"mappings":[{"job_id":"42","device":{"uuid":"GPU-1"},"sharing":"bogus","authoritative":true}]}`)}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := refreshFixtureFS(t)
			tt.mutate(m)
			if _, err := nvtelemetry.ReadSlurmAttributionSnapshot(m, refreshSource()); err == nil {
				t.Fatal("read succeeded, want error")
			}
		})
	}
}

// TestJobAttributionTableServesPastSourceAge records the gap the refresher
// closes: a table alone keeps serving an assignment after its source stops
// refreshing, until the table TTL.
func TestJobAttributionTableServesPastSourceAge(t *testing.T) {
	got, err := nvtelemetry.ReadSlurmAttributionSnapshot(refreshFixtureFS(t), refreshSource())
	if err != nil {
		t.Fatal(err)
	}
	table, err := nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := table.Refresh(got.Snapshot, 200); err != nil {
		t.Fatal(err)
	}
	maxAge := int64(5 * time.Second)
	if r := table.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200+maxAge+1); r.Assignment.State != "assigned" {
		t.Fatalf("table alone = %q %q, want assigned until its TTL", r.Assignment.State, r.Reason)
	}
}

type attributionHarness struct {
	clock *nvtelemetry.FakeClock
	table *nvtelemetry.JobAttributionTable
	mu    sync.Mutex
	fsys  fs.FS
	fail  error
	reads int
	r     *nvtelemetry.AttributionRefresher
}

func newAttributionHarness(t *testing.T, cfg nvtelemetry.RefreshConfig) *attributionHarness {
	t.Helper()
	h := &attributionHarness{clock: nvtelemetry.NewFakeClock(0, 200), fsys: refreshFixtureFS(t)}
	var err error
	h.table, err = nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	read := func(ctx context.Context) (nvtelemetry.AttributionRead, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.reads++
		if h.fail != nil {
			return nvtelemetry.AttributionRead{}, h.fail
		}
		return nvtelemetry.ReadSlurmAttributionSnapshot(h.fsys, refreshSource())
	}
	h.r, err = nvtelemetry.NewAttributionRefresher(h.table, read, h.clock, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { h.r.Close() })
	return h
}

func (h *attributionHarness) set(fsys fs.FS, fail error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.fsys, h.fail = fsys, fail
}

var refreshTestConfig = nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: time.Second, MaxAge: 5 * time.Second}

func TestAttributionRefresherReadErrorKeepsTableAndGoesStale(t *testing.T) {
	h := newAttributionHarness(t, refreshTestConfig)
	if h.r.Ready() {
		t.Fatal("ready before any read")
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State != "unknown" || r.Reason != "source_stale" {
		t.Fatalf("before first read = %q %q, want unknown source_stale", r.Assignment.State, r.Reason)
	}
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !h.r.Ready() {
		t.Fatal("not ready after successful read")
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State != "assigned" || r.Assignment.Owner.JobID != "42" {
		t.Fatalf("after read = %+v", r)
	}
	entries, last, fails := h.table.Stats()

	h.set(nil, errors.New("squeue unavailable"))
	h.clock.Advance(time.Second)
	if err := h.r.Step(context.Background()); err == nil {
		t.Fatal("step succeeded with failing read")
	}
	// A read error must not refresh or clear the table.
	if e, l, f := h.table.Stats(); e != entries || l != last || f != fails || e != 1 {
		t.Fatalf("table stats after read error = %d %d %d, want %d %d %d", e, l, f, entries, last, fails)
	}
	if !h.r.Ready() {
		t.Fatal("not ready within MaxAge of last success")
	}
	st := h.r.Status()
	if st.Cycles != 2 || st.Failures != 1 || st.LastError == "" || !st.HasSuccess || st.Records != 1 {
		t.Fatalf("status = %+v", st)
	}

	h.clock.Advance(5 * time.Second) // 6s since success; table TTL is 1h.
	if h.r.Ready() {
		t.Fatal("ready past MaxAge")
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State != "unknown" || r.Reason != "source_stale" {
		t.Fatalf("past MaxAge = %q %q, want unknown source_stale", r.Assignment.State, r.Reason)
	}
	if r := h.table.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State != "assigned" {
		t.Fatalf("table entry lost: %+v", r)
	}

	h.set(refreshFixtureFS(t), nil)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !h.r.Ready() {
		t.Fatal("not ready after recovery")
	}
}

func TestAttributionRefresherInvalidSnapshotKeepsTable(t *testing.T) {
	h := newAttributionHarness(t, refreshTestConfig)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	m := refreshFixtureFS(t)
	// The proof's interval is valid JSON but its state fails validation in
	// ReadSlurmAttributionSnapshot, so the read fails before table.Refresh.
	m["executions.json"] = &fstest.MapFile{Data: []byte(strings.Replace(string(m["executions.json"].Data), `"state":"assigned"`, `"state":"bogus"`, 1))}
	h.set(m, nil)
	h.clock.Advance(time.Second)
	if err := h.r.Step(context.Background()); err == nil {
		t.Fatal("step succeeded with invalid snapshot")
	}
	if e, _, _ := h.table.Stats(); e != 1 {
		t.Fatalf("entries = %d, want 1", e)
	}
}

// A snapshot that reads cleanly but fails table.Refresh counts as a failure
// and ages the source out.
func TestAttributionRefresherTableRefreshErrorGoesStale(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(0, 200)
	table, err := nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	var bad atomic.Bool
	read := func(context.Context) (nvtelemetry.AttributionRead, error) {
		if bad.Load() {
			// An allocation without a job ID passes the reader's contract
			// but is refused by table.Refresh.
			return nvtelemetry.AttributionRead{Snapshot: nvtelemetry.JobAttributionSnapshot{Allocations: []nvtelemetry.SlurmAllocation{{}}}}, nil
		}
		return nvtelemetry.ReadSlurmAttributionSnapshot(refreshFixtureFS(t), refreshSource())
	}
	r, err := nvtelemetry.NewAttributionRefresher(table, read, clock, refreshTestConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	bad.Store(true)
	for range 6 {
		clock.Advance(time.Second)
		if err := r.Step(context.Background()); err == nil || !strings.Contains(err.Error(), "missing allocation job id") {
			t.Fatalf("step = %v, want table refresh error", err)
		}
	}
	if st := r.Status(); st.Failures != 6 || r.Ready() {
		t.Fatalf("after 6s of table refresh errors: ready=%v status=%+v", r.Ready(), st)
	}
	if e, _, _ := table.Stats(); e != 1 {
		t.Fatalf("entries = %d, want 1", e)
	}
	if res := r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); res.Assignment.State != "unknown" || res.Reason != "source_stale" {
		t.Fatalf("resolve = %q %q, want unknown source_stale", res.Assignment.State, res.Reason)
	}
}

func TestAttributionRefresherMissingProcessWithholds(t *testing.T) {
	h := newAttributionHarness(t, refreshTestConfig)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	m := refreshFixtureFS(t)
	delete(m, "proc/1234/stat")
	h.set(m, nil)
	h.clock.Advance(time.Second)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st := h.r.Status(); st.Withheld != 1 || st.Records != 0 {
		t.Fatalf("status = %+v, want 1 withheld 0 records", st)
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State == "assigned" {
		t.Fatalf("withheld process still assigned: %+v", r)
	}
}

func TestAttributionRefresherPIDReuse(t *testing.T) {
	h := newAttributionHarness(t, refreshTestConfig)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	m := refreshFixtureFS(t)
	stat := strings.Replace(string(m["proc/1234/stat"].Data), " 1000 1000000 ", " 2000 1000000 ", 1)
	m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(stat)}
	h.set(m, nil)
	h.clock.Advance(time.Second)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State == "assigned" {
		t.Fatalf("previous generation still assigned after PID reuse: %+v", r)
	}
	if r := h.r.Resolve(refreshQueryProcess(2000), refreshDeviceUUID, 200); r.Assignment.State == "assigned" {
		t.Fatalf("new generation borrowed old proof: %+v", r)
	}
}

func TestAttributionRefresherBlockingReadTimesOut(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(0, 200)
	table, _ := nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	release := make(chan struct{})
	var mu sync.Mutex
	block := true
	read := func(ctx context.Context) (nvtelemetry.AttributionRead, error) {
		mu.Lock()
		b := block
		mu.Unlock()
		if b {
			<-release // ignores ctx, like a read stuck in the kernel
		}
		return nvtelemetry.ReadSlurmAttributionSnapshot(refreshFixtureFS(t), refreshSource())
	}
	r, err := nvtelemetry.NewAttributionRefresher(table, read, clock, nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: 20 * time.Millisecond, MaxAge: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	start := time.Now()
	if err := r.Step(context.Background()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("step = %v, want deadline exceeded", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Fatalf("step took %v", d)
	}
	if err := r.Step(context.Background()); err == nil || !strings.Contains(err.Error(), "still running") {
		t.Fatalf("overlapping step = %v, want refusal", err)
	}
	if r.Ready() {
		t.Fatal("ready without a completed read")
	}
	if e, _, _ := table.Stats(); e != 0 {
		t.Fatalf("abandoned read reached the table: %d entries", e)
	}
	mu.Lock()
	block = false
	mu.Unlock()
	close(release)
	deadline := time.Now().Add(5 * time.Second)
	for {
		err := r.Step(context.Background())
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("step after release: %v", err)
		}
		time.Sleep(time.Millisecond)
	}
	if !r.Ready() {
		t.Fatal("not ready after release")
	}
}

func TestAttributionRefresherRunClose(t *testing.T) {
	h := newAttributionHarness(t, nvtelemetry.RefreshConfig{Interval: time.Millisecond, Timeout: time.Second, MaxAge: time.Hour})
	done := make(chan error, 1)
	go func() { done <- h.r.Run(context.Background()) }()
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200)
				h.r.Status()
				h.r.Ready()
			}
		}()
	}
	for h.r.Status().Cycles < 3 {
		time.Sleep(time.Millisecond)
	}
	if err := h.r.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run = %v, want nil after close", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return after close")
	}
	cycles := h.r.Status().Cycles
	if err := h.r.Step(context.Background()); err == nil {
		t.Fatal("step after close succeeded")
	}
	if h.r.Status().Cycles != cycles {
		t.Fatal("cycle counted after close")
	}
	if r := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); r.Assignment.State != "unknown" || r.Reason != "source_closed" {
		t.Fatalf("after close = %q %q", r.Assignment.State, r.Reason)
	}
	if h.r.Ready() || !h.r.Status().Closed {
		t.Fatal("closed refresher reports ready")
	}
}

func TestAttributionRefresherRunContext(t *testing.T) {
	h := newAttributionHarness(t, nvtelemetry.RefreshConfig{Interval: time.Millisecond, Timeout: time.Second, MaxAge: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.r.Run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("run = %v, want canceled", err)
	}
}

func TestNewAttributionRefresherRefusals(t *testing.T) {
	table, _ := nvtelemetry.NewJobAttributionTable(1, 1)
	read := func(context.Context) (nvtelemetry.AttributionRead, error) { return nvtelemetry.AttributionRead{}, nil }
	clock := nvtelemetry.NewFakeClock(0, 0)
	tests := []struct {
		name  string
		table *nvtelemetry.JobAttributionTable
		read  func(context.Context) (nvtelemetry.AttributionRead, error)
		clock nvtelemetry.Clock
		cfg   nvtelemetry.RefreshConfig
	}{
		{"nil table", nil, read, clock, refreshTestConfig},
		{"nil read", table, nil, clock, refreshTestConfig},
		{"nil clock", table, read, nil, refreshTestConfig},
		{"zero interval", table, read, clock, nvtelemetry.RefreshConfig{Timeout: 1, MaxAge: 1}},
		{"zero timeout", table, read, clock, nvtelemetry.RefreshConfig{Interval: 1, MaxAge: 1}},
		{"zero max age", table, read, clock, nvtelemetry.RefreshConfig{Interval: 1, Timeout: 1}},
		{"negative max age", table, read, clock, nvtelemetry.RefreshConfig{Interval: 1, Timeout: 1, MaxAge: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := nvtelemetry.NewAttributionRefresher(tt.table, tt.read, tt.clock, tt.cfg); err == nil {
				t.Fatal("constructor succeeded")
			}
		})
	}
}

func TestAttributionRefresherCloseDuringRead(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(0, 200)
	table, err := nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	read := func(ctx context.Context) (nvtelemetry.AttributionRead, error) {
		close(started)
		<-ctx.Done()
		return nvtelemetry.AttributionRead{}, ctx.Err()
	}
	r, err := nvtelemetry.NewAttributionRefresher(table, read, clock, refreshTestConfig)
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() { errc <- r.Step(context.Background()) }()
	<-started
	r.Close()
	if err := <-errc; err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("step = %v, want closed", err)
	}
	if st := r.Status(); st.Failures != 0 || st.HasSuccess || !st.Closed {
		t.Fatalf("status after close during read = %+v", st)
	}
}
