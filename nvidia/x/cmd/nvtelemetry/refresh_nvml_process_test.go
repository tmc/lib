package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// These controls drive newAttributionSource with synthetic NVML queries and
// synthetic Slurm and proc fixtures. They load no NVML library and read no
// live scheduler, procfs or GPU.

const (
	liveGPUA = "GPU-11111111-2222-3333-4444-555555555555"
	liveGPUB = "GPU-22222222-3333-4444-5555-666666666666"

	liveCompute  = "nvml_compute_v3"
	liveGraphics = "nvml_graphics_v3"
	liveMPS      = "nvml_mps_compute_v3"
)

func liveFixtureFS() fstest.MapFS {
	file := func(s string) *fstest.MapFile { return &fstest.MapFile{Data: []byte(s)} }
	return fstest.MapFS{
		"squeue.json":       file(`{"jobs":[{"job_id":42,"name":"worker","job_state":["RUNNING"],"nodes":"node-1","tres_alloc_str":"cpu=4,gres/gpu=1"}],"errors":[]}`),
		"mappings.json":     file(`{"version":1,"mappings":[{"job_id":"42","step_id":"0","device":{"uuid":"` + liveGPUA + `","index":0},"sharing":"exclusive","allocation_id":"alloc-42","attempt":"attempt-1","authoritative":true,"clock_domain":"synthetic_unix_utc","clock_epoch":"fixture-epoch-1","start_nanos":100,"end_nanos":1000000000000}]}`),
		"executions.json":   file(`{"version":1,"assignments":[{"process":{"process":{"pid":1234,"generation":1000},"boot_id":"boot-1","node_id":"node-1","pid_namespace":"pid:[4026531836]","exec_generation":1,"has_exec_generation":true},"device":{"uuid":"` + liveGPUA + `","index":0},"owner":{"job_id":"42","allocation_id":"alloc-42","attempt":"attempt-1"},"source":"fixture_execution_proof","source_revision":"fixture-1","authoritative":true,"state":"assigned","clock_domain":"synthetic_unix_utc","clock_epoch":"fixture-epoch-1","start_nanos":100,"end_nanos":1000000000000,"sharing":"exclusive"}]}`),
		"proc/1234/stat":    file("1234 (python) S 1 1234 1234 0 -1 4194560 100 0 0 0 1 1 0 0 20 0 1 0 1000 1000000 100\n"),
		"proc/1234/status":  file("Name:\tpython\nNSpid:\t1234\n"),
		"proc/1234/cgroup":  file("0::/system.slice/slurmstepd.scope/job_42/step_0/task_0\n"),
		"proc/1234/environ": file("SLURM_JOB_ID=99\n"),
		"proc/1234/cmdline": file("python\x00train.py\x00"),
		"proc/5678/stat":    file("5678 (other) S 1 5678 5678 0 -1 4194560 100 0 0 0 1 1 0 0 20 0 1 0 3000 1000000 100\n"),
		"proc/5678/status":  file("Name:\tother\nNSpid:\t5678\n"),
		"proc/5678/cgroup":  file("0::/user.slice\n"),
	}
}

var liveReads = []string{"squeue.json", "mappings.json", "executions.json", "proc/1234/stat", "proc/1234/status", "proc/1234/cgroup"}

// liveFS records every open and refuses any name outside the shared files
// and PID 1234's stat, status and cgroup: directories, environ, cmdline and
// unrelated PIDs fail the read.
type liveFS struct {
	fsys  fs.FS
	mu    sync.Mutex
	opens []string
}

func (l *liveFS) Open(name string) (fs.File, error) {
	l.mu.Lock()
	l.opens = append(l.opens, name)
	l.mu.Unlock()
	if !slices.Contains(liveReads, name) {
		return nil, fmt.Errorf("live fs: refused open %s", name)
	}
	return l.fsys.Open(name)
}

func (l *liveFS) names() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.opens)
}

// liveQuery is a synthetic NVMLProcessQuery that serves fixed records per
// API. If block is non-nil, the compute query waits for it, ignoring any
// deadline, like a native call that does not return.
type liveQuery struct {
	mu      sync.Mutex
	records map[string][]nvml.ProcessInfo
	err     error
	calls   map[string]int
	block   chan struct{}
	started chan struct{}
}

func (q *liveQuery) query(source string, buf []nvml.ProcessInfo) (int, error) {
	q.mu.Lock()
	if q.calls == nil {
		q.calls = make(map[string]int)
	}
	q.calls[source]++
	block, started := q.block, q.started
	r, err := q.records[source], q.err
	q.mu.Unlock()
	if block != nil && source == liveCompute {
		close(started)
		<-block
	}
	if err != nil {
		return 0, err
	}
	if len(r) > len(buf) {
		return 0, fmt.Errorf("synthetic query: %w", nvml.ErrInsufficientSize)
	}
	return copy(buf, r), nil
}

func (q *liveQuery) ComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(liveCompute, buf)
}

func (q *liveQuery) GraphicsRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(liveGraphics, buf)
}

func (q *liveQuery) MPSComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(liveMPS, buf)
}

func (q *liveQuery) set(records map[string][]nvml.ProcessInfo, err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.records, q.err = records, err
}

func (q *liveQuery) callCounts() map[string]int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return maps.Clone(q.calls)
}

// liveNVML is a synthetic acquisition: it hands out the queries by UUID and
// counts acquisitions and releases.
type liveNVML struct {
	mu       sync.Mutex
	queries  map[string]*liveQuery
	acquired [][]string
	released int
	done     chan struct{} // closed by the first release
}

func newLiveNVML() *liveNVML {
	pid := func(pid uint32) nvml.ProcessInfo {
		return nvml.ProcessInfo{Pid: pid, UsedGpuMemory: ^uint64(0), GpuInstanceId: ^uint32(0), ComputeInstanceId: ^uint32(0)}
	}
	a, b := &liveQuery{}, &liveQuery{}
	// PID 1234 is seen by compute and MPS on A and by graphics on B.
	a.set(map[string][]nvml.ProcessInfo{liveCompute: {pid(1234)}, liveMPS: {pid(1234)}}, nil)
	b.set(map[string][]nvml.ProcessInfo{liveGraphics: {pid(1234)}}, nil)
	return &liveNVML{queries: map[string]*liveQuery{liveGPUA: a, liveGPUB: b}, done: make(chan struct{})}
}

func (n *liveNVML) acquire(uuids []string) ([]nvtelemetry.NVMLProcessDevice, func() error, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.acquired = append(n.acquired, slices.Clone(uuids))
	var devices []nvtelemetry.NVMLProcessDevice
	for _, uuid := range uuids {
		q, ok := n.queries[uuid]
		if !ok {
			return nil, nil, errors.New("synthetic acquisition: unknown device")
		}
		devices = append(devices, nvtelemetry.NVMLProcessDevice{Device: nvtelemetry.DeviceMetadata{UUID: uuid}, Query: q})
	}
	return devices, n.release, nil
}

func (n *liveNVML) release() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.released++
	if n.released == 1 {
		close(n.done)
	}
	return nil
}

func (n *liveNVML) counts() (acquired, released int) {
	n.mu.Lock()
	defer n.mu.Unlock()
	return len(n.acquired), n.released
}

func liveSnapshot(revision string) snapshotConfig {
	return snapshotConfig{SourceRevision: revision, Authoritative: true, NodeID: "node-1", ClockDomain: "synthetic_unix_utc", ClockEpoch: "fixture-epoch-1", StartNanos: 100, EndNanos: 1_000_000_000_000}
}

func liveAttribution(fsys fs.FS, n *liveNVML) *attributionConfig {
	nvmlSnapshot := liveSnapshot("fixture-nvml-1")
	return &attributionConfig{
		Root:             "fixtures",
		Squeue:           "squeue.json",
		Mappings:         "mappings.json",
		Executions:       "executions.json",
		ProcDir:          "proc",
		BootID:           "boot-1",
		NodeID:           "node-1",
		PIDNamespace:     "pid:[4026531836]",
		Snapshot:         liveSnapshot("fixture-squeue-1"),
		TableCapacity:    16,
		TableTTL:         time.Hour,
		NVMLUUIDs:        []string{liveGPUA, liveGPUB},
		NVMLSources:      []string{liveCompute, liveGraphics, liveMPS},
		NVMLMaxPerCall:   8,
		NVMLMaxRecords:   16,
		NVMLBootID:       "boot-1",
		NVMLPIDNamespace: "pid:[4026531836]",
		NVMLSnapshot:     &nvmlSnapshot,
		fsys:             fsys,
		acquireNVML:      n.acquire,
	}
}

var liveRefresh = nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: time.Second, MaxAge: 5 * time.Second}

func newLiveSource(t *testing.T, a *attributionConfig, cfg nvtelemetry.RefreshConfig) refreshSource {
	t.Helper()
	s, err := newAttributionSource(a, nvtelemetry.NewFakeClock(0, 200), cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.close() })
	return s
}

// The configured source names no PID. The production reader opens the proc
// files of the PID the queries observed, once although three facts on two
// devices name it, and the scheduler proof yields one attribution record.
func TestRefreshNVMLProcessesComposed(t *testing.T) {
	n, fsys := newLiveNVML(), &liveFS{fsys: liveFixtureFS()}
	a := liveAttribution(fsys, n)
	s := newLiveSource(t, a, liveRefresh)
	if acquired, released := n.counts(); acquired != 1 || released != 0 || !slices.Equal(n.acquired[0], []string{liveGPUA, liveGPUB}) {
		t.Fatalf("acquired %v, released %d", n.acquired, released)
	}
	if err := s.step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st := s.status(); !s.ready() || st.Records != 1 || st.Withheld != 0 || st.Failures != 0 {
		t.Fatalf("ready %v status %+v, want one attribution record", s.ready(), st)
	}
	if opens := fsys.names(); !slices.Equal(opens, liveReads) {
		t.Fatalf("opened %q, want %q", opens, liveReads)
	}
	for uuid, q := range n.queries {
		if c := q.callCounts(); len(c) != 3 || c[liveCompute] != 1 || c[liveGraphics] != 1 || c[liveMPS] != 1 {
			t.Fatalf("%s calls = %v, want one per selected API", uuid, c)
		}
	}
	if err := s.close(); err != nil {
		t.Fatal(err)
	}
	s.close()
	if _, released := n.counts(); released != 1 {
		t.Fatalf("released %d times, want once", released)
	}
	if err := s.step(context.Background()); err == nil {
		t.Fatal("step after close succeeded")
	}
	if c := n.queries[liveGPUA].callCounts(); c[liveCompute] != 1 {
		t.Fatalf("closed source queried: %v", c)
	}
}

func TestRefreshNVMLProcessesRefusals(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*attributionConfig)
	}{
		{"live and static pids", func(a *attributionConfig) { a.PIDs = []uint32{5678} }},
		{"only snapshot", func(a *attributionConfig) {
			a.NVMLUUIDs, a.NVMLSources, a.NVMLMaxPerCall, a.NVMLMaxRecords = nil, nil, 0, 0
		}},
		{"empty uuid list", func(a *attributionConfig) { a.NVMLUUIDs = []string{} }},
		{"empty uuid", func(a *attributionConfig) { a.NVMLUUIDs = []string{liveGPUA, ""} }},
		{"duplicate uuid", func(a *attributionConfig) { a.NVMLUUIDs = []string{liveGPUA, liveGPUA} }},
		{"no sources", func(a *attributionConfig) { a.NVMLSources = nil }},
		{"unknown source", func(a *attributionConfig) { a.NVMLSources = []string{"nvml_running_v3"} }},
		{"duplicate source", func(a *attributionConfig) { a.NVMLSources = []string{liveMPS, liveMPS} }},
		{"zero per call", func(a *attributionConfig) { a.NVMLMaxPerCall = 0 }},
		{"per call over cap", func(a *attributionConfig) { a.NVMLMaxPerCall = 65537 }},
		{"negative records", func(a *attributionConfig) { a.NVMLMaxRecords = -1 }},
		{"records over cap", func(a *attributionConfig) { a.NVMLMaxRecords = 65537 }},
		{"missing snapshot", func(a *attributionConfig) { a.NVMLSnapshot = nil }},
		{"unpinned snapshot", func(a *attributionConfig) { a.NVMLSnapshot.SourceRevision = "" }},
		{"node differs", func(a *attributionConfig) { a.NVMLSnapshot.NodeID = "node-2" }},
		{"boot differs", func(a *attributionConfig) { a.NVMLBootID = "boot-2" }},
		{"pid namespace differs", func(a *attributionConfig) { a.NVMLPIDNamespace = "pid:[1]" }},
		{"invalid static source", func(a *attributionConfig) { a.Squeue = "../squeue.json" }},
		{"zero table capacity", func(a *attributionConfig) { a.TableCapacity = 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n, fsys := newLiveNVML(), &liveFS{fsys: liveFixtureFS()}
			a := liveAttribution(fsys, n)
			tt.mutate(a)
			if _, err := newAttributionSource(a, nvtelemetry.NewFakeClock(0, 200), liveRefresh); err == nil {
				t.Fatal("source accepted")
			}
			if acquired, _ := n.counts(); acquired != 0 || len(fsys.names()) != 0 {
				t.Fatalf("refused configuration acquired NVML %d times or opened %q", acquired, fsys.names())
			}
		})
	}
	n := newLiveNVML()
	a := liveAttribution(&liveFS{fsys: liveFixtureFS()}, n)
	a.NVMLMaxPerCall, a.NVMLMaxRecords = 65536, 65536
	newLiveSource(t, a, liveRefresh)
	if acquired, _ := n.counts(); acquired != 1 {
		t.Fatalf("maximum caps acquired %d times", acquired)
	}
}

// The command decodes the live fields and refuses a mixed configuration
// before NVML is initialized: no acquisition seam is set here.
func TestRefreshNVMLProcessesConfig(t *testing.T) {
	snapshot := `{"source_revision":"r","node_id":"n","clock_domain":"d","clock_epoch":"e","start_nanos":1,"end_nanos":2}`
	live := fmt.Sprintf(`"nvml_uuids":["GPU-a"],"nvml_sources":["nvml_compute_v3"],"nvml_max_per_call":4,"nvml_max_records":8,"nvml_boot_id":"b","nvml_pid_namespace":"p","nvml_snapshot":%s`, snapshot)
	config := func(pids string) string {
		return fmt.Sprintf(`{"interval_nanos":1000000,"timeout_nanos":1000000000,"max_age_nanos":1000000000,"attribution":{"root":%q,"squeue":"s.json","mappings":"m.json","executions":"e.json","proc_dir":"proc",%s"boot_id":"b","node_id":"n","pid_namespace":"p","snapshot":%s,"table_capacity":16,"table_ttl_nanos":1000000000,%s}}`, t.TempDir(), pids, snapshot, live)
	}
	c, err := readRefreshConfig(writeRefreshFile(t, "refresh.json", config("")))
	if err != nil {
		t.Fatal(err)
	}
	a := c.Attribution
	sel, ok := a.nvmlSelection()
	if !ok || !slices.Equal(sel.UUIDs, []string{"GPU-a"}) || !slices.Equal(sel.Sources, []string{"nvml_compute_v3"}) || sel.MaxPerCall != 4 || sel.MaxRecords != 8 ||
		sel.BootID != "b" || sel.PIDNamespace != "p" || sel.Snapshot.SourceRevision != "r" || sel.Snapshot.NodeID != "n" || sel.Snapshot.EndNanos != 2 {
		t.Fatalf("selection = %+v, %v", sel, ok)
	}
	if err := sel.Validate(a.source()); err != nil {
		t.Fatalf("decoded selection: %v", err)
	}

	var out bytes.Buffer
	err = runWithin(t, []string{"refresh", "-config", writeRefreshFile(t, "refresh.json", config(`"pids":[1],`)), "-count", "1"}, &out, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "excludes configured pids") || out.Len() != 0 {
		t.Fatalf("mixed configuration = %v, output %q", err, out.String())
	}
	if _, err := readRefreshConfig(writeRefreshFile(t, "refresh.json", strings.Replace(config(""), `"nvml_boot_id"`, `"nvml_environ"`, 1))); err == nil {
		t.Fatal("unknown live field accepted")
	}
}

func TestRefreshNVMLProcessesEmptyAndFailure(t *testing.T) {
	n, fsys := newLiveNVML(), &liveFS{fsys: liveFixtureFS()}
	s := newLiveSource(t, liveAttribution(fsys, n), liveRefresh)
	if err := s.step(context.Background()); err != nil {
		t.Fatal(err)
	}

	// A failed query is not an empty success: the read fails before any file
	// is opened, and the last successful cycle stays current.
	n.queries[liveGPUB].set(nil, &nvml.CallError{Func: "nvmlDeviceGetGraphicsRunningProcesses_v3", Result: nvml.ErrNoPermission})
	err := s.step(context.Background())
	if !errors.Is(err, nvml.ErrNoPermission) {
		t.Fatalf("step = %v, want permission error", err)
	}
	if st := s.status(); !s.ready() || st.Failures != 1 || st.Records != 1 {
		t.Fatalf("after failed query: ready %v status %+v", s.ready(), st)
	}
	if opens := fsys.names(); len(opens) != len(liveReads) {
		t.Fatalf("failed query opened %q", opens[len(liveReads):])
	}

	// A successful empty query reads only the shared files and clears the
	// attribution records.
	n.queries[liveGPUA].set(nil, nil)
	n.queries[liveGPUB].set(nil, nil)
	if err := s.step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st := s.status(); !s.ready() || st.Records != 0 || st.Withheld != 0 {
		t.Fatalf("after empty query: ready %v status %+v", s.ready(), st)
	}
	if opens := fsys.names(); !slices.Equal(opens[len(liveReads):], liveReads[:3]) {
		t.Fatalf("empty query opened %q", opens[len(liveReads):])
	}
}

func TestRefreshNVMLProcessesOwnership(t *testing.T) {
	n, fsys := newLiveNVML(), &liveFS{fsys: liveFixtureFS()}
	a := liveAttribution(fsys, n)
	s := newLiveSource(t, a, liveRefresh)
	a.NVMLUUIDs[0] = "GPU-mutated"
	a.NVMLSources[0] = "nvml_mutated"
	a.NVMLSnapshot.NodeID = "node-9"
	a.PIDs = []uint32{5678}
	if err := s.step(context.Background()); err != nil {
		t.Fatalf("step after config mutation: %v", err)
	}
	if st := s.status(); st.Records != 1 {
		t.Fatalf("status = %+v", st)
	}

	// Static mode keeps its own copy of the configured PIDs and never
	// acquires NVML.
	static := liveAttribution(&liveFS{fsys: liveFixtureFS()}, n)
	static.NVMLUUIDs, static.NVMLSources, static.NVMLMaxPerCall, static.NVMLMaxRecords = nil, nil, 0, 0
	static.NVMLBootID, static.NVMLPIDNamespace, static.NVMLSnapshot = "", "", nil
	static.PIDs = []uint32{1234}
	static.acquireNVML = func([]string) ([]nvtelemetry.NVMLProcessDevice, func() error, error) {
		t.Fatal("static source acquired NVML")
		return nil, nil, nil
	}
	ss := newLiveSource(t, static, liveRefresh)
	static.PIDs[0] = 5678
	if err := ss.step(context.Background()); err != nil {
		t.Fatalf("static step after config mutation: %v", err)
	}
	if opens := static.fsys.(*liveFS).names(); !slices.Equal(opens, liveReads) {
		t.Fatalf("static source opened %q", opens)
	}
}

// A query that ignores its deadline is abandoned by Step, later Steps are
// refused instead of stacking goroutines, and NVML is released only after the
// query returns.
func TestRefreshNVMLProcessesTimeout(t *testing.T) {
	n, fsys := newLiveNVML(), &liveFS{fsys: liveFixtureFS()}
	q := n.queries[liveGPUA]
	q.block, q.started = make(chan struct{}), make(chan struct{})
	s := newLiveSource(t, liveAttribution(fsys, n), nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: 20 * time.Millisecond, MaxAge: time.Second})
	if err := s.step(context.Background()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("step = %v, want deadline exceeded", err)
	}
	<-q.started
	if err := s.step(context.Background()); err == nil || !strings.Contains(err.Error(), "still running") {
		t.Fatalf("overlapping step = %v, want refusal", err)
	}
	if err := s.close(); err != nil {
		t.Fatal(err)
	}
	if _, released := n.counts(); released != 0 {
		t.Fatal("NVML released while a query was running")
	}
	close(q.block)
	select {
	case <-n.done:
	case <-time.After(10 * time.Second):
		t.Fatal("NVML not released after the query returned")
	}
	if c := q.callCounts(); len(c) != 1 || c[liveCompute] != 1 {
		t.Fatalf("calls = %v, want only the abandoned compute query", c)
	}
	if c := n.queries[liveGPUB].callCounts(); len(c) != 0 {
		t.Fatalf("second device queried after cancellation: %v", c)
	}
	if len(fsys.names()) != 0 {
		t.Fatalf("abandoned read opened %q", fsys.names())
	}
	s.close()
	if _, released := n.counts(); released != 1 {
		t.Fatalf("released %d times, want once", released)
	}
}

func TestRefreshNVMLProcessesImports(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "refresh.go", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatal(err)
	}
	for _, imp := range f.Imports {
		if p, _ := strconv.Unquote(imp.Path.Value); p == "os/exec" {
			t.Fatal("refresh imports os/exec")
		}
	}
}
