package nvtelemetry_test

import (
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

const (
	nvmlCompute  = "nvml_compute_v3"
	nvmlGraphics = "nvml_graphics_v3"
	nvmlMPS      = "nvml_mps_compute_v3"

	secondDeviceUUID = "GPU-22222222-3333-4444-5555-666666666666"
)

// processQuery is a synthetic NVMLProcessQuery. It serves fixed records per
// API with the capacity rules of the nvml Into methods and counts calls.
type processQuery struct {
	mu      sync.Mutex
	records map[string][]nvml.ProcessInfo
	errs    map[string]error
	count   map[string]int // a count reported instead of the records
	calls   map[string]int
	lens    []int
}

func (q *processQuery) query(source string, buf []nvml.ProcessInfo) (int, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.calls == nil {
		q.calls = make(map[string]int)
	}
	q.calls[source]++
	q.lens = append(q.lens, len(buf))
	if err := q.errs[source]; err != nil {
		return 0, err
	}
	if n, ok := q.count[source]; ok {
		return n, nil
	}
	r := q.records[source]
	if len(r) > len(buf) {
		return 0, fmt.Errorf("synthetic query: %w", nvml.ErrInsufficientSize)
	}
	return copy(buf, r), nil
}

func (q *processQuery) ComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(nvmlCompute, buf)
}

func (q *processQuery) GraphicsRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(nvmlGraphics, buf)
}

func (q *processQuery) MPSComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error) {
	return q.query(nvmlMPS, buf)
}

func (q *processQuery) set(source string, records ...nvml.ProcessInfo) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.records == nil {
		q.records = make(map[string][]nvml.ProcessInfo)
	}
	q.records[source] = records
}

func (q *processQuery) fail(source string, err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.errs == nil {
		q.errs = make(map[string]error)
	}
	q.errs[source] = err
}

func (q *processQuery) callCounts() map[string]int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return maps.Clone(q.calls)
}

func unknownProcess(pid uint32) nvml.ProcessInfo {
	return nvml.ProcessInfo{Pid: pid, UsedGpuMemory: ^uint64(0), GpuInstanceId: ^uint32(0), ComputeInstanceId: ^uint32(0)}
}

func processSnapshot() nvtelemetry.EntitySnapshot {
	return nvtelemetry.EntitySnapshot{
		SourceRevision: "fixture-nvml-1",
		NodeID:         "node-1",
		ClockDomain:    "synthetic_unix_utc",
		ClockEpoch:     "fixture-epoch-1",
		StartNanos:     100,
		EndNanos:       1_000_000_000_000,
	}
}

func processSelection(uuids ...string) nvtelemetry.NVMLProcessSelection {
	return nvtelemetry.NVMLProcessSelection{
		UUIDs:        uuids,
		Sources:      []string{nvmlCompute, nvmlGraphics, nvmlMPS},
		MaxPerCall:   8,
		MaxRecords:   16,
		BootID:       "boot-1",
		PIDNamespace: "pid:[4026531836]",
		Snapshot:     processSnapshot(),
	}
}

func liveSource() nvtelemetry.SlurmAttributionSource {
	src := refreshSource()
	src.PIDs = nil
	return src
}

// twoDevices returns the fixture device, whose compute and MPS queries see PID
// 1234, and a second device whose graphics query sees PID 1234 too.
func twoDevices() ([]nvtelemetry.NVMLProcessDevice, *processQuery, *processQuery) {
	a, b := &processQuery{}, &processQuery{}
	a.set(nvmlCompute, nvml.ProcessInfo{Pid: 1234, UsedGpuMemory: 4096, GpuInstanceId: 1, ComputeInstanceId: 0}, nvml.ProcessInfo{Pid: 1234, UsedGpuMemory: 1024, GpuInstanceId: 2, ComputeInstanceId: 0})
	a.set(nvmlMPS, unknownProcess(1234))
	b.set(nvmlGraphics, unknownProcess(1234))
	return []nvtelemetry.NVMLProcessDevice{
		{Device: nvtelemetry.DeviceMetadata{UUID: refreshDeviceUUID, FieldCapabilities: []string{"processes"}}, Query: a},
		{Device: nvtelemetry.DeviceMetadata{UUID: secondDeviceUUID}, Query: b},
	}, a, b
}

// strictFS records every open and refuses any name not in allow, including
// directories, environ, cmdline and the files of unrelated PIDs.
type strictFS struct {
	recordingFS
	allow map[string]bool
}

var errRefusedOpen = errors.New("strict fs: refused open")

func newStrictFS(fsys fs.FS, pids ...uint32) *strictFS {
	s := &strictFS{recordingFS: recordingFS{fsys: fsys}, allow: map[string]bool{"squeue.json": true, "mappings.json": true, "executions.json": true}}
	for _, pid := range pids {
		for _, name := range []string{"stat", "status", "cgroup"} {
			s.allow["proc/"+strconv.FormatUint(uint64(pid), 10)+"/"+name] = true
		}
	}
	return s
}

func (s *strictFS) Open(name string) (fs.File, error) {
	if !s.allow[name] {
		s.recordingFS.mu.Lock()
		s.recordingFS.opens = append(s.recordingFS.opens, name)
		s.recordingFS.mu.Unlock()
		return nil, fmt.Errorf("%w: %s", errRefusedOpen, name)
	}
	return s.recordingFS.Open(name)
}

var procReads1234 = []string{"squeue.json", "mappings.json", "executions.json", "proc/1234/stat", "proc/1234/status", "proc/1234/cgroup"}

func TestQueryNVMLProcessesPreservesFacts(t *testing.T) {
	devices, a, b := twoDevices()
	facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), processSelection(refreshDeviceUUID, secondDeviceUUID), devices)
	if err != nil {
		t.Fatal(err)
	}
	type fact struct {
		device, source string
		pid            uint32
		mem            uint64
		hasMem         bool
		gi             uint32
		hasGI, hasCI   bool
	}
	var got []fact
	for _, f := range facts {
		got = append(got, fact{f.Device.UUID, f.ObservationSource, f.PID, f.UsedMemoryBytes, f.HasUsedMemory, f.GPUInstanceID, f.HasGPUInstanceID, f.HasComputeInstanceID})
		if f.Snapshot != processSnapshot() {
			t.Fatalf("fact snapshot = %+v", f.Snapshot)
		}
	}
	want := []fact{
		{refreshDeviceUUID, nvmlCompute, 1234, 4096, true, 1, true, true},
		{refreshDeviceUUID, nvmlCompute, 1234, 1024, true, 2, true, true},
		{refreshDeviceUUID, nvmlMPS, 1234, 0, false, 0, false, false},
		{secondDeviceUUID, nvmlGraphics, 1234, 0, false, 0, false, false},
	}
	if !slices.Equal(got, want) {
		t.Fatalf("facts =\n%+v\nwant\n%+v", got, want)
	}
	for _, q := range []*processQuery{a, b} {
		if c := q.callCounts(); len(c) != 3 || c[nvmlCompute] != 1 || c[nvmlGraphics] != 1 || c[nvmlMPS] != 1 {
			t.Fatalf("calls = %v, want one per selected API", c)
		}
		if !slices.Equal(q.lens, []int{8, 8, 8}) {
			t.Fatalf("buffer lengths = %v, want MaxPerCall", q.lens)
		}
	}

	// Facts own their device metadata and records.
	devices[0].Device.FieldCapabilities[0] = "mutated"
	a.set(nvmlCompute, unknownProcess(9))
	if facts[0].Device.FieldCapabilities[0] != "processes" || facts[0].PID != 1234 {
		t.Fatalf("fact shares caller storage: %+v", facts[0])
	}
}

func TestQueryNVMLProcessesSelectedSourcesOnly(t *testing.T) {
	devices, a, b := twoDevices()
	sel := processSelection(refreshDeviceUUID, secondDeviceUUID)
	sel.Sources = []string{nvmlGraphics}
	facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), sel, devices)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Device.UUID != secondDeviceUUID || facts[0].ObservationSource != nvmlGraphics {
		t.Fatalf("facts = %+v", facts)
	}
	for _, q := range []*processQuery{a, b} {
		if c := q.callCounts(); len(c) != 1 || c[nvmlGraphics] != 1 {
			t.Fatalf("calls = %v, want only graphics", c)
		}
	}
}

func TestQueryNVMLProcessesRefusesBeforeQuery(t *testing.T) {
	tests := []struct {
		name   string
		sel    func(*nvtelemetry.NVMLProcessSelection)
		device func([]nvtelemetry.NVMLProcessDevice) []nvtelemetry.NVMLProcessDevice
	}{
		{name: "no uuids", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.UUIDs = nil }},
		{name: "empty uuid", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.UUIDs = []string{refreshDeviceUUID, ""} }},
		{name: "duplicate uuid", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.UUIDs = []string{refreshDeviceUUID, refreshDeviceUUID} }},
		{name: "no sources", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Sources = nil }},
		{name: "empty source", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Sources = []string{""} }},
		{name: "unknown source", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Sources = []string{"nvml_compute_v2"} }},
		{name: "duplicate source", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Sources = []string{nvmlCompute, nvmlMPS, nvmlCompute} }},
		{name: "zero per call", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxPerCall = 0 }},
		{name: "negative per call", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxPerCall = -1 }},
		{name: "per call over cap", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxPerCall = 65537 }},
		{name: "zero records", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxRecords = 0 }},
		{name: "records over cap", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxRecords = 65537 }},
		{name: "unpinned snapshot", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Snapshot.SourceRevision = "" }},
		{name: "empty interval", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Snapshot.EndNanos = s.Snapshot.StartNanos }},
		{name: "missing boot", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.BootID = "" }},
		{name: "missing node", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Snapshot.NodeID = "" }},
		{name: "missing pid namespace", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.PIDNamespace = "" }},
		{name: "missing device", device: func(d []nvtelemetry.NVMLProcessDevice) []nvtelemetry.NVMLProcessDevice { return d[:1] }},
		{name: "reordered devices", device: func(d []nvtelemetry.NVMLProcessDevice) []nvtelemetry.NVMLProcessDevice {
			return []nvtelemetry.NVMLProcessDevice{d[1], d[0]}
		}},
		{name: "unselected device", device: func(d []nvtelemetry.NVMLProcessDevice) []nvtelemetry.NVMLProcessDevice {
			d[1].Device.UUID = "GPU-other"
			return d
		}},
		{name: "nil query", device: func(d []nvtelemetry.NVMLProcessDevice) []nvtelemetry.NVMLProcessDevice {
			d[1].Query = nil
			return d
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devices, a, b := twoDevices()
			sel := processSelection(refreshDeviceUUID, secondDeviceUUID)
			if tt.sel != nil {
				tt.sel(&sel)
			}
			if tt.device != nil {
				devices = tt.device(devices)
			}
			facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), sel, devices)
			if err == nil || facts != nil {
				t.Fatalf("query = %v, %v, want refusal", facts, err)
			}
			if len(a.callCounts())+len(b.callCounts()) != 0 {
				t.Fatalf("refused selection queried: %v %v", a.callCounts(), b.callCounts())
			}
		})
	}
}

func TestQueryNVMLProcessesCaps(t *testing.T) {
	records := func(n int) []nvml.ProcessInfo {
		var r []nvml.ProcessInfo
		for i := range n {
			r = append(r, unknownProcess(uint32(1000+i)))
		}
		return r
	}
	tests := []struct {
		name       string
		perCall    int
		total      int
		a, b       int // compute records on each device
		wantRecord int
		wantErr    error
	}{
		{name: "per call at cap", perCall: 3, total: 16, a: 3, b: 0, wantRecord: 3},
		{name: "per call over cap", perCall: 3, total: 16, a: 4, b: 0, wantErr: nvml.ErrInsufficientSize},
		{name: "aggregate at cap", perCall: 3, total: 5, a: 3, b: 2, wantRecord: 5},
		{name: "aggregate over cap", perCall: 3, total: 5, a: 3, b: 3},
		{name: "maximum caps", perCall: 65536, total: 65536, a: 3, b: 3, wantRecord: 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := &processQuery{}, &processQuery{}
			a.set(nvmlCompute, records(tt.a)...)
			b.set(nvmlCompute, records(tt.b)...)
			devices := []nvtelemetry.NVMLProcessDevice{
				{Device: nvtelemetry.DeviceMetadata{UUID: refreshDeviceUUID}, Query: a},
				{Device: nvtelemetry.DeviceMetadata{UUID: secondDeviceUUID}, Query: b},
			}
			sel := processSelection(refreshDeviceUUID, secondDeviceUUID)
			sel.Sources = []string{nvmlCompute}
			sel.MaxPerCall, sel.MaxRecords = tt.perCall, tt.total
			facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), sel, devices)
			if tt.wantRecord != 0 {
				if err != nil || len(facts) != tt.wantRecord {
					t.Fatalf("query = %d facts, %v, want %d", len(facts), err, tt.wantRecord)
				}
				return
			}
			if err == nil || facts != nil {
				t.Fatalf("query = %d facts, %v, want refusal without truncation", len(facts), err)
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			// An over-cap result is refused before any file is opened.
			rec := newStrictFS(refreshFixtureFS(t))
			if _, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), rec, liveSource(), sel, devices); err == nil {
				t.Fatal("read accepted an over-cap result")
			}
			if opens := rec.names(); len(opens) != 0 {
				t.Fatalf("refused query opened %q", opens)
			}
		})
	}
}

func TestQueryNVMLProcessesFailures(t *testing.T) {
	missing := &nvml.SymbolError{Symbol: "nvmlDeviceGetGraphicsRunningProcesses_v3"}
	tests := []struct {
		name    string
		prepare func(a, b *processQuery)
		wantErr error
	}{
		{"permission", func(a, b *processQuery) {
			b.fail(nvmlGraphics, &nvml.CallError{Func: "f", Result: nvml.ErrNoPermission})
		}, nvml.ErrNoPermission},
		{"unsupported", func(a, b *processQuery) { b.fail(nvmlGraphics, nvml.ErrNotSupported) }, nvml.ErrNotSupported},
		{"missing symbol", func(a, b *processQuery) { b.fail(nvmlGraphics, missing) }, missing},
		{"lost device", func(a, b *processQuery) { a.fail(nvmlMPS, nvml.ErrGpuIsLost) }, nvml.ErrGpuIsLost},
		{"count beyond buffer", func(a, b *processQuery) { a.count = map[string]int{nvmlCompute: 9} }, nil},
		{"negative count", func(a, b *processQuery) { a.count = map[string]int{nvmlCompute: -1} }, nil},
		{"zero pid", func(a, b *processQuery) { b.set(nvmlGraphics, unknownProcess(0)) }, nil},
		{"duplicate tuple", func(a, b *processQuery) { b.set(nvmlGraphics, unknownProcess(7), unknownProcess(7)) }, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devices, a, b := twoDevices()
			tt.prepare(a, b)
			facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), processSelection(refreshDeviceUUID, secondDeviceUUID), devices)
			if err == nil || facts != nil {
				t.Fatalf("query = %v, %v, want failure, not an empty success", facts, err)
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want errors.Is %v", err, tt.wantErr)
			}
		})
	}

	// A failure stops the remaining queries.
	devices, a, b := twoDevices()
	a.fail(nvmlCompute, nvml.ErrNoPermission)
	if _, err := nvtelemetry.QueryNVMLProcesses(context.Background(), processSelection(refreshDeviceUUID, secondDeviceUUID), devices); err == nil {
		t.Fatal("query succeeded")
	}
	if c := a.callCounts(); len(c) != 1 || len(b.callCounts()) != 0 {
		t.Fatalf("calls after failure: %v %v", c, b.callCounts())
	}

	// A canceled context refuses every query.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	devices, a, b = twoDevices()
	if _, err := nvtelemetry.QueryNVMLProcesses(ctx, processSelection(refreshDeviceUUID, secondDeviceUUID), devices); !errors.Is(err, context.Canceled) {
		t.Fatalf("query = %v, want canceled", err)
	}
	if len(a.callCounts())+len(b.callCounts()) != 0 {
		t.Fatal("canceled query called the device")
	}
}

// liveHarness drives the real reader and refresher with synthetic queries.
type liveHarness struct {
	clock   *nvtelemetry.FakeClock
	table   *nvtelemetry.JobAttributionTable
	fsys    *strictFS
	devices []nvtelemetry.NVMLProcessDevice
	a, b    *processQuery
	r       *nvtelemetry.AttributionRefresher
}

func newLiveHarness(t *testing.T, fsys *strictFS) *liveHarness {
	t.Helper()
	h := &liveHarness{clock: nvtelemetry.NewFakeClock(0, 200), fsys: fsys}
	h.devices, h.a, h.b = twoDevices()
	var err error
	h.table, err = nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	sel := processSelection(refreshDeviceUUID, secondDeviceUUID)
	read := func(ctx context.Context) (nvtelemetry.AttributionRead, error) {
		return nvtelemetry.ReadNVMLAttributionSnapshot(ctx, h.fsys, liveSource(), sel, h.devices)
	}
	h.r, err = nvtelemetry.NewAttributionRefresher(h.table, read, h.clock, refreshTestConfig)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { h.r.Close() })
	return h
}

func (h *liveHarness) resolve(generation uint64, device string) nvtelemetry.JobAttributionResult {
	return h.r.Resolve(refreshQueryProcess(generation), device, 200)
}

// The configured source names no PID. The queried PID selects which proc
// files the production reader opens, and the scheduler proof alone resolves
// the owner; the second device has no mapping and stays unknown.
func TestReadNVMLAttributionSnapshotResolves(t *testing.T) {
	h := newLiveHarness(t, newStrictFS(refreshFixtureFS(t), 1234))
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r := h.resolve(1000, refreshDeviceUUID); r.Assignment.State != "assigned" || r.Assignment.Owner.JobID != "42" || r.Assignment.Process.Process.PID != 1234 {
		t.Fatalf("resolve = %+v, want pid 1234 generation 1000 assigned to job 42", r)
	}
	if r := h.resolve(1000, secondDeviceUUID); r.Assignment.State == "assigned" {
		t.Fatalf("second device = %+v, want no relationship from an NVML observation", r)
	}
	if opens := h.fsys.names(); !slices.Equal(opens, procReads1234) {
		t.Fatalf("opened %q, want one proc read %q", opens, procReads1234)
	}
	for _, q := range []*processQuery{h.a, h.b} {
		if c := q.callCounts(); len(c) != 3 || c[nvmlCompute] != 1 || c[nvmlGraphics] != 1 || c[nvmlMPS] != 1 {
			t.Fatalf("calls = %v", c)
		}
	}
	if st := h.r.Status(); st.Records != 1 || st.Withheld != 0 {
		t.Fatalf("status = %+v", st)
	}
}

func TestReadNVMLAttributionSnapshotDoesNotModifySource(t *testing.T) {
	devices, _, _ := twoDevices()
	src := liveSource()
	read, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), newStrictFS(refreshFixtureFS(t), 1234), src, processSelection(refreshDeviceUUID, secondDeviceUUID), devices)
	if err != nil || len(read.Snapshot.Processes) != 1 {
		t.Fatalf("read = %+v, %v", read, err)
	}
	if src.PIDs != nil {
		t.Fatalf("source pids = %v, want untouched", src.PIDs)
	}
}

func TestReadNVMLAttributionSnapshotRefusals(t *testing.T) {
	tests := []struct {
		name string
		src  func(*nvtelemetry.SlurmAttributionSource)
		sel  func(*nvtelemetry.NVMLProcessSelection)
	}{
		{name: "configured pids", src: func(s *nvtelemetry.SlurmAttributionSource) { s.PIDs = []uint32{1234} }},
		{name: "invalid source", src: func(s *nvtelemetry.SlurmAttributionSource) { s.Squeue = "../squeue.json" }},
		{name: "boot differs", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.BootID = "boot-2" }},
		{name: "node differs", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.Snapshot.NodeID = "node-2" }},
		{name: "pid namespace differs", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.PIDNamespace = "pid:[1]" }},
		{name: "invalid selection", sel: func(s *nvtelemetry.NVMLProcessSelection) { s.MaxRecords = 65537 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devices, a, b := twoDevices()
			src, sel := liveSource(), processSelection(refreshDeviceUUID, secondDeviceUUID)
			if tt.src != nil {
				tt.src(&src)
			}
			if tt.sel != nil {
				tt.sel(&sel)
			}
			if err := sel.Validate(src); err == nil {
				t.Fatal("Validate accepted")
			}
			rec := newStrictFS(refreshFixtureFS(t), 1234)
			if _, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), rec, src, sel, devices); err == nil {
				t.Fatal("read accepted")
			}
			if len(a.callCounts())+len(b.callCounts()) != 0 || len(rec.names()) != 0 {
				t.Fatalf("refusal queried %v %v or opened %q", a.callCounts(), b.callCounts(), rec.names())
			}
		})
	}
	devices, a, _ := twoDevices()
	if _, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), nil, liveSource(), processSelection(refreshDeviceUUID, secondDeviceUUID), devices); err == nil || len(a.callCounts()) != 0 {
		t.Fatalf("nil file system = %v, calls %v", err, a.callCounts())
	}
	if err := processSelection(refreshDeviceUUID).Validate(liveSource()); err != nil {
		t.Fatalf("Validate = %v", err)
	}
}

func TestReadNVMLAttributionSnapshotEmptyClears(t *testing.T) {
	h := newLiveHarness(t, newStrictFS(refreshFixtureFS(t), 1234))
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{nvmlCompute, nvmlGraphics, nvmlMPS} {
		h.a.set(source)
		h.b.set(source)
	}
	h.fsys.allow = map[string]bool{"squeue.json": true, "mappings.json": true, "executions.json": true}
	h.clock.Advance(time.Second)
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r := h.resolve(1000, refreshDeviceUUID); r.Assignment.State != "unknown" || r.Reason != "missing_attribution" || r.Assignment.Owner.JobID != "" {
		t.Fatalf("after empty query = %+v, want unknown missing_attribution", r)
	}
	if e, _, _ := h.table.Stats(); e != 0 {
		t.Fatalf("entries = %d after empty query", e)
	}
	if st := h.r.Status(); st.Records != 0 || st.Withheld != 0 || !h.r.Ready() {
		t.Fatalf("status = %+v, want a successful empty cycle", st)
	}
	if opens := h.fsys.names(); !slices.Equal(opens[len(procReads1234):], procReads1234[:3]) {
		t.Fatalf("empty query opened %q", opens[len(procReads1234):])
	}
}

func TestReadNVMLAttributionSnapshotNoScheduler(t *testing.T) {
	m := refreshFixtureFS(t)
	m["squeue.json"] = &fstest.MapFile{Data: []byte(`{"jobs":[],"errors":[]}`)}
	m["mappings.json"] = &fstest.MapFile{Data: []byte(`{"version":1,"mappings":[]}`)}
	m["executions.json"] = &fstest.MapFile{Data: []byte(`{"version":1,"assignments":[]}`)}
	h := newLiveHarness(t, newStrictFS(m, 1234))
	if err := h.r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, device := range []string{refreshDeviceUUID, secondDeviceUUID} {
		if r := h.resolve(1000, device); r.Assignment.State != "unknown" || r.Assignment.Owner != (nvtelemetry.EntityOwner{}) {
			t.Fatalf("%s without scheduler proof = %+v, want unknown", device, r)
		}
	}
	if opens := h.fsys.names(); !slices.Equal(opens, procReads1234) {
		t.Fatalf("opened %q", opens)
	}
}

func TestReadNVMLAttributionSnapshotFailedQueryKeepsTable(t *testing.T) {
	for _, err := range []error{nvml.ErrNoPermission, nvml.ErrNotSupported, &nvml.SymbolError{Symbol: "nvmlDeviceGetMPSComputeRunningProcesses_v3"}} {
		t.Run(err.Error(), func(t *testing.T) {
			h := newLiveHarness(t, newStrictFS(refreshFixtureFS(t), 1234))
			if err := h.r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			h.a.fail(nvmlMPS, err)
			h.clock.Advance(time.Second)
			if got := h.r.Step(context.Background()); !errors.Is(got, err) {
				t.Fatalf("step = %v, want %v", got, err)
			}
			if e, _, _ := h.table.Stats(); e != 1 {
				t.Fatalf("entries = %d, want the last successful table", e)
			}
			if r := h.resolve(1000, refreshDeviceUUID); r.Assignment.State != "assigned" {
				t.Fatalf("failed query cleared attribution: %+v", r)
			}
			if st := h.r.Status(); st.Failures != 1 || st.Records != 1 {
				t.Fatalf("status = %+v", st)
			}
			if opens := h.fsys.names(); len(opens) != len(procReads1234) {
				t.Fatalf("failed query opened files: %q", opens[len(procReads1234):])
			}
		})
	}
}

func TestReadNVMLAttributionSnapshotProcessChanges(t *testing.T) {
	stat := func(m fstest.MapFS) string { return string(m["proc/1234/stat"].Data) }
	tests := []struct {
		name     string
		pid      uint32
		mutate   func(fstest.MapFS)
		wantErr  bool
		withheld int
		reused   uint64 // a later generation that must not resolve
	}{
		{name: "pid gone", pid: 77, withheld: 1},
		{name: "zombie", pid: 1234, withheld: 1, mutate: func(m fstest.MapFS) {
			m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(strings.Replace(stat(m), ") S ", ") Z ", 1))}
		}},
		{name: "stat pid mismatch", pid: 1234, wantErr: true, mutate: func(m fstest.MapFS) {
			m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(strings.Replace(stat(m), "1234 (", "999 (", 1))}
		}},
		{name: "pid reuse", pid: 1234, reused: 2000, mutate: func(m fstest.MapFS) {
			m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(strings.Replace(stat(m), " 1000 1000000 ", " 2000 1000000 ", 1))}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := refreshFixtureFS(t)
			h := newLiveHarness(t, newStrictFS(m, 1234, 77))
			if err := h.r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if tt.mutate != nil {
				tt.mutate(m)
			}
			h.a.set(nvmlCompute, unknownProcess(tt.pid))
			h.a.set(nvmlMPS)
			h.b.set(nvmlGraphics)
			h.clock.Advance(time.Second)
			err := h.r.Step(context.Background())
			if tt.wantErr {
				if err == nil {
					t.Fatal("step succeeded")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if st := h.r.Status(); st.Withheld != tt.withheld {
				t.Fatalf("status = %+v, want %d withheld", st, tt.withheld)
			}
			if r := h.resolve(1000, refreshDeviceUUID); r.Assignment.State == "assigned" {
				t.Fatalf("earlier generation still assigned: %+v", r)
			}
			if tt.reused != 0 {
				if r := h.resolve(tt.reused, refreshDeviceUUID); r.Assignment.State == "assigned" {
					t.Fatalf("new generation borrowed the old proof: %+v", r)
				}
			}
		})
	}
}

func TestProcessSourceImports(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "process_source.go", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatal(err)
	}
	for _, imp := range f.Imports {
		if p, _ := strconv.Unquote(imp.Path.Value); p == "os" || p == "os/exec" || p == "syscall" {
			t.Fatalf("process source imports %s", p)
		}
	}
}
