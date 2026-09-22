package cudacapture_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/pprof/profile"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestActivityRecordUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantGrid  [3]uint32
		wantBlock [3]uint32
		wantRegs  uint32
		wantSmem  uint32
	}{
		{
			name:      "array fields",
			input:     `{"subtype":"kernel","valid":true,"grid":[128,2,1],"block":[32,4,1],"registers_per_thread":48,"dynamic_shared_memory":2048}`,
			wantGrid:  [3]uint32{128, 2, 1},
			wantBlock: [3]uint32{32, 4, 1},
			wantRegs:  48,
			wantSmem:  2048,
		},
		{
			name:      "coordinate and alias fields",
			input:     `{"subtype":"kernel","valid":true,"grid_x":64,"grid_y":1,"grid_z":1,"block_x":256,"block_y":1,"block_z":1,"registers":32,"smem":1024}`,
			wantGrid:  [3]uint32{64, 1, 1},
			wantBlock: [3]uint32{256, 1, 1},
			wantRegs:  32,
			wantSmem:  1024,
		},
		{
			name:      "grid_dim and block_dim arrays",
			input:     `{"subtype":"kernel","valid":true,"grid_dim":[16,1,1],"block_dim":[64,1,1],"registers_per_thread":16,"dynamic_shared_memory":512}`,
			wantGrid:  [3]uint32{16, 1, 1},
			wantBlock: [3]uint32{64, 1, 1},
			wantRegs:  16,
			wantSmem:  512,
		},
		{
			name:      "memcpy without geometry",
			input:     `{"subtype":"memcpy","valid":true,"bytes":4096}`,
			wantGrid:  [3]uint32{0, 0, 0},
			wantBlock: [3]uint32{0, 0, 0},
			wantRegs:  0,
			wantSmem:  0,
		},
		{
			name:      "legacy kernel without geometry",
			input:     `{"subtype":"kernel","valid":true,"start":100,"end":200}`,
			wantGrid:  [3]uint32{0, 0, 0},
			wantBlock: [3]uint32{0, 0, 0},
			wantRegs:  0,
			wantSmem:  0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var rec cudacapture.ActivityRecord
			if err := json.Unmarshal([]byte(tc.input), &rec); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}
			if rec.GridDim != tc.wantGrid {
				t.Errorf("GridDim = %v, want %v", rec.GridDim, tc.wantGrid)
			}
			if rec.BlockDim != tc.wantBlock {
				t.Errorf("BlockDim = %v, want %v", rec.BlockDim, tc.wantBlock)
			}
			if rec.RegistersPerThread != tc.wantRegs {
				t.Errorf("RegistersPerThread = %d, want %d", rec.RegistersPerThread, tc.wantRegs)
			}
			if rec.DynamicSharedMemory != tc.wantSmem {
				t.Errorf("DynamicSharedMemory = %d, want %d", rec.DynamicSharedMemory, tc.wantSmem)
			}
		})
	}
}

func TestActivityRecordMarshalJSON(t *testing.T) {
	tests := []struct {
		name         string
		record       cudacapture.ActivityRecord
		wantContains []string
		wantOmit     []string
	}{
		{
			name: "kernel serializes geometry and register allocations",
			record: cudacapture.ActivityRecord{
				Subtype:             "kernel",
				Name:                "matmul",
				Valid:               true,
				GridDim:             [3]uint32{128, 1, 1},
				BlockDim:            [3]uint32{256, 1, 1},
				RegistersPerThread:  40,
				DynamicSharedMemory: 4096,
			},
			wantContains: []string{
				`"grid":[128,1,1]`,
				`"block":[256,1,1]`,
				`"registers_per_thread":40`,
				`"dynamic_shared_memory":4096`,
			},
		},
		{
			name: "memcpy omits geometry and register allocations",
			record: cudacapture.ActivityRecord{
				Subtype: "memcpy",
				Valid:   true,
				Bytes:   1024,
			},
			wantOmit: []string{
				`"grid"`,
				`"block"`,
				`"registers_per_thread"`,
				`"dynamic_shared_memory"`,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.record)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}
			s := string(b)
			for _, exp := range tc.wantContains {
				if !contains(s, exp) {
					t.Errorf("JSON output %s does not contain %s", s, exp)
				}
			}
			for _, omit := range tc.wantOmit {
				if contains(s, omit) {
					t.Errorf("JSON output %s contains omitted key %s", s, omit)
				}
			}
		})
	}
}

func TestAttachKernelLabels(t *testing.T) {
	tests := []struct {
		name       string
		record     cudacapture.ActivityRecord
		initial    map[string][]string
		wantLabels map[string][]string
	}{
		{
			name: "kernel attaches launch geometry and registers",
			record: cudacapture.ActivityRecord{
				Subtype:             "kernel",
				GridDim:             [3]uint32{64, 2, 1},
				BlockDim:            [3]uint32{32, 1, 1},
				RegistersPerThread:  32,
				DynamicSharedMemory: 1024,
			},
			initial: map[string][]string{"session": {"test"}},
			wantLabels: map[string][]string{
				"session":   {"test"},
				"grid":      {"64,2,1"},
				"block":     {"32,1,1"},
				"registers": {"32"},
				"smem":      {"1024"},
			},
		},
		{
			name: "memcpy does not attach kernel labels",
			record: cudacapture.ActivityRecord{
				Subtype: "memcpy",
			},
			initial: map[string][]string{"session": {"test"}},
			wantLabels: map[string][]string{
				"session": {"test"},
			},
		},
		{
			name: "nil labels map does not panic",
			record: cudacapture.ActivityRecord{
				Subtype: "kernel",
			},
			initial:    nil,
			wantLabels: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cudacapture.AttachKernelLabels(tc.initial, tc.record)
			if tc.wantLabels != nil && !reflect.DeepEqual(tc.initial, tc.wantLabels) {
				t.Errorf("labels = %v, want %v", tc.initial, tc.wantLabels)
			}
		})
	}
}

func TestBuildActivityProfile(t *testing.T) {
	activities := []cudacapture.ActivityRecord{
		{
			Subtype:             "kernel",
			Name:                "vectorAdd",
			Valid:               true,
			Start:               1000,
			End:                 2000,
			Device:              0,
			GridDim:             [3]uint32{256, 1, 1},
			BlockDim:            [3]uint32{64, 1, 1},
			RegistersPerThread:  24,
			DynamicSharedMemory: 512,
		},
		{
			Subtype:  "memcpy",
			Valid:    true,
			Start:    3000,
			End:      3500,
			Bytes:    8192,
			CopyKind: 1,
			Device:   0,
		},
	}

	p := cudacapture.BuildActivityProfile("session-42", 100, activities, nil)
	if p == nil {
		t.Fatal("BuildActivityProfile returned nil")
	}
	if len(p.Sample) != 2 {
		t.Fatalf("sample count = %d, want 2", len(p.Sample))
	}

	// First sample is kernel
	kSample := p.Sample[0]
	if kSample.Value[0] != 1000 { // kernel_time = End - Start
		t.Errorf("kernel_time = %d, want 1000", kSample.Value[0])
	}
	wantKernelLabels := map[string]string{
		"subtype":   "kernel",
		"name":      "vectorAdd",
		"grid":      "256,1,1",
		"block":     "64,1,1",
		"registers": "24",
		"smem":      "512",
	}
	for k, want := range wantKernelLabels {
		got := kSample.Label[k]
		if len(got) != 1 || got[0] != want {
			t.Errorf("kernel label %s = %v, want %s", k, got, want)
		}
	}

	// Second sample is memcpy
	mSample := p.Sample[1]
	if mSample.Value[1] != 500 { // copy_time = End - Start
		t.Errorf("copy_time = %d, want 500", mSample.Value[1])
	}
	if mSample.Value[2] != 8192 { // copy_bytes
		t.Errorf("copy_bytes = %d, want 8192", mSample.Value[2])
	}
	for _, unwanted := range []string{"grid", "block", "registers", "smem"} {
		if got, exists := mSample.Label[unwanted]; exists {
			t.Errorf("memcpy sample unexpectedly contains label %s = %v", unwanted, got)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (len(s) > 0 && len(substr) > 0 && containsSubstr(s, substr)))
}

func containsSubstr(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// TestBuildActivityProfileDefaultSampleType pins the sample type a viewer
// opens. pprof defaults to the last type when none is named, which here is
// "activity" -- a launch count, not a duration -- so every consumer had to
// pass -sample_index=0 to see device time.
func TestBuildActivityProfileDefaultSampleType(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session-1", 1, []cudacapture.ActivityRecord{{
		Subtype: "kernel",
		Name:    "vectorAdd",
		Valid:   true,
		Start:   1000,
		End:     2000,
	}}, nil)
	if p.DefaultSampleType != "kernel_time" {
		t.Errorf("DefaultSampleType = %q, want %q", p.DefaultSampleType, "kernel_time")
	}
	// A default naming a type the profile does not carry is worse than none:
	// pprof reports an error instead of silently picking the wrong column.
	index := -1
	for i, st := range p.SampleType {
		if st.Type == p.DefaultSampleType {
			index = i
			break
		}
	}
	if index < 0 {
		t.Fatalf("DefaultSampleType %q is not among the configured sample types", p.DefaultSampleType)
	}
	if p.Sample[0].Value[index] != 1000 {
		t.Errorf("default sample type carries %d, want the kernel duration 1000", p.Sample[0].Value[index])
	}
	if err := p.CheckValid(); err != nil {
		t.Errorf("CheckValid: %v", err)
	}
}

// A kernel that ran as a graph node carries graph attribution into its labels,
// and one that did not carries none. The distinction matters because a
// zero-valued graph_id label would make every ordinary launch look graph-like
// to a consumer that tests for the label's presence.
func TestAttachKernelLabelsGraphAttribution(t *testing.T) {
	tests := []struct {
		name       string
		record     cudacapture.ActivityRecord
		wantLabels map[string]string
		wantAbsent []string
	}{
		{
			name: "graph node carries both identifiers",
			record: cudacapture.ActivityRecord{
				Subtype: "kernel", Name: "bump", GraphID: 42, GraphNodeID: 987654321,
			},
			wantLabels: map[string]string{"graph_id": "42", "graph_node_id": "987654321"},
		},
		{
			name:       "ordinary launch carries neither",
			record:     cudacapture.ActivityRecord{Subtype: "kernel", Name: "bump"},
			wantAbsent: []string{"graph_id", "graph_node_id"},
		},
		{
			name: "a node ID without a graph ID still records what it has",
			record: cudacapture.ActivityRecord{
				Subtype: "kernel", Name: "bump", GraphNodeID: 7,
			},
			wantLabels: map[string]string{"graph_node_id": "7"},
			wantAbsent: []string{"graph_id"},
		},
		{
			name:       "memcpy is not a kernel and gets no kernel labels",
			record:     cudacapture.ActivityRecord{Subtype: "memcpy", GraphID: 42},
			wantAbsent: []string{"graph_id", "graph_node_id", "registers", "smem"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			labels := map[string][]string{}
			cudacapture.AttachKernelLabels(labels, tt.record)
			for key, want := range tt.wantLabels {
				got, ok := labels[key]
				if !ok {
					t.Errorf("label %q missing, want %q", key, want)
					continue
				}
				if len(got) != 1 || got[0] != want {
					t.Errorf("label %q = %v, want [%q]", key, got, want)
				}
			}
			for _, key := range tt.wantAbsent {
				if got, ok := labels[key]; ok {
					t.Errorf("label %q = %v, want it absent", key, got)
				}
			}
		})
	}
}

// TestBuildActivityProfileSkipsNVTXMarkers covers the population boundary. The
// profile's four sample types are three device measurements and a count of the
// activities they were measured over, so the count has to count the same
// population. An NVTX marker is a host annotation carried on the activity
// record kind: it contributes zero to kernel_time, zero to copy_time and zero
// to copy_bytes, so including it would leave "activity" counting a population
// the three durations were never measured over.
func TestBuildActivityProfileSkipsNVTXMarkers(t *testing.T) {
	activities := []cudacapture.ActivityRecord{
		{Subtype: "kernel", Name: "k", Valid: true, Start: 1000, End: 3000},
		{Subtype: "marker", Name: "prefill", Valid: true, Start: 500, End: 500},
		{Subtype: "marker", Name: "prefill", Valid: true, Start: 9000, End: 9000},
	}
	p := cudacapture.BuildActivityProfile("session", 4242, activities, nil)
	if len(p.Sample) != 1 {
		t.Fatalf("got %d samples, want 1: only the kernel is device activity", len(p.Sample))
	}
	if got := p.Sample[0].Value[3]; got != 1 {
		t.Errorf("activity count = %d, want 1", got)
	}
	for _, s := range p.Sample {
		for _, v := range s.Label["subtype"] {
			if v == "marker" {
				t.Errorf("a marker reached the profile as a sample; it carries no device time, so it would count in the activity dimension and nowhere else")
			}
		}
	}
}

// TestBuildActivityProfileSkipsCaptureRangeEdges guards the capture-range half
// of the same filter. A range edge records when the producer started and
// stopped collecting; it contributes to the activity count dimension and to
// neither duration beside it, so admitting one inflates the count alone.
//
// The edges are marked valid deliberately: a fixture matching what the
// producer emits today carries no Valid field, would be dropped by the
// validity check before the subtype is ever consulted, and would pass with the
// filter deleted.
func TestBuildActivityProfileSkipsCaptureRangeEdges(t *testing.T) {
	activities := []cudacapture.ActivityRecord{
		{Subtype: "kernel", Name: "k", Valid: true, Start: 1000, End: 3000},
		{Subtype: "capture_range", Name: "cudaProfilerStart", Valid: true, Start: 500, End: 600},
		{Subtype: "capture_range", Name: "cudaProfilerStop", Valid: true, Start: 9000, End: 9100},
	}
	p := cudacapture.BuildActivityProfile("session", 4242, activities, nil)
	if len(p.Sample) != 1 {
		t.Fatalf("got %d samples, want 1: only the kernel is device activity", len(p.Sample))
	}
	if got := p.Sample[0].Value[3]; got != 1 {
		t.Errorf("activity count = %d, want 1", got)
	}
	for _, s := range p.Sample {
		for _, v := range s.Label["subtype"] {
			if v == "capture_range" {
				t.Errorf("a capture-range edge reached the profile as a sample; it is the producer's own bookkeeping, not device work")
			}
		}
	}
}

// nvtxOpen and nvtxClose build the two records CUPTI reports for one NVTX
// range pushed on host thread tid. A range is never a single record: the open
// and the close are separate, share a marker id, and carry the timestamp in
// both Start and End.
func nvtxOpen(tid, id uint32, name string, at uint64) cudacapture.ActivityRecord {
	return cudacapture.ActivityRecord{Subtype: "marker", Name: name, Valid: true, Start: at, End: at, MarkerID: id, MarkerFlags: 1 << 1, ThreadID: tid, ProcessID: 1}
}

func nvtxClose(tid, id uint32, name string, at uint64) cudacapture.ActivityRecord {
	return cudacapture.ActivityRecord{Subtype: "marker", Name: name, Valid: true, Start: at, End: at, MarkerID: id, MarkerFlags: 1 << 2, ThreadID: tid, ProcessID: 1}
}

// kernel is device work with correlation corr that ran on the device over
// [start, end).
func kernel(name string, corr uint32, start, end uint64) cudacapture.ActivityRecord {
	return cudacapture.ActivityRecord{Subtype: "kernel", Name: name, Valid: true, Start: start, End: end, Correlation: corr}
}

// launchCall is the host API call with correlation corr, made on thread tid
// of process 1 at host time at.
func launchCall(tid, corr uint32, at uint64) cudacapture.APIRecord {
	return cudacapture.APIRecord{Domain: "runtime", Start: at, End: at + 5, ProcessID: 1, ThreadID: tid, Correlation: corr}
}

func labelOf(t *testing.T, p *profile.Profile, kernel, key string) []string {
	t.Helper()
	for _, s := range p.Sample {
		if len(s.Label["name"]) == 1 && s.Label["name"][0] == kernel {
			return s.Label[key]
		}
	}
	t.Fatalf("no sample named %q", kernel)
	return nil
}

func nvtxLabelOf(t *testing.T, p *profile.Profile, kernel string) []string {
	t.Helper()
	return labelOf(t, p, kernel, "nvtx")
}

// TestBuildActivityProfileAssignsNVTXByLaunch pins the join that places
// device work in an NVTX range. A range is pushed and popped on the host
// around the launch call; the device runs the work whenever its queue gets
// there. So the range is the one open on the launching thread when the launch
// call started, found through the correlation id the kernel shares with that
// call, and never the one that happens to be open when the device started.
func TestBuildActivityProfileAssignsNVTXByLaunch(t *testing.T) {
	type want struct {
		nvtx       []string
		unassigned string
	}
	tests := []struct {
		name       string
		activities []cudacapture.ActivityRecord
		apis       []cudacapture.APIRecord
		want       map[string]want
	}{
		{
			// The device start of k falls long after the range closed, and
			// that of late falls inside a range opened after it was launched.
			// By device time both would be wrong.
			name: "device start after the host range closed",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "step", 100),
				nvtxClose(7, 1, "step", 200),
				nvtxOpen(7, 2, "next", 300),
				nvtxClose(7, 2, "next", 400),
				kernel("k", 1, 5000, 5100),
				kernel("late", 2, 350, 360),
			},
			apis: []cudacapture.APIRecord{launchCall(7, 1, 150), launchCall(7, 2, 250)},
			want: map[string]want{
				"k":    {nvtx: []string{"step"}},
				"late": {},
			},
		},
		{
			// Thread 7 and thread 8 each hold a range over the same interval.
			// Each kernel belongs to its own launching thread's range only.
			name: "two threads with overlapping ranges",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "A", 100),
				nvtxOpen(8, 1, "B", 150),
				kernel("k7", 1, 320, 330),
				kernel("k8", 2, 330, 340),
				nvtxClose(7, 1, "A", 500),
				nvtxClose(8, 1, "B", 600),
			},
			apis: []cudacapture.APIRecord{launchCall(7, 1, 300), launchCall(8, 2, 310)},
			want: map[string]want{
				"k7": {nvtx: []string{"A"}},
				"k8": {nvtx: []string{"B"}},
			},
		},
		{
			// The path is outermost first, so the innermost range is last.
			name: "nested ranges",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "outer", 100),
				nvtxOpen(7, 2, "inner", 200),
				nvtxClose(7, 2, "inner", 500),
				nvtxClose(7, 1, "outer", 1000),
				kernel("in", 1, 2000, 2100),
				kernel("out", 2, 2200, 2300),
			},
			apis: []cudacapture.APIRecord{launchCall(7, 1, 300), launchCall(7, 2, 600)},
			want: map[string]want{
				"in":  {nvtx: []string{"outer", "inner"}},
				"out": {nvtx: []string{"outer"}},
			},
		},
		{
			// Both kernels start on the device inside the range. Neither has
			// a launch to join, so neither may be placed in it.
			name: "no matching api record",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "step", 100),
				kernel("orphan", 99, 150, 160),
				kernel("uncorrelated", 0, 170, 180),
				nvtxClose(7, 1, "step", 500),
			},
			apis: []cudacapture.APIRecord{launchCall(7, 1, 120)},
			want: map[string]want{
				"orphan":       {unassigned: "no api record"},
				"uncorrelated": {unassigned: "no correlation id"},
			},
		},
		{
			// A graph node kernel carries the correlation of the graph launch
			// that replayed it. With that launch recorded it is placed; a node
			// whose launch was not recorded is not guessed at.
			name: "graph nodes",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "decode", 100),
				nvtxClose(7, 1, "decode", 200),
				{Subtype: "kernel", Name: "node", Valid: true, Start: 900, End: 950, Correlation: 3, GraphID: 5, GraphNodeID: 1},
				{Subtype: "kernel", Name: "lost", Valid: true, Start: 150, End: 160, Correlation: 4, GraphID: 5, GraphNodeID: 2},
			},
			apis: []cudacapture.APIRecord{launchCall(7, 3, 150)},
			want: map[string]want{
				"node": {nvtx: []string{"decode"}},
				"lost": {unassigned: "no api record"},
			},
		},
		{
			// The runtime call and the driver call beneath it share the id;
			// the runtime call is the outer one and starts first.
			name: "runtime and driver records for one launch",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "step", 100),
				nvtxClose(7, 1, "step", 160),
				kernel("k", 1, 900, 950),
			},
			apis: []cudacapture.APIRecord{
				{Domain: "driver", Start: 170, End: 175, ProcessID: 1, ThreadID: 7, Correlation: 1},
				{Domain: "runtime", Start: 150, End: 180, ProcessID: 1, ThreadID: 7, Correlation: 1},
			},
			want: map[string]want{"k": {nvtx: []string{"step"}}},
		},
		{
			name: "one correlation on two threads",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "A", 100),
				nvtxClose(7, 1, "A", 500),
				kernel("k", 1, 900, 950),
			},
			apis: []cudacapture.APIRecord{launchCall(7, 1, 150), launchCall(8, 1, 160)},
			want: map[string]want{"k": {unassigned: "api records on different threads"}},
		},
		{
			// Correlation ids are per process. Process 2 reuses id 1 on its
			// own thread 8; its kernel joins its own call, not process 1's.
			name: "worker process",
			activities: []cudacapture.ActivityRecord{
				nvtxOpen(7, 1, "A", 100),
				nvtxClose(7, 1, "A", 500),
				{Subtype: "marker", Name: "B", Valid: true, Start: 100, End: 100, MarkerID: 1, MarkerFlags: 1 << 1, ThreadID: 8, ProcessID: 2},
				{Subtype: "marker", Name: "B", Valid: true, Start: 500, End: 500, MarkerID: 1, MarkerFlags: 1 << 2, ThreadID: 8, ProcessID: 2},
				{Subtype: "kernel", Name: "k", Valid: true, Start: 900, End: 950, Correlation: 1, Process: 2},
			},
			apis: []cudacapture.APIRecord{
				launchCall(7, 1, 150),
				{Domain: "runtime", Start: 160, End: 165, ProcessID: 2, ThreadID: 8, Correlation: 1},
			},
			want: map[string]want{"k": {nvtx: []string{"B"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := cudacapture.BuildActivityProfile("session", 1, tt.activities, tt.apis)
			for name, w := range tt.want {
				if got := nvtxLabelOf(t, p, name); !reflect.DeepEqual(got, w.nvtx) {
					t.Errorf("kernel %q: nvtx = %v, want %v", name, got, w.nvtx)
				}
				var wantUnassigned []string
				if w.unassigned != "" {
					wantUnassigned = []string{w.unassigned}
				}
				if got := labelOf(t, p, name, "nvtx_unassigned"); !reflect.DeepEqual(got, wantUnassigned) {
					t.Errorf("kernel %q: nvtx_unassigned = %v, want %v", name, got, wantUnassigned)
				}
			}
		})
	}
}

// TestBuildActivityProfileWithoutNVTXLabelsNothing covers the capture with no
// ranges at all. There is nothing to assign, so no sample is unassigned
// either: marking every sample of an unannotated capture would bury the label
// that means the join failed.
func TestBuildActivityProfileWithoutNVTXLabelsNothing(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{kernel("k", 1, 100, 200)}, nil)
	for _, key := range []string{"nvtx", "nvtx_unassigned"} {
		if got := labelOf(t, p, "k", key); got != nil {
			t.Errorf("%s = %v, want absent in a capture with no ranges", key, got)
		}
	}
}

// TestBuildActivityProfileAttributesNVTXOnce pins the property the launch
// instant buys: an activity is attributed at one moment, so it counts under
// exactly one set of ranges and the per-range device time sums to the total.
//
// The kernel that runs past the close is the point. Attributing it by overlap
// would count it under the range and again outside, making the parts sum to
// more than the device ever spent.
func TestBuildActivityProfileAttributesNVTXOnce(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		kernel("before", 1, 50, 80),
		nvtxOpen(7, 1, "forward", 100),
		kernel("straddles", 2, 200, 1000),
		nvtxClose(7, 1, "forward", 500),
		kernel("after", 3, 600, 700),
	}, []cudacapture.APIRecord{launchCall(7, 1, 40), launchCall(7, 2, 150), launchCall(7, 3, 550)})
	if got := nvtxLabelOf(t, p, "straddles"); !reflect.DeepEqual(got, []string{"forward"}) {
		t.Errorf("a kernel launched inside the range is labelled %v, want [forward]", got)
	}
	for _, k := range []string{"before", "after"} {
		if got := nvtxLabelOf(t, p, k); got != nil {
			t.Errorf("kernel %q was launched outside every range but is labelled %v", k, got)
		}
	}

	var inRange, outside, total int64
	for _, s := range p.Sample {
		total += s.Value[0]
		if len(s.Label["nvtx"]) > 0 {
			inRange += s.Value[0]
		} else {
			outside += s.Value[0]
		}
	}
	if want := int64(30 + 800 + 100); total != want {
		t.Fatalf("total kernel time = %d, want %d", total, want)
	}
	if inRange+outside != total {
		t.Errorf("in-range %d + outside %d = %d, want %d: an activity was attributed twice or not at all", inRange, outside, inRange+outside, total)
	}
	if inRange != 800 {
		t.Errorf("in-range kernel time = %d, want 800: the straddling kernel counts wholly to the range it was launched in", inRange)
	}
}

// TestNVTXRangesIgnoreUnpairedMarkers covers the half-open windows a real
// capture ends with: a range the target never closed, and a close with no open
// (its start fell outside the capture). Neither is a window, so neither can
// enclose anything, and inventing an end for one would attribute work to a
// range whose extent nothing recorded.
func TestNVTXRangesIgnoreUnpairedMarkers(t *testing.T) {
	// The launch sits AFTER the orphan open and BEFORE the orphan close, so
	// that either of the two tempting repairs -- running an unclosed open to
	// the end of the capture, or backdating an unopened close to its start --
	// would enclose it. A fixture where the launch falls outside the invented
	// window passes whether or not the window is invented.
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		nvtxOpen(7, 1, "never_closed", 100),
		kernel("k", 1, 200, 300),
		nvtxClose(7, 2, "never_opened", 400),
		nvtxOpen(7, 3, "closed", 1000),
		nvtxClose(7, 3, "closed", 1100),
	}, []cudacapture.APIRecord{launchCall(7, 1, 200)})
	if got := nvtxLabelOf(t, p, "k"); got != nil {
		t.Errorf("nvtx = %v, want none: neither an unclosed open nor an unopened close is a window", got)
	}
}

// TestBuildActivityProfileCountsMemsetTime pins the memset arm of the sample
// value. The arms are a chain with no default: a subtype none of them names
// still produces a sample, with a count of one and both duration dimensions
// left at zero. A memset would then appear in the profile as device work that
// took no time and moved no bytes -- present, countable, and invisible to
// every `-sample_index` a reader would use to find it.
func TestBuildActivityProfileCountsMemsetTime(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session-42", 100, []cudacapture.ActivityRecord{
		{Subtype: "memset", Valid: true, Start: 3000, End: 3500, Bytes: 8192, Value: 0, Device: 0},
	}, nil)
	if p == nil || len(p.Sample) != 1 {
		t.Fatalf("samples = %d, want 1", len(p.Sample))
	}
	s := p.Sample[0]
	if len(s.Value) != 4 {
		t.Fatalf("value dimensions = %d, want 4", len(s.Value))
	}
	if s.Value[0] != 0 {
		t.Errorf("kernel time = %d, want 0: a memset is not a kernel", s.Value[0])
	}
	if s.Value[1] != 500 {
		t.Errorf("memory time = %d, want 500: the memset contributed no device time", s.Value[1])
	}
	if s.Value[2] != 8192 {
		t.Errorf("bytes = %d, want 8192", s.Value[2])
	}
	if s.Value[3] != 1 {
		t.Errorf("count = %d, want 1", s.Value[3])
	}
	// A memset carries no symbol, so the profile has to name it something a
	// reader can find; an empty function name is not one.
	if got := s.Location[0].Line[0].Function.Name; got != "cudaMemset" {
		t.Errorf("function name = %q, want %q", got, "cudaMemset")
	}
	if got := s.Label["subtype"]; len(got) != 1 || got[0] != "memset" {
		t.Errorf("subtype label = %v, want [memset]", got)
	}
}

// TestBuildActivityProfileSkipsExternalCorrelation pins the third population
// boundary, beside the NVTX and capture-range ones. An external-correlation
// record is a link between a framework scope and a native correlation ID: it
// travels in the activity slice so the join can see it next to the device work
// it explains, and it must never become a sample, which would count a link as
// a launch in the activity dimension.
//
// Valid is deliberately true. The producer leaves it false, so the invalid
// check above would skip such a record anyway and hide whether the subtype is
// handled at all; a caller building the record itself gets no such accident.
func TestBuildActivityProfileSkipsExternalCorrelation(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		{Subtype: "kernel", Name: "k", Valid: true, Start: 100, End: 200, Correlation: 9},
		{Subtype: "external_correlation", Valid: true, Correlation: 9, ExternalKind: 3, ExternalID: 770},
	}, nil)
	if len(p.Sample) != 1 {
		t.Fatalf("got %d samples, want 1: an external correlation link is not device work", len(p.Sample))
	}
	if got := p.Sample[0].Label["subtype"]; len(got) != 1 || got[0] != "kernel" {
		t.Errorf("sample subtype = %v, want kernel", got)
	}
}

// TestBuildActivityProfileSkipsTheIdleExplainingTiers pins the exclusion of
// the three tiers that are not device work.
//
// The value arms in BuildActivityProfile are a chain with no default, so a
// subtype none of them names still produces a sample: count one, kernel time
// zero, memory time zero, bytes zero. A CUPTI overhead span, a graph replay
// envelope and a cudaStreamSynchronize would each land in the profile as
// device work that took no time, inflating the count dimension against the
// duration dimensions beside it -- the same defect the marker exclusion
// already exists for. The graph trace would be worse than that: its span
// contains the kernel records of the nodes it replayed, so admitting it would
// charge their device time twice.
func TestBuildActivityProfileSkipsTheIdleExplainingTiers(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session-42", 100, []cudacapture.ActivityRecord{
		{Subtype: "kernel", Valid: true, Start: 1000, End: 1500, Device: 1},
		{Subtype: "overhead", Valid: true, Start: 2000, End: 2070, OverheadKind: 1 << 16, ObjectKind: 2},
		{Subtype: "graph_trace", Valid: true, Start: 900, End: 1600, Device: 1, GraphID: 5},
		{Subtype: "sync", Valid: true, Start: 3000, End: 3060, SyncType: 3, Context: 4},
	}, nil)
	if p == nil {
		t.Fatal("no profile")
	}
	if len(p.Sample) != 1 {
		var subtypes []string
		for _, s := range p.Sample {
			subtypes = append(subtypes, s.Label["subtype"]...)
		}
		t.Fatalf("samples = %d %v, want 1 (the kernel alone): a record that is not device work became device work that took no time", len(p.Sample), subtypes)
	}
	s := p.Sample[0]
	if got := s.Label["subtype"]; len(got) != 1 || got[0] != "kernel" {
		t.Fatalf("surviving sample subtype = %v, want [kernel]", got)
	}
	if s.Value[0] != 500 {
		t.Errorf("kernel time = %d, want 500", s.Value[0])
	}
	if s.Value[3] != 1 {
		t.Errorf("count = %d, want 1", s.Value[3])
	}
}
