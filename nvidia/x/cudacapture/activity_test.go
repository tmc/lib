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

	p := cudacapture.BuildActivityProfile("session-42", 100, activities)
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
	}})
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
	p := cudacapture.BuildActivityProfile("session", 4242, activities)
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
	p := cudacapture.BuildActivityProfile("session", 4242, activities)
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

// marker builds the two records CUPTI reports for one NVTX range. A range is
// never a single record: the open and the close are separate, share a marker
// id, and carry the timestamp in both Start and End.
func nvtxOpen(id uint32, name string, at uint64) cudacapture.ActivityRecord {
	return cudacapture.ActivityRecord{Subtype: "marker", Name: name, Valid: true, Start: at, End: at, MarkerID: id, MarkerFlags: 1 << 1, ThreadID: 7, ProcessID: 1}
}

func nvtxClose(id uint32, name string, at uint64) cudacapture.ActivityRecord {
	return cudacapture.ActivityRecord{Subtype: "marker", Name: name, Valid: true, Start: at, End: at, MarkerID: id, MarkerFlags: 1 << 2, ThreadID: 7, ProcessID: 1}
}

func nvtxLabelOf(t *testing.T, p *profile.Profile, kernel string) []string {
	t.Helper()
	for _, s := range p.Sample {
		if len(s.Label["name"]) == 1 && s.Label["name"][0] == kernel {
			return s.Label["nvtx"]
		}
	}
	t.Fatalf("no sample named %q", kernel)
	return nil
}

// TestBuildActivityProfileLabelsEnclosingNVTXRanges covers nesting: a kernel
// inside several ranges is labelled with all of them, outermost first, so that
// focusing an outer range in pprof also finds the work inside its children.
// Labelling only the innermost would make -tagfocus=train_step silently miss
// every kernel that ran in a nested range.
func TestBuildActivityProfileLabelsEnclosingNVTXRanges(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		nvtxOpen(1, "train_step", 100),
		nvtxOpen(2, "forward", 200),
		nvtxOpen(3, "attention", 300),
		{Subtype: "kernel", Name: "inner", Valid: true, Start: 400, End: 450},
		nvtxClose(3, "attention", 500),
		{Subtype: "kernel", Name: "mid", Valid: true, Start: 520, End: 540},
		nvtxClose(2, "forward", 600),
		{Subtype: "kernel", Name: "outer", Valid: true, Start: 700, End: 750},
		nvtxClose(1, "train_step", 1000),
		{Subtype: "kernel", Name: "after", Valid: true, Start: 1100, End: 1200},
	})
	for _, tt := range []struct {
		kernel string
		want   []string
	}{
		{"inner", []string{"train_step", "forward", "attention"}},
		{"mid", []string{"train_step", "forward"}},
		{"outer", []string{"train_step"}},
		{"after", nil},
	} {
		if got := nvtxLabelOf(t, p, tt.kernel); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("kernel %q: nvtx = %v, want %v", tt.kernel, got, tt.want)
		}
	}
}

// TestBuildActivityProfileAttributesNVTXByStartTime pins the containment rule
// and the property it exists to buy: an activity belongs to the ranges open
// when it STARTED, so every activity is attributed exactly once and the
// per-range device time sums to the total.
//
// The kernel that straddles the close is the whole point. Attributing it by
// overlap would count it under the range and again outside, making the parts
// sum to more than the device ever spent; requiring full containment would
// drop it from both, making them sum to less, with the shortfall invisible.
func TestBuildActivityProfileAttributesNVTXByStartTime(t *testing.T) {
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		{Subtype: "kernel", Name: "before", Valid: true, Start: 50, End: 80},
		nvtxOpen(1, "forward", 100),
		{Subtype: "kernel", Name: "straddles", Valid: true, Start: 200, End: 1000},
		nvtxClose(1, "forward", 500),
		{Subtype: "kernel", Name: "after", Valid: true, Start: 600, End: 700},
	})
	if got := nvtxLabelOf(t, p, "straddles"); !reflect.DeepEqual(got, []string{"forward"}) {
		t.Errorf("a kernel that began inside the range is labelled %v, want [forward]", got)
	}
	for _, k := range []string{"before", "after"} {
		if got := nvtxLabelOf(t, p, k); got != nil {
			t.Errorf("kernel %q ran outside every range but is labelled %v", k, got)
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
		t.Errorf("in-range kernel time = %d, want 800: the straddling kernel counts wholly to the range it began in", inRange)
	}
}

// TestNVTXRangesIgnoreUnpairedMarkers covers the half-open windows a real
// capture ends with: a range the target never closed, and a close with no open
// (its start fell outside the capture). Neither is a window, so neither can
// enclose anything, and inventing an end for one would attribute work to a
// range whose extent nothing recorded.
func TestNVTXRangesIgnoreUnpairedMarkers(t *testing.T) {
	// The kernel sits AFTER the orphan open and BEFORE the orphan close, so
	// that either of the two tempting repairs -- running an unclosed open to
	// the end of the capture, or backdating an unopened close to its start --
	// would enclose it. A fixture where the kernel falls outside the invented
	// window passes whether or not the window is invented.
	p := cudacapture.BuildActivityProfile("session", 1, []cudacapture.ActivityRecord{
		nvtxOpen(1, "never_closed", 100),
		{Subtype: "kernel", Name: "k", Valid: true, Start: 200, End: 300},
		nvtxClose(2, "never_opened", 400),
	})
	if got := nvtxLabelOf(t, p, "k"); got != nil {
		t.Errorf("nvtx = %v, want none: neither an unclosed open nor an unopened close is a window", got)
	}
}
