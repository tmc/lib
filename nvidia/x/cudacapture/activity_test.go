package cudacapture_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestActivityRecordUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantGrid [3]uint32
		wantBlock [3]uint32
		wantRegs uint32
		wantSmem uint32
	}{
		{
			name:     "array fields",
			input:    `{"subtype":"kernel","valid":true,"grid":[128,2,1],"block":[32,4,1],"registers_per_thread":48,"dynamic_shared_memory":2048}`,
			wantGrid: [3]uint32{128, 2, 1},
			wantBlock: [3]uint32{32, 4, 1},
			wantRegs: 48,
			wantSmem: 2048,
		},
		{
			name:     "coordinate and alias fields",
			input:    `{"subtype":"kernel","valid":true,"grid_x":64,"grid_y":1,"grid_z":1,"block_x":256,"block_y":1,"block_z":1,"registers":32,"smem":1024}`,
			wantGrid: [3]uint32{64, 1, 1},
			wantBlock: [3]uint32{256, 1, 1},
			wantRegs: 32,
			wantSmem: 1024,
		},
		{
			name:     "grid_dim and block_dim arrays",
			input:    `{"subtype":"kernel","valid":true,"grid_dim":[16,1,1],"block_dim":[64,1,1],"registers_per_thread":16,"dynamic_shared_memory":512}`,
			wantGrid: [3]uint32{16, 1, 1},
			wantBlock: [3]uint32{64, 1, 1},
			wantRegs: 16,
			wantSmem: 512,
		},
		{
			name:     "memcpy without geometry",
			input:    `{"subtype":"memcpy","valid":true,"bytes":4096}`,
			wantGrid: [3]uint32{0, 0, 0},
			wantBlock: [3]uint32{0, 0, 0},
			wantRegs: 0,
			wantSmem: 0,
		},
		{
			name:     "legacy kernel without geometry",
			input:    `{"subtype":"kernel","valid":true,"start":100,"end":200}`,
			wantGrid: [3]uint32{0, 0, 0},
			wantBlock: [3]uint32{0, 0, 0},
			wantRegs: 0,
			wantSmem: 0,
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
		name          string
		record        cudacapture.ActivityRecord
		wantContains  []string
		wantOmit      []string
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
