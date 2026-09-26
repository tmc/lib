package timeline

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

func TestCallName(t *testing.T) {
	tests := []struct {
		domain string
		cbid   uint32
		want   string
	}{
		{"driver", 243, "cuMemAlloc_v2"},
		{"driver", 245, "cuMemFree_v2"},
		{"driver", 502, "cuGraphAddKernelNode"},
		{"runtime", 20, "cudaMalloc"},
		{"runtime", 131, "cudaStreamSynchronize"},
		{"driver", 1 << 30, "driver:1073741824"},
		{"nvtx", 1, "nvtx:1"},
	}
	for _, tt := range tests {
		if got := CallName(tt.domain, tt.cbid); got != tt.want {
			t.Errorf("CallName(%q, %d) = %q, want %q", tt.domain, tt.cbid, got, tt.want)
		}
	}
}

func TestDecodeCallsAndMarkers(t *testing.T) {
	journal := journaltest.New().
		Marker("step", 7, MarkerStart, 5).
		Call("driver", 243, 11, 6, 9, 1).
		Kernel(10, 20, 2).
		Call("runtime", 131, 11, 21, 30, 3).
		Marker("", 7, MarkerEnd, 31).
		Finalize()
	r, err := Decode(map[string][]byte{"capture.journal": journal}, MaxRecords, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Activities) != 1 {
		t.Errorf("Activities = %d, want 1: a marker is not device work", len(r.Activities))
	}
	want := []Call{
		{Domain: "driver", CallbackID: 243, Thread: 11, Correlation: 1, Start: 6, End: 9},
		{Domain: "runtime", CallbackID: 131, Thread: 11, Correlation: 3, Start: 21, End: 30},
	}
	if len(r.Calls) != len(want) {
		t.Fatalf("Calls = %+v, want %+v", r.Calls, want)
	}
	for i := range want {
		if r.Calls[i] != want[i] {
			t.Errorf("Calls[%d] = %+v, want %+v", i, r.Calls[i], want[i])
		}
	}
	if got := r.Calls[0].Name(); got != "cuMemAlloc_v2" {
		t.Errorf("Calls[0].Name() = %q, want cuMemAlloc_v2", got)
	}
	wantMarkers := []Marker{{Name: "step", ID: 7, Flags: MarkerStart, Time: 5}, {ID: 7, Flags: MarkerEnd, Time: 31}}
	if len(r.Markers) != 2 || r.Markers[0] != wantMarkers[0] || r.Markers[1] != wantMarkers[1] {
		t.Errorf("Markers = %+v, want %+v", r.Markers, wantMarkers)
	}
}
