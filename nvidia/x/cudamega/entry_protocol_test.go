package cudamega_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

func TestEntrySourceBeforeSelection(t *testing.T) {
	f := crossFixture(t)
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	entry, selection := strings.Index(src, "// entry protocol begin"), strings.Index(src, "if (owns_0 && !done_0)")
	if entry < 0 || selection < entry {
		t.Fatal("task selection precedes entry")
	}
	protocol := src[entry:selection]
	for _, want := range []string{"unsigned long long i=blockIdx.x", "i<task_count; i+=grid_ctas", "atomicAdd(completed+i,0u)", "atomicAdd(arrivals,1ull)", "atomicExch(released,1ull)", "while (atomicAdd(released,0ull)==0)", "entry_abort=atomicAdd((unsigned*)abort_code,0u)", "if (entry_abort) return"} {
		if !strings.Contains(protocol, want) {
			t.Fatalf("entry lacks %q", want)
		}
	}
	// This is a source contract check, not a CUDA compile or launch result.
	harness := nativeMain(t, f)
	for _, want := range []string{"cudaDevAttrCooperativeLaunch", "cooperative launch unsupported", "cudaOccupancyMaxActiveBlocksPerMultiprocessor", "cudaLaunchCooperativeKernel", "if (launch_worker(1ull)!=cudaSuccess)", "if (launch_worker(2ull)!=cudaSuccess)", "if (launch_worker(1ull<<32)!=cudaSuccess)"} {
		if !strings.Contains(harness, want) {
			t.Fatalf("qualification harness lacks %q", want)
		}
	}
	if strings.Contains(harness, "dag_worker<<<") {
		t.Fatal("persistent worker uses ordinary launch")
	}
}
