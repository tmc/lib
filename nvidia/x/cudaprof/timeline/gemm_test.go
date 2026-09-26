package timeline

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

func TestDecodeGEMMs(t *testing.T) {
	journal := journaltest.New().
		GEMM(1, "cublasLtMatmul", "split-K call").
		GEMMLaunch(1, 10).
		GEMMLaunch(1, 11).
		GEMM(2, "cublasLtTSTMatmul", "graph-captured call").
		GEMMLaunch(2, 12).
		GEMMLaunch(9, 13). // no such call
		Named("kernel", "gemm_main", 100, 200, 10).
		Named("kernel", "splitk_reduce", 200, 220, 11).
		Named("memcpy", "", 220, 230, 12).
		Named("kernel", "other", 300, 400, 13).
		Finalize()
	got, err := Decode(map[string][]byte{"capture.journal": journal}, 0, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.GEMMs) != 2 {
		t.Fatalf("decoded %d GEMMs, want 2: %+v", len(got.GEMMs), got.GEMMs)
	}
	g := got.GEMMs[0]
	if g.ID != 1 || g.API != "cublasLtMatmul" || g.Log != "split-K call" || g.Thread != 7 || g.Journal != "capture.journal" {
		t.Errorf("GEMM 1 = %+v", g)
	}
	if g.Launches != 2 || len(g.Kernels) != 2 || g.Kernels[0].Name != "gemm_main" || g.Kernels[1].Name != "splitk_reduce" {
		t.Errorf("GEMM 1 launches %d, kernels %+v; want 2 launches joined to gemm_main and splitk_reduce", g.Launches, g.Kernels)
	}
	// A launch whose record is not a kernel is counted and not joined.
	if g := got.GEMMs[1]; g.Launches != 1 || len(g.Kernels) != 0 {
		t.Errorf("GEMM 2 launches %d, kernels %+v; want 1 launch and no kernel", g.Launches, g.Kernels)
	}
}

func TestDecodeWithoutGEMMs(t *testing.T) {
	journal := journaltest.New().Kernel(10, 20, 1).Finalize()
	got, err := Decode(map[string][]byte{"capture.journal": journal}, 0, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if got.GEMMs != nil {
		t.Errorf("GEMMs = %+v, want none from a capture made without --gemms", got.GEMMs)
	}
}

// Correlation ids are unique only within a process, so a launch in one
// journal never joins a kernel in another.
func TestDecodeGEMMsPerJournal(t *testing.T) {
	a := journaltest.New().GEMM(1, "cublasLtMatmul", "a").GEMMLaunch(1, 5).Finalize()
	b := journaltest.New().Named("kernel", "b_kernel", 10, 20, 5).Finalize()
	got, err := Decode(map[string][]byte{"a.journal": a, "b.journal": b}, 0, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.GEMMs) != 1 || len(got.GEMMs[0].Kernels) != 0 {
		t.Errorf("GEMMs = %+v, want one call with no kernel", got.GEMMs)
	}
}
