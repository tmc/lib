//go:build cuda_tma_multicast_qualification

package cudatma

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This separate opt-in compiler packet is source only until the coordinator
// admits a pinned toolkit/host lease. It performs no GPU execution. The native
// numerical/participation packet still needs a concrete launch/allocation owner
// and physical completion receipts; the compiler packet cannot establish that.
func TestMulticastCompilePTX(t *testing.T) {
	if os.Getenv("CUDA_TMA_MULTICAST_COMPILE") != "1" {
		t.Skip("compiler qualification requires explicit opt-in")
	}
	// No discovery/default compiler: a qualified caller supplies the exact path.
	nvcc := os.Getenv("CUDA_TMA_MULTICAST_NVCC")
	if !filepath.IsAbs(nvcc) {
		t.Fatal("explicit absolute qualified nvcc path required")
	}
	for rank := 1; rank <= 5; rank++ {
		p, err := NewMulticastPlan(multicastConfig(rank, 4, 128))
		if err != nil {
			t.Fatal(err)
		}
		s, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		input := filepath.Join(dir, "multicast.cu")
		output := filepath.Join(dir, "multicast.ptx")
		if err := os.WriteFile(input, []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		result, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_100a", "-ptx", input, "-o", output).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("rank%d compile: %v\n%s", rank, err, result)
		}
		b, err := os.ReadFile(output)
		if err != nil {
			t.Fatal(err)
		}
		text := string(b)
		for _, want := range []string{".multicast::cluster", "cvta.param", "fence.mbarrier_init.release.cluster", "mbarrier.arrive.expect_tx", "mbarrier.try_wait.parity", "barrier.cluster"} {
			if !strings.Contains(text, want) {
				t.Fatalf("rank%d PTX missing %s", rank, want)
			}
		}
		// A missing toolkit/unsupported target is a failed requested gate, never
		// accepted as a skipped compiler qualification.
	}
}
