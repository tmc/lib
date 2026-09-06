package cudalaunch_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestOwnedKernelRejectsClosed(t *testing.T) {
	for _, kernel := range []*cuda.Kernel{nil, new(cuda.Kernel)} {
		var count cudalaunch.Counter
		if err := cudalaunch.OwnedKernel(kernel).Count(&count).Launch(); err == nil {
			t.Fatal("closed kernel launched")
		}
		if count.Value() != 0 {
			t.Fatal("failed launch counted")
		}
	}
}

func TestBindOwnedInvalidTarget(t *testing.T) {
	for _, target := range []any{nil, new(int), &struct {
		Bad int `cuda:"Bad"`
	}{}, &struct{ hidden *cuda.Kernel }{}} {
		if err := cudalaunch.BindOwned(nil, target); err == nil {
			t.Fatalf("BindOwned(nil, %T) succeeded", target)
		}
	}
	var kernels struct {
		Embed *cuda.Kernel `cuda:"EmbeddingLookup"`
	}
	err := cudalaunch.BindOwned(new(cuda.Module), &kernels)
	if err == nil || !strings.Contains(err.Error(), "EmbeddingLookup") {
		t.Fatalf("BindOwned: %v", err)
	}
}

func ExampleOwnedKernel() {
	var kernel cuda.Kernel // A zero kernel has no live module.
	err := cudalaunch.OwnedKernel(&kernel).Grid(1).Block(32).Launch()
	fmt.Println(err != nil)
	// Output: true
}

func ExampleBindOwned() {
	var module cuda.Module // A zero module is closed.
	var kernels struct {
		Add *cuda.Kernel `cuda:"VectorAdd"`
	}
	fmt.Println(cudalaunch.BindOwned(&module, &kernels) != nil)
	// Output: true
}
