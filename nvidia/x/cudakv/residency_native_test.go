//go:build native

package cudakv_test

import (
	"context"
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestResidencyNativeRoundTrip is device qualification, not a host fake. Its
// source is compiled by host gates; native execution requires its own admitted
// target/lease tuple. No overlap, timing or HBM-allocation reclamation is claimed.
func TestResidencyNativeRoundTrip(t *testing.T) {
	lockOwner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		t.Fatalf("native residency requires canonical inherited GPU ownership: %v", err)
	}
	if os.Getenv("GPU_LOCK_DISABLED") != "" || os.Getenv("CUDATEST_NO_LOCK") != "" {
		t.Fatal("native residency refuses GPU lock bypass flags")
	}
	// Register first so this check runs after all native-resource and nested
	// Require cleanup. Owner is borrowed metadata, not a handle to release.
	t.Cleanup(func() {
		current, err := cudalock.Inherited(os.Getenv)
		if err != nil || current.PID != lockOwner.PID || current.Host != lockOwner.Host || current.Epoch != lockOwner.Epoch || current.Label != lockOwner.Label || !current.At.Equal(lockOwner.At) {
			t.Errorf("native residency inherited owner changed during scope: %v", err)
		}
	})
	dev := cudatest.Require(t)
	pc, err := cudakv.NewPagedCache(dev, 2, 1, 2, 2, 4)
	if pc != nil {
		t.Cleanup(func() {
			if err := pc.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if stream != nil {
		t.Cleanup(func() {
			if err := stream.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.AllocSequence(7, 3); err != nil {
		t.Fatal(err)
	}
	oldGen, _ := pc.AllocationGeneration(7)
	old := cudakv.ResidencyBinding{SequenceID: 7, Generation: oldGen, Blocks: pc.BlockTable(7), Tokens: 3}
	var want [][]float32
	for i, slot := range old.Blocks {
		for layer := 0; layer < 2; layer++ {
			page, err := pc.Page(layer, slot)
			if err != nil {
				t.Fatal(err)
			}
			for j, buffer := range []cudamem.Slice[float32]{page.K, page.V} {
				data := make([]float32, buffer.Len())
				for k := range data {
					data[k] = float32(100*i + 10*layer + 3*j + k + 1)
				}
				want = append(want, data)
				if err := buffer.CopyFromHost(data); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	owner, err := cudakv.NewResidency(pc, 7, oldGen)
	if owner != nil {
		t.Cleanup(func() {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.Close(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("pool lifetime escaped residency", err)
	}
	if err := owner.Evict(stream); err != nil {
		t.Fatal(err)
	}
	if err := owner.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if owner.Tier() != cudakv.ResidencyHost || pc.NumFreeBlocks() != 4 || pc.ValidateSequence(old) == nil {
		t.Fatal("eviction did not unbind/free pool slots")
	}
	if err := pc.AllocSequence(99, 8); err != nil {
		t.Fatal(err)
	}
	for _, slot := range pc.BlockTable(99) {
		for layer := 0; layer < 2; layer++ {
			p, err := pc.Page(layer, slot)
			if err != nil {
				t.Fatal(err)
			}
			for _, buffer := range []cudamem.Slice[float32]{p.K, p.V} {
				if err := buffer.CopyFromHost(make([]float32, buffer.Len())); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	if err := owner.Prefetch(stream); err == nil {
		t.Fatal("full pool admitted prefetch")
	}
	pc.FreeSequence(99)
	if err := owner.Prefetch(stream); err != nil {
		t.Fatal(err)
	}
	if err := owner.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	binding, err := owner.Binding()
	if err != nil {
		t.Fatal(err)
	}
	if binding.Generation == oldGen || pc.ValidateSequence(binding) != nil || pc.ValidateSequence(old) == nil {
		t.Fatal("restoration reused stale allocation identity")
	}
	n := 0
	for _, slot := range binding.Blocks {
		for layer := 0; layer < 2; layer++ {
			p, err := pc.Page(layer, slot)
			if err != nil {
				t.Fatal(err)
			}
			for _, buffer := range []cudamem.Slice[float32]{p.K, p.V} {
				got := make([]float32, buffer.Len())
				if err := buffer.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want[n]) {
					t.Fatalf("page %d parity: got %v want %v", n, got, want[n])
				}
				n++
			}
		}
	}
}
