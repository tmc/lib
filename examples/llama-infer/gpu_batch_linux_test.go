//go:build !tpu

package main

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDABatchArgmaxTieAndLowestFiniteValue(t *testing.T) {
	dev := requireCUDAHardware(t)
	modules, err := CompileGPUModules(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer modules.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	const rows, vocab = 2, 513
	logits := make([]float32, rows*vocab)
	for i := range logits {
		logits[i] = -math.MaxFloat32
	}
	// Indices 1 and 257 are reduced by different threads. Equal maxima must
	// choose the lowest vocabulary index, matching the scalar sampler.
	logits[1], logits[257] = 7, 7
	deviceLogits, err := cudamem.Alloc[float32](dev, len(logits))
	if err != nil {
		t.Fatal(err)
	}
	defer deviceLogits.Free()
	deviceTokens, err := cudamem.Alloc[int32](dev, rows)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceTokens.Free()
	if err := deviceLogits.CopyFromHost(logits); err != nil {
		t.Fatal(err)
	}
	if err := cudalaunch.OwnedKernel(modules.BatchArgmax).Grid(rows).Block(256).On(stream).
		Launch(deviceLogits, deviceTokens, int32(vocab), int32(rows)); err != nil {
		t.Fatal(err)
	}
	got := make([]int32, rows)
	if err := deviceTokens.CopyToHostAsync(got, stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}
	want := []int32{1, 0}
	for row := range got {
		if got[row] != want[row] {
			t.Fatalf("row %d argmax = %d, want %d", row, got[row], want[row])
		}
	}
	t.Logf("receipt cross_thread_tie=%d lowest_finite_row=%d", got[0], got[1])
}
