package cudakvlease

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/lm/kvcache"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCUDAAllocationEventCopyAndReuse(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice: %v", err)
	}
	defer dev.Close()
	name, err := dev.Name()
	if err != nil {
		t.Fatalf("Name: %v", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("ComputeCapability: %v", err)
	}
	t.Logf("device=%q compute=sm_%d%d", name, major, minor)

	a, err := NewCUDA(dev, Config{NumLayers: 2, KVHeads: 2, HeadDim: 8, BlockSize: 4, TotalPages: 1, DType: "float32", Domain: "gpu0"})
	if err != nil {
		t.Fatalf("NewCUDA: %v", err)
	}
	first, err := a.Allocate("first", kvcache.Range{Start: 0, End: 4})
	if err != nil {
		t.Fatalf("Allocate first: %v", err)
	}
	firstID, firstGeneration := first.ID()
	use, err := a.Bind(firstID, firstGeneration)
	if err != nil {
		t.Fatalf("Bind first: %v", err)
	}
	firstSlot, err := use.Slot()
	if err != nil {
		t.Fatalf("Slot first: %v", err)
	}
	page, err := use.Page(1)
	if err != nil {
		t.Fatalf("Page: %v", err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()
	wantK := make([]float32, page.K.Len())
	wantV := make([]float32, page.V.Len())
	for i := range wantK {
		wantK[i] = float32(i) + 0.25
		wantV[i] = -float32(i) - 0.5
	}
	if err := page.K.CopyFromHostAsync(wantK, stream); err != nil {
		t.Fatalf("copy K: %v", err)
	}
	if err := page.V.CopyFromHostAsync(wantV, stream); err != nil {
		t.Fatalf("copy V: %v", err)
	}
	gotK := make([]float32, page.K.Len())
	gotV := make([]float32, page.V.Len())
	if err := page.K.CopyToHostAsync(gotK, stream); err != nil {
		t.Fatalf("read K: %v", err)
	}
	if err := page.V.CopyToHostAsync(gotV, stream); err != nil {
		t.Fatalf("read V: %v", err)
	}
	consumerDone, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent consumer: %v", err)
	}
	defer consumerDone.Destroy()
	if err := consumerDone.Record(stream); err != nil {
		t.Fatalf("Record consumer: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := use.Complete(ctx, consumerDone); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for i := range wantK {
		if gotK[i] != wantK[i] || gotV[i] != wantV[i] {
			t.Fatalf("copy mismatch at %d: got K/V %g/%g, want %g/%g", i, gotK[i], gotV[i], wantK[i], wantV[i])
		}
	}
	retireDone, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent retire: %v", err)
	}
	defer retireDone.Destroy()
	if err := retireDone.Record(stream); err != nil {
		t.Fatalf("Record retire: %v", err)
	}
	if err := a.Retire(ctx, first, retireDone); err != nil {
		t.Fatalf("Retire first: %v", err)
	}
	second, err := a.Allocate("second", kvcache.Range{Start: 4, End: 8})
	if err != nil {
		t.Fatalf("Allocate second: %v", err)
	}
	secondID, secondGeneration := second.ID()
	secondUse, err := a.Bind(secondID, secondGeneration)
	if err != nil {
		t.Fatalf("Bind second: %v", err)
	}
	secondSlot, err := secondUse.Slot()
	if err != nil {
		t.Fatalf("Slot second: %v", err)
	}
	if secondSlot != firstSlot || secondID != firstID || secondGeneration == firstGeneration {
		t.Fatalf("physical reuse receipt: slots %d/%d ids %d/%d generations %d/%d", firstSlot, secondSlot, firstID, secondID, firstGeneration, secondGeneration)
	}
	if err := secondUse.Complete(ctx, retireDone); err != nil {
		t.Fatalf("Complete second: %v", err)
	}
	if err := a.Retire(ctx, second, retireDone); err != nil {
		t.Fatalf("Retire second: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	t.Logf("receipt slot=%d logical_id=%d generations=%d,%d copied_elements=%d", firstSlot, firstID, firstGeneration, secondGeneration, len(wantK)+len(wantV))
}
