package cudakvlease

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/lm/kvcache"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
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

func TestCUDAOffloadRestoreFeedsPagedAttention(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice: %v", err)
	}
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	const (
		qHeads    = 2
		kvHeads   = 1
		headDim   = 2
		blockSize = 4
	)
	a, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: kvHeads, HeadDim: headDim, BlockSize: blockSize, TotalPages: 1, DType: "float32", Domain: "gpu0"})
	if err != nil {
		t.Fatal(err)
	}
	allocation, err := a.Allocate("attention", kvcache.Range{Start: 0, End: blockSize})
	if err != nil {
		t.Fatal(err)
	}
	id, generation := allocation.ID()
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	page, err := use.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	wantK := []float32{1, 0, 0, 1, 1, 1, -1, 1}
	wantV := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := dev.WithContext(func() error {
		if err := page.K.CopyFromHost(wantK); err != nil {
			return err
		}
		return page.V.CopyFromHost(wantV)
	}); err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	done, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer done.Destroy()
	if err := done.Record(stream); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := use.Complete(ctx, done); err != nil {
		t.Fatal(err)
	}
	if err := a.Offload(ctx, allocation, done); err != nil {
		t.Fatalf("Offload: %v", err)
	}
	if stats := a.Stats(); stats.DevicePages != 0 || stats.HostBlocks != 1 || stats.PinnedHostBytes != 64 {
		t.Fatalf("offloaded stats = %+v", stats)
	}
	if err := a.Restore(ctx, allocation); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	use, err = a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	page, err = use.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	gotK, gotV := make([]float32, len(wantK)), make([]float32, len(wantV))
	if err := dev.WithContext(func() error {
		if err := page.K.CopyToHost(gotK); err != nil {
			return err
		}
		return page.V.CopyToHost(gotV)
	}); err != nil {
		t.Fatal(err)
	}
	for i := range wantK {
		if gotK[i] != wantK[i] || gotV[i] != wantV[i] {
			t.Fatalf("restored element %d = %g/%g, want %g/%g", i, gotK[i], gotV[i], wantK[i], wantV[i])
		}
	}

	compiled, err := cudaattn.CompilePaged(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	module, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer module.Unload()
	attention, err := module.GetFunction("BatchedPagedAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	q := []float32{1, 0, 0, 1}
	dq := cudaTestSlice(t, dev, q)
	dtable := cudaTestSlice(t, dev, []int32{0})
	dlens := cudaTestSlice(t, dev, []int32{blockSize})
	dout := cudaTestSlice(t, dev, make([]float32, len(q)))
	dstatus := cudaTestSlice(t, dev, []int32{0})
	for _, free := range []func() error{dq.Free, dtable.Free, dlens.Free, dout.Free, dstatus.Free} {
		defer free()
	}
	shared, err := cudaattn.PagedSharedMemoryBytes(blockSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudalaunch.Kernel(attention).Grid(qHeads, 1).Block(32).SharedMem(uint32(shared)).On(stream).Launch(
		dq, page.K, page.V, dtable, dlens, dout,
		int32(1), int32(blockSize), int32(qHeads), int32(kvHeads), int32(headDim),
		float32(1/math.Sqrt(headDim)), int32(1), int32(1), int32(blockSize), dstatus,
	); err != nil {
		t.Fatal(err)
	}
	if err := done.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := use.Complete(ctx, done); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(q))
	status := []int32{0}
	if err := dev.WithContext(func() error {
		if err := dout.CopyToHost(got); err != nil {
			return err
		}
		return dstatus.CopyToHost(status)
	}); err != nil {
		t.Fatal(err)
	}
	if status[0] != 0 {
		t.Fatalf("attention status = %d", status[0])
	}
	want := referenceAttention(q, wantK, wantV, qHeads, kvHeads, headDim)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("attention[%d] = %g, want %g", i, got[i], want[i])
		}
	}
	if err := a.Retire(ctx, allocation, done); err != nil {
		t.Fatal(err)
	}
	t.Logf("receipt generation=%d device_pages_max=1 pinned_bytes=64 restored_values=%d attention=%v", generation, len(wantK)+len(wantV), got)
}

func cudaTestSlice[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(values))
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(func() error { return s.CopyFromHost(values) }); err != nil {
		_ = s.Free()
		t.Fatal(err)
	}
	return s
}

func referenceAttention(q, k, v []float32, qHeads, kvHeads, headDim int) []float32 {
	out := make([]float32, len(q))
	group := qHeads / kvHeads
	for head := 0; head < qHeads; head++ {
		kvHead := head / group
		scores := make([]float64, blockSizeFor(k, kvHeads, headDim))
		maxScore := math.Inf(-1)
		for token := range scores {
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(q[head*headDim+d] * k[(token*kvHeads+kvHead)*headDim+d])
			}
			scores[token] = dot / math.Sqrt(float64(headDim))
			maxScore = max(maxScore, scores[token])
		}
		var sum float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - maxScore)
			sum += scores[i]
		}
		for d := 0; d < headDim; d++ {
			var value float64
			for token, score := range scores {
				value += score / sum * float64(v[(token*kvHeads+kvHead)*headDim+d])
			}
			out[head*headDim+d] = float32(value)
		}
	}
	return out
}

func blockSizeFor(k []float32, kvHeads, headDim int) int {
	return len(k) / (kvHeads * headDim)
}
