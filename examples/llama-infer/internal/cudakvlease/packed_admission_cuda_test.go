package cudakvlease

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/lm/kvcache"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCUDAPackedRaggedAdmission(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	tests := []struct {
		name    string
		lengths []int
	}{
		{"B1", []int{3}},
		{"B2", []int{3, 5}},
		{"B4", []int{3, 5, 2, 6}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const blockSize = 2
			pagesNeeded, tokens := 0, 0
			for _, length := range test.lengths {
				pagesNeeded += (length + blockSize - 1) / blockSize
				tokens += length
			}
			dev, err := cuda.NewDevice(0)
			if err != nil {
				t.Fatal(err)
			}
			defer dev.Close()
			a, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: 1, HeadDim: 4, BlockSize: blockSize, TotalPages: pagesNeeded, DType: "float32", Domain: "cuda:0"})
			if err != nil {
				t.Fatal(err)
			}
			victimCount := min(2, pagesNeeded)
			victims := make([]Allocation, victimCount)
			for i := range victims {
				victims[i], err = a.Allocate(fmt.Sprintf("victim-%d", i), kvcache.Range{Start: 0, End: blockSize})
				if err != nil {
					t.Fatal(err)
				}
			}
			metadata, err := NewPackedMetadata(a, dev, len(test.lengths), tokens, pagesNeeded)
			if err != nil {
				t.Fatal(err)
			}
			requests := make([]PackedRequest, len(test.lengths))
			for row, length := range test.lengths {
				positions := make([]int, length)
				causal := make([]int, length)
				for i := range positions {
					positions[i], causal[i] = i, i+1
				}
				requests[row] = PackedRequest{BatchRequest: BatchRequest{Key: fmt.Sprintf("row-%d", row), Generation: uint64(row + 1), Length: length, Position: length - 1}, Positions: positions, CausalLengths: causal}
			}
			batch, stats, err := metadata.Admit(t.Context(), requests, blockSize, &fakeCompletion{})
			if err != nil {
				t.Fatal(err)
			}
			if stats.AllocatedPages != pagesNeeded || stats.OffloadedPages != victimCount || stats.RestoredPages != 0 {
				t.Fatalf("admission stats = %+v", stats)
			}
			if got := a.Stats(); got.DevicePages != pagesNeeded || got.HostBlocks != victimCount || got.FreeDevicePages != 0 {
				t.Fatalf("residency after admission = %+v", got)
			}
			admitted := make([]Allocation, 0, pagesNeeded)
			for row := range requests {
				allocations, err := batch.Allocations(requests[row].Key)
				if err != nil {
					t.Fatal(err)
				}
				admitted = append(admitted, allocations...)
			}
			if err := batch.Complete(t.Context(), &fakeCompletion{}); err != nil {
				t.Fatal(err)
			}
			for i := len(admitted) - 1; i >= 0; i-- {
				if err := a.Retire(t.Context(), admitted[i], &fakeCompletion{}); err != nil {
					t.Fatal(err)
				}
			}
			for i := len(victims) - 1; i >= 0; i-- {
				if err := a.Restore(t.Context(), victims[i]); err != nil {
					t.Fatal(err)
				}
				if err := a.Retire(t.Context(), victims[i], &fakeCompletion{}); err != nil {
					t.Fatal(err)
				}
			}
			if got := a.Stats(); got.DevicePages != 0 || got.HostBlocks != 0 || got.FreeDevicePages != pagesNeeded {
				t.Fatalf("final residency = %+v", got)
			}
			if err := metadata.Close(); err != nil {
				t.Fatal(err)
			}
			if err := a.Close(); err != nil {
				t.Fatal(err)
			}
			t.Logf("receipt batch=%d lengths=%v pages=%d victims=%d migration_bytes=%d reclaimed=true", len(test.lengths), test.lengths, pagesNeeded, victimCount, stats.MigrationBytes)
		})
	}
}

func TestCUDAPackedAdmissionFailure(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	a, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: 1, HeadDim: 4, BlockSize: 2, TotalPages: 2, DType: "float32", Domain: "cuda:0"})
	if err != nil {
		t.Fatal(err)
	}
	victims := make([]Allocation, 2)
	for i := range victims {
		victims[i], err = a.Allocate(fmt.Sprintf("victim-%d", i), kvcache.Range{Start: 0, End: 2})
		if err != nil {
			t.Fatal(err)
		}
	}
	metadata, err := NewPackedMetadata(a, dev, 1, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	request := []PackedRequest{{BatchRequest: BatchRequest{Key: "row", Generation: 1, Length: 3, Position: 2}, Positions: []int{0, 1, 2}}}
	if _, _, err := metadata.Admit(t.Context(), request, 2, &nthCompletion{fail: 2}); err == nil {
		t.Fatal("admission succeeded after second victim completion failed")
	}
	if got := a.Stats(); got.DevicePages != 2 || got.HostBlocks != 0 || got.FreeDevicePages != 0 {
		t.Fatalf("failed admission residency = %+v", got)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, _, err := metadata.Admit(canceled, request, 2, &fakeCompletion{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled admission = %v, want context canceled", err)
	}
	if got := a.Stats(); got.DevicePages != 2 || got.HostBlocks != 0 {
		t.Fatalf("canceled admission residency = %+v", got)
	}
	for _, victim := range victims {
		if err := a.Retire(t.Context(), victim, &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	if err := metadata.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	t.Log("receipt partial_victims=2 rollback_device_pages=2 rollback_host_pages=0 canceled_mutations=0 reclaimed=true")
}

func TestCUDAPackedConcurrentAdmission(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	a, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: 1, HeadDim: 4, BlockSize: 2, TotalPages: 4, DType: "float32", Domain: "cuda:0"})
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := NewPackedMetadata(a, dev, 1, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		batch *PackedBatch
		err   error
	}
	start := make(chan struct{})
	results := make(chan result, 2)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			request := PackedRequest{
				BatchRequest: BatchRequest{Key: fmt.Sprintf("row-%d", i), Generation: 1, Length: 3, Position: 2},
				Positions:    []int{0, 1, 2},
			}
			batch, _, err := metadata.Admit(t.Context(), []PackedRequest{request}, 2, &fakeCompletion{})
			results <- result{batch: batch, err: err}
		}(i)
	}
	close(start)
	wg.Wait()
	close(results)
	var winner *PackedBatch
	successes, busy := 0, 0
	for result := range results {
		if result.err == nil {
			successes++
			winner = result.batch
		} else if strings.Contains(result.err.Error(), "packed metadata busy") {
			busy++
		} else {
			t.Fatalf("concurrent admission error = %v", result.err)
		}
	}
	if successes != 1 || busy != 1 {
		t.Fatalf("concurrent admissions: successes=%d busy=%d, want 1/1", successes, busy)
	}
	if got := a.Stats(); got.DevicePages != 2 || got.HostBlocks != 0 || got.FreeDevicePages != 2 {
		t.Fatalf("concurrent admission residency = %+v", got)
	}
	allocations, err := winner.Allocations("row-0")
	if err != nil {
		allocations, err = winner.Allocations("row-1")
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := winner.Complete(t.Context(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	for _, allocation := range allocations {
		if err := a.Retire(t.Context(), allocation, &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	if err := metadata.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	t.Log("receipt concurrent=2 published=1 rolled_back=1 leaked_pages=0")
}
