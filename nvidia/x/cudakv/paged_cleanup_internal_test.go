package cudakv

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestPagedConstructionOwnership(t *testing.T) {
	failure := errors.New("allocation failed")
	for failAt := range 4 {
		calls := 0
		pc, err := newPagedCache(nil, nil, 2, 2, 4, 8, 2, func(n int) (cudamem.Slice[float32], error) {
			if n != 128 {
				t.Fatalf("elements=%d", n)
			}
			index := calls
			calls++
			if index == failAt {
				return cudamem.Slice[float32]{}, failure
			}
			return cudamem.Slice[float32]{}, nil
		})
		if !errors.Is(err, failure) || pc == nil || calls != failAt+1 {
			t.Fatalf("calls=%d error=%v", calls, err)
		}
		if pc.NumFreeBlocks() != 0 {
			t.Fatal("partial cache advertised capacity")
		}
		if err := pc.AllocSequence(1, 1); err == nil {
			t.Fatal("partial cache admitted sequence")
		}
		if _, err := pc.Page(0, 0); err == nil {
			t.Fatal("partial page exposed")
		}
		if err := pc.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPagedCloseRetry(t *testing.T) {
	failure := errors.New("free failed")
	for failAt := range 4 {
		pc, err := NewTestPagedCache(2, 1, 1, 2, 4)
		if err != nil {
			t.Fatal(err)
		}
		pc.kBlocks = make([]cudamem.Slice[float32], 2)
		pc.vBlocks = make([]cudamem.Slice[float32], 2)
		if err := pc.AllocSequence(1, 1); err != nil {
			t.Fatal(err)
		}
		calls := 0
		if err := pc.closeWith(func(cudamem.Slice[float32]) error {
			index := calls
			calls++
			if index == failAt {
				return failure
			}
			return nil
		}); !errors.Is(err, failure) {
			t.Fatal(err)
		}
		if pc.freed != failAt || pc.kBlocks == nil {
			t.Fatal("lost cleanup progress")
		}
		if pc.NumFreeBlocks() != 0 || len(pc.BlockTable(1)) != 0 {
			t.Fatal("closed cache published blocks")
		}
		if _, err := pc.Page(0, 0); err == nil {
			t.Fatal("closing page accessible")
		}
		if err := pc.AppendToken(1); err == nil {
			t.Fatal("closing cache advanced")
		}
		if err := pc.copyBlockAsync(0, 1, nil); err == nil {
			t.Fatal("closing pool copied")
		}
		var wg sync.WaitGroup
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := pc.closeWith(func(cudamem.Slice[float32]) error { calls++; return nil }); err != nil {
					t.Error(err)
				}
			}()
		}
		wg.Wait()
		if calls != 5 || pc.kBlocks != nil || pc.vBlocks != nil || pc.hostK != nil || pc.hostV != nil {
			t.Fatalf("cleanup calls=%d", calls)
		}
	}
}
