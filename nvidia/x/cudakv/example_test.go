package cudakv_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleRollingKVCache() {
	cache, err := cudakv.NewRollingKVCache(cudakv.SlidingWindowConfig{
		SinkTokens: 1,
		WindowSize: 2,
	})
	if err != nil {
		panic(err)
	}
	// This example tracks positions only; tensor storage is caller-owned.
	if err := cache.Advance(4); err != nil {
		panic(err)
	}
	fmt.Println(cache.Slot(0)) // The sink token stays resident.
	fmt.Println(cache.Slot(1)) // The rolling window has overwritten token 1.
	fmt.Println(cache.Slot(3))
	// Output:
	// 0 true
	// -1 false
	// 1 true
}

func ExampleCompactor() {
	// Create a pool of 5 blocks, 2 tokens per block.
	cache, err := cudakv.NewTestPagedCache(1, 1, 4, 2, 5)
	if err != nil {
		panic(err)
	}
	defer cache.Close()

	// Allocate 5 sequences filling all 5 blocks.
	for s := int64(1); s <= 5; s++ {
		_ = cache.AllocSequence(s, 2)
	}

	// Free sequences 2 and 4 to create interleaved holes.
	cache.FreeSequence(2)
	cache.FreeSequence(4)

	fmt.Printf("Before compaction: %.2f\n", cache.FragmentationIndex())

	compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy())
	plan, err := compactor.Compact(cache)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Compacted moves: %d\n", len(plan.Moves))
	fmt.Printf("After compaction: %.2f\n", cache.FragmentationIndex())
	fmt.Printf("Seq 5 block table: %v\n", cache.BlockTable(5))

	// Output:
	// Before compaction: 0.50
	// Compacted moves: 1
	// After compaction: 0.00
	// Seq 5 block table: [0]
}

func Example() {
	// Create a paged KV cache with 4 blocks of 2 tokens each.
	cache, err := cudakv.NewTestPagedCache(1, 1, 4, 2, 4)
	if err != nil {
		panic(err)
	}
	defer cache.Close()

	_ = cache.AllocSequence(1, 2)
	_ = cache.AllocSequence(2, 2)
	cache.FreeSequence(1)

	// Proactively compact the paged KV pool.
	plan, _ := cache.Defrag()
	fmt.Printf("Compacted %d blocks, frag index: %.2f\n", len(plan.Moves), cache.FragmentationIndex())
	// Output:
	// Compacted 1 blocks, frag index: 0.00
}
