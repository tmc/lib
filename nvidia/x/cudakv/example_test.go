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
