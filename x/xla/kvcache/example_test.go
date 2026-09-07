package kvcache_test

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/lib/x/xla"
	"github.com/tmc/lib/x/xla/kvcache"
)

// This example runs when a caller supplies a CPU plugin. Inputs and output are
// one unbatched f32 value, encoded as little-endian host bytes.
func Example() {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		return
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer engine.Close()
	cache, err := kvcache.New(engine, kvcache.Config{Capacity: 8, KeySize: 1, ValueSize: 1, ElementType: xla.F32})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cache.Close()
	device, err := cache.Device()
	if err != nil {
		fmt.Println(err)
		return
	}
	input, err := engine.NewBufferFromHost([]byte{0, 0, 128, 63}, xla.F32, []int64{1, 1}, device)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer input.Close()
	output, err := cache.Append(context.Background(), input, input, input)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer output.Close()
	if cache.Len() != 1 {
		fmt.Println("unexpected cache length")
	}
	// Output:
}

func ExampleCache_Len() {
	var cache kvcache.Cache
	fmt.Println(cache.Len()) // Output: 0
}
func ExampleCache_Close() {
	var cache kvcache.Cache
	fmt.Println(cache.Close()) // Output: <nil>
}
func ExampleCache_Reset() {
	var cache kvcache.Cache
	fmt.Println(cache.Reset()) // Output: kvcache: cache is closed
}
func ExampleCache_Device() {
	var cache kvcache.Cache
	_, err := cache.Device()
	fmt.Println(err) // Output: kvcache: cache is closed
}
func ExampleCache_Append() {
	var cache kvcache.Cache
	_, err := cache.Append(context.Background(), nil, nil, nil)
	fmt.Println(err) // Output: kvcache: cache is closed
}
func ExampleNew() {
	_, err := kvcache.New(nil, kvcache.Config{})
	fmt.Println(err) // Output: kvcache: f32, f16 or bf16 elements required
}
