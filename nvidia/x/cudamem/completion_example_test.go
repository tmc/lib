package cudamem_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func ExampleCompletion() {
	var completion cudamem.Completion // An empty transfer is already complete.
	fmt.Println(completion.Close())
	// Output: <nil>
}

func ExampleCompletion_Wait() {
	var completion cudamem.Completion
	fmt.Println(completion.Wait(context.Background()))
	// Output: <nil>
}

func ExampleCompletion_Close() {
	var completion cudamem.Completion
	fmt.Println(completion.Close(), completion.Close())
	// Output: <nil> <nil>
}

func ExampleSlice_CopyFromPinnedAsync() {
	var dst cudamem.Slice[float32]
	var src cudamem.HostAllocation[float32]
	_, err := dst.CopyFromPinnedAsync(src, nil)
	fmt.Println(err)
	// Output: cudamem: nil transfer stream
}

func ExampleSlice_CopyToPinnedAsync() {
	var src cudamem.Slice[float32]
	var dst cudamem.HostAllocation[float32]
	_, err := src.CopyToPinnedAsync(dst, nil)
	fmt.Println(err)
	// Output: cudamem: nil transfer stream
}
