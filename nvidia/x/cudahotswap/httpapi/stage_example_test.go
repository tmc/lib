package httpapi

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

func ExampleStageService() {
	// The real service implements the optional extension without adding
	// stage methods to kernel-only transport implementations.
	var service StageService = (*cudahotswap.Service)(nil)
	fmt.Printf("%T\n", service)
	// Output: *cudahotswap.Service
}

func ExampleStageBinding() {
	// Supply the application's retained Builder before passing this binding
	// to New. A host-only C stage has no buffers, functions or CUDA stream.
	binding := StageBinding{Schema: cudahotswap.StageSpec{HostOnly: true}}
	fmt.Println(binding.Schema.HostOnly, len(binding.BufferNames), binding.CUDA)
	// Output: true 0 false
}
