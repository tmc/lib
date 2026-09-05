package cutensornet_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cutensornet"
)

// Example creates a cuTensorNet library handle and reports the loaded version.
// It returns early when libcutensornet is not available.
//
// These examples have no Output comment, so they are compiled but not run: they
// need an NVIDIA GPU and a cuQuantum installation, and a machine without them
// would print a different thing.
func Example() {
	h, err := cutensornet.Create()
	if err != nil {
		fmt.Println("cuTensorNet unavailable:", err)
		return
	}
	defer h.Destroy()
	fmt.Println("cuTensorNet version:", cutensornet.GetVersion())
	fmt.Println("loaded from:", cutensornet.Path())
}

// ExampleHandle_NewNetwork describes the matrix product C[m,n] = A[m,k] B[k,n]
// as a two-tensor network. The operands are stored column-major, stated with
// explicit strides: cuTensorNet operands are strided, and a row-major buffer
// would need strides of {K, 1} and {N, 1} instead.
func ExampleHandle_NewNetwork() {
	h, err := cutensornet.Create()
	if err != nil {
		fmt.Println("cuTensorNet unavailable:", err)
		return
	}
	defer h.Destroy()

	const M, K, N = 64, 32, 48
	a := cutensornet.Tensor{
		Modes:   []int32{'m', 'k'},
		Extents: []int64{M, K},
		Strides: []int64{1, M},
	}
	b := cutensornet.Tensor{
		Modes:   []int32{'k', 'n'},
		Extents: []int64{K, N},
		Strides: []int64{1, K},
	}
	c := cutensornet.Tensor{
		Modes:   []int32{'m', 'n'},
		Extents: []int64{M, N},
		Strides: []int64{1, M},
	}

	net, err := h.NewNetwork([]cutensornet.Tensor{a, b}, &c, cutensornet.R32F, cutensornet.Compute32F)
	if err != nil {
		fmt.Println("describe network:", err)
		return
	}
	defer net.Close()
	fmt.Println("network with", net.NumInputs(), "input tensors")
}

// ExampleWorkspace shows the workspace protocol: find a contraction path, ask
// how much scratch memory it needs, allocate that much, hand it over, and let
// the [cutensornet.Workspace] decide when it may be released.
//
// alloc and free stand in for a CUDA device allocator, such as MemAlloc and
// DevicePtr.MemFree in github.com/tmc/lib/nvidia/driver. Nothing in the
// cutensornet package allocates device memory.
func ExampleWorkspace() {
	var (
		alloc = func(n int64) (uintptr, error) { return 0, fmt.Errorf("no allocator in this example") }
		free  = func(uintptr) {}
	)

	h, err := cutensornet.Create()
	if err != nil {
		fmt.Println("cuTensorNet unavailable:", err)
		return
	}
	defer h.Destroy()

	inputs := []cutensornet.Tensor{
		{Modes: []int32{'m', 'k'}, Extents: []int64{64, 32}},
		{Modes: []int32{'k', 'n'}, Extents: []int64{32, 48}},
	}
	net, err := h.NewNetwork(inputs, nil, cutensornet.R32F, cutensornet.Compute32F)
	if err != nil {
		fmt.Println("describe network:", err)
		return
	}
	defer net.Close()

	cfg, err := h.CreateContractionOptimizerConfig()
	if err != nil {
		fmt.Println("optimizer config:", err)
		return
	}
	defer cfg.Destroy()
	info, err := h.CreateContractionOptimizerInfo(net.Descriptor())
	if err != nil {
		fmt.Println("optimizer info:", err)
		return
	}
	defer info.Destroy()
	if err := h.ContractionOptimize(net.Descriptor(), cfg, 64<<20, info); err != nil {
		fmt.Println("find contraction path:", err)
		return
	}

	ws, err := h.NewWorkspace()
	if err != nil {
		fmt.Println("workspace:", err)
		return
	}
	if err := ws.ComputeContractionSizes(net, info); err != nil {
		fmt.Println("size workspace:", err)
		return
	}
	size, err := ws.Size(cutensornet.WorksizePrefRecommended, cutensornet.MemspaceDevice, cutensornet.WorkspaceScratch)
	if err != nil {
		fmt.Println("workspace size:", err)
		return
	}
	buf, err := alloc(size)
	if err != nil {
		fmt.Println("allocate workspace:", err)
		return
	}
	// The deallocator is handed to the workspace, not deferred here: the
	// library holds the buffer until Close, which refuses while a plan is live.
	if err := ws.Provide(cutensornet.MemspaceDevice, cutensornet.WorkspaceScratch,
		buf, size, func() { free(buf) }); err != nil {
		free(buf)
		fmt.Println("provide workspace:", err)
		return
	}
	defer ws.Close()

	fmt.Printf("workspace of %d bytes provided\n", size)
}
