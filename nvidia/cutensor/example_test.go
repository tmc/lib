package cutensor_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cutensor"
	"github.com/tmc/lib/nvidia/driver"
)

// Example creates a cuTENSOR handle and describes a rank-2 tensor.
// It returns early when the cuTENSOR library is not available. It has no
// Output comment, so it is compiled but not run: what it prints depends on
// the host's cuTENSOR.
func Example() {
	h, err := cutensor.Create()
	if err != nil {
		fmt.Println("cuTENSOR unavailable:", err)
		return
	}
	defer h.Destroy()
	version, err := cutensor.GetVersion()
	if err != nil {
		fmt.Println("version:", err)
		return
	}
	cudart, err := cutensor.GetCudartVersion()
	if err != nil {
		fmt.Println("CUDA runtime version:", err)
		return
	}
	fmt.Println("cuTENSOR version:", version)
	fmt.Println("CUDA runtime version:", cudart)

	// A 64x32 float32 tensor with dense (nil) strides, 128-byte aligned.
	extent := []int64{64, 32}
	desc, err := h.CreateTensorDescriptor(extent, nil, cutensor.R32F, 128)
	if err != nil {
		fmt.Println("tensor descriptor:", err)
		return
	}
	defer desc.Destroy()
	fmt.Println("tensor descriptor ready")
}

// ExampleHandle_Contract computes the tensor contraction
// C[m,n] = A[m,k] * B[k,n] in single precision. It returns early when the
// cuTENSOR library is not available. The device buffers dA, dB, and dC
// are CUDA device pointers, e.g. from the driver package's MemAlloc.
func ExampleHandle_Contract() {
	h, err := cutensor.Create()
	if err != nil {
		fmt.Println("cuTENSOR unavailable:", err)
		return
	}
	defer h.Destroy()

	compute32f, err := cutensor.ComputeDesc32F()
	if err != nil {
		fmt.Println("compute descriptor:", err)
		return
	}

	const m, n, k = 64, 48, 32
	newDesc := func(extent ...int64) cutensor.TensorDescriptor {
		d, err := h.CreateTensorDescriptor(extent, nil, cutensor.R32F, 128)
		if err != nil {
			panic(err)
		}
		return d
	}
	descA := newDesc(m, k)
	descB := newDesc(k, n)
	descC := newDesc(m, n)
	defer descA.Destroy()
	defer descB.Destroy()
	defer descC.Destroy()

	modeA := []int32{'m', 'k'}
	modeB := []int32{'k', 'n'}
	modeC := []int32{'m', 'n'}
	op, err := h.CreateContraction(
		descA, modeA, cutensor.OpIdentity,
		descB, modeB, cutensor.OpIdentity,
		descC, modeC, cutensor.OpIdentity,
		descC, modeC, compute32f)
	if err != nil {
		fmt.Println("contraction:", err)
		return
	}
	defer op.Destroy()

	pref, err := h.CreatePlanPreference(cutensor.AlgoDefault, cutensor.JitModeNone)
	if err != nil {
		fmt.Println("plan preference:", err)
		return
	}
	defer pref.Destroy()

	wsSize, err := h.EstimateWorkspaceSize(op, pref, cutensor.WorkspaceDefault)
	if err != nil {
		fmt.Println("workspace:", err)
		return
	}
	plan, err := h.CreatePlan(op, pref, wsSize)
	if err != nil {
		fmt.Println("plan:", err)
		return
	}
	defer plan.Destroy()

	var dA, dB, dC, dWorkspace driver.DevicePtr // device allocations
	alpha, beta := float32(1), float32(0)
	err = h.Contract(plan,
		cutensor.Float32Scalar(alpha), dA, dB,
		cutensor.Float32Scalar(beta), dC, dC,
		dWorkspace, wsSize, driver.Stream(0))
	if err != nil {
		fmt.Println("contract:", err)
		return
	}
	fmt.Println("contraction enqueued")
}
