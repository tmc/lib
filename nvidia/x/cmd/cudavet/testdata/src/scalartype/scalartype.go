package scalartype

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cutensor"
	"github.com/tmc/lib/nvidia/driver"
)

func contract(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc32F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, dB, []int32{'k', 'j'}, dC, []int32{'i', 'j'}, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Destroy()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Destroy()
	alpha, beta := 1.0, 0.0
	return h.Contract(plan, unsafe.Pointer(&alpha), a, b, unsafe.Pointer(&beta), c, c, 0, 0, s) // want `alpha is float64, but the operation computes in ComputeDesc32F and reads it as float32` `beta is float64, but the operation computes in ComputeDesc32F and reads it as float32`
}

func contract32(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc32F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, dB, []int32{'k', 'j'}, dC, []int32{'i', 'j'}, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Destroy()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Destroy()
	alpha, beta := float32(1), float32(0)
	return h.Contract(plan, unsafe.Pointer(&alpha), a, b, unsafe.Pointer(&beta), c, c, 0, 0, s)
}

func gemm(h cublas.Handle, a, b, c driver.DevicePtr, n int32) error {
	alpha, beta := float32(1), float32(0)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, uintptr(unsafe.Pointer(&alpha)), a, cublas.R32F, n, b, cublas.R32F, n, uintptr(unsafe.Pointer(&beta)), c, cublas.R32F, n, cublas.Compute64F) // want `alpha is float32, but the operation computes in Compute64F and reads it as float64` `beta is float32, but the operation computes in Compute64F and reads it as float64`
}

func gemm32(h cublas.Handle, a, b, c driver.DevicePtr, n int32) error {
	alpha, beta := float32(1), float32(0)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, uintptr(unsafe.Pointer(&alpha)), a, cublas.R32F, n, b, cublas.R32F, n, uintptr(unsafe.Pointer(&beta)), c, cublas.R32F, n, cublas.Compute32F)
}
