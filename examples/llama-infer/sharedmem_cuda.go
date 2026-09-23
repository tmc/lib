//go:build !darwin && !tpu

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// boundedKernel is a kernel whose dynamic shared-memory ceiling has been
// raised to the device's opt-in maximum and recorded, so that a launch can be
// checked against it before it is submitted.
//
// UNQUALIFIED: no host that has built this has a CUDA device. The opt-in
// itself is the documented CUDA mechanism, cuFuncSetAttribute with
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES; what has not been observed
// is this binding of it succeeding, or a dense decode running past 12288
// tokens because of it. TestDynamicSharedLimit and TestSharedMemFits cover the
// arithmetic on any host.
type boundedKernel struct {
	fn *cuda.Kernel

	// sharedLimit is the most dynamic shared memory one block may request
	// after the opt-in.
	sharedLimit int
}

// newBoundedKernel raises fn's dynamic shared-memory ceiling as far as dev
// allows and returns the kernel with the resulting limit.
func newBoundedKernel(dev *cuda.Device, fn *cuda.Kernel) (*boundedKernel, error) {
	if dev == nil || fn == nil {
		return nil, fmt.Errorf("bounded kernel: nil device or kernel")
	}
	perBlock, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlock)
	if err != nil {
		return nil, err
	}
	optin, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		return nil, err
	}
	static, err := fn.Attribute(driver.FuncAttributeSharedSizeBytes)
	if err != nil {
		return nil, err
	}
	limit := dynamicSharedLimit(optin, perBlock, static)
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, limit); err != nil {
		return nil, fmt.Errorf("raise dynamic shared memory to %d bytes: %w", limit, err)
	}
	return &boundedKernel{fn: fn, sharedLimit: limit}, nil
}

// sharedMem checks a dynamic shared-memory request against the kernel's
// limit and returns it in the width the launcher takes.
func (k *boundedKernel) sharedMem(bytes int) (uint32, error) {
	if k == nil {
		return 0, fmt.Errorf("bounded kernel: nil")
	}
	if err := sharedMemFits(bytes, k.sharedLimit); err != nil {
		return 0, err
	}
	return uint32(bytes), nil
}
