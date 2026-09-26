// Package cutensor is a declaration-only stand-in for the real cutensor
// package.
package cutensor

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

type DataType int32

const R32F DataType = 0

type ComputeDescriptor uintptr

func ComputeDesc32F() (ComputeDescriptor, error) { return 0, nil }
func ComputeDesc64F() (ComputeDescriptor, error) { return 0, nil }

type Handle uintptr

func Create() (Handle, error)  { return 0, nil }
func (h *Handle) Close() error { return nil }

type TensorDescriptor uintptr

func (d *TensorDescriptor) Close() error { return nil }

func (h Handle) CreateTensorDescriptor(extent []int64, stride []int64, dataType DataType, alignment uint32) (TensorDescriptor, error) {
	return 0, nil
}

type OperationDescriptor uintptr

func (d *OperationDescriptor) Close() error { return nil }

func (h Handle) CreateContraction(descA TensorDescriptor, modeA []int32, descB TensorDescriptor, modeB []int32, descC TensorDescriptor, modeC []int32, descD TensorDescriptor, modeD []int32, descCompute ComputeDescriptor) (OperationDescriptor, error) {
	return 0, nil
}

type PlanPreference uintptr

type Plan uintptr

func (p *Plan) Close() error { return nil }

func (h Handle) EstimateWorkspaceSize(desc OperationDescriptor, pref PlanPreference) (uint64, error) {
	return 0, nil
}

func (h Handle) CreatePlan(desc OperationDescriptor, pref PlanPreference, workspaceSizeLimit uint64) (Plan, error) {
	return 0, nil
}

func (h Handle) Contract(plan Plan, alpha unsafe.Pointer, a, b driver.DevicePtr, beta unsafe.Pointer, c, d, workspace driver.DevicePtr, workspaceSize uint64, s driver.Stream) error {
	return nil
}
