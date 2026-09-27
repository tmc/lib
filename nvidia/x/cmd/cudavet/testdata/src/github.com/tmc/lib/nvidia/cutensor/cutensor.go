// Package cutensor is a declaration-only stand-in for the real cutensor
// package.
package cutensor

import "github.com/tmc/lib/nvidia/driver"

type DataType int32

const (
	R32F DataType = 0
	C32F DataType = 4
	R64F DataType = 1
	C64F DataType = 5
)

type OperatorT int32

const OpIdentity OperatorT = 1

type WorksizepreferenceT int32

const WorkspaceDefault WorksizepreferenceT = 2

type ComputeDescriptor uintptr

func ComputeDesc16F() (ComputeDescriptor, error)    { return 0, nil }
func ComputeDesc32F() (ComputeDescriptor, error)    { return 0, nil }
func ComputeDesc64F() (ComputeDescriptor, error)    { return 0, nil }
func ComputeDesc4X16F() (ComputeDescriptor, error)  { return 0, nil }
func ComputeDesc8XInt8() (ComputeDescriptor, error) { return 0, nil }
func ComputeDesc9X16BF() (ComputeDescriptor, error) { return 0, nil }

type Scalar struct{}

func Float32Scalar(x float32) Scalar       { return Scalar{} }
func Float64Scalar(x float64) Scalar       { return Scalar{} }
func Complex64Scalar(x complex64) Scalar   { return Scalar{} }
func Complex128Scalar(x complex128) Scalar { return Scalar{} }
func Int32Scalar(x int32) Scalar           { return Scalar{} }

func GetVersion() (uint, error) { return 0, nil }

type Handle uintptr

func Create() (Handle, error)  { return 0, nil }
func (h *Handle) Close() error { return nil }

type TensorDescriptor uintptr

func (t *TensorDescriptor) Close() error { return nil }

func (h Handle) CreateTensorDescriptor(extent []int64, stride []int64, dataType DataType, alignmentRequirement uint32) (TensorDescriptor, error) {
	return 0, nil
}

type OperationDescriptor uintptr

func (o *OperationDescriptor) Close() error { return nil }

func (h Handle) CreateContraction(descA TensorDescriptor, modeA []int32, opA OperatorT, descB TensorDescriptor, modeB []int32, opB OperatorT, descC TensorDescriptor, modeC []int32, opC OperatorT, descD TensorDescriptor, modeD []int32, descCompute ComputeDescriptor) (OperationDescriptor, error) {
	return 0, nil
}

func (h Handle) CreateContractionTrinary(descA TensorDescriptor, modeA []int32, opA OperatorT, descB TensorDescriptor, modeB []int32, opB OperatorT, descC TensorDescriptor, modeC []int32, opC OperatorT, descD TensorDescriptor, modeD []int32, opD OperatorT, descE TensorDescriptor, modeE []int32, descCompute ComputeDescriptor) (OperationDescriptor, error) {
	return 0, nil
}

type PlanPreference uintptr

type Plan uintptr

func (p *Plan) Close() error { return nil }

func (h Handle) EstimateWorkspaceSize(desc OperationDescriptor, planPref PlanPreference, workspacePref WorksizepreferenceT) (uint64, error) {
	return 0, nil
}

func (h Handle) CreatePlan(desc OperationDescriptor, pref PlanPreference, workspaceSizeLimit uint64) (Plan, error) {
	return 0, nil
}

func (h Handle) Contract(plan Plan, alpha Scalar, a driver.DevicePtr, b driver.DevicePtr, beta Scalar, c driver.DevicePtr, d driver.DevicePtr, workspace driver.DevicePtr, workspaceSize uint64, stream driver.Stream) error {
	return nil
}

func (h Handle) ContractTrinary(plan Plan, alpha Scalar, a driver.DevicePtr, b driver.DevicePtr, c driver.DevicePtr, beta Scalar, d driver.DevicePtr, e driver.DevicePtr, workspace driver.DevicePtr, workspaceSize uint64, stream driver.Stream) error {
	return nil
}
