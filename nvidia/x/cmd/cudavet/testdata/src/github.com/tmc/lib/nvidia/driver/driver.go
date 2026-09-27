// Package driver is a declaration-only stand-in for the real driver package,
// carrying the handle types, calls and signatures the checks look for.
package driver

import "unsafe"

type Result int32

func (r Result) Error() string { return "" }

const (
	Success         Result = 0
	ErrInvalidValue Result = 1
	ErrNotReady     Result = 600
)

type CallError struct {
	Func   string
	Result Result
}

func (e *CallError) Error() string { return "" }
func (e *CallError) Unwrap() error { return e.Result }

func Init(flags uint32) error { return nil }
func CtxSynchronize() error   { return nil }
func ProfilerStart() error    { return nil }
func ProfilerStop() error     { return nil }

type Device int32

func DeviceGet(ordinal int) (Device, error)                       { return 0, nil }
func (d Device) ComputeCapability() (major, minor int, err error) { return 0, 0, nil }
func (d Device) PrimaryCtxRetain() (Context, error)               { return 0, nil }
func (d Device) PrimaryCtxRelease() error                         { return nil }
func (d Device) PrimaryCtxReset() error                           { return nil }

type Limit uint32

const LimitPersistingL2CacheSize Limit = 0x06

func CtxSetLimit(limit Limit, value uint) error { return nil }

type CtxFlags uint32

type Context uintptr

func NewContext(dev Device, flags CtxFlags) (Context, error) { return 0, nil }
func (c Context) SetCurrent() error                          { return nil }
func (c Context) PushCurrent() error                         { return nil }
func (c *Context) Close() error                              { return nil }

type DevicePtr uint64

func MemAlloc[T any](n int) (DevicePtr, error)                              { return 0, nil }
func MemAllocBytes(bytesize uint) (DevicePtr, error)                        { return 0, nil }
func MemAllocManaged[T any](n int, flags MemattachFlags) (DevicePtr, error) { return 0, nil }

type MemattachFlags int32

const MemAttachGlobal MemattachFlags = 0x1

func (d DevicePtr) Free() error                                                { return nil }
func (d DevicePtr) FreeAsync(hStream Stream) error                             { return nil }
func (d DevicePtr) CopyFromHost[T any](srcHost []T) error                      { return nil }
func (d DevicePtr) CopyToHost[T any](dstHost []T) error                        { return nil }
func (d DevicePtr) CopyToHostPtr(dstHost unsafe.Pointer, byteCount uint) error { return nil }
func (d DevicePtr) CopyFromHostAsync[T any](srcHost []T, hStream Stream) error { return nil }
func (d DevicePtr) CopyToHostAsync[T any](dstHost []T, hStream Stream) error   { return nil }
func (d DevicePtr) CopyFromHostAsyncPtr(srcHost unsafe.Pointer, byteCount uint, hStream Stream) error {
	return nil
}
func (d DevicePtr) CopyToHostAsyncPtr(dstHost unsafe.Pointer, byteCount uint, hStream Stream) error {
	return nil
}
func (d DevicePtr) MemsetD8Async(uc uint8, n uint, hStream Stream) error { return nil }

type Array uintptr

func (a Array) CopyFromHostAsyncPtr(dstOffset uint, srcHost unsafe.Pointer, byteCount uint, hStream Stream) error {
	return nil
}

type StreamFlags uint32

const (
	StreamDefault     StreamFlags = 0x0
	StreamNonBlocking StreamFlags = 0x1
)

type Stream uintptr

func NewStream(flags StreamFlags) (Stream, error)                   { return 0, nil }
func (s *Stream) Close() error                                      { return nil }
func (s Stream) Synchronize() error                                 { return nil }
func (s Stream) Query() error                                       { return nil }
func (s Stream) WaitEvent(hEvent Event, flags EventWaitFlags) error { return nil }
func (s Stream) MemAllocAsync[T any](n int) (DevicePtr, error)      { return 0, nil }

type Event uintptr

type EventFlags uint32

const EventDefault EventFlags = 0x0

type EventWaitFlags int32

const EventWaitDefault EventWaitFlags = 0x0

func NewEvent(flags EventFlags) (Event, error) { return 0, nil }
func (e *Event) Close() error                  { return nil }
func (e Event) Record(hStream Stream) error    { return nil }
func (e Event) Synchronize() error             { return nil }
func (e Event) Query() error                   { return nil }

type Dim3 struct{ X, Y, Z uint32 }

func Dims(x, y, z uint32) Dim3     { return Dim3{x, y, z} }
func GridFor(n, block uint32) Dim3 { return Dim3{} }

type FunctionAttribute int32

const FuncAttributeMaxDynamicSharedSizeBytes FunctionAttribute = 8

type Function uintptr

func (f Function) LaunchKernel(gridDim Dim3, blockDim Dim3, sharedMemBytes uint32, hStream Stream, kernelParams []unsafe.Pointer) error {
	return nil
}
func (f Function) LaunchKernelEx(config *LaunchConfig, kernelParams []unsafe.Pointer, extra *uintptr) error {
	return nil
}
func (f Function) LaunchGrid(gridWidth int32, gridHeight int32) error       { return nil }
func (f Function) SetAttribute(attrib FunctionAttribute, value int32) error { return nil }

type LaunchConfig struct {
	GridDim        Dim3
	BlockDim       Dim3
	SharedMemBytes uint32
	HStream        Stream
}

type Graph uintptr

type GraphNode uintptr

type KernelNodeParams struct {
	KernelParams uintptr
	Extra        uintptr
}

type KernelNode struct{}

func (k KernelNode) SetParams(nodeParams *KernelNodeParams) error { return nil }

func (g Graph) AddKernelNode(dependencies *GraphNode, numDependencies uint, nodeParams *KernelNodeParams) (KernelNode, error) {
	return KernelNode{}, nil
}

type GraphExec uintptr

type GraphinstantiateFlags int32

func NewGraph(flags uint32) (Graph, error)                                 { return 0, nil }
func (g Graph) Clone() (Graph, error)                                      { return 0, nil }
func (g Graph) Instantiate(flags GraphinstantiateFlags) (GraphExec, error) { return 0, nil }
func (g Graph) Destroy() error                                             { return nil }
func (g GraphExec) Launch(hStream Stream) error                            { return nil }
func (g *GraphExec) Close() error                                          { return nil }

type JITOption int32

const (
	JITInfoLogBuffer          JITOption = 3
	JITInfoLogBufferSizeBytes JITOption = 4
)

type Module uintptr

func ModuleLoadDataEx(image []byte, numOptions uint32, options *JITOption, optionValues *uintptr) (Module, error) {
	return 0, nil
}
func (m *Module) Close() error { return nil }

type LinkState uintptr

func LinkCreate(numOptions uint32, options *JITOption, optionValues *uintptr) (LinkState, error) {
	return 0, nil
}
func (l LinkState) Complete() (cubinOut uintptr, sizeOut uint, err error) { return 0, 0, nil }
func (l *LinkState) Close() error                                         { return nil }

type TensorMap struct{ Opaque [16]uint64 }

type TensorMapDataType int32

const TensorMapDataTypeFloat32 TensorMapDataType = 7

type TensorMapInterleave int32

type TensorMapSwizzle int32

type TensorMapL2promotion int32

type TensorMapFloatOOBfill int32

func TensorMapEncodeTiled(tensorMap *TensorMap, tensorDataType TensorMapDataType, tensorRank uint32, globalAddress uintptr, globalDim *uint64, globalStrides *uint64, boxDim *uint32, elementStrides *uint32, interleave TensorMapInterleave, swizzle TensorMapSwizzle, l2Promotion TensorMapL2promotion, oobFill TensorMapFloatOOBfill) error {
	return nil
}
