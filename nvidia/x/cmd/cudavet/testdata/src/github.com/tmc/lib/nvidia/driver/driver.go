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

type DevicePtr uintptr

func MemAlloc[T any](n int) (DevicePtr, error)                      { return 0, nil }
func MemAllocBytes(bytesize uint) (DevicePtr, error)                { return 0, nil }
func MemAllocManaged[T any](n int, flags uint32) (DevicePtr, error) { return 0, nil }

func (d DevicePtr) Free() error                                                         { return nil }
func (d DevicePtr) FreeAsync(s Stream) error                                            { return nil }
func (d DevicePtr) CopyFromHost[T any](srcHost []T) error                               { return nil }
func (d DevicePtr) CopyToHost[T any](dstHost []T) error                                 { return nil }
func (d DevicePtr) CopyToHostPtr(dstHost unsafe.Pointer, n uint) error                  { return nil }
func (d DevicePtr) CopyFromHostAsync[T any](srcHost []T, s Stream) error                { return nil }
func (d DevicePtr) CopyToHostAsync[T any](dstHost []T, s Stream) error                  { return nil }
func (d DevicePtr) CopyFromHostAsyncPtr(srcHost unsafe.Pointer, n uint, s Stream) error { return nil }
func (d DevicePtr) CopyToHostAsyncPtr(dstHost unsafe.Pointer, n uint, s Stream) error   { return nil }
func (d DevicePtr) MemsetD8Async(uc uint8, n uint, s Stream) error                      { return nil }

type Array uintptr

func (a Array) CopyFromHostAsyncPtr(dstOffset uint, srcHost unsafe.Pointer, n uint, s Stream) error {
	return nil
}

type StreamFlags uint32

type Stream uintptr

func NewStream(flags uint32) (Stream, error)                   { return 0, nil }
func (s *Stream) Close() error                                 { return nil }
func (s Stream) Synchronize() error                            { return nil }
func (s Stream) Query() (done bool, err error)                 { return false, nil }
func (s Stream) WaitEvent(e Event, flags uint32) error         { return nil }
func (s Stream) MemAllocAsync[T any](n int) (DevicePtr, error) { return 0, nil }

type Event uintptr

func NewEvent(flags uint32) (Event, error)    { return 0, nil }
func (e *Event) Close() error                 { return nil }
func (e Event) Record(s Stream) error         { return nil }
func (e Event) Synchronize() error            { return nil }
func (e Event) Query() (done bool, err error) { return false, nil }

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

type GraphExec uintptr

func NewGraph(flags uint32) (Graph, error)                  { return 0, nil }
func (g Graph) Clone() (Graph, error)                       { return 0, nil }
func (g Graph) Instantiate(flags uint64) (GraphExec, error) { return 0, nil }
func (g Graph) Destroy() error                              { return nil }
func (g GraphExec) Launch(s Stream) error                   { return nil }
func (g *GraphExec) Close() error                           { return nil }

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

func TensorMapEncodeTiled(tensorMap *TensorMap, tensorDataType int32, tensorRank uint32, globalAddress uintptr, globalDim *uint64, globalStrides *uint64, boxDim *uint32, elementStrides *uint32) error {
	return nil
}

// Batch has the count-after shape: the count follows its own array.
func Batch(ids *int32, count uint32, values *float64) error { return nil }
