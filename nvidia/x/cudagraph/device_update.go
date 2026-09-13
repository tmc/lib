package cudagraph

import (
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// DeviceNode represents a CUDA device node handle (cudaGraphDeviceNode_t)
// used for device-side graph updates.
type DeviceNode uintptr

// Uint64 returns the handle as a uint64.
func (d DeviceNode) Uint64() uint64 {
	return uint64(d)
}

// String returns the hexadecimal representation of the device node handle.
func (d DeviceNode) String() string {
	return fmt.Sprintf("%#x", uintptr(d))
}

// KernelNodeField specifies the field to update when performing node updates
// via cudaGraphKernelNodeUpdatesApply.
type KernelNodeField uint32

const (
	KernelNodeFieldInvalid KernelNodeField = 0
	KernelNodeFieldGridDim KernelNodeField = 1
	KernelNodeFieldParam   KernelNodeField = 2
	KernelNodeFieldEnabled KernelNodeField = 3
)

func (f KernelNodeField) String() string {
	switch f {
	case KernelNodeFieldInvalid:
		return "KernelNodeFieldInvalid"
	case KernelNodeFieldGridDim:
		return "KernelNodeFieldGridDim"
	case KernelNodeFieldParam:
		return "KernelNodeFieldParam"
	case KernelNodeFieldEnabled:
		return "KernelNodeFieldEnabled"
	default:
		return fmt.Sprintf("KernelNodeField(%d)", f)
	}
}

// KernelNodeUpdate specifies a single kernel node update, matching the semantics of
// CUDA's cudaGraphKernelNodeUpdate structure for device-side updates.
type KernelNodeUpdate struct {
	Field     KernelNodeField
	Node      DeviceNode
	GridDim   cudalaunch.Dim3
	Offset    uintptr
	Value     unsafe.Pointer
	Size      uintptr
	IsEnabled uint32
}

// NewGridDimUpdate creates a KernelNodeUpdate to update the grid dimensions of node.
func NewGridDimUpdate(node DeviceNode, grid cudalaunch.Dim3) KernelNodeUpdate {
	return KernelNodeUpdate{
		Field:   KernelNodeFieldGridDim,
		Node:    node,
		GridDim: grid,
	}
}

// NewParamUpdate creates a KernelNodeUpdate to update a parameter at offset with size bytes.
func NewParamUpdate(node DeviceNode, offset uintptr, val unsafe.Pointer, size uintptr) KernelNodeUpdate {
	return KernelNodeUpdate{
		Field:  KernelNodeFieldParam,
		Node:   node,
		Offset: offset,
		Value:  val,
		Size:   size,
	}
}

// NewEnabledUpdate creates a KernelNodeUpdate to enable or disable node.
func NewEnabledUpdate(node DeviceNode, enabled bool) KernelNodeUpdate {
	var val uint32
	if enabled {
		val = 1
	}
	return KernelNodeUpdate{
		Field:     KernelNodeFieldEnabled,
		Node:      node,
		IsEnabled: val,
	}
}

// NewParamUpdateForFunc queries the offset and size of the parameter at paramIndex
// of fn using cuFuncGetParamInfo and returns a typed KernelNodeUpdate.
func NewParamUpdateForFunc(node DeviceNode, fn driver.Function, paramIndex uint, val unsafe.Pointer) (KernelNodeUpdate, error) {
	offset, size, err := fn.GetParamInfo(paramIndex)
	if err != nil {
		return KernelNodeUpdate{}, fmt.Errorf("cudagraph: get param info: %w", err)
	}
	return NewParamUpdate(node, uintptr(offset), val, uintptr(size)), nil
}

// DeviceKernelNodeUpdate matches the exact 40-byte C-ABI layout of
// cudaGraphKernelNodeUpdate in the CUDA Runtime.
// It can be copied directly to GPU device memory for consumption by kernels calling
// cudaGraphKernelNodeUpdatesApply.
type DeviceKernelNodeUpdate struct {
	Field      uint32
	_pad       uint32
	Node       uint64
	UpdateData [24]byte
}

// DeviceUpdate converts u into the C-ABI compliant DeviceKernelNodeUpdate struct.
func (u KernelNodeUpdate) DeviceUpdate() DeviceKernelNodeUpdate {
	dev := DeviceKernelNodeUpdate{
		Field: uint32(u.Field),
		Node:  uint64(u.Node),
	}
	switch u.Field {
	case KernelNodeFieldGridDim:
		binary.LittleEndian.PutUint32(dev.UpdateData[0:4], u.GridDim.X)
		binary.LittleEndian.PutUint32(dev.UpdateData[4:8], u.GridDim.Y)
		binary.LittleEndian.PutUint32(dev.UpdateData[8:12], u.GridDim.Z)
	case KernelNodeFieldParam:
		binary.LittleEndian.PutUint64(dev.UpdateData[0:8], uint64(u.Offset))
		binary.LittleEndian.PutUint64(dev.UpdateData[8:16], uint64(uintptr(u.Value)))
		binary.LittleEndian.PutUint64(dev.UpdateData[16:24], uint64(u.Size))
	case KernelNodeFieldEnabled:
		binary.LittleEndian.PutUint32(dev.UpdateData[0:4], u.IsEnabled)
	}
	return dev
}

// Bytes returns the raw 40-byte slice of the device update.
func (d DeviceKernelNodeUpdate) Bytes() []byte {
	b := make([]byte, 40)
	binary.LittleEndian.PutUint32(b[0:4], d.Field)
	binary.LittleEndian.PutUint32(b[4:8], d._pad)
	binary.LittleEndian.PutUint64(b[8:16], d.Node)
	copy(b[16:40], d.UpdateData[:])
	return b
}

// EncodeDeviceUpdates converts a slice of KernelNodeUpdate into a slice of DeviceKernelNodeUpdate.
func EncodeDeviceUpdates(updates []KernelNodeUpdate) []DeviceKernelNodeUpdate {
	if len(updates) == 0 {
		return nil
	}
	res := make([]DeviceKernelNodeUpdate, len(updates))
	for i, u := range updates {
		res[i] = u.DeviceUpdate()
	}
	return res
}

// Params returns a copy of the kernel node's parameters.
func (n *Node) Params() (driver.KernelNodeParams, error) {
	if n == nil || n.raw == 0 {
		return driver.KernelNodeParams{}, errors.New("cudagraph: nil or closed node")
	}
	if n.graph == nil {
		return driver.KernelNodeParams{}, errors.New("cudagraph: node has no owning graph")
	}
	var params driver.KernelNodeParams
	err := n.graph.withContext(func() error {
		var err error
		params, err = n.raw.GraphKernelNodeGetParams()
		return err
	})
	if err != nil {
		return driver.KernelNodeParams{}, fmt.Errorf("cudagraph: get kernel node params: %w", err)
	}
	return params, nil
}

// SetParams sets the kernel node's parameters in the non-instantiated graph.
func (n *Node) SetParams(params *driver.KernelNodeParams) error {
	if n == nil || n.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	if n.graph == nil {
		return errors.New("cudagraph: node has no owning graph")
	}
	if params == nil {
		return errors.New("cudagraph: nil kernel node params")
	}
	err := n.graph.withContext(func() error {
		return n.raw.GraphKernelNodeSetParams(params)
	})
	if err != nil {
		return fmt.Errorf("cudagraph: set kernel node params: %w", err)
	}
	return nil
}

// SetGrid updates the grid dimensions of the kernel node in the non-instantiated graph.
func (n *Node) SetGrid(grid cudalaunch.Dim3) error {
	if grid.X == 0 || grid.Y == 0 || grid.Z == 0 {
		return errors.New("cudagraph: invalid zero grid dimension")
	}
	params, err := n.Params()
	if err != nil {
		return err
	}
	params.GridDimX = grid.X
	params.GridDimY = grid.Y
	params.GridDimZ = grid.Z
	return n.SetParams(&params)
}

// SetDeviceUpdatable marks the kernel node as device-updatable.
// Once enabled, CUDA does not allow opting out, and device-updatable nodes
// cannot be destroyed individually via GraphDestroyNode.
func (n *Node) SetDeviceUpdatable(enable bool) error {
	if n == nil || n.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	if n.graph == nil {
		return errors.New("cudagraph: node has no owning graph")
	}
	var val [8]uint64
	if enable {
		val[0] = 1
	}
	err := n.graph.withContext(func() error {
		return n.raw.GraphKernelNodeSetAttribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode), &val)
	})
	if err != nil {
		return fmt.Errorf("cudagraph: set device updatable: %w", err)
	}
	return nil
}

// IsDeviceUpdatable reports whether this kernel node is configured as device-updatable.
func (n *Node) IsDeviceUpdatable() (bool, error) {
	if n == nil || n.raw == 0 {
		return false, errors.New("cudagraph: nil or closed node")
	}
	if n.graph == nil {
		return false, errors.New("cudagraph: node has no owning graph")
	}
	var val [8]uint64
	err := n.graph.withContext(func() error {
		var err error
		val, err = n.raw.GraphKernelNodeGetAttribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
		return err
	})
	if err != nil {
		return false, fmt.Errorf("cudagraph: get device updatable: %w", err)
	}
	return (val[0] & 0xFFFFFFFF) != 0, nil
}

// DeviceNode returns the device-side node handle (cudaGraphDeviceNode_t) for this node.
// Returns an error if the node is not device-updatable.
func (n *Node) DeviceNode() (DeviceNode, error) {
	if n == nil || n.raw == 0 {
		return 0, errors.New("cudagraph: nil or closed node")
	}
	if n.graph == nil {
		return 0, errors.New("cudagraph: node has no owning graph")
	}
	var val [8]uint64
	err := n.graph.withContext(func() error {
		var err error
		val, err = n.raw.GraphKernelNodeGetAttribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
		return err
	})
	if err != nil {
		return 0, fmt.Errorf("cudagraph: get device node: %w", err)
	}
	if val[0]&0xFFFFFFFF == 0 {
		return 0, errors.New("cudagraph: node is not device updatable")
	}
	return DeviceNode(val[1]), nil
}

// SetKernelNodeParams sets the parameters of a kernel node in an instantiated graph
// using cuGraphExecKernelNodeSetParams without validating topology or re-instantiating.
func (e *Exec) SetKernelNodeParams(node *Node, params *driver.KernelNodeParams) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if node == nil || node.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	if params == nil {
		return errors.New("cudagraph: nil kernel node params")
	}
	err := e.withContext(func() error {
		return e.raw.KernelNodeSetParams(node.raw, params)
	})
	if err != nil {
		return fmt.Errorf("cudagraph: set exec kernel node params: %w", err)
	}
	return nil
}

// SetKernelNodeGrid updates the grid dimensions of a kernel node in an instantiated graph.
func (e *Exec) SetKernelNodeGrid(node *Node, grid cudalaunch.Dim3) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if node == nil || node.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	if grid.X == 0 || grid.Y == 0 || grid.Z == 0 {
		return errors.New("cudagraph: invalid zero grid dimension")
	}
	params, err := node.Params()
	if err != nil {
		return err
	}
	params.GridDimX = grid.X
	params.GridDimY = grid.Y
	params.GridDimZ = grid.Z
	return e.SetKernelNodeParams(node, &params)
}

// UpdateKernelNode updates the grid dimensions and kernel arguments of node in the
// instantiated graph without re-instantiating or validating topology.
// If grid.X is non-zero, grid dimensions are updated. If args are provided,
// they are packed and updated.
func (e *Exec) UpdateKernelNode(node *Node, grid cudalaunch.Dim3, args ...any) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if node == nil || node.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	params, err := node.Params()
	if err != nil {
		return err
	}
	if grid.X != 0 {
		if grid.Y == 0 || grid.Z == 0 {
			return errors.New("cudagraph: invalid zero grid dimension")
		}
		params.GridDimX = grid.X
		params.GridDimY = grid.Y
		params.GridDimZ = grid.Z
	}
	if len(args) > 0 {
		kernelParams, err := cudalaunch.PackArgs(args...)
		if err != nil {
			return fmt.Errorf("cudagraph: pack args: %w", err)
		}
		// KernelParams is a uintptr, which the garbage collector does not
		// trace, so kernelParams must be kept alive explicitly until the
		// driver has read through it. See x/cudamlx.CommandEncoder.AddKernelNode.
		defer runtime.KeepAlive(kernelParams)
		params.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0]))
		params.Extra = 0
		return e.SetKernelNodeParams(node, &params)
	}
	return e.SetKernelNodeParams(node, &params)
}

// ApplyKernelNodeUpdate applies a typed KernelNodeUpdate to node in the executable graph.
func (e *Exec) ApplyKernelNodeUpdate(node *Node, update KernelNodeUpdate) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if node == nil || node.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	switch update.Field {
	case KernelNodeFieldGridDim:
		return e.SetKernelNodeGrid(node, update.GridDim)
	case KernelNodeFieldParam:
		params, err := node.Params()
		if err != nil {
			return err
		}
		if params.KernelParams == 0 {
			return errors.New("cudagraph: kernel params pointer is null")
		}
		if update.Value == nil || update.Size == 0 {
			return errors.New("cudagraph: invalid update param value or size")
		}
		basePtr := *(*unsafe.Pointer)(unsafe.Pointer(&params.KernelParams))
		dst := unsafe.Add(basePtr, update.Offset)
		srcSlice := unsafe.Slice((*byte)(update.Value), update.Size)
		dstSlice := unsafe.Slice((*byte)(dst), update.Size)
		copy(dstSlice, srcSlice)
		return e.SetKernelNodeParams(node, &params)
	default:
		return fmt.Errorf("cudagraph: unsupported kernel node update field %v", update.Field)
	}
}
