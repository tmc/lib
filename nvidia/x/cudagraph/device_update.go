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
	GridDim   driver.Dim3
	Offset    uintptr
	Value     unsafe.Pointer
	Size      uintptr
	IsEnabled uint32
}

// NewGridDimUpdate creates a KernelNodeUpdate to update the grid dimensions of node.
func NewGridDimUpdate(node DeviceNode, grid driver.Dim3) KernelNodeUpdate {
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
	offset, size, err := fn.ParamInfo(paramIndex)
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

// kernel returns the node as a kernel node. The driver rejects the kernel
// node calls on a node of another kind.
func (n *Node) kernel() driver.KernelNode {
	return driver.KernelNode{GraphNode: n.raw}
}

// Params returns the kernel node's parameters. CUDA owns the returned
// argument array and values; they must not be written and remain valid only
// until the original node changes or is destroyed.
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
		params, err = n.kernel().Params()
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
		if err := n.kernel().SetParams(params); err != nil {
			return err
		}
		n.graph.recordKernelNode(n.raw, *params, nil)
		return nil
	})
	if err != nil {
		return fmt.Errorf("cudagraph: set kernel node params: %w", err)
	}
	return nil
}

// SetGrid updates the grid dimensions of the kernel node in the non-instantiated graph.
func (n *Node) SetGrid(grid driver.Dim3) error {
	if grid.X == 0 || grid.Y == 0 || grid.Z == 0 {
		return errors.New("cudagraph: invalid zero grid dimension")
	}
	params, err := n.Params()
	if err != nil {
		return err
	}
	params.GridDim = grid
	return n.SetParams(&params)
}

// deviceUpdatableVersion is the first driver version, as
// [driver.DriverGetVersion] reports it, with device-updatable kernel nodes.
const deviceUpdatableVersion = 12040

// SetDeviceUpdatable marks the kernel node as device-updatable.
// Once enabled, CUDA does not allow opting out, and device-updatable nodes
// cannot be destroyed individually via DestroyNode.
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
		return n.kernel().SetAttribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode), &val)
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
		val, err = n.kernel().Attribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
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
		val, err = n.kernel().Attribute(int32(driver.LaunchAttributeDeviceUpdatableKernelNode))
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

// SetKernelNodeParams updates future launches, without modifying the original
// graph node. Raw argument arrays remain caller-owned: retain them until the
// next successful replacement or destruction. Calls on a graph or Exec must
// be serialized. Before replacing retained storage, ensure no launches are in
// flight and no device-side updates are concurrent. Modules and device memory
// must remain valid independently of the host argument storage.
func (e *Exec) SetKernelNodeParams(node *Node, params *driver.KernelNodeParams) error {
	if err := e.checkKernelNode(node); err != nil {
		return err
	}
	if params == nil {
		return errors.New("cudagraph: nil kernel node params")
	}
	var owner *launchArgs
	if old, ok := e.kernelNodes[node.raw]; ok && old.params.KernelParams == params.KernelParams && old.params.Extra == params.Extra {
		owner = old.args
	}
	return e.setKernelNodeParams(node, *params, owner, e.withContext, func(p *driver.KernelNodeParams) error {
		return e.raw.KernelNodeSetParams(node.kernel(), p)
	})
}

func (e *Exec) checkKernelNode(node *Node) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if node == nil || node.raw == 0 {
		return errors.New("cudagraph: nil or closed node")
	}
	if e.kernelOrigins != nil {
		if _, ok := e.kernelOrigins[node.raw]; !ok {
			return errors.New("cudagraph: node is not from the originally instantiated graph")
		}
	}
	return nil
}

// setKernelNodeParams commits only after native success. Context restoration
// can subsequently fail; in that case the retained snapshot follows CUDA's
// already committed state even though the public call returns an error.
func (e *Exec) setKernelNodeParams(node *Node, params driver.KernelNodeParams, owner *launchArgs, run func(func() error) error, set func(*driver.KernelNodeParams) error) error {
	err := run(func() error {
		if err := set(&params); err != nil {
			return err
		}
		if e.kernelNodes == nil {
			e.kernelNodes = make(map[driver.GraphNode]kernelNodeState)
		}
		e.kernelNodes[node.raw] = kernelNodeState{params, owner}
		e.kernelOrder = nil
		return nil
	})
	runtime.KeepAlive(owner)
	if err != nil {
		return fmt.Errorf("cudagraph: set exec kernel node params: %w", err)
	}
	return nil
}

func (e *Exec) kernelState(node *Node) (kernelNodeState, error) {
	if err := e.checkKernelNode(node); err != nil {
		return kernelNodeState{}, err
	}
	state, ok := e.kernelNodes[node.raw]
	if !ok {
		return kernelNodeState{}, errors.New("cudagraph: executable kernel parameters are not retained; supply complete parameters")
	}
	return state, nil
}

// SetKernelNodeGrid preserves the Exec's current function and arguments.
// Nodes built from raw argument arrays or by stream capture require
// SetKernelNodeParams with a fresh complete set: the Exec does not own their
// argument storage.
func (e *Exec) SetKernelNodeGrid(node *Node, grid driver.Dim3) error {
	state, err := e.kernelState(node)
	if err != nil {
		return err
	}
	if err := checkDim("grid", grid); err != nil {
		return err
	}
	if state.args == nil && (state.params.KernelParams != 0 || state.params.Extra != 0) {
		return errors.New("cudagraph: raw kernel argument storage is not owned")
	}
	state.params.GridDim = grid
	return e.SetKernelNodeParams(node, &state.params)
}

// UpdateKernelNode replaces arguments when provided and otherwise preserves
// the current executable parameters. A zero grid leaves its dimensions alone.
// Superseded argument storage is released, so callers must serialize updates
// and ensure no launches or device-side updates are in flight.
func (e *Exec) UpdateKernelNode(node *Node, grid driver.Dim3, args ...any) error {
	return e.updateKernelNode(node, 0, grid, args, e.withContext,
		paramCount,
		func(fn driver.Function, index uint) (uint, uint, error) { return fn.ParamInfo(index) },
		func(p *driver.KernelNodeParams) error {
			return e.raw.KernelNodeSetParams(node.kernel(), p)
		})
}

// HotSwapNode replaces the function while preserving current arguments when
// none are supplied. The caller owns the new function's module lifetime;
// changing a raw handle cannot acquire a module reference. The retained arguments
// must match the new function's queried count and sizes. CUDA validates
// function context and graph capability restrictions. See UpdateKernelNode
// for the serialized, no-inflight update requirement.
func (e *Exec) HotSwapNode(node *Node, fn driver.Function, grid driver.Dim3, args ...any) error {
	if fn == 0 {
		return errors.New("cudagraph: nil kernel function")
	}
	return e.updateKernelNode(node, fn, grid, args, e.withContext,
		paramCount,
		func(fn driver.Function, index uint) (uint, uint, error) { return fn.ParamInfo(index) },
		func(p *driver.KernelNodeParams) error {
			return e.raw.KernelNodeSetParams(node.kernel(), p)
		})
}

func (e *Exec) updateKernelNode(node *Node, fn driver.Function, grid driver.Dim3, args []any, run func(func() error) error, count func(driver.Function) (uint, error), info func(driver.Function, uint) (uint, uint, error), set func(*driver.KernelNodeParams) error) error {
	state, err := e.kernelState(node)
	if err != nil {
		return err
	}
	if grid != (driver.Dim3{}) {
		if err := checkDim("grid", grid); err != nil {
			return err
		}
		state.params.GridDim = grid
	}
	if fn != 0 {
		state.params.Func = fn
	}
	if len(args) != 0 {
		if len(args) > 128 {
			return errors.New("cudagraph: kernel argument count exceeds 128")
		}
		// Partition expansion has no owned byte-layout contract here. Reject it
		// before PackArgs can allocate an unbounded expanded argument array.
		sizes := argumentSizes(args, len(args))
		if sizes == nil {
			return errors.New("cudagraph: kernel arguments exceed owned layout bounds")
		}
		params, err := cudalaunch.PackArgs(args...)
		if err != nil {
			return fmt.Errorf("cudagraph: pack args: %w", err)
		}
		if len(params) == 0 || len(params) != len(args) || len(params) > 128 {
			return errors.New("cudagraph: kernel argument count exceeds 128")
		}
		owner := &launchArgs{params: params, sizes: sizes}
		if state.args != nil {
			owner.desc = state.args.desc
		}
		owner.desc.Function = state.params.Func
		owner.desc.Grid, owner.desc.Block = state.params.GridDim, state.params.BlockDim
		owner.desc.SharedMem = state.params.SharedMemBytes
		state.args = owner
		state.params.KernelParams = uintptr(unsafe.Pointer(&params[0]))
		state.params.Extra = 0
	} else if state.args == nil && (state.params.KernelParams != 0 || state.params.Extra != 0) {
		return errors.New("cudagraph: raw kernel argument storage is not owned")
	}
	err = run(func() error {
		if len(args) != 0 || fn != 0 {
			if err := checkKernelArguments(state.params, state.args, count, info); err != nil {
				return err
			}
		}
		return e.setKernelNodeParams(node, state.params, state.args, func(f func() error) error { return f() }, set)
	})
	runtime.KeepAlive(state.args)
	if err != nil {
		return fmt.Errorf("cudagraph: update exec kernel node: %w", err)
	}
	return nil
}

// ApplyKernelNodeUpdate applies a host update. Parameter offsets are CUDA's
// packed-device parameter offsets, not offsets into the void** pointer array.
// Only owned, known argument sizes are writable. The current state remains
// unchanged on native rejection. Caller must provide readable Value bytes,
// serialize the update and ensure there are no launches/device updates in flight.
func (e *Exec) ApplyKernelNodeUpdate(node *Node, update KernelNodeUpdate) error {
	if err := e.checkKernelNode(node); err != nil {
		return err
	}
	switch update.Field {
	case KernelNodeFieldGridDim:
		return e.SetKernelNodeGrid(node, update.GridDim)
	case KernelNodeFieldParam:
		return e.applyKernelParam(node, update, e.withContext,
			paramCount,
			func(fn driver.Function, index uint) (uint, uint, error) { return fn.ParamInfo(index) },
			func(p *driver.KernelNodeParams) error { return e.raw.KernelNodeSetParams(node.kernel(), p) })
	default:
		return fmt.Errorf("cudagraph: unsupported kernel node update field %v", update.Field)
	}
}

func (e *Exec) applyKernelParam(node *Node, update KernelNodeUpdate, run func(func() error) error, count func(driver.Function) (uint, error), info func(driver.Function, uint) (uint, uint, error), set func(*driver.KernelNodeParams) error) error {
	state, err := e.kernelState(node)
	if err != nil {
		return err
	}
	owner := state.args
	if owner == nil || len(owner.sizes) != len(owner.params) || len(owner.params) == 0 || state.params.Extra != 0 || state.params.KernelParams != uintptr(unsafe.Pointer(&owner.params[0])) {
		return errors.New("cudagraph: kernel parameter layout is not owned")
	}
	if update.Value == nil || update.Size == 0 || update.Size > 65536 || update.Offset > 65536-update.Size {
		return errors.New("cudagraph: invalid update parameter range")
	}
	var replacement *launchArgs
	err = run(func() error {
		n, err := count(state.params.Func)
		if err != nil {
			return err
		}
		if n != uint(len(owner.params)) || n > 128 {
			return errors.New("cudagraph: kernel parameter count differs from owned storage")
		}
		index := -1
		var relative uintptr
		var previousEnd uintptr
		for i, size := range owner.sizes {
			offset, actualSize, err := info(state.params.Func, uint(i))
			if err != nil {
				return err
			}
			start, length := uintptr(offset), uintptr(actualSize)
			if length == 0 || length > 65536 || length != size || start > 65536-length || start < previousEnd {
				return errors.New("cudagraph: native parameter layout differs from owned storage")
			}
			previousEnd = start + length
			if update.Offset >= start && update.Offset-start <= length && update.Size <= length-(update.Offset-start) {
				if index != -1 {
					return errors.New("cudagraph: ambiguous kernel parameter range")
				}
				index, relative = i, update.Offset-start
			}
		}
		if index == -1 {
			return errors.New("cudagraph: update crosses a parameter or padding boundary")
		}
		replacement = &launchArgs{params: make([]unsafe.Pointer, n), sizes: append([]uintptr(nil), owner.sizes...), buffers: make([][]byte, n), desc: owner.desc}
		for i, size := range owner.sizes {
			if owner.params[i] == nil {
				return errors.New("cudagraph: nil owned parameter storage")
			}
			// Keep struct parameters aligned just as PackArgs does.
			buf := make([]byte, size+15)
			shift := (16 - uintptr(unsafe.Pointer(&buf[0]))%16) % 16
			value := unsafe.Pointer(&buf[shift])
			copy(unsafe.Slice((*byte)(value), size), unsafe.Slice((*byte)(owner.params[i]), size))
			replacement.params[i], replacement.buffers[i] = value, buf
		}
		copy(unsafe.Slice((*byte)(replacement.params[index]), owner.sizes[index])[relative:relative+update.Size], unsafe.Slice((*byte)(update.Value), update.Size))
		state.params.KernelParams = uintptr(unsafe.Pointer(&replacement.params[0]))
		return e.setKernelNodeParams(node, state.params, replacement, func(fn func() error) error { return fn() }, set)
	})
	runtime.KeepAlive(owner)
	runtime.KeepAlive(replacement)
	if err != nil {
		return fmt.Errorf("cudagraph: update kernel parameter: %w", err)
	}
	return nil
}

// checkKernelArguments validates every pointee before CUDA reads the image's
// parameter table. A table carries neither its count nor storage lengths.
func checkKernelArguments(params driver.KernelNodeParams, owner *launchArgs, count func(driver.Function) (uint, error), info func(driver.Function, uint) (uint, uint, error)) error {
	n, err := count(params.Func)
	if err != nil {
		return fmt.Errorf("query kernel parameter count: %w", err)
	}
	if n > 128 {
		return errors.New("kernel parameter count exceeds 128")
	}
	if owner == nil {
		if n == 0 && params.KernelParams == 0 && params.Extra == 0 {
			return nil
		}
		return errors.New("kernel parameter layout is not owned")
	}
	if n != uint(len(owner.params)) || len(owner.sizes) != len(owner.params) || params.Extra != 0 {
		return errors.New("kernel parameter count differs from owned storage")
	}
	if n == 0 {
		if params.KernelParams != 0 {
			return errors.New("nonempty zero-argument parameter table")
		}
		return nil
	}
	if params.KernelParams != uintptr(unsafe.Pointer(&owner.params[0])) {
		return errors.New("kernel parameter table differs from owned storage")
	}
	var end uintptr
	for i, size := range owner.sizes {
		offset, actual, err := info(params.Func, uint(i))
		if err != nil {
			return fmt.Errorf("query kernel parameter %d: %w", i, err)
		}
		start, length := uintptr(offset), uintptr(actual)
		if owner.params[i] == nil || length == 0 || length > 65536 || length != size || start > 65536-length || start < end {
			return errors.New("native parameter layout differs from owned storage")
		}
		end = start + length
	}
	return nil
}

// paramCount returns fn's parameter count. A driver that predates
// cuFuncGetParamCount (the CUDA 13.0 r580 series) still exports
// cuFuncGetParamInfo, which rejects an index at or past the parameter count
// with ErrInvalidValue, so the count is the first index it rejects.
func paramCount(fn driver.Function) (uint, error) {
	return countParams(fn, driver.Function.ParamCount, driver.Function.ParamInfo)
}

func countParams(fn driver.Function, count func(driver.Function) (uint, error), info func(driver.Function, uint) (uint, uint, error)) (uint, error) {
	n, err := count(fn)
	var missing *driver.SymbolError
	if !errors.As(err, &missing) || missing.Err != nil {
		return n, err
	}
	for i := uint(0); i <= 128; i++ {
		_, _, err := info(fn, i)
		if errors.Is(err, driver.ErrInvalidValue) {
			return i, nil
		}
		if err != nil {
			return 0, fmt.Errorf("count parameters without %s: %w", missing.Symbol, err)
		}
	}
	return 0, errors.New("kernel parameter count exceeds 128")
}
