package cutensornet

import (
	"errors"
	"fmt"
	"runtime"
)

// Errors reported by the workspace and plan ownership layer.
var (
	// ErrWorkspaceInUse reports an attempt to release a workspace, or the
	// memory behind it, while a plan that the library may still read it
	// through is live.
	ErrWorkspaceInUse = errors.New("cutensornet: workspace still referenced by a live plan")

	// ErrWorkspaceClosed reports use of a workspace after Close.
	ErrWorkspaceClosed = errors.New("cutensornet: workspace is closed")

	// ErrPlanClosed reports use of a plan after Close.
	ErrPlanClosed = errors.New("cutensornet: plan is closed")
)

// A Workspace owns the scratch and cache memory cuTensorNet uses while
// planning, autotuning and contracting a network.
//
// The library's protocol is: create a workspace descriptor, ask it how large a
// buffer of a given memory space and kind must be, allocate that buffer, and
// hand the address back through the descriptor. From that point the library
// holds the address. It reads and writes the buffer during planning and during
// every contraction that uses the descriptor, so freeing the buffer before the
// last such call has completed on the stream is a use-after-free that this
// package cannot detect for you at the moment it happens.
//
// Workspace makes the ordering explicit instead. The deallocator handed to
// [Workspace.Provide] is not run until [Workspace.Close], and Close refuses
// while any [Plan] created from the workspace is still live. Stream ordering is
// still the caller's problem: Close only proves that no Go-side plan remains,
// not that the device has finished with the buffer. Synchronize the stream
// before calling it.
//
// This package never allocates device memory. Pair it with a CUDA allocator
// such as github.com/tmc/lib/nvidia/driver.
//
// A Workspace is not safe for concurrent use.
type Workspace struct {
	h      Handle
	desc   WorkspaceDescriptor
	frees  []func()
	plans  int
	closed bool
}

// NewWorkspace creates an empty workspace descriptor.
//
// The returned Workspace must be released with [Workspace.Close].
func (h Handle) NewWorkspace() (*Workspace, error) {
	desc, err := h.CreateWorkspaceDescriptor()
	if err != nil {
		return nil, err
	}
	return &Workspace{h: h, desc: desc}, nil
}

// Descriptor returns the raw descriptor, for the generated calls that this
// package does not wrap. The descriptor is owned by w and is invalid after
// [Workspace.Close].
func (w *Workspace) Descriptor() WorkspaceDescriptor { return w.desc }

// ComputeContractionSizes records in the workspace descriptor the scratch and
// cache sizes needed to contract net along the path in info. Call it before
// [Workspace.Size]; the size query reports what was last computed, not what a
// particular plan will need.
func (w *Workspace) ComputeContractionSizes(net *Network, info ContractionOptimizerInfo) error {
	if w.closed {
		return ErrWorkspaceClosed
	}
	if net == nil || net.closed {
		return ErrNetworkClosed
	}
	return w.h.WorkspaceComputeContractionSizes(net.desc, info, w.desc)
}

// Size returns the workspace size in bytes that the last
// [Workspace.ComputeContractionSizes] recorded for the given preference, memory
// space and kind.
//
// [WorksizePrefMin] is the smallest buffer for which some algorithm is
// available for every pairwise contraction; [WorksizePrefRecommended] is the
// size at which the most suitable algorithm is available. Providing less than
// the Min size for [WorkspaceScratch] is rejected by the library when the
// workspace is used. [WorkspaceCache] accepts any size, including zero, which
// disables it.
func (w *Workspace) Size(pref WorksizePref, space Memspace, kind WorkspaceKind) (int64, error) {
	if w.closed {
		return 0, ErrWorkspaceClosed
	}
	return w.h.WorkspaceGetMemorySize(w.desc, pref, space, kind)
}

// Provide hands a caller-allocated buffer of size bytes at address ptr to the
// workspace for the given memory space and kind, and takes over the decision of
// when it may be released.
//
// free, if non-nil, is called by [Workspace.Close] after the descriptor has been
// destroyed, in reverse order of the Provide calls. Passing nil leaves the
// buffer entirely to the caller, who then must not release it until after Close
// returns and the stream has been synchronized.
//
// The buffer must live in the memory space named by space: device-accessible
// memory for [MemspaceDevice], host-accessible memory for [MemspaceHost]. This
// package cannot check that. It must also stay unmodified by anything other
// than the library for as long as the workspace holds it; a [WorkspaceCache]
// buffer additionally must survive unmodified across the iterations that reuse
// it.
//
// A ptr of 0 with size > 0 asks the library to draw the buffer from its own
// memory pool instead; free must then be nil, since there is nothing for the
// caller to release. A ptr of 0 with size 0 disables the workspace of that kind.
func (w *Workspace) Provide(space Memspace, kind WorkspaceKind, ptr uintptr, size int64, free func()) error {
	if w.closed {
		return ErrWorkspaceClosed
	}
	if w.plans > 0 {
		return ErrWorkspaceInUse
	}
	if ptr == 0 && free != nil {
		return errors.New("cutensornet: workspace deallocator given for a null buffer")
	}
	if ptr != 0 && size <= 0 {
		return fmt.Errorf("cutensornet: workspace buffer at %#x has size %d, want > 0", ptr, size)
	}
	if err := w.h.WorkspaceSetMemory(w.desc, space, kind, ptr, size); err != nil {
		return err
	}
	if free != nil {
		w.frees = append(w.frees, free)
	}
	return nil
}

// PurgeCache invalidates the cached data in the [WorkspaceCache] buffer for the
// given memory space. It does not release the buffer or return it to a pool.
func (w *Workspace) PurgeCache(space Memspace) error {
	if w.closed {
		return ErrWorkspaceClosed
	}
	return w.h.WorkspacePurgeCache(w.desc, space)
}

// Close destroys the workspace descriptor and then runs the deallocators
// registered with [Workspace.Provide], in reverse order.
//
// It reports [ErrWorkspaceInUse] and releases nothing while a [Plan] created
// from w is still live. Close does not synchronize any stream: the caller must
// have done that, or the deallocators race with device work still reading the
// buffers. Close is idempotent once it has succeeded.
func (w *Workspace) Close() error {
	if w.closed {
		return nil
	}
	if w.plans > 0 {
		return ErrWorkspaceInUse
	}
	w.closed = true
	err := w.desc.Destroy()
	for i := len(w.frees) - 1; i >= 0; i-- {
		w.frees[i]()
	}
	w.frees = nil
	return err
}

// A Plan is a contraction plan bound to the network and workspace it was built
// from.
//
// The library reads the network descriptor and the workspace buffers during
// every contraction the plan performs, so a Plan keeps both alive: closing
// either while a plan is live reports an error rather than freeing memory the
// library still holds.
//
// A Plan is not safe for concurrent use.
type Plan struct {
	ws     *Workspace
	net    *Network
	plan   ContractionPlan
	closed bool
}

// NewPlan builds a contraction plan for net along the path in info, sized
// against the buffers already provided to w.
//
// Call [Workspace.ComputeContractionSizes] and [Workspace.Provide] first: the
// plan is validated against the workspace as it stands at this moment, and the
// library holds those buffers from here until [Plan.Close].
//
// The returned Plan must be released with [Plan.Close], before the workspace or
// the network.
func (w *Workspace) NewPlan(net *Network, info ContractionOptimizerInfo) (*Plan, error) {
	if w.closed {
		return nil, ErrWorkspaceClosed
	}
	if net == nil || net.closed {
		return nil, ErrNetworkClosed
	}
	plan, err := w.h.CreateContractionPlan(net.desc, info, w.desc)
	if err != nil {
		return nil, err
	}
	w.plans++
	net.plans++
	return &Plan{ws: w, net: net, plan: plan}, nil
}

// Plan returns the raw plan handle, for the generated calls that this package
// does not wrap. It is owned by p and is invalid after [Plan.Close].
func (p *Plan) Plan() ContractionPlan { return p.plan }

// Contract executes the contraction on stream.
//
// in holds one device address per input operand, in the order the operands were
// given to [Handle.NewNetwork]; out is the device address of the output
// operand. When accumulate is true the result is added to whatever out already
// holds, otherwise out is overwritten.
//
// slices selects which slices of the contraction to execute. Pass a zero
// [SliceGroup] to execute all of them.
//
// Contract is asynchronous with respect to the host: it enqueues work on stream
// and returns. Every buffer named here, and every workspace buffer, must stay
// valid and unmodified until that work completes. Nothing in this package
// synchronizes the stream for you.
//
// All the buffers must be accessible from the device that is current on the
// calling thread, and that must be the same device the [Handle] was created on.
func (p *Plan) Contract(in []uintptr, out uintptr, accumulate bool, slices SliceGroup, stream Stream) error {
	if err := p.check(in, out); err != nil {
		return err
	}
	accumulateOutput := int32(0)
	if accumulate {
		accumulateOutput = 1
	}
	err := p.ws.h.ContractSlices(p.plan, &in[0], out, accumulateOutput, p.ws.desc, slices, stream)
	runtime.KeepAlive(in)
	return err
}

// Autotune runs the contraction repeatedly on stream to pick the fastest
// algorithm for each pairwise contraction, and records the choice in the plan.
//
// It writes to out and to the workspace exactly as [Plan.Contract] does, and is
// asynchronous in the same way. Pass a zero [ContractionAutotunePreference] to
// use the library defaults.
func (p *Plan) Autotune(in []uintptr, out uintptr, pref ContractionAutotunePreference, stream Stream) error {
	if err := p.check(in, out); err != nil {
		return err
	}
	err := p.ws.h.ContractionAutotune(p.plan, &in[0], out, p.ws.desc, pref, stream)
	runtime.KeepAlive(in)
	return err
}

// check validates the operand addresses against the network the plan was built
// from.
func (p *Plan) check(in []uintptr, out uintptr) error {
	if p.closed {
		return ErrPlanClosed
	}
	if len(in) != p.net.numInputs {
		return fmt.Errorf("cutensornet: %d input addresses for a network with %d input tensors", len(in), p.net.numInputs)
	}
	for i, ptr := range in {
		if ptr == 0 {
			return fmt.Errorf("cutensornet: input address %d is null", i)
		}
	}
	if out == 0 {
		return errors.New("cutensornet: output address is null")
	}
	return nil
}

// Close destroys the plan and releases its claim on the workspace and the
// network, which may then be closed.
//
// Close does not synchronize the stream. Work enqueued by [Plan.Contract] may
// still be reading the workspace; the caller must synchronize before releasing
// any of it. Close is idempotent.
func (p *Plan) Close() error {
	if p.closed {
		return nil
	}
	p.releaseClaims()
	return p.plan.Destroy()
}

// releaseClaims marks p closed and drops its holds on the workspace and the
// network. It is idempotent.
func (p *Plan) releaseClaims() {
	if p.closed {
		return
	}
	p.closed = true
	p.ws.plans--
	p.net.plans--
}
