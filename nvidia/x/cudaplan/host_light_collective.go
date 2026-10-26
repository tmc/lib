package cudaplan

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

var (
	// ErrHostLightCollectiveBinding reports a host command, payload or
	// collective binding that does not match its registration. A command
	// refused with it was neither prepared nor dispatched.
	ErrHostLightCollectiveBinding = errors.New("cudaplan: host-light collective binding mismatch")

	// ErrHostLightCollectiveUnsupported reports an operation the host-light
	// collective binding does not provide. Only AllReduce over Int32 without a
	// pipeline slot is supported; nothing is converted to fit.
	ErrHostLightCollectiveUnsupported = errors.New("cudaplan: host-light collective operation unsupported")

	// ErrHostLightCollectiveBusy reports a Run or Close refused because
	// another Run or Close on the same registry is in progress.
	ErrHostLightCollectiveBusy = errors.New("cudaplan: host-light collective busy")

	// ErrHostLightCollectiveClosed reports a Run after Close.
	ErrHostLightCollectiveClosed = errors.New("cudaplan: host-light collective closed")
)

// Host-light collective limits. The count bound matches the cudakv per-batch
// word bound, one word per element.
const (
	maxHostLightCollectiveBindings = 1024
	maxHostLightCollectiveCount    = 1 << 20
)

// HostLightCollectiveBinding registers one host command for one collective
// step.
//
// SeqID and Generation identify the command: a [cudakv.Descriptor] runs the
// binding only if both match. Generation is the ring or application
// generation; CommunicatorGeneration, Rank and World must equal the
// communicator's. Op is the complete operation executed, and Op.Count and
// Op.DataType are the bound element count and type. Only AllReduce over
// [cudair.Int32] with PipelineSlot 0 is supported.
//
// Send and Recv are borrowed caller device addresses. The collective plan
// checks their extents and overlap; nothing checks their allocation or
// device. Generation and buffer values are caller assertions, not device
// attestation.
type HostLightCollectiveBinding struct {
	SeqID                  uint64
	Generation             uint64
	Rank                   int
	World                  int
	CommunicatorGeneration cudacollective.Generation
	Op                     cudacollective.OperationDescriptor
	Send, Recv             driver.DevicePtr
}

// HostLightCollectiveCall is a validated command passed to a prepare
// function. Values holds one Int32 element per input word, decoded as 32-bit
// two's complement, and is a fresh copy the function may keep.
type HostLightCollectiveCall struct {
	SeqID      uint64
	Generation uint64
	Rank       int
	Op         cudacollective.OperationDescriptor
	Send, Recv driver.DevicePtr
	Values     []int32
}

// HostLightCollectiveConfig configures [NewHostLightCollective].
//
// Communicator is borrowed and must stay open until Close returns nil.
// Bindings are copied. Prepare, if non-nil, is caller code, typically an
// upload of Values to Send. It runs only after every check on a command has
// passed and before the collective is dispatched.
type HostLightCollectiveConfig struct {
	Communicator *cudacollective.Communicator
	Bindings     []HostLightCollectiveBinding
	Prepare      func(context.Context, HostLightCollectiveCall) error
}

// A HostLightCollective is a bounded registry binding host commands to
// collective plans it owns. Its Run method is a
// [cudakv.HostCommandConsumer.ConsumeBatch] callback: each consumed
// descriptor drives the private [cudacollective.Plan] created from its
// binding, through the communicator's transport.
//
// It is host control flow only. It starts no goroutines and adds no queue,
// scheduler or transport, and a successful Run is evidence of nothing beyond
// what the transport reports.
//
// The registry owns its plans and borrows the communicator, its transport,
// the bound buffers and whatever Prepare touches. It never sees the ring.
// The caller keeps them valid until Close returns nil, and only then closes
// the communicator. The zero value is unusable; use [NewHostLightCollective].
type HostLightCollective struct {
	mu      sync.Mutex // held for one Run or Close, never waited on
	comm    *cudacollective.Communicator
	prepare func(context.Context, HostLightCollectiveCall) error
	entries []hostLightEntry
	index   map[hostLightKey]int
	closed  bool
	pending *cudacollective.CompletionError
}

type hostLightKey struct {
	seq, gen uint64
}

type hostLightEntry struct {
	binding HostLightCollectiveBinding
	plan    *cudacollective.Plan // nil once closed
}

// NewHostLightCollective returns a registry for config.Communicator. It
// checks every binding before creating any plan, then creates one plan per
// binding with a single collective step. If the plan refuses a step, the
// plans already created are closed and the errors are joined.
func NewHostLightCollective(config HostLightCollectiveConfig) (*HostLightCollective, error) {
	comm := config.Communicator
	if comm == nil {
		return nil, errors.New("cudaplan: nil host-light collective communicator")
	}
	n := len(config.Bindings)
	if n == 0 || n > maxHostLightCollectiveBindings {
		return nil, fmt.Errorf("cudaplan: %d host-light collective bindings outside [1, %d]", n, maxHostLightCollectiveBindings)
	}
	index := make(map[hostLightKey]int, n)
	for i, b := range config.Bindings {
		if err := checkHostLightBinding(comm, b); err != nil {
			return nil, err
		}
		k := hostLightKey{b.SeqID, b.Generation}
		if _, dup := index[k]; dup {
			return nil, fmt.Errorf("%w: duplicate sequence %d generation %d", ErrHostLightCollectiveBinding, b.SeqID, b.Generation)
		}
		index[k] = i
	}
	r := &HostLightCollective{
		comm:    comm,
		prepare: config.Prepare,
		entries: make([]hostLightEntry, 0, n),
		index:   index,
	}
	for _, b := range config.Bindings {
		plan, err := newHostLightPlan(comm, b)
		if err != nil {
			err = fmt.Errorf("cudaplan: host-light collective sequence %d generation %d: %w", b.SeqID, b.Generation, err)
			return nil, errors.Join(err, r.closePlans())
		}
		r.entries = append(r.entries, hostLightEntry{binding: b, plan: plan})
	}
	return r, nil
}

// checkHostLightBinding checks b against comm without side effects.
func checkHostLightBinding(comm *cudacollective.Communicator, b HostLightCollectiveBinding) error {
	world := comm.Membership().WorldSize()
	switch {
	case b.SeqID == 0 || b.Generation == 0:
		return fmt.Errorf("%w: sequence %d generation %d must be nonzero", ErrHostLightCollectiveBinding, b.SeqID, b.Generation)
	case b.Rank != comm.Rank() || b.World != world:
		return fmt.Errorf("%w: sequence %d bound to rank %d of %d, communicator rank %d of %d", ErrHostLightCollectiveBinding, b.SeqID, b.Rank, b.World, comm.Rank(), world)
	case b.CommunicatorGeneration != comm.Generation():
		return fmt.Errorf("%w: sequence %d bound to communicator generation %d, communicator generation %d", ErrHostLightCollectiveBinding, b.SeqID, b.CommunicatorGeneration, comm.Generation())
	}
	if err := b.Op.Validate(world); err != nil {
		return fmt.Errorf("cudaplan: host-light collective sequence %d: %w", b.SeqID, err)
	}
	switch {
	case b.Op.Kind != cudacollective.KindAllReduce || b.Op.DataType != cudair.Int32 || b.Op.PipelineSlot != 0:
		return fmt.Errorf("%w: sequence %d %v %v pipeline slot %d", ErrHostLightCollectiveUnsupported, b.SeqID, b.Op.Kind, b.Op.DataType, b.Op.PipelineSlot)
	case b.Op.Count > maxHostLightCollectiveCount:
		return fmt.Errorf("%w: sequence %d count %d exceeds %d", ErrHostLightCollectiveBinding, b.SeqID, b.Op.Count, maxHostLightCollectiveCount)
	}
	return nil
}

func newHostLightPlan(comm *cudacollective.Communicator, b HostLightCollectiveBinding) (*cudacollective.Plan, error) {
	plan, err := cudacollective.NewPlan(comm, nil)
	if err != nil {
		return nil, err
	}
	name := fmt.Sprintf("host-light sequence %d generation %d", b.SeqID, b.Generation)
	if err := plan.AddCollectiveStep(name, b.Op, b.Send, b.Recv); err != nil {
		return nil, errors.Join(err, plan.Close())
	}
	return plan, nil
}

// Run executes the binding registered for d.SeqID and d.Generation with the
// command's input words, one Int32 element per word.
//
// Before any side effect it checks that the command is registered, that
// d.Flags is zero, that the payload has exactly Op.Count words, that the
// communicator is alive and that ctx is not done. Only then does it call
// Prepare, check ctx again, and execute the binding's plan once.
// d.PromptToken is not interpreted.
//
// Run never retries or requeues: a refused, failed or cancelled command stays
// consumed, and cancellation after Prepare does not roll it back. If the plan
// reports an operation whose completion is unproven, Run refuses every later
// command with [cudacollective.ErrCompletionPending] until that operation's
// CompletionError stops reporting Pending. A concurrent Run or Close is
// refused with [ErrHostLightCollectiveBusy]. Callers sharing one registry
// across distinct consumers must serialize their ConsumeBatch calls: a Busy
// refusal consumes a command its consumer has already removed.
func (r *HostLightCollective) Run(ctx context.Context, d cudakv.Descriptor, words []uint32) error {
	if r == nil || r.comm == nil {
		return errors.New("cudaplan: host-light collective must be created with NewHostLightCollective")
	}
	if ctx == nil {
		return errors.New("cudaplan: nil host-light collective context")
	}
	if !r.mu.TryLock() {
		return ErrHostLightCollectiveBusy
	}
	defer r.mu.Unlock()
	if r.closed {
		return ErrHostLightCollectiveClosed
	}
	if r.pending != nil {
		if r.pending.Pending() {
			return fmt.Errorf("%w: earlier host-light collective operation not proven complete", cudacollective.ErrCompletionPending)
		}
		r.pending = nil
	}
	i, ok := r.index[hostLightKey{d.SeqID, d.Generation}]
	if !ok {
		return fmt.Errorf("%w: sequence %d generation %d not registered", ErrHostLightCollectiveBinding, d.SeqID, d.Generation)
	}
	e := &r.entries[i]
	b := e.binding
	if d.Flags != 0 {
		return fmt.Errorf("%w: sequence %d flags %#x", ErrHostLightCollectiveBinding, d.SeqID, d.Flags)
	}
	if len(words) != b.Op.Count {
		return fmt.Errorf("%w: sequence %d payload of %d words, registered %d Int32 elements", ErrHostLightCollectiveBinding, d.SeqID, len(words), b.Op.Count)
	}
	if err := r.comm.CheckAlive(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.prepare != nil {
		values := make([]int32, len(words))
		for j, w := range words {
			values[j] = int32(w)
		}
		call := HostLightCollectiveCall{
			SeqID:      b.SeqID,
			Generation: b.Generation,
			Rank:       b.Rank,
			Op:         b.Op,
			Send:       b.Send,
			Recv:       b.Recv,
			Values:     values,
		}
		if err := r.prepare(ctx, call); err != nil {
			return fmt.Errorf("cudaplan: host-light collective prepare: %w", err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	err := e.plan.ExecuteSteps(ctx)
	var pending *cudacollective.CompletionError
	if errors.As(err, &pending) && pending.Pending() {
		r.pending = pending
	}
	return err
}

// Close stops Run and closes each plan the registry still owns. A plan whose
// Close fails, such as one retaining an operation of unproven completion,
// stays owned; the errors are joined and a later Close retries only those
// plans. Close does not close or fail the communicator. A Close concurrent
// with Run or Close is refused with [ErrHostLightCollectiveBusy].
func (r *HostLightCollective) Close() error {
	if r == nil {
		return nil
	}
	if !r.mu.TryLock() {
		return ErrHostLightCollectiveBusy
	}
	defer r.mu.Unlock()
	r.closed = true
	return r.closePlans()
}

// closePlans closes every owned plan, keeping those whose Close fails.
func (r *HostLightCollective) closePlans() error {
	var errs []error
	for i := range r.entries {
		e := &r.entries[i]
		if e.plan == nil {
			continue
		}
		if err := e.plan.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cudaplan: host-light collective sequence %d generation %d: %w", e.binding.SeqID, e.binding.Generation, err))
			continue
		}
		e.plan = nil
	}
	if len(errs) == 0 {
		r.pending = nil
	}
	return errors.Join(errs...)
}
