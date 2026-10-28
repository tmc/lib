package cudacollective

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// NCCLTransport executes on caller-owned rank-ordered communicators and streams.
// Comms and Streams must each contain WorldSize entries and remain unchanged.
// The caller must establish the rank's CUDA context and retain buffers and
// handles until Execute establishes completion. An error containing a
// CompletionError retains the affected buffers and handles until its Close or
// the transport's Close succeeds. Close never destroys borrowed handles.
// Stream zero selects the CUDA default stream.
//
// The caller coordinates admission of all ranks: cancelling one rank before
// submission can strand admitted peers. After submission Execute waits for
// stream completion before reporting cancellation. Native calls and Close do
// not promise bounded cancellation; coordinated all-rank recovery belongs to
// the caller. Use blocking NCCL communicators; nonblocking enqueue is unsupported.
type NCCLTransport struct {
	Comms   []nccl.Comm
	Streams []driver.Stream
	mu      sync.Mutex
	closed  bool
	active  sync.WaitGroup
	closeMu sync.Mutex
	ranks   []sync.Mutex
	pending []*CompletionError // at most one incomplete operation per rank
	calls   *ncclCalls
}

// ErrCompletionPending means submitted work has not been proven quiescent.
// Returning from a host call, cancellation, or a failed synchronization does
// not permit reuse or destruction of its buffers and borrowed native handles.
var ErrCompletionPending = errors.New("cudacollective: completion pending")

// CompletionError retains one operation whose completion check failed. Use
// errors.As to find it through wrapping. Close retries only that operation's
// completion check, never submission, and returns nil only after it succeeds.
// A failed Close retains ownership for another explicit attempt. The original
// submission, completion and cancellation errors remain reachable via Unwrap,
// including after successful cleanup. This is not proof the operation worked.
// The zero value holds no operation.
type CompletionError struct {
	cause          error
	transport      *NCCLTransport
	rank           int
	stream         driver.Stream
	complete       func(driver.Stream) error
	context        driver.Context
	currentContext func() (driver.Context, error)
	done           bool // protected by transport.mu
}

func (e *CompletionError) Error() string {
	if e == nil || e.cause == nil {
		return ErrCompletionPending.Error()
	}
	return e.cause.Error()
}
func (e *CompletionError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Pending reports whether this operation still requires completion recovery.
func (e *CompletionError) Pending() bool {
	if e == nil || e.transport == nil {
		return false
	}
	e.transport.mu.Lock()
	defer e.transport.mu.Unlock()
	return !e.done
}

// Close retries completion once. Calls are serialized with rank execution.
// The caller must establish the same CUDA context used for submission. Native
// synchronization can block; Close does not promise bounded cancellation.
func (e *CompletionError) Close() error {
	if e == nil || e.transport == nil {
		return nil
	}
	return e.transport.completePending(e)
}

func pendingCompletion(err error) *CompletionError {
	if pending, ok := err.(*CompletionError); ok {
		if pending.Pending() {
			return pending
		}
		return nil
	}
	switch e := err.(type) {
	case interface{ Unwrap() []error }:
		for _, child := range e.Unwrap() {
			if pending := pendingCompletion(child); pending != nil {
				return pending
			}
		}
	case interface{ Unwrap() error }:
		return pendingCompletion(e.Unwrap())
	}
	return nil
}

func (t *NCCLTransport) completePending(p *CompletionError) error {
	t.ranks[p.rank].Lock()
	defer t.ranks[p.rank].Unlock()
	t.mu.Lock()
	if p.done {
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if p.currentContext != nil {
		current, err := p.currentContext()
		if err != nil {
			return errors.Join(p, fmt.Errorf("nccl recovery context: %w", err))
		}
		if current != p.context {
			return errors.Join(p, errors.New("nccl recovery context differs from submission"))
		}
	}
	if err := p.complete(p.stream); err != nil {
		return errors.Join(p, err)
	}
	t.mu.Lock()
	p.done = true
	t.pending[p.rank] = nil
	t.mu.Unlock()
	return nil
}

type ncclInvocation struct {
	comm                 nccl.Comm
	stream               driver.Stream
	op                   OperationDescriptor
	dtype                nccl.DataType
	reduction            nccl.RedOp
	send, recv           driver.DevicePtr
	sendBytes, recvBytes uint64
}
type ncclCalls struct {
	info           func(nccl.Comm) (int, int, error)
	invoke         func(ncclInvocation) error
	complete       func(driver.Stream) error
	currentContext func() (driver.Context, error)
}

var nativeNCCLCalls = ncclCalls{
	currentContext: driver.CtxGetCurrent,
	info: func(c nccl.Comm) (int, int, error) {
		n, err := c.Count()
		if err != nil {
			return 0, 0, err
		}
		r, err := c.UserRank()
		return int(n), int(r), err
	},
	invoke: func(q ncclInvocation) error {
		n := uint(q.op.Count)
		switch q.op.Kind {
		case KindAllReduce:
			return q.comm.AllReduce(q.send, q.recv, n, q.dtype, q.reduction, q.stream)
		case KindAllGather:
			return q.comm.AllGather(q.send, q.recv, n, q.dtype, q.stream)
		case KindBroadcast:
			return q.comm.Broadcast(q.send, q.recv, n, q.dtype, int32(q.op.Root), q.stream)
		case KindReduceScatter:
			return q.comm.ReduceScatter(q.send, q.recv, n, q.dtype, q.reduction, q.stream)
		case KindReduce:
			return q.comm.Reduce(q.send, q.recv, n, q.dtype, q.reduction, int32(q.op.Root), q.stream)
		default:
			return ErrExecutionUnavailable
		}
	},
	complete: func(s driver.Stream) error { return s.Synchronize() },
}

func (t *NCCLTransport) Type() TransportType { return TransportTypeNCCLTest }
func (t *NCCLTransport) Available(m Membership) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed || len(t.Comms) != m.WorldSize() || len(t.Streams) != m.WorldSize() || m.WorldSize() == 0 {
		return false
	}
	for _, p := range t.pending {
		if p != nil {
			return false
		}
	}
	for _, c := range t.Comms {
		if c == 0 {
			return false
		}
	}
	if t.calls != nil {
		return true
	}
	v, err := nccl.GetVersion()
	return err == nil && v > 0
}
func (t *NCCLTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, send, recv driver.DevicePtr) error {
	if ctx == nil {
		return errors.New("cudacollective: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if comm == nil {
		return ErrCommunicatorClosed
	}
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	world, rank := comm.Membership().WorldSize(), comm.Rank()
	q, err := ncclRequest(op, rank, world, send, recv)
	if err != nil {
		return err
	}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return ErrCommunicatorClosed
	}
	for _, p := range t.pending {
		if p != nil {
			t.mu.Unlock()
			return ErrCompletionPending
		}
	}
	if len(t.Comms) != world || len(t.Streams) != world || t.Comms[rank] == 0 {
		t.mu.Unlock()
		return ErrExecutionUnavailable
	}
	if t.ranks == nil {
		t.ranks = make([]sync.Mutex, world)
		t.pending = make([]*CompletionError, world)
	}
	q.comm, q.stream = t.Comms[rank], t.Streams[rank]
	calls := t.calls
	if calls == nil {
		calls = &nativeNCCLCalls
	}
	rankMu := &t.ranks[rank]
	t.active.Add(1)
	t.mu.Unlock()
	defer t.active.Done()
	rankMu.Lock()
	defer rankMu.Unlock()
	t.mu.Lock()
	pending := t.pending[rank]
	t.mu.Unlock()
	if pending != nil {
		return ErrCompletionPending
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	n, r, err := calls.info(q.comm)
	if err != nil {
		return fmt.Errorf("nccl communicator: %w", err)
	}
	if n != world || r != rank {
		return ErrRankOrderMismatch
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Enqueue errors may follow partial submission. Drain before Plan releases
	// its in-flight buffer, including when cancellation arrives during enqueue.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var submissionContext driver.Context
	if calls.currentContext != nil {
		submissionContext, err = calls.currentContext()
		if err != nil {
			return fmt.Errorf("nccl submission context: %w", err)
		}
		if submissionContext == 0 {
			return errors.New("nccl submission context unavailable")
		}
	}
	submitErr := calls.invoke(q)
	completeErr := calls.complete(q.stream)
	err = errors.Join(submitErr, completeErr, ctx.Err())
	if completeErr != nil {
		pending := &CompletionError{cause: errors.Join(ErrCompletionPending, err), transport: t, rank: rank, stream: q.stream, complete: calls.complete, context: submissionContext, currentContext: calls.currentContext}
		t.mu.Lock()
		t.pending[rank] = pending
		t.mu.Unlock()
		err = pending
	}
	if err != nil {
		// Publish the operation failure before active.Done lets transport Close
		// finish. Plan may not yet have had a chance to retire its pipeline
		// stage, and communicator Close must not replace this error meanwhile.
		comm.inFlightMu.Lock()
		for _, transfer := range comm.inFlight {
			lo, hi := uint64(transfer.BufferAddr), uint64(transfer.BufferAddr)+uint64(transfer.Bytes)
			if uint64(q.send) < hi && lo < uint64(q.send)+q.sendBytes || uint64(q.recv) < hi && lo < uint64(q.recv)+q.recvBytes {
				transfer.Err = errors.Join(transfer.Err, err)
			}
		}
		comm.inFlightMu.Unlock()
	}
	return err
}

// Close stops admission, waits for host calls, then checks each incomplete
// rank once. A failure retains that rank for an explicit retry. A nil result
// establishes quiescence only; it neither destroys handles nor repairs peers.
// Recovery requires the submission CUDA context. For ranks using distinct
// contexts, recover each CompletionError in its own context before Close.
func (t *NCCLTransport) Close() error {
	t.closeMu.Lock()
	defer t.closeMu.Unlock()
	t.mu.Lock()
	t.closed = true
	t.mu.Unlock()
	t.active.Wait()
	t.mu.Lock()
	pending := append([]*CompletionError(nil), t.pending...)
	t.mu.Unlock()
	var err error
	for _, p := range pending {
		if p != nil {
			err = errors.Join(err, p.Close())
		}
	}
	return err
}
func ncclRequest(op OperationDescriptor, rank, world int, send, recv driver.DevicePtr) (ncclInvocation, error) {
	q := ncclInvocation{op: op, send: send, recv: recv}
	if err := op.Validate(world); err != nil {
		return q, err
	}
	if rank < 0 || rank >= world {
		return q, ErrInvalidRank
	}
	if send == 0 || recv == 0 {
		return q, ErrNilBuffer
	}
	if op.Topology != TopologyAny {
		return q, ErrTopologyMismatch
	}
	switch op.DataType {
	case cudair.Float32:
		q.dtype = nccl.Float32
	case cudair.Float16:
		q.dtype = nccl.Float16
	case cudair.BFloat16:
		q.dtype = nccl.Bfloat16
	case cudair.Int32:
		q.dtype = nccl.Int32
	case cudair.Int64:
		q.dtype = nccl.Int64
	case cudair.Uint8:
		q.dtype = nccl.Uint8
	case cudair.Int8:
		q.dtype = nccl.Int8
	default:
		return q, ErrExecutionUnavailable
	}
	switch op.Kind {
	case KindAllReduce, KindReduceScatter, KindReduce:
		switch op.Reduction {
		case ReductionSum:
			q.reduction = nccl.Sum
		case ReductionProd:
			q.reduction = nccl.Prod
		case ReductionMin:
			q.reduction = nccl.Min
		case ReductionMax:
			q.reduction = nccl.Max
		case ReductionAvg:
			q.reduction = nccl.Avg
		default:
			return q, ErrExecutionUnavailable
		}
	case KindAllGather, KindBroadcast:
	default:
		return q, ErrExecutionUnavailable
	}
	max, size := uint64(^uint(0)>>1), uint64(op.DataType.SizeBytes())
	if uint64(op.Count) > max/size {
		return q, ErrBufferLengthMismatch
	}
	q.sendBytes, q.recvBytes = uint64(op.Count)*size, uint64(op.Count)*size
	if op.Kind == KindAllGather {
		if q.recvBytes > max/uint64(world) {
			return q, ErrBufferLengthMismatch
		}
		q.recvBytes *= uint64(world)
	}
	if op.Kind == KindReduceScatter {
		if q.sendBytes > max/uint64(world) {
			return q, ErrBufferLengthMismatch
		}
		q.sendBytes *= uint64(world)
	}
	if uint64(send) > ^uint64(0)-q.sendBytes || uint64(recv) > ^uint64(0)-q.recvBytes {
		return q, ErrBufferLengthMismatch
	}
	wantSend, wantRecv := send, recv
	if op.Kind == KindAllGather {
		wantSend = recv + driver.DevicePtr(uint64(rank)*q.sendBytes)
	}
	if op.Kind == KindReduceScatter {
		wantRecv = send + driver.DevicePtr(uint64(rank)*q.recvBytes)
	}
	if op.InPlace {
		if send != wantSend || recv != wantRecv || (op.Kind != KindAllGather && op.Kind != KindReduceScatter && send != recv) {
			return q, ErrBufferOverlap
		}
	} else if uint64(send) < uint64(recv)+q.recvBytes && uint64(recv) < uint64(send)+q.sendBytes {
		return q, ErrBufferOverlap
	}
	return q, nil
}
