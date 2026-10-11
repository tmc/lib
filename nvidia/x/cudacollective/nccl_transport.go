package cudacollective

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// NCCLTransport executes on caller-owned rank-ordered communicators and streams.
// Comms and Streams must each contain WorldSize entries and remain unchanged.
// The caller must establish the rank's CUDA context and retain buffers and
// handles until Execute returns. Close waits for admitted work without
// destroying borrowed handles. Stream zero selects the CUDA default stream.
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
	ranks   []sync.Mutex
	calls   *ncclCalls
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
	info     func(nccl.Comm) (int, int, error)
	invoke   func(ncclInvocation) error
	complete func(driver.Stream) error
}

var nativeNCCLCalls = ncclCalls{
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
	if len(t.Comms) != world || len(t.Streams) != world || t.Comms[rank] == 0 {
		t.mu.Unlock()
		return ErrExecutionUnavailable
	}
	if t.ranks == nil {
		t.ranks = make([]sync.Mutex, world)
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
	submitErr := calls.invoke(q)
	completeErr := calls.complete(q.stream)
	return errors.Join(submitErr, completeErr, ctx.Err())
}
func (t *NCCLTransport) Close() error {
	t.mu.Lock()
	t.closed = true
	t.mu.Unlock()
	t.active.Wait()
	return nil
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
