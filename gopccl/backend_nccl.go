//go:build gopccl_nccl

package gopccl

import (
	"context"
	"fmt"

	nccl "github.com/tmc/lib/nvidia/nccl"
)

func init() {
	registerBackend("nccl", newNCCLBackend)
}

type ncclBackend struct {
	comm   nccl.Comm
	rank   int
	size   int
	closed bool
}

func newNCCLBackend(ctx context.Context, cfg Config) (backend, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ncclValidateConfig(cfg); err != nil {
		return nil, err
	}
	id, err := ncclUniqueID(ctx, cfg)
	if err != nil {
		return nil, err
	}
	comm, err := nccl.NewCommInitRank(int32(cfg.Size), id, int32(cfg.Rank))
	if err != nil {
		return nil, fmt.Errorf("nccl init rank: %w", err)
	}
	return &ncclBackend{comm: comm, rank: cfg.Rank, size: cfg.Size}, nil
}

func (b *ncclBackend) Name() string { return "nccl" }

func (b *ncclBackend) Close() error {
	if b == nil || b.closed || b.comm == 0 {
		return nil
	}
	b.closed = true
	if err := b.comm.Destroy(); err != nil {
		return fmt.Errorf("nccl destroy: %w", err)
	}
	return nil
}

func (b *ncclBackend) Capabilities() Capabilities {
	ready := b != nil && !b.closed
	return Capabilities{
		Collectives:  []string{"all_reduce", "all_gather", "broadcast", "all_to_all", "reduce_scatter"},
		PointToPoint: true,
		DeviceKinds:  []DeviceKind{DeviceCUDA},
		Provider:     ProviderInfo{Name: "nccl", Runtime: "cuda"},
		Health:       HealthSnapshot{Ready: ready},
	}
}

func (b *ncclBackend) barrier(ctx context.Context) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	return ErrUnsupported
}

func (b *ncclBackend) allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := ncclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	redop, err := ncclReduceOp(op)
	if err != nil {
		return err
	}
	if err := b.comm.AllReduce(srcPtr, dstPtr, src.Count, dtype, redop, ncclStream(dst, src)); err != nil {
		return fmt.Errorf("nccl all reduce: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) allGather(ctx context.Context, dst Buffer, src Buffer) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := ncclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.AllGather(srcPtr, dstPtr, src.Count, dtype, ncclStream(dst, src)); err != nil {
		return fmt.Errorf("nccl all gather: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := ncclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.Broadcast(srcPtr, dstPtr, src.Count, dtype, int32(root), ncclStream(dst, src)); err != nil {
		return fmt.Errorf("nccl broadcast: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) allToAll(ctx context.Context, dst Buffer, src Buffer) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	if b == nil || b.size <= 0 {
		return fmt.Errorf("%w: nccl backend is closed", ErrClosed)
	}
	if src.Count%uint(b.size) != 0 {
		return fmt.Errorf("%w: all-to-all count %d is not divisible by size %d", ErrInvalidArgument, src.Count, b.size)
	}
	srcPtr, dstPtr, dtype, err := ncclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.AlltoAll(srcPtr, dstPtr, src.Count/uint(b.size), dtype, ncclStream(dst, src)); err != nil {
		return fmt.Errorf("nccl all-to-all: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := ncclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	redop, err := ncclReduceOp(op)
	if err != nil {
		return err
	}
	if err := b.comm.ReduceScatter(srcPtr, dstPtr, dst.Count, dtype, redop, ncclStream(dst, src)); err != nil {
		return fmt.Errorf("nccl reduce scatter: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) send(ctx context.Context, peer int, src Buffer) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	srcPtr, err := ncclBufferPtr(src)
	if err != nil {
		return err
	}
	dtype, err := ncclDataType(src.DType)
	if err != nil {
		return err
	}
	if err := b.comm.Send(srcPtr, src.Count, dtype, int32(peer), ncclSingleStream(src)); err != nil {
		return fmt.Errorf("nccl send: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) recv(ctx context.Context, peer int, dst Buffer) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	dstPtr, err := ncclBufferPtr(dst)
	if err != nil {
		return err
	}
	dtype, err := ncclDataType(dst.DType)
	if err != nil {
		return err
	}
	if err := b.comm.Recv(dstPtr, dst.Count, dtype, int32(peer), ncclSingleStream(dst)); err != nil {
		return fmt.Errorf("nccl recv: %w", err)
	}
	return ncclContext(ctx)
}

func (b *ncclBackend) peersPending(ctx context.Context) (bool, error) {
	if err := ncclContext(ctx); err != nil {
		return false, err
	}
	return false, ErrUnsupported
}

func (b *ncclBackend) updateTopology(ctx context.Context) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	return ErrUnsupported
}

func (b *ncclBackend) optimizeTopology(ctx context.Context) error {
	if err := ncclContext(ctx); err != nil {
		return err
	}
	return ErrUnsupported
}

func (b *ncclBackend) synchronizeState(ctx context.Context, revision uint64, tensors []StateTensor, strategy SyncStrategy) (SyncInfo, error) {
	if err := ncclContext(ctx); err != nil {
		return SyncInfo{}, err
	}
	return SyncInfo{}, ErrUnsupported
}

func ncclValidateConfig(cfg Config) error {
	if cfg.Size <= 0 {
		return fmt.Errorf("%w: size must be positive", ErrInvalidArgument)
	}
	if cfg.Rank < 0 || cfg.Rank >= cfg.Size {
		return fmt.Errorf("%w: rank %d out of range for size %d", ErrInvalidArgument, cfg.Rank, cfg.Size)
	}
	if cfg.Size > 1 && cfg.Store == nil {
		return fmt.Errorf("%w: nccl store required for size %d", ErrInvalidArgument, cfg.Size)
	}
	return nil
}

func ncclUniqueID(ctx context.Context, cfg Config) (nccl.UniqueId, error) {
	if err := ncclContext(ctx); err != nil {
		var zero nccl.UniqueId
		return zero, err
	}
	key := cfg.Rendezvous
	if key == "" {
		key = "default"
	}
	key += "/nccl/unique-id"
	if cfg.Rank == 0 {
		id, err := ncclGetUniqueID()
		if err != nil {
			var zero nccl.UniqueId
			return zero, fmt.Errorf("nccl get unique id: %w", err)
		}
		if cfg.Store != nil {
			if err := cfg.Store.Set(ctx, key, id.Internal[:]); err != nil {
				var zero nccl.UniqueId
				return zero, fmt.Errorf("nccl store unique id: %w", err)
			}
		}
		return id, nil
	}
	raw, err := cfg.Store.Get(ctx, key)
	if err != nil {
		var zero nccl.UniqueId
		return zero, fmt.Errorf("nccl get unique id: %w", err)
	}
	var id nccl.UniqueId
	if len(raw) != len(id.Internal) {
		return id, fmt.Errorf("%w: nccl unique id length %d, want %d", ErrInvalidArgument, len(raw), len(id.Internal))
	}
	copy(id.Internal[:], raw)
	return id, nil
}

func ncclGetUniqueID() (nccl.UniqueId, error) {
	return nccl.GetUniqueId()
}

func ncclCollectiveArgs(dst Buffer, src Buffer) (uintptr, uintptr, nccl.DataType, error) {
	srcPtr, err := ncclBufferPtr(src)
	if err != nil {
		return 0, 0, 0, err
	}
	dstPtr, err := ncclBufferPtr(dst)
	if err != nil {
		return 0, 0, 0, err
	}
	dtype, err := ncclDataType(src.DType)
	if err != nil {
		return 0, 0, 0, err
	}
	return srcPtr, dstPtr, dtype, nil
}

func ncclBufferPtr(buf Buffer) (uintptr, error) {
	if buf.Count == 0 {
		return 0, nil
	}
	if buf.Device.Kind != DeviceCUDA {
		return 0, fmt.Errorf("%w: buffer device %d, want %d", ErrInvalidArgument, buf.Device.Kind, DeviceCUDA)
	}
	if buf.Ptr == nil {
		return 0, fmt.Errorf("%w: nil device pointer", ErrInvalidArgument)
	}
	return uintptr(buf.Ptr), nil
}

func ncclDataType(dtype DType) (nccl.DataType, error) {
	switch dtype {
	case Int8:
		return nccl.Int8, nil
	case Int32:
		return nccl.Int32, nil
	case Int64:
		return nccl.Int64, nil
	case Uint8:
		return nccl.Uint8, nil
	case Uint32:
		return nccl.Uint32, nil
	case Uint64:
		return nccl.Uint64, nil
	case Float32:
		return nccl.Float32, nil
	case Float64:
		return nccl.Float64, nil
	default:
		return 0, fmt.Errorf("%w: unsupported dtype %d", ErrInvalidArgument, dtype)
	}
}

func ncclReduceOp(op ReduceOp) (nccl.RedOp, error) {
	switch op {
	case Sum:
		return nccl.Sum, nil
	case Max:
		return nccl.Max, nil
	case Min:
		return nccl.Min, nil
	case Prod:
		return nccl.Prod, nil
	default:
		return 0, fmt.Errorf("%w: unsupported reduce op %d", ErrInvalidArgument, op)
	}
}

func ncclStream(dst Buffer, src Buffer) uintptr {
	stream := src.Stream
	if dst.Stream != 0 {
		stream = dst.Stream
	}
	return stream
}

func ncclSingleStream(buf Buffer) uintptr {
	stream := buf.Stream
	return stream
}

func ncclContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
