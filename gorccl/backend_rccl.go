//go:build gorccl_hardware

package gorccl

import (
	"context"
	"fmt"

	rccl "github.com/tmc/lib/rocm/rccl"
)

func init() {
	registerBackend("rccl", newRCCLBackend)
}

type rcclBackend struct {
	comm   rccl.Comm
	rank   int
	size   int
	closed bool
}

func newRCCLBackend(ctx context.Context, cfg Config) (backend, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := rcclValidateConfig(cfg); err != nil {
		return nil, err
	}
	id, err := rcclUniqueID(ctx, cfg)
	if err != nil {
		return nil, err
	}
	comm, err := rccl.NewCommRank(int32(cfg.Size), id, int32(cfg.Rank))
	if err != nil {
		return nil, fmt.Errorf("rccl init rank: %w", err)
	}
	return &rcclBackend{comm: comm, rank: cfg.Rank, size: cfg.Size}, nil
}

func (b *rcclBackend) Name() string { return "rccl" }

func (b *rcclBackend) Close() error {
	if b == nil || b.closed || b.comm == 0 {
		return nil
	}
	b.closed = true
	if err := b.comm.Destroy(); err != nil {
		return fmt.Errorf("rccl destroy: %w", err)
	}
	return nil
}

func (b *rcclBackend) Capabilities() Capabilities {
	ready := b != nil && !b.closed
	return Capabilities{
		Collectives:  []string{"all_reduce", "all_gather", "broadcast", "all_to_all", "reduce_scatter"},
		PointToPoint: true,
		DeviceKinds:  []DeviceKind{DeviceROCM},
		Provider:     ProviderInfo{Name: "rccl", Runtime: "rocm"},
		Health:       HealthSnapshot{Ready: ready},
	}
}

func (b *rcclBackend) barrier(ctx context.Context) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	return ErrUnsupported
}

func (b *rcclBackend) allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := rcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	redop, err := rcclReduceOp(op)
	if err != nil {
		return err
	}
	if err := b.comm.AllReduce(srcPtr, dstPtr, src.Count, dtype, redop, rcclStream(dst, src)); err != nil {
		return fmt.Errorf("rccl all reduce: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) allGather(ctx context.Context, dst Buffer, src Buffer) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := rcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.AllGather(srcPtr, dstPtr, src.Count, dtype, rcclStream(dst, src)); err != nil {
		return fmt.Errorf("rccl all gather: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := rcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.Broadcast(srcPtr, dstPtr, src.Count, dtype, int32(root), rcclStream(dst, src)); err != nil {
		return fmt.Errorf("rccl broadcast: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) allToAll(ctx context.Context, dst Buffer, src Buffer) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	if b == nil || b.size <= 0 {
		return fmt.Errorf("%w: rccl backend is closed", ErrClosed)
	}
	if src.Count%uint(b.size) != 0 {
		return fmt.Errorf("%w: all-to-all count %d is not divisible by size %d", ErrInvalidArgument, src.Count, b.size)
	}
	srcPtr, dstPtr, dtype, err := rcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	if err := b.comm.AlltoAll(srcPtr, dstPtr, src.Count/uint(b.size), dtype, rcclStream(dst, src)); err != nil {
		return fmt.Errorf("rccl all-to-all: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := rcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	redop, err := rcclReduceOp(op)
	if err != nil {
		return err
	}
	if err := b.comm.ReduceScatter(srcPtr, dstPtr, dst.Count, dtype, redop, rcclStream(dst, src)); err != nil {
		return fmt.Errorf("rccl reduce scatter: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) send(ctx context.Context, peer int, src Buffer) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	srcPtr, err := rcclBufferPtr(src)
	if err != nil {
		return err
	}
	dtype, err := rcclDataType(src.DType)
	if err != nil {
		return err
	}
	if err := b.comm.Send(srcPtr, src.Count, dtype, int32(peer), rcclSingleStream(src)); err != nil {
		return fmt.Errorf("rccl send: %w", err)
	}
	return rcclContext(ctx)
}

func (b *rcclBackend) recv(ctx context.Context, peer int, dst Buffer) error {
	if err := rcclContext(ctx); err != nil {
		return err
	}
	dstPtr, err := rcclBufferPtr(dst)
	if err != nil {
		return err
	}
	dtype, err := rcclDataType(dst.DType)
	if err != nil {
		return err
	}
	if err := b.comm.Recv(dstPtr, dst.Count, dtype, int32(peer), rcclSingleStream(dst)); err != nil {
		return fmt.Errorf("rccl recv: %w", err)
	}
	return rcclContext(ctx)
}

func rcclValidateConfig(cfg Config) error {
	if cfg.Size <= 0 {
		return fmt.Errorf("%w: size must be positive", ErrInvalidArgument)
	}
	if cfg.Rank < 0 || cfg.Rank >= cfg.Size {
		return fmt.Errorf("%w: rank %d out of range for size %d", ErrInvalidArgument, cfg.Rank, cfg.Size)
	}
	if cfg.Size > 1 && cfg.Store == nil {
		return fmt.Errorf("%w: rccl store required for size %d", ErrInvalidArgument, cfg.Size)
	}
	return nil
}

func rcclUniqueID(ctx context.Context, cfg Config) (rccl.UniqueId, error) {
	if err := rcclContext(ctx); err != nil {
		var zero rccl.UniqueId
		return zero, err
	}
	key := cfg.Rendezvous
	if key == "" {
		key = "default"
	}
	key += "/rccl/unique-id"
	if cfg.Rank == 0 {
		id, err := rcclGetUniqueID()
		if err != nil {
			var zero rccl.UniqueId
			return zero, fmt.Errorf("rccl get unique id: %w", err)
		}
		if cfg.Store != nil {
			if err := cfg.Store.Set(ctx, key, id.Internal[:]); err != nil {
				var zero rccl.UniqueId
				return zero, fmt.Errorf("rccl store unique id: %w", err)
			}
		}
		return id, nil
	}
	raw, err := cfg.Store.Get(ctx, key)
	if err != nil {
		var zero rccl.UniqueId
		return zero, fmt.Errorf("rccl get unique id: %w", err)
	}
	var id rccl.UniqueId
	if len(raw) != len(id.Internal) {
		return id, fmt.Errorf("%w: rccl unique id length %d, want %d", ErrInvalidArgument, len(raw), len(id.Internal))
	}
	copy(id.Internal[:], raw)
	return id, nil
}

func rcclGetUniqueID() (rccl.UniqueId, error) {
	var id rccl.UniqueId
	if err := rccl.GetUniqueId(&id); err != nil {
		return id, err
	}
	return id, nil
}

func rcclCollectiveArgs(dst Buffer, src Buffer) (uintptr, uintptr, rccl.DataType, error) {
	srcPtr, err := rcclBufferPtr(src)
	if err != nil {
		return 0, 0, 0, err
	}
	dstPtr, err := rcclBufferPtr(dst)
	if err != nil {
		return 0, 0, 0, err
	}
	dtype, err := rcclDataType(src.DType)
	if err != nil {
		return 0, 0, 0, err
	}
	return srcPtr, dstPtr, dtype, nil
}

func rcclBufferPtr(buf Buffer) (uintptr, error) {
	if buf.Count == 0 {
		return 0, nil
	}
	if buf.Device.Kind != DeviceROCM {
		return 0, fmt.Errorf("%w: buffer device %d, want %d", ErrInvalidArgument, buf.Device.Kind, DeviceROCM)
	}
	if buf.Ptr == nil {
		return 0, fmt.Errorf("%w: nil device pointer", ErrInvalidArgument)
	}
	return uintptr(buf.Ptr), nil
}

func rcclDataType(dtype DType) (rccl.DataType, error) {
	switch dtype {
	case Int8:
		return rccl.Int8, nil
	case Int32:
		return rccl.Int32, nil
	case Int64:
		return rccl.Int64, nil
	case Uint8:
		return rccl.Uint8, nil
	case Uint32:
		return rccl.Uint32, nil
	case Uint64:
		return rccl.Uint64, nil
	case Float32:
		return rccl.Float32, nil
	case Float64:
		return rccl.Float64, nil
	default:
		return 0, fmt.Errorf("%w: unsupported dtype %d", ErrInvalidArgument, dtype)
	}
}

func rcclReduceOp(op ReduceOp) (rccl.RedOp, error) {
	switch op {
	case Sum:
		return rccl.Sum, nil
	case Max:
		return rccl.Max, nil
	case Min:
		return rccl.Min, nil
	case Prod:
		return rccl.Prod, nil
	default:
		return 0, fmt.Errorf("%w: unsupported reduce op %d", ErrInvalidArgument, op)
	}
}

func rcclStream(dst Buffer, src Buffer) rccl.Stream {
	stream := src.Stream
	if dst.Stream != 0 {
		stream = dst.Stream
	}
	return rccl.Stream(stream)
}

func rcclSingleStream(buf Buffer) rccl.Stream {
	stream := buf.Stream
	return rccl.Stream(stream)
}

func rcclContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
