package gorccl

import (
	"context"
	"fmt"
)

func (g *Group) Barrier(ctx context.Context) error {
	if g == nil {
		return &OperationError{Op: "Barrier", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "Barrier", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if err := b.barrier(ctx); err != nil {
		return &OperationError{Op: "Barrier", Rank: g.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) AllReduce(ctx context.Context, dst, src Buffer, op ReduceOp) error {
	if g == nil {
		return &OperationError{Op: "AllReduce", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "AllReduce", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if dst.Count != src.Count {
		return &OperationError{Op: "AllReduce", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("allreduce: %w: destination count %d, want %d", ErrInvalidArgument, dst.Count, src.Count)}
	}
	if dst.DType != src.DType {
		return &OperationError{Op: "AllReduce", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("allreduce: %w: source and destination dtype differ", ErrInvalidArgument)}
	}
	if err := b.allReduce(ctx, dst, src, op); err != nil {
		return &OperationError{Op: "AllReduce", Rank: g.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) AllGather(ctx context.Context, dst, src Buffer) error {
	if g == nil {
		return &OperationError{Op: "AllGather", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "AllGather", Rank: g.Rank(), Peer: -1, Err: err}
	}
	wantDstCount := src.Count * uint(g.Size())
	if dst.Count != wantDstCount {
		return &OperationError{Op: "AllGather", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("allgather: %w: dst count %d, want %d", ErrInvalidArgument, dst.Count, wantDstCount)}
	}
	if dst.DType != src.DType {
		return &OperationError{Op: "AllGather", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("allgather: %w: source and destination dtype differ", ErrInvalidArgument)}
	}
	if err := b.allGather(ctx, dst, src); err != nil {
		return &OperationError{Op: "AllGather", Rank: g.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) Broadcast(ctx context.Context, dst, src Buffer, root int) error {
	if g == nil {
		return &OperationError{Op: "Broadcast", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "Broadcast", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if root < 0 || root >= g.Size() {
		return &OperationError{Op: "Broadcast", Rank: g.Rank(), Peer: root, Err: fmt.Errorf("broadcast: %w: rank %d out of range", ErrInvalidArgument, root)}
	}
	if dst.Count != src.Count {
		return &OperationError{Op: "Broadcast", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("broadcast: %w: destination count %d, want %d", ErrInvalidArgument, dst.Count, src.Count)}
	}
	if dst.DType != src.DType {
		return &OperationError{Op: "Broadcast", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("broadcast: %w: source and destination dtype differ", ErrInvalidArgument)}
	}
	if err := b.broadcast(ctx, dst, src, root); err != nil {
		return &OperationError{Op: "Broadcast", Rank: g.Rank(), Peer: root, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) AllToAll(ctx context.Context, dst, src Buffer) error {
	if g == nil {
		return &OperationError{Op: "AllToAll", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "AllToAll", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if dst.Count != src.Count {
		return &OperationError{Op: "AllToAll", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("alltoall: %w: destination count %d, want %d", ErrInvalidArgument, dst.Count, src.Count)}
	}
	if dst.DType != src.DType {
		return &OperationError{Op: "AllToAll", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("alltoall: %w: source and destination dtype differ", ErrInvalidArgument)}
	}
	if err := b.allToAll(ctx, dst, src); err != nil {
		return &OperationError{Op: "AllToAll", Rank: g.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) ReduceScatter(ctx context.Context, dst, src Buffer, op ReduceOp) error {
	if g == nil {
		return &OperationError{Op: "ReduceScatter", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "ReduceScatter", Rank: g.Rank(), Peer: -1, Err: err}
	}
	wantSrcCount := dst.Count * uint(g.Size())
	if src.Count != wantSrcCount {
		return &OperationError{Op: "ReduceScatter", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("reducescatter: %w: src count %d, want %d", ErrInvalidArgument, src.Count, wantSrcCount)}
	}
	if dst.DType != src.DType {
		return &OperationError{Op: "ReduceScatter", Rank: g.Rank(), Peer: -1, Err: fmt.Errorf("reducescatter: %w: source and destination dtype differ", ErrInvalidArgument)}
	}
	if err := b.reduceScatter(ctx, dst, src, op); err != nil {
		return &OperationError{Op: "ReduceScatter", Rank: g.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) Send(ctx context.Context, peer int, src Buffer) error {
	if g == nil {
		return &OperationError{Op: "Send", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "Send", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if peer < 0 || peer >= g.Size() {
		return &OperationError{Op: "Send", Rank: g.Rank(), Peer: peer, Err: fmt.Errorf("send: %w: rank %d out of range", ErrInvalidArgument, peer)}
	}
	if err := b.send(ctx, peer, src); err != nil {
		return &OperationError{Op: "Send", Rank: g.Rank(), Peer: peer, Backend: b.Name(), Err: err}
	}
	return nil
}

func (g *Group) Recv(ctx context.Context, peer int, dst Buffer) error {
	if g == nil {
		return &OperationError{Op: "Recv", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := g.acquire()
	if err != nil {
		return &OperationError{Op: "Recv", Rank: g.Rank(), Peer: -1, Err: err}
	}
	if peer < 0 || peer >= g.Size() {
		return &OperationError{Op: "Recv", Rank: g.Rank(), Peer: peer, Err: fmt.Errorf("recv: %w: rank %d out of range", ErrInvalidArgument, peer)}
	}
	if err := b.recv(ctx, peer, dst); err != nil {
		return &OperationError{Op: "Recv", Rank: g.Rank(), Peer: peer, Backend: b.Name(), Err: err}
	}
	return nil
}
