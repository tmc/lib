package ggml

import (
	"context"
	"unsafe"
)

// NewTensor1D allocates a one-dimensional tensor in the context.
func (c *Context) NewTensor1D(t DType, n0 int64) (*Tensor, error) {
	if c == nil {
		return nil, &OperationError{Op: "NewTensor1D", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := c.acquire()
	if err != nil {
		return nil, &OperationError{Op: "NewTensor1D", Rank: c.Rank(), Peer: -1, Err: err}
	}
	result0, err := b.newTensor1D(t, n0)
	if err != nil {
		return nil, &OperationError{Op: "NewTensor1D", Rank: c.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return result0, nil
}

// TensorData returns the backend-owned data pointer.
func (t *Tensor) TensorData() unsafe.Pointer {
	if t == nil {
		return nil
	}
	b, err := t.acquire()
	if err != nil {
		return nil
	}
	return b.tensorData(t)
}

// Add appends an elementwise add node to the graph.
func (c *Context) Add(a, rhs *Tensor) (*Tensor, error) {
	if c == nil {
		return nil, &OperationError{Op: "Add", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := c.acquire()
	if err != nil {
		return nil, &OperationError{Op: "Add", Rank: c.Rank(), Peer: -1, Err: err}
	}
	result0, err := b.add(a, rhs)
	if err != nil {
		return nil, &OperationError{Op: "Add", Rank: c.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return result0, nil
}

// Mul appends an elementwise multiply node to the graph.
func (c *Context) Mul(a, rhs *Tensor) (*Tensor, error) {
	if c == nil {
		return nil, &OperationError{Op: "Mul", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := c.acquire()
	if err != nil {
		return nil, &OperationError{Op: "Mul", Rank: c.Rank(), Peer: -1, Err: err}
	}
	result0, err := b.mul(a, rhs)
	if err != nil {
		return nil, &OperationError{Op: "Mul", Rank: c.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return result0, nil
}

// BuildForward builds a forward compute graph rooted at t.
func BuildForward(t *Tensor) (*Graph, error) {
	if t == nil {
		return nil, &OperationError{Op: "BuildForward", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := t.acquire()
	if err != nil {
		return nil, &OperationError{Op: "BuildForward", Rank: t.Rank(), Peer: -1, Err: err}
	}
	result0, err := b.buildForward(t)
	if err != nil {
		return nil, &OperationError{Op: "BuildForward", Rank: t.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return result0, nil
}

// GraphCompute computes g on the CPU backend and stops waiting when ctx is canceled.
func (c *Context) GraphCompute(ctx context.Context, g *Graph) error {
	if c == nil {
		return &OperationError{Op: "GraphCompute", Rank: -1, Peer: -1, Err: ErrClosed}
	}
	b, err := c.acquire()
	if err != nil {
		return &OperationError{Op: "GraphCompute", Rank: c.Rank(), Peer: -1, Err: err}
	}
	if err := b.graphCompute(ctx, g); err != nil {
		return &OperationError{Op: "GraphCompute", Rank: c.Rank(), Peer: -1, Backend: b.Name(), Err: err}
	}
	return nil
}
