package ggml

import "context"

type Graph struct {
	b      backend
	closed bool
	rank   int
	size   int
}

func NewGraph(ctx context.Context, cfg Config) (*Graph, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b, err := newBackend(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return &Graph{b: b, rank: cfg.Rank, size: cfg.Size}, nil
}

func (x *Graph) Close() error {
	if x == nil || x.closed {
		return nil
	}
	x.closed = true
	if x.b == nil {
		return nil
	}
	x.b = nil
	return nil
}

func (x *Graph) Rank() int {
	if x == nil {
		return -1
	}
	return x.rank
}

func (x *Graph) Size() int {
	if x == nil {
		return 0
	}
	return x.size
}

func (x *Graph) Capabilities() Capabilities {
	if x == nil || x.b == nil {
		return Capabilities{}
	}
	return x.b.Capabilities()
}

func (x *Graph) acquire() (backend, error) {
	if x == nil || x.closed {
		return nil, ErrClosed
	}
	if x.b == nil {
		return nil, ErrUnsupported
	}
	return x.b, nil
}
