package ggml

import "context"

type Tensor struct {
	b      backend
	closed bool
	rank   int
	size   int
}

func NewTensor(ctx context.Context, cfg Config) (*Tensor, error) {
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
	return &Tensor{b: b, rank: cfg.Rank, size: cfg.Size}, nil
}

func (x *Tensor) Close() error {
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

func (x *Tensor) Rank() int {
	if x == nil {
		return -1
	}
	return x.rank
}

func (x *Tensor) Size() int {
	if x == nil {
		return 0
	}
	return x.size
}

func (x *Tensor) Capabilities() Capabilities {
	if x == nil || x.b == nil {
		return Capabilities{}
	}
	return x.b.Capabilities()
}

func (x *Tensor) acquire() (backend, error) {
	if x == nil || x.closed {
		return nil, ErrClosed
	}
	if x.b == nil {
		return nil, ErrUnsupported
	}
	return x.b, nil
}
