package ggml

import "context"

type Context struct {
	b      backend
	closed bool
	rank   int
	size   int
}

func NewContext(ctx context.Context, cfg Config) (*Context, error) {
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
	return &Context{b: b, rank: cfg.Rank, size: cfg.Size}, nil
}

func (x *Context) Close() error {
	if x == nil || x.closed {
		return nil
	}
	x.closed = true
	if x.b == nil {
		return nil
	}
	return x.b.Close()
}

func (x *Context) Rank() int {
	if x == nil {
		return -1
	}
	return x.rank
}

func (x *Context) Size() int {
	if x == nil {
		return 0
	}
	return x.size
}

func (x *Context) Capabilities() Capabilities {
	if x == nil || x.b == nil {
		return Capabilities{}
	}
	return x.b.Capabilities()
}

func (x *Context) acquire() (backend, error) {
	if x == nil || x.closed {
		return nil, ErrClosed
	}
	if x.b == nil {
		return nil, ErrUnsupported
	}
	return x.b, nil
}
