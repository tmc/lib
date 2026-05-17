package ggml

import "context"

type BackendBuffer struct {
	b      backend
	closed bool
	rank   int
	size   int
}

func NewBackendBuffer(ctx context.Context, cfg Config) (*BackendBuffer, error) {
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
	return &BackendBuffer{b: b, rank: cfg.Rank, size: cfg.Size}, nil
}

func (x *BackendBuffer) Close() error {
	if x == nil || x.closed {
		return nil
	}
	x.closed = true
	if x.b == nil {
		return nil
	}
	return x.b.Close()
}

func (x *BackendBuffer) Rank() int {
	if x == nil {
		return -1
	}
	return x.rank
}

func (x *BackendBuffer) Size() int {
	if x == nil {
		return 0
	}
	return x.size
}

func (x *BackendBuffer) Capabilities() Capabilities {
	if x == nil || x.b == nil {
		return Capabilities{}
	}
	return x.b.Capabilities()
}

func (x *BackendBuffer) acquire() (backend, error) {
	if x == nil || x.closed {
		return nil, ErrClosed
	}
	if x.b == nil {
		return nil, ErrUnsupported
	}
	return x.b, nil
}
