package gonccl

import "context"

type Group struct {
	b      backend
	closed bool
	rank   int
	size   int
}

func Open(ctx context.Context, cfg Config) (*Group, error) {
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
	return &Group{b: b, rank: cfg.Rank, size: cfg.Size}, nil
}

func (x *Group) Close() error {
	if x == nil || x.closed {
		return nil
	}
	x.closed = true
	if x.b == nil {
		return nil
	}
	return x.b.Close()
}

func (x *Group) Rank() int {
	if x == nil {
		return -1
	}
	return x.rank
}

func (x *Group) Size() int {
	if x == nil {
		return 0
	}
	return x.size
}

func (x *Group) Capabilities() Capabilities {
	if x == nil || x.b == nil {
		return Capabilities{}
	}
	return x.b.Capabilities()
}

func (x *Group) acquire() (backend, error) {
	if x == nil || x.closed {
		return nil, ErrClosed
	}
	if x.b == nil {
		return nil, ErrUnsupported
	}
	return x.b, nil
}
