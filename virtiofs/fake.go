package virtiofs

import (
	"context"
	"sync"

	"github.com/tmc/lib/fuse"
)

type MemoryTransport struct {
	mu     sync.Mutex
	server *Server
	closed bool
}

func NewMemoryTransport(fs fuse.FileSystem) (*MemoryTransport, error) {
	server, err := NewServer(fs)
	if err != nil {
		return nil, err
	}
	return &MemoryTransport{server: server}, nil
}

func (t *MemoryTransport) RoundTrip(ctx context.Context, request []byte) ([]byte, error) {
	if t == nil {
		return nil, ErrClosed
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil, ErrClosed
	}
	return t.server.Handle(ctx, request)
}

func (t *MemoryTransport) Close() error {
	if t == nil {
		return ErrClosed
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	return nil
}
