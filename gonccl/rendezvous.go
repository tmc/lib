package gonccl

import "context"

// Store is a context-aware rendezvous key/value store.
type Store interface {
	Set(ctx context.Context, key string, value []byte) error
	Get(ctx context.Context, key string) ([]byte, error)
	Delete(ctx context.Context, key string) error
}
