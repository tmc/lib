package gopccl

import (
	"context"
	"fmt"
)

type backend interface {
	Name() string
	Close() error
	Capabilities() Capabilities
	barrier(context.Context) error
	allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error
	allGather(ctx context.Context, dst Buffer, src Buffer) error
	broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error
	allToAll(ctx context.Context, dst Buffer, src Buffer) error
	reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error
	send(ctx context.Context, peer int, src Buffer) error
	recv(ctx context.Context, peer int, dst Buffer) error
	peersPending(context.Context) (bool, error)
	updateTopology(context.Context) error
	optimizeTopology(context.Context) error
	synchronizeState(ctx context.Context, revision uint64, tensors []StateTensor, strategy SyncStrategy) (SyncInfo, error)
}

type unsupportedBackend struct{}

var backendFactory = func(context.Context, Config) (backend, error) { return unsupportedBackend{}, nil }

type backendProvider func(context.Context, Config) (backend, error)

var backendProviders = map[string]backendProvider{}

func registerBackend(name string, provider backendProvider) {
	if name == "" || provider == nil {
		return
	}
	backendProviders[name] = provider
}

func newBackend(ctx context.Context, cfg Config) (backend, error) {
	if cfg.Backend != "" && cfg.Backend != "auto" {
		provider := backendProviders[cfg.Backend]
		if provider == nil {
			return nil, fmt.Errorf("%w: backend %q", ErrUnsupported, cfg.Backend)
		}
		return provider(ctx, cfg)
	}
	if provider := backendProviders["auto"]; provider != nil {
		return provider(ctx, cfg)
	}
	return backendFactory(ctx, cfg)
}

func (unsupportedBackend) Name() string                  { return "unsupported" }
func (unsupportedBackend) Close() error                  { return nil }
func (unsupportedBackend) Capabilities() Capabilities    { return Capabilities{} }
func (unsupportedBackend) barrier(context.Context) error { return ErrUnsupported }
func (unsupportedBackend) allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	return ErrUnsupported
}
func (unsupportedBackend) allGather(ctx context.Context, dst Buffer, src Buffer) error {
	return ErrUnsupported
}
func (unsupportedBackend) broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error {
	return ErrUnsupported
}
func (unsupportedBackend) allToAll(ctx context.Context, dst Buffer, src Buffer) error {
	return ErrUnsupported
}
func (unsupportedBackend) reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	return ErrUnsupported
}
func (unsupportedBackend) send(ctx context.Context, peer int, src Buffer) error {
	return ErrUnsupported
}
func (unsupportedBackend) recv(ctx context.Context, peer int, dst Buffer) error {
	return ErrUnsupported
}
func (unsupportedBackend) peersPending(context.Context) (bool, error) { return false, ErrUnsupported }
func (unsupportedBackend) updateTopology(context.Context) error       { return ErrUnsupported }
func (unsupportedBackend) optimizeTopology(context.Context) error     { return ErrUnsupported }
func (unsupportedBackend) synchronizeState(ctx context.Context, revision uint64, tensors []StateTensor, strategy SyncStrategy) (SyncInfo, error) {
	return SyncInfo{}, ErrUnsupported
}
