package ggml

import (
	"context"
	"fmt"
	"unsafe"
)

type backend interface {
	Name() string
	Close() error
	Capabilities() Capabilities
	newTensor1D(t DType, n0 int64) (*Tensor, error)
	tensorData(t *Tensor) unsafe.Pointer
	add(a *Tensor, rhs *Tensor) (*Tensor, error)
	mul(a *Tensor, rhs *Tensor) (*Tensor, error)
	buildForward(t *Tensor) (*Graph, error)
	graphCompute(ctx context.Context, g *Graph) error
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

func (unsupportedBackend) Name() string                                     { return "unsupported" }
func (unsupportedBackend) Close() error                                     { return nil }
func (unsupportedBackend) Capabilities() Capabilities                       { return Capabilities{} }
func (unsupportedBackend) newTensor1D(t DType, n0 int64) (*Tensor, error)   { return nil, ErrUnsupported }
func (unsupportedBackend) tensorData(t *Tensor) unsafe.Pointer              { return nil }
func (unsupportedBackend) add(a *Tensor, rhs *Tensor) (*Tensor, error)      { return nil, ErrUnsupported }
func (unsupportedBackend) mul(a *Tensor, rhs *Tensor) (*Tensor, error)      { return nil, ErrUnsupported }
func (unsupportedBackend) buildForward(t *Tensor) (*Graph, error)           { return nil, ErrUnsupported }
func (unsupportedBackend) graphCompute(ctx context.Context, g *Graph) error { return ErrUnsupported }
