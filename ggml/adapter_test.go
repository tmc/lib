package ggml

import (
	"context"
	"errors"
	"testing"
)

func TestGeneratedBackendProviderRegistry(t *testing.T) {
	oldProviders := backendProviders
	backendProviders = map[string]backendProvider{}
	defer func() { backendProviders = oldProviders }()

	called := false
	registerBackend("cuda", func(ctx context.Context, cfg Config) (backend, error) {
		called = true
		if cfg.Backend != "cuda" {
			t.Fatalf("Backend = %q, want cuda", cfg.Backend)
		}
		return unsupportedBackend{}, nil
	})
	b, err := newBackend(context.Background(), Config{Backend: "cuda"})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("provider was not called")
	}
	if b.Name() != "unsupported" {
		t.Fatalf("backend name = %q, want unsupported", b.Name())
	}
	if _, err := newBackend(context.Background(), Config{Backend: "missing"}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("missing backend error = %v, want ErrUnsupported", err)
	}
}
