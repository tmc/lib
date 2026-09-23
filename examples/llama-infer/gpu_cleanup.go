package main

import (
	"errors"
	"fmt"
	"io"
	"sync"
)

// cudaInitError keeps failed cleanup and its dependencies alive for retry.
// Cleanup stops at the first failure, before dependent contexts are closed.
type cudaInitError struct {
	mu      sync.Mutex
	cause   error
	pending []func() error
}

func (e *cudaInitError) Error() string { return e.cause.Error() }
func (e *cudaInitError) Unwrap() error { return e.cause }
func (e *cudaInitError) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	for len(e.pending) != 0 {
		if err := e.pending[0](); err != nil {
			return err
		}
		e.pending = e.pending[1:]
	}
	return nil
}

func finishCUDAInit(cause error, cleanup ...func() error) error {
	var retry io.Closer
	if errors.As(cause, &retry) {
		cleanup = append([]func() error{retry.Close}, cleanup...)
	}
	e := &cudaInitError{cause: cause, pending: cleanup}
	if err := e.Close(); err != nil {
		e.cause = errors.Join(cause, fmt.Errorf("CUDA initialization cleanup: %w", err))
		return e
	}
	return cause
}
