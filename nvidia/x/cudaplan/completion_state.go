package cudaplan

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type completionState struct {
	mu       sync.Mutex
	done     atomic.Bool
	wait     func(context.Context) error
	destroy  func() error
	releases []func()
	once     sync.Once
}

func (s *completionState) complete() {
	s.once.Do(func() {
		for _, rel := range s.releases {
			if rel != nil {
				rel()
			}
		}
		s.releases = nil
		s.done.Store(true)
	})
}

func (s *completionState) Wait(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.done.Load() {
		if s.wait != nil {
			if err := s.wait(ctx); err != nil {
				return fmt.Errorf("cudaplan: wait: %w", err)
			}
		}
		s.complete()
	}

	if s.destroy != nil {
		if err := s.destroy(); err != nil {
			return fmt.Errorf("cudaplan: destroy completion event: %w", err)
		}
		s.destroy = nil
	}

	return nil
}

// newEventCompletion creates a Completion backed by a CUDA stream and event.
func newEventCompletion(dev *cuda.Device, stream *cuda.Stream, event *cuda.Event, releases []func()) (*Completion, error) {
	s := &completionState{
		releases: releases,
		destroy: func() error {
			if event != nil {
				return event.Close()
			}
			return nil
		},
	}

	s.wait = func(ctx context.Context) error {
		if stream == nil || event == nil {
			return nil
		}

		if ctx == nil || ctx.Done() == nil {
			return event.Synchronize(nil)
		}

		// Context-aware wait with backoff polling
		if err := ctx.Err(); err != nil {
			return err
		}

		backoff := 50 * time.Microsecond
		maxBackoff := 5 * time.Millisecond

		for {
			idle, err := event.Query()
			if err != nil {
				return err
			}
			if idle {
				return nil
			}

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				backoff *= 2
				if backoff > maxBackoff {
					backoff = maxBackoff
				}
			}
		}
	}

	return &Completion{state: s}, nil
}

// newHostCompletion creates a completion for an owned drain function.
// Native execution passes Stream.Synchronize; tests may supply a controlled
// wait. A canceled or failed drain retains every release callback.
func newHostCompletion(waitFn func(context.Context) error, destroyFn func() error, releases ...func()) *Completion {
	s := &completionState{
		wait:     waitFn,
		destroy:  destroyFn,
		releases: releases,
	}
	return &Completion{state: s}
}
