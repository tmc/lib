package cudaagreement

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// megaInvocation holds real allocation borrows across attempted enqueues.
// Its private join is always stream.Synchronize in the actual consumer.
// The executor mutex protects this state, including releases and join retries.
type megaInvocation struct {
	stream    *cuda.Stream
	join      func(context.Context) error
	releases  []func()
	attempted bool
}

func (p *megaInvocation) borrow(borrow func() (func(), error)) error {
	release, err := borrow()
	if err != nil {
		return err
	}
	p.releases = append(p.releases, release)
	return nil
}

func (p *megaInvocation) release() {
	for i := len(p.releases) - 1; i >= 0; i-- {
		p.releases[i]()
	}
	p.releases = nil
}

func (p *megaInvocation) enqueue(submit func() error) error {
	// Even an enqueue reporting an error may have submitted work, or may
	// report a prior asynchronous error. Only a successful join releases it.
	p.attempted = true
	return submit()
}

func (e *MegaExecutor) rollbackSubmission() {
	if e.pending != nil && !e.pending.attempted {
		e.pending.release()
		e.pending = nil
	}
}

func (e *MegaExecutor) drain(ctx context.Context) error {
	if e.pending == nil {
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if e.pending.attempted {
		if err := e.pending.join(ctx); err != nil {
			return fmt.Errorf("cudaagreement: drain persistent work: %w", err)
		}
	}
	e.pending.release()
	e.pending = nil
	return nil
}

// Drain waits for all attempted persistent work and final output copies on
// their actual stream, then releases allocation borrows. Cancellation stops
// waiting, not device work; failed waits retain ownership for a later retry.
// Drain never closes the caller's stream, device or module cache.
func (e *MegaExecutor) Drain(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.drain(ctx)
}

func (e *MegaExecutor) idle() error {
	if e.pending != nil && e.pending.attempted {
		return cudamem.ErrInUse
	}
	return nil
}
