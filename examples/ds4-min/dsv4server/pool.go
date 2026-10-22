package dsv4server

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// sessionPool reuses a bounded number of independent host-side sessions. It
// deliberately does not claim to batch device work: each lease still runs
// the ordinary Session API in order.
type sessionPool struct {
	model Model

	mu      sync.Mutex
	free    []*dsv4.Session
	waiters []*sessionWaiter
	err     error
	closed  bool
}

type sessionWaiter struct {
	result chan sessionResult
}

type sessionResult struct {
	session *dsv4.Session
	err     error
}

type sessionLease struct {
	pool    *sessionPool
	session *dsv4.Session
	once    sync.Once
}

func newSessionPool(model Model, slots int) (*sessionPool, error) {
	p := &sessionPool{model: model}
	for i := 0; i < slots; i++ {
		session, err := model.NewSession()
		if err != nil {
			p.closeSessions()
			return nil, err
		}
		if session == nil {
			p.closeSessions()
			return nil, errors.New("model returned nil session")
		}
		p.free = append(p.free, session)
	}
	return p, nil
}

func (p *sessionPool) acquire(ctx context.Context) (*sessionLease, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	w := &sessionWaiter{result: make(chan sessionResult, 1)}
	p.mu.Lock()
	if p.err != nil {
		err := p.err
		p.mu.Unlock()
		return nil, err
	}
	p.waiters = append(p.waiters, w)
	p.dispatchLocked()
	p.mu.Unlock()

	select {
	case result := <-w.result:
		if result.err != nil {
			return nil, result.err
		}
		return &sessionLease{pool: p, session: result.session}, nil
	case <-ctx.Done():
		p.mu.Lock()
		for i, waiter := range p.waiters {
			if waiter == w {
				copy(p.waiters[i:], p.waiters[i+1:])
				p.waiters = p.waiters[:len(p.waiters)-1]
				p.mu.Unlock()
				return nil, ctx.Err()
			}
		}
		p.mu.Unlock()
		// The dispatcher granted the lease concurrently with cancellation.
		result := <-w.result
		if result.session != nil {
			p.release(result.session)
		}
		return nil, ctx.Err()
	}
}

func (p *sessionPool) dispatchLocked() {
	for p.err == nil && len(p.free) != 0 && len(p.waiters) != 0 {
		session := p.free[0]
		p.free = p.free[1:]
		waiter := p.waiters[0]
		p.waiters = p.waiters[1:]
		waiter.result <- sessionResult{session: session}
	}
	if p.err != nil {
		for _, waiter := range p.waiters {
			waiter.result <- sessionResult{err: p.err}
		}
		p.waiters = nil
	}
}

func (p *sessionPool) release(session *dsv4.Session) {
	if session == nil {
		return
	}
	// Session.Sync compares the next prompt with the retained timeline. Keeping
	// that timeline here enables common-prefix reuse; a changed suffix causes
	// Session.Sync to reset the state before evaluating the new suffix.
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		_ = session.Close()
		return
	}
	p.free = append(p.free, session)
	p.dispatchLocked()
	p.mu.Unlock()
}

// close stops new leases and releases sessions that are not in use. A lease
// returned after close releases its session directly, so an in-flight request
// can finish without racing a close of its mutable state.
func (p *sessionPool) close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.err = ErrClosed
	waiters := p.waiters
	p.waiters = nil
	free := p.free
	p.free = nil
	p.mu.Unlock()

	for _, waiter := range waiters {
		waiter.result <- sessionResult{err: ErrClosed}
	}
	var first error
	for _, session := range free {
		if err := session.Close(); err != nil && first == nil {
			first = fmt.Errorf("close session: %w", err)
		}
	}
	return first
}

func (p *sessionPool) isClosed() bool {
	p.mu.Lock()
	closed := p.closed
	p.mu.Unlock()
	return closed
}

func (l *sessionLease) Release() {
	if l == nil {
		return
	}
	l.once.Do(func() { l.pool.release(l.session) })
}

func (p *sessionPool) closeSessions() {
	for _, session := range p.free {
		_ = session.Close()
	}
	p.free = nil
}
