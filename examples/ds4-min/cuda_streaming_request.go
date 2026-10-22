package main

import (
	"errors"
	"io"
	"sync"
)

// The coordinator owns one bounded upload cache. The active request remains
// installed until the final consumer and argument-table cleanup complete.
// Errors close execution admission, but never erase retryable owners.
type cudaStreamingCoordinator struct {
	mu              sync.Mutex
	active          *cudaStreamingRequest
	closeCache      func() error
	closing, closed bool
}

type cudaStreamingRequest struct {
	owner   *cudaStreamingCoordinator
	pending io.Closer
	failed  error
	closed  bool
}

func (o *cudaStreamingCoordinator) request() *cudaStreamingRequest {
	if o == nil {
		return nil
	}
	return &cudaStreamingRequest{owner: o}
}

func (q *cudaStreamingRequest) run(selectExpert func() (io.Closer, error), consume func(io.Closer) error) error {
	if q == nil || q.owner == nil || selectExpert == nil || consume == nil {
		return errors.New("missing CUDA streaming request")
	}
	o := q.owner
	o.mu.Lock()
	defer o.mu.Unlock()
	if q.closed || q.failed != nil || o.closing || o.closed || o.active != nil {
		return errors.New("CUDA streaming request unavailable")
	}
	o.active = q
	var err error
	q.pending, err = selectExpert()
	if err == nil && q.pending == nil {
		err = errors.New("CUDA streaming selection missing owner")
	}
	if err == nil {
		err = consume(q.pending)
	}
	q.failed = err
	return errors.Join(err, q.finishLocked())
}

func (q *cudaStreamingRequest) finishLocked() error {
	if q.pending != nil {
		if err := q.pending.Close(); err != nil {
			return err
		}
		q.pending = nil
	}
	if q.owner.active == q {
		q.owner.active = nil
	}
	return nil
}

func (q *cudaStreamingRequest) reset() error {
	if q == nil {
		return nil
	}
	if q.owner == nil {
		return errors.New("missing CUDA streaming owner")
	}
	q.owner.mu.Lock()
	defer q.owner.mu.Unlock()
	if q.closed || q.owner.closed || q.owner.closing {
		return errors.New("CUDA streaming request closed")
	}
	if err := q.finishLocked(); err != nil {
		return err
	}
	q.failed = nil
	return nil
}

func (q *cudaStreamingRequest) Close() error {
	if q == nil {
		return nil
	}
	if q.owner == nil {
		return errors.New("missing CUDA streaming owner")
	}
	q.owner.mu.Lock()
	defer q.owner.mu.Unlock()
	q.closed = true
	return q.finishLocked()
}

func (o *cudaStreamingCoordinator) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil
	}
	o.closing = true
	if o.active != nil {
		if err := o.active.finishLocked(); err != nil {
			return err
		}
	}
	if o.closeCache != nil {
		if err := o.closeCache(); err != nil {
			return err
		}
		o.closeCache = nil
	}
	o.closed = true
	return nil
}
