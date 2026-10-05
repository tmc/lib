package cudahotswap

import (
	"context"
	"errors"
	"io"
)

// ErrSlowSubscriber means a subscriber stopped reading before its bounded
// queue filled. Resume with Subscribe and the last consumed sequence to receive
// any missing range as a Gap event.
var ErrSlowSubscriber = errSlowSubscriber

// Subscription receives detached reports until canceled, closed or dropped.
// Its zero value is an already closed subscription. It may be used concurrently.
type Subscription struct {
	sub *reportSubscription
}

// Subscribe replays retained reports after since and streams subsequent ones.
// Each service retains 64 reports, admits 32 subscribers and buffers 65 events
// per subscriber. Reports exceeding the 64 KiB retained-value budget are refused.
// A cursor older than the ring receives an explicit Gap before retained reports.
// This method takes no GPU lease.
func (s *Service) Subscribe(ctx context.Context, since uint64) (*Subscription, error) {
	if s == nil || s.loader == nil || s.reports == nil {
		return nil, errors.New("subscribe reports: nil service or report owner")
	}
	s.mutations.RLock()
	defer s.mutations.RUnlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	sub, err := s.reports.subscribe(ctx, since)
	if err != nil {
		return nil, err
	}
	return &Subscription{sub: sub}, nil
}

// Events returns the receive-only report stream. A nil or zero Subscription
// returns an already closed stream. Previously queued events remain readable
// after the stream closes.
func (s *Subscription) Events() <-chan Event {
	if s == nil || s.sub == nil {
		return closedReportStream
	}
	return s.sub.events
}

var closedReportStream = func() <-chan Event {
	c := make(chan Event)
	close(c)
	return c
}()

// Err returns the cause of termination, or nil before termination or explicit
// Close. Context cancellation returns its context error, a dropped subscriber
// returns ErrSlowSubscriber, and Service.Close returns io.ErrClosedPipe.
func (s *Subscription) Err() error {
	if s == nil || s.sub == nil {
		return nil
	}
	return s.sub.err()
}

// Close stops delivery and releases the subscriber slot. Queued events remain
// readable. Close is idempotent and does not take a GPU lease.
func (s *Subscription) Close() error {
	if s != nil && s.sub != nil {
		s.sub.cancel(nil)
	}
	return nil
}

// publishReport stamps the exact request snapshot, not the state at some later
// delivery time. It can publish a terminal fault after observeError poisons the
// service, but cannot publish after shutdown starts. Push supplies its snapshot;
// this helper does not obtain or qualify a candidate verdict.
func (s *Service) publishReport(state State, report *Report) (uint64, error) {
	if s == nil || report == nil {
		return 0, errors.New("publish report: service and report are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, io.ErrClosedPipe
	}
	if s.reports == nil {
		return 0, errors.New("publish report: missing report owner")
	}
	r := *report
	r.Generation = state.Generation
	r.ConfigRevision = state.ConfigRevision
	return s.reports.append(state, &r)
}

// Close holds the mutation gate and marks the service closed before this call,
// preventing new public subscriptions or report publications during shutdown.
func (s *Service) stopReports() {
	if s.reports == nil {
		return
	}
	r := s.reports
	r.mu.Lock()
	defer r.mu.Unlock()
	for sub := range r.subs {
		r.dropLocked(sub, io.ErrClosedPipe)
	}
}
