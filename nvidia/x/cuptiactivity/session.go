package cuptiactivity

import "errors"

// Session owns an activity collection. A Session must not be copied after use.
// Close drains activity buffers but does not synchronize application GPU work.
type Session struct {
	owner      *activityOwner
	collection *collection
}

// Result is a snapshot of collection data and losses. Records own their data.
// A clean result requires a closed session, nonzero complete records, no pending
// buffers, no losses, and no errors; it is not a performance qualification.
type Result struct {
	Records           []Record
	Closed            bool
	PendingBuffers    int
	RefusedBuffers    uint64
	IncompleteRecords uint64
	DiscardedRecords  uint64
	DroppedRecords    uint64
	CallbackErrors    uint64
	FirstError        error
}

// Close stops owned activity kinds and flushes their buffers. Native cleanup
// failures retain ownership for a later retry. Callers must synchronize their
// workload before Close if they require complete device timestamps.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	if s.owner == nil {
		return errors.New("cuptiactivity: uninitialized session")
	}
	return s.owner.close()
}

// Result returns a copied snapshot. A snapshot taken before Close is not final.
func (s *Session) Result() Result {
	if s == nil || s.owner == nil || s.collection == nil {
		return Result{FirstError: errors.New("cuptiactivity: uninitialized session")}
	}
	s.owner.mu.Lock()
	closed := s.owner.closed
	s.owner.mu.Unlock()
	c := s.collection
	c.mu.Lock()
	r := Result{
		Records: append([]Record(nil), c.records...), Closed: closed,
		IncompleteRecords: c.incomplete, DiscardedRecords: c.discarded,
		DroppedRecords: c.nativeDropped, CallbackErrors: c.failures, FirstError: c.firstError,
	}
	c.mu.Unlock()
	r.PendingBuffers, r.RefusedBuffers = c.pool.status()
	return r
}
