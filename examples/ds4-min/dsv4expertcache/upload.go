package dsv4expertcache

import "errors"

type uploadCompletion struct{ events []Completion }

func (e *uploadCompletion) Wait() error {
	var err error
	for _, event := range e.events {
		err = errors.Join(err, event.Wait())
	}
	return err
}

// Stage reads and uploads all missing triplets with one bounded staging buffer.
// The backend owns stage (pinned where required) and transfer. It waits before
// every buffer reuse and commits only after all upload events complete. Graph
// capture is rejected initially; this is not a permanent feature exclusion.
// Preflight errors preserve the lease, so callers must Abort it. After upload
// registration, failures abort and return joined submission/cleanup errors.
func (s *Source) Stage(l *Lease, stage []byte, transfer Transfer, graphCapture bool) error {
	if graphCapture {
		return errors.New("expert streaming graph capture unqualified")
	}
	if s == nil || l == nil || nilCompletion(transfer) || len(stage) == 0 {
		return errors.New("invalid expert staging")
	}
	if err := l.check(); err != nil {
		return err
	}
	if l.cache.identity != s.identity {
		return errors.New("source identity mismatch")
	}
	for _, u := range l.Uploads() {
		if u.Key.Identity != s.identity {
			return errors.New("source identity mismatch")
		}
		tr, ok := s.layers[u.Key.Layer]
		if !ok {
			return errors.New("missing expert layer")
		}
		for _, t := range []Tensor{tr.Gate, tr.Up, tr.Down} {
			if t.Stride > uint64(len(stage)) {
				return errors.New("staging buffer too small")
			}
		}
	}
	for _, u := range l.Uploads() {
		event := &uploadCompletion{}
		if err := l.Started(u, event); err != nil {
			return errors.Join(err, l.Abort())
		}
		tr := s.layers[u.Key.Layer]
		for component, t := range []Tensor{tr.Gate, tr.Up, tr.Down} {
			buf := stage[:int(t.Stride)]
			if err := s.Read(u.Key, component, buf); err != nil {
				return errors.Join(err, l.Abort())
			}
			done, err := transfer.Submit(u, component, buf)
			if !nilCompletion(done) {
				event.events = append(event.events, done)
			} else {
				err = errors.Join(err, errors.New("submission lacks completion"), l.Abort())
				l.cache.closed = true
				return err
			}
			if err != nil {
				return errors.Join(err, l.Abort())
			}
			if err = done.Wait(); err != nil {
				return errors.Join(err, l.Abort())
			}
		}
	}
	if err := l.Commit(); err != nil {
		return errors.Join(err, l.Abort())
	}
	return nil
}
