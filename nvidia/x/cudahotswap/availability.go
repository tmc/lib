package cudahotswap

import "errors"

// ErrPoisoned identifies a service whose device context failed. An error that
// wraps ErrPoisoned also retains the original device error.
var ErrPoisoned = errors.New("service context is poisoned")

// Err reports whether the service currently accepts requests. It returns nil
// when available, an error wrapping ErrPoisoned when poisoned, or an error when
// closed. This is a snapshot, not a reservation: callers must also check the
// error returned by each request. A nil Service returns an error.
func (s *Service) Err() error {
	if s == nil {
		return errors.New("nil service")
	}
	if err := s.available(); err != nil {
		return err
	}
	if s.loader == nil {
		return errors.New("service is not initialized")
	}
	return nil
}
