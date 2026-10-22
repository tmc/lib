//go:build linux

package main

import "github.com/tmc/lib/examples/ds4-min/dsv4"

// completeObservedCacheReplacement is only for Load and Restore. Eval advances
// the same epoch but must preserve its successfully evaluated token history.
func (s *cudaRequestState) completeObservedCacheReplacement(err error) error {
	if err != nil {
		return err
	}
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	// The cache already changed. Invalidate even if epoch advancement fails.
	s.runtime.headObserver.invalidate()
	return s.advanceSpeculativeEpoch()
}
