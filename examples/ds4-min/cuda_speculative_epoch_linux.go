//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// lifecycleEpoch is never restored from a snapshot. Prepared drafts bind
// it, so even restoring an identical position invalidates an old proposal.
func (s *cudaRequestState) advanceSpeculativeEpoch() error {
	if s.lifecycleEpoch == ^uint64(0) {
		s.runtime.failed = errors.Join(dsv4.ErrDesynchronized, fmt.Errorf("cuda state: speculative lifecycle epoch exhausted"))
		return s.runtime.failed
	}
	s.lifecycleEpoch++
	return nil
}

func (s *cudaRequestState) completeSpeculativeTransition(err error) error {
	if err != nil {
		return err
	}
	return s.advanceSpeculativeEpoch()
}

func (s *cudaRequestState) speculativeEpoch() (uint64, error) {
	if s == nil || s.closed || s.runtime == nil {
		return 0, dsv4.ErrClosed
	}
	if s.runtime.failed != nil {
		return 0, s.runtime.failed
	}
	return s.lifecycleEpoch, nil
}
