package cuptipcsampling

import (
	"errors"

	"github.com/tmc/lib/nvidia/cupti"
)

// samplingSession joins control, configured output, and a distinct drain
// buffer. Its owner serializes access and records each returned batch with
// range/process/module identity. It does not synchronize or replay a workload.
type samplingSession struct {
	control         *samplingControl
	drain           *samplingBuffer
	names           *samplingNames
	query           func(*cupti.PCSamplingGetDataParams) error
	maxReads, reads uint
	closed          bool
	lossObserved    bool
	final           samplingBatch
	finalErr        error
}

func (s *samplingSession) begin() error {
	if s.control == nil || s.control.config == nil || s.drain == nil || s.names == nil || s.query == nil || s.maxReads == 0 || s.closed || s.drain == s.control.config.output {
		return errors.New("cuptipcsampling: invalid sampling session")
	}
	if err := s.reserveRead(); err != nil {
		return err
	}
	if err := s.acquire(); err != nil {
		return err
	}
	if err := s.control.prepare(); err != nil {
		s.control.failed = true
		return err
	}
	if err := s.control.start(); err != nil {
		return err
	}
	return nil
}

func (s *samplingSession) reserveRead() error {
	// Preserve worst-case name storage for disable's configured buffer, even
	// when all names returned by this read are new to the session.
	a, b := len(s.drain.pcs), len(s.control.config.output.pcs)
	if a > int(^uint(0)>>1)-b {
		return errors.New("cuptipcsampling: name reservation overflows")
	}
	return s.names.reserve(a + b)
}

func (s *samplingSession) read() (samplingBatch, error) {
	if s.closed || s.control == nil || !s.control.enabled || s.control.failed {
		return samplingBatch{}, errors.New("cuptipcsampling: sampling session cannot read")
	}
	if s.reads >= s.maxReads {
		s.control.failed = true
		return samplingBatch{}, errors.New("cuptipcsampling: sampling read budget exhausted")
	}
	if err := s.reserveRead(); err != nil {
		s.control.failed = true
		return samplingBatch{}, err
	}
	s.reads++
	batch, err := s.drain.drainOwned(s.control.ctx, s.control.version, s.query, s.names)
	if batch.dropped != 0 || batch.hardwareFull != 0 {
		s.lossObserved = true
	}
	if err != nil {
		s.control.failed = true
	}
	return batch, err
}

// finish requires the declared workload boundary and completion of every
// consumer sharing native names. Failed disable retains all storage for an
// explicit retry. After successful disable, final output is copied before
// freeing tracked names or unpinning buffers. Invalid native output remains
// diagnostic; unvalidated addresses are never passed to free.
func (s *samplingSession) finish(sharingComplete bool) (samplingBatch, error) {
	if s.closed {
		return s.final, s.finalErr
	}
	if !sharingComplete || s.control == nil || s.control.config == nil || s.drain == nil || s.names == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: sampling consumers have not completed")
	}
	stopErr := s.control.shutdown()
	if !s.control.canRelease() {
		return samplingBatch{}, stopErr
	}
	batch, copyErr := s.control.config.output.snapshotOwned(s.names)
	var lossErr error
	if s.lossObserved || batch.remaining != 0 || batch.dropped != 0 || batch.hardwareFull != 0 {
		lossErr = errors.New("cuptipcsampling: final sampling output has loss or remaining PCs")
	}
	s.final, s.finalErr = batch, errors.Join(stopErr, copyErr, lossErr)
	if s.control.failed && s.finalErr == nil {
		s.finalErr = errors.New("cuptipcsampling: sampling session previously failed")
	}
	if s.finalErr != nil {
		s.control.failed = true
	}
	s.names.release()
	s.drain.release()
	s.control.config.output.release()
	s.control.config.release()
	s.closed = true
	s.releaseOwnership()
	return s.final, s.finalErr
}
