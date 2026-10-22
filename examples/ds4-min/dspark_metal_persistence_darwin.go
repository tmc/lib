//go:build darwin && arm64

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func metalSupportEnvelope(v *metalDSparkFeatureSnapshot) (*dsparkSupportEnvelope, error) {
	if v == nil {
		return nil, errors.New("metal support save: missing snapshot")
	}
	identity, err := hex.DecodeString(v.modelSHA)
	if err != nil || len(identity) != 32 {
		return nil, errors.New("metal support save: invalid model identity")
	}
	e := &dsparkSupportEnvelope{context: v.context, block: v.block, targets: v.targets, journal: v.journal, history: v.data}
	copy(e.model[:], identity)
	return e, nil
}

func (s *metalRequestState) saveSupport(w io.Writer) (err error) {
	snapshot, err := s.Snapshot()
	if err != nil {
		return err
	}
	defer func() { err = finishBackendCleanup(err, snapshot) }()
	v := snapshot.(*metalSnapshot)
	return writeMetalStateSnapshot(w, v)
}

func writeMetalStateSnapshot(w io.Writer, v *metalSnapshot) error {
	if v == nil || v.closed {
		return dsv4.ErrClosed
	}
	if len(v.data) < metalStateHeaderSize || int64(len(v.data)) > metalStateMaxBytes {
		return errors.New("metal snapshot: invalid core payload")
	}
	if v.dspark == nil {
		return writeMetalStateFull(w, v.data)
	}
	e, err := metalSupportEnvelope(v.dspark)
	if err != nil {
		return err
	}
	return writeDSparkSupportContainer(w, v.data, e, v.dspark.journal.frontier, 0, uint64(metalStateMaxBytes))
}

func (s *metalRequestState) loadSupport(r io.Reader) error {
	if s.runtime.speculativeEpoch == ^uint64(0) {
		return errors.New("metal support load: speculative epoch overflow")
	}
	core, support, _, err := readDSparkSupportContainer(r, false, uint64(metalStateMaxBytes))
	if err != nil {
		return err
	}
	if len(core) < metalStateHeaderSize || string(core[:8]) != metalStateMagic {
		return errors.New("metal support load: invalid core header")
	}
	position := binary.LittleEndian.Uint64(core[16:24])
	if position > uint64(s.runtime.ctx) {
		return errors.New("metal support load: invalid frontier")
	}
	f := s.runtime.dsparkFeatures
	shape := &metalDSparkFeatureSnapshot{modelSHA: f.weights.model.sha256, context: s.runtime.ctx, block: f.weights.model.block, targets: f.targets}
	expected, err := metalSupportEnvelope(shape)
	if err != nil {
		return err
	}
	e, err := readDSparkSupport(support, expected, int(position), 0)
	if err != nil {
		return err
	}
	v := &metalDSparkFeatureSnapshot{owner: f, journal: e.journal, data: e.history, modelSHA: shape.modelSHA, context: e.context, block: e.block, targets: e.targets}
	// Both envelope identity and the actual destination owner are checked before
	// the transactional core loader can replace any destination allocation.
	if err := f.validateSnapshot(v, int(position)); err != nil {
		return fmt.Errorf("metal support load: %w", err)
	}
	if err := loadMetalState(s.runtime, bytes.NewReader(core)); err != nil {
		return err
	}
	// Publication cannot fail after validateSnapshot. No transfer or allocation
	// is left between the core commit and the support commit.
	f.publishSnapshot(v)
	s.runtime.speculativeEpoch++
	return nil
}
