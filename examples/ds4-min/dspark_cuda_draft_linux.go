//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// cudaDSparkDrafter binds the native producer to the existing verifier and
// one exact request/history. It retains no predicted KV as trusted state.
// This is not an enabled CLI route or a physical qualification receipt.
type cudaDSparkDrafter struct {
	forward   *cudaDSparkForward
	verifier  *cudaSpeculativeVerifier
	history   []int
	pending   []int
	hash      string
	epoch     uint64
	prepared  bool
	drafted   bool
	threshold float32
}

var _ dsv4.DraftLifecycle = (*cudaDSparkDrafter)(nil)
var _ dsv4.Drafter = (*cudaDSparkDrafter)(nil)

func newCUDADSparkSpeculator(backend *cudaStateAdapter, target *cudaRequestState, weights *cudaDSparkWeights, threshold float32) (_ *cudaDSparkDrafter, _ *dsv4.Speculator, err error) {
	if backend == nil || backend.closed || target == nil || target.closed || target.owner != backend || target.runtime == nil || weights == nil || weights.closed || weights.model == nil || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) {
		return nil, nil, fmt.Errorf("cuda dspark scheduler: invalid owner or policy")
	}
	createdFeatures := false
	if target.dspark == nil {
		if _, err := newCUDADSparkFeatures(target, weights.model); err != nil {
			return nil, nil, err
		}
		createdFeatures = true
	} else if !target.dspark.bound() || target.dspark.modelSHA != weights.model.sha256 {
		return nil, nil, fmt.Errorf("cuda dspark scheduler: foreign support capture")
	}
	forward, err := newCUDADSparkForward(target, weights)
	if err != nil {
		if createdFeatures {
			err = finishBackendCleanup(err, target.dspark)
		}
		return nil, nil, err
	}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, forward)
			if createdFeatures {
				err = finishBackendCleanup(err, target.dspark)
			}
		}
	}()
	v, err := backend.NewSpeculativeVerifier(target)
	if err != nil {
		return nil, nil, err
	}
	verifier, ok := v.(*cudaSpeculativeVerifier)
	if !ok {
		return nil, nil, fmt.Errorf("cuda dspark scheduler: unsupported verifier ownership")
	}
	d := &cudaDSparkDrafter{forward: forward, verifier: verifier, threshold: threshold}
	speculator, err := dsv4.NewSpeculator(verifier, d)
	if err != nil {
		return nil, nil, err
	}
	return d, speculator, nil
}

func dsparkFeaturePrefix(history []int) ([32]byte, error) {
	var prefix [32]byte
	for _, token := range history {
		if token < 0 || token >= Flash.Vocab {
			return [32]byte{}, fmt.Errorf("dspark prepare: history token outside vocabulary")
		}
		var row [36]byte
		copy(row[:32], prefix[:])
		binary.LittleEndian.PutUint32(row[32:], uint32(token))
		prefix = sha256.Sum256(row[:])
	}
	return prefix, nil
}

func (d *cudaDSparkDrafter) prepare(history []int) error {
	if d == nil || d.forward == nil || d.forward.closed || d.forward.failed || d.verifier == nil || d.drafted || len(history) == 0 || d.verifier.state != d.forward.target {
		return fmt.Errorf("cuda dspark prepare: unavailable request frontier")
	}
	s := d.forward.target
	f := s.dspark
	if f == nil || !f.bound() || !f.published || f.history == nil || f.history.journal == nil || s.runtime.pos != len(history) || f.position != len(history)-1 || f.token != history[len(history)-1] || f.generation == ^uint64(0) || s.generation != f.generation+1 || f.modelSHA != d.forward.weights.model.sha256 {
		return fmt.Errorf("cuda dspark prepare: unpublished or foreign history")
	}
	if err := f.history.journal.validate(len(history)); err != nil {
		return err
	}
	prefix, err := dsparkFeaturePrefix(history)
	if err != nil || prefix != f.history.journal.prefix {
		return fmt.Errorf("cuda dspark prepare: committed history mismatch")
	}
	hash, err := dsv4.SpeculativeHistorySHA256(history)
	if err != nil {
		return err
	}
	epoch, err := s.speculativeEpoch()
	if err != nil {
		return err
	}
	d.history, d.hash, d.epoch = append([]int(nil), history...), hash, epoch
	d.prepared = true
	return nil
}

func (d *cudaDSparkDrafter) PreparedSeed() (int, int, error) {
	if d == nil || !d.prepared || d.forward == nil || d.forward.closed || d.forward.failed || len(d.history) == 0 || len(d.hash) != 64 {
		return 0, 0, fmt.Errorf("cuda dspark: no prepared frontier")
	}
	s := d.forward.target
	epoch, err := s.speculativeEpoch()
	if err != nil || epoch != d.epoch || s.runtime.pos != len(d.history) {
		return 0, 0, fmt.Errorf("cuda dspark: stale prepared frontier")
	}
	return d.history[len(d.history)-1], len(d.history), nil
}

func (d *cudaDSparkDrafter) Draft(token, position, maxDrafts int) ([]int, error) {
	seed, next, err := d.PreparedSeed()
	if err != nil || seed != token || next != position || maxDrafts < 1 || d.drafted {
		return nil, fmt.Errorf("cuda dspark draft: invalid preparation or limit")
	}
	count := min(maxDrafts, d.forward.weights.model.block)
	tokens, _, err := d.forward.proposal(seed, count, d.threshold)
	if err != nil {
		return nil, err
	}
	d.pending, d.drafted = append([]int(nil), tokens...), true
	return append([]int(nil), tokens...), nil
}

func (d *cudaDSparkDrafter) CommitDrafts(n int) error {
	if d == nil || !d.prepared || !d.drafted || n < 1 || n > len(d.pending) || d.forward == nil || d.verifier == nil || d.verifier.state != d.forward.target {
		return fmt.Errorf("cuda dspark commit: invalid accepted prefix")
	}
	s := d.forward.target
	epoch, err := s.speculativeEpoch()
	if err != nil {
		return err
	}
	// The bound verifier evaluates lookahead, restores its frontier, then
	// replays exactly n accepted target rows. No unrelated State transition
	// may be hidden by restoring the same position/history.
	steps := len(d.pending) + 1 + n
	if d.verifier.batched && len(d.pending) > 1 {
		steps = 2 + n
	}
	if uint64(steps) > ^uint64(0)-d.epoch || epoch != d.epoch+uint64(steps) || s.runtime.pos != len(d.history)+n {
		return fmt.Errorf("cuda dspark commit: foreign target lifecycle")
	}
	history := append(append([]int(nil), d.history...), d.pending[:n]...)
	prefix, err := dsparkFeaturePrefix(history)
	if err != nil || s.dspark == nil || s.dspark.history == nil || s.dspark.history.journal == nil || s.dspark.history.journal.prefix != prefix || s.dspark.token != history[len(history)-1] {
		return fmt.Errorf("cuda dspark commit: accepted target history mismatch")
	}
	return d.DiscardDrafts()
}

func (d *cudaDSparkDrafter) DiscardDrafts() error {
	if d == nil {
		return nil
	}
	d.pending, d.drafted = nil, false
	// A free decline has not changed the trusted target frontier. Preserve its
	// preparation so the caller can retry or fall back without reconstructing
	// support state. A target transition makes that binding stale.
	if d.prepared {
		if _, _, err := d.PreparedSeed(); err != nil {
			d.history, d.hash, d.prepared = nil, "", false
		}
	}
	return nil
}

// Close stops proposals before releasing request-private support scratch.
// Selected features remain owned by the target State. Failed cleanup retains
// its owners for a later Close; it never reopens proposal admission.
func (d *cudaDSparkDrafter) Close() error {
	if d == nil {
		return nil
	}
	d.history, d.pending, d.hash, d.prepared, d.drafted = nil, nil, "", false, false
	var err error
	if d.verifier != nil {
		if closeErr := d.verifier.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		} else {
			d.verifier = nil
		}
	}
	if d.forward != nil {
		if closeErr := d.forward.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		} else {
			d.forward = nil
		}
	}
	return err
}
