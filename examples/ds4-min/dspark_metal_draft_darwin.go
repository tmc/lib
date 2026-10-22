//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// Native support DraftLifecycle. Target features, support-private KV and
// target verifier state are distinct owners. No predicted row becomes trusted.
type metalDSparkDrafter struct {
	forward                     *metalDSparkNative
	verifier                    *metalSpeculativeVerifier
	speculator                  *dsv4.Speculator
	history, pending            []int
	hash                        string
	epoch                       uint64
	seed                        int
	prepared, drafted, incoming bool
	ready                       bool
	threshold                   float32
}

var _ dsv4.Drafter = (*metalDSparkDrafter)(nil)
var _ dsv4.DraftLifecycle = (*metalDSparkDrafter)(nil)

func newMetalDSparkSpeculator(backend *metalStateAdapter, target *metalRequestState, w *metalDSparkWeights, threshold float32) (_ *metalDSparkDrafter, _ *dsv4.Speculator, err error) {
	return newMetalDSparkSpeculatorWithFactory(backend, target, w, threshold, metalDSparkConstructorFactory{
		features: func(t *metalRequestState, w *metalDSparkWeights) (*metalDSparkFeatures, error) {
			return attachMetalDSparkFeatures(t, w, 1)
		},
		forward:  newMetalDSparkNative,
		verifier: func(t *metalRequestState) (dsv4.SpeculativeVerifier, error) { return backend.NewSpeculativeVerifier(t) },
	})
}

type metalDSparkConstructorFactory struct {
	features func(*metalRequestState, *metalDSparkWeights) (*metalDSparkFeatures, error)
	forward  func(*metalRequestState, *metalDSparkWeights) (*metalDSparkNative, error)
	verifier func(*metalRequestState) (dsv4.SpeculativeVerifier, error)
}

func newMetalDSparkSpeculatorWithFactory(backend *metalStateAdapter, target *metalRequestState, w *metalDSparkWeights, threshold float32, factory metalDSparkConstructorFactory) (_ *metalDSparkDrafter, _ *dsv4.Speculator, err error) {
	if backend == nil || backend.closed || target == nil || target.closed || target.owner != backend || target.runtime == nil || w == nil || w.closed || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) {
		return nil, nil, fmt.Errorf("metal dspark scheduler: invalid owner or threshold")
	}
	created := false
	if target.runtime.dsparkFeatures == nil {
		if _, err := factory.features(target, w); err != nil {
			return nil, nil, err
		}
		created = true
	}
	forward, err := factory.forward(target, w)
	if err != nil {
		if created {
			err = finishBackendCleanup(err, target.runtime.dsparkFeatures)
		}
		return nil, nil, err
	}
	d := &metalDSparkDrafter{forward: forward, threshold: threshold}
	defer func() {
		if err != nil {
			if created && target.runtime.dsparkFeatures != nil {
				err = finishMetalDSparkConstructor(err, d, target.runtime.dsparkFeatures)
			} else {
				err = finishMetalDSparkConstructor(err, d, nil)
			}
		}
	}()
	v, err := factory.verifier(target)
	if err != nil {
		return nil, nil, err
	}
	var ok bool
	d.verifier, ok = v.(*metalSpeculativeVerifier)
	if !ok {
		return nil, nil, fmt.Errorf("metal dspark scheduler: foreign verifier")
	}
	speculator, err := dsv4.NewSpeculator(d.verifier, d)
	if err != nil {
		return nil, nil, err
	}
	d.speculator = speculator
	return d, speculator, nil
}

// A failed constructor transfers its remaining owners through the error.
// The consumer must finish before the captured features can be released.
func finishMetalDSparkConstructor(err error, consumer, features io.Closer) error {
	if features == nil {
		return finishBackendCleanup(err, consumer)
	}
	var request io.Closer = consumer
	if owners := backendErrorOwners(err); len(owners) > 0 {
		request = &backendCleanupOwner{cause: err, request: &backendCleanupGroup{owners}, backend: consumer}
	}
	return finishBackendCleanup(&backendCleanupOwner{cause: err, request: request}, features)
}

func metalDSparkHistoryPrefix(history []int) ([32]byte, error) {
	var prefix [32]byte
	for _, token := range history {
		if token < 0 || token >= Flash.Vocab {
			return [32]byte{}, fmt.Errorf("metal dspark prepare: invalid history token")
		}
		var row [36]byte
		copy(row[:32], prefix[:])
		binary.LittleEndian.PutUint32(row[32:], uint32(token))
		prefix = sha256.Sum256(row[:])
	}
	return prefix, nil
}

func (d *metalDSparkDrafter) prepare(history []int) error {
	if d == nil || d.forward == nil || d.forward.closed || d.forward.failed || d.verifier == nil || d.drafted || len(history) < 2 || d.verifier.state != d.forward.target {
		return fmt.Errorf("metal dspark prepare: unavailable minimum-two frontier")
	}
	r := d.forward.target.runtime
	f := r.dsparkFeatures
	if f == nil || f.closed || f.batch != nil || f.weights != d.forward.weights || r.pos != len(history) {
		return fmt.Errorf("metal dspark prepare: unpublished or foreign target history")
	}
	if err := f.journal.validate(len(history)); err != nil {
		return err
	}
	prefix, err := metalDSparkHistoryPrefix(history)
	if err != nil || prefix != f.journal.prefix {
		return fmt.Errorf("metal dspark prepare: committed prefix mismatch")
	}
	hash, err := dsv4.SpeculativeHistorySHA256(history)
	if err != nil {
		return err
	}
	d.history, d.hash, d.epoch = append([]int(nil), history...), hash, r.speculativeEpoch
	d.seed, d.prepared, d.incoming = history[len(history)-1], true, false
	return nil
}

func (d *metalDSparkDrafter) prepareIncoming(history []int, seed int) error {
	if err := d.prepare(history); err != nil {
		return err
	}
	best, err := metalSpeculativeFrontier(d.verifier.CurrentLogits())
	if err != nil || seed != best {
		d.prepared = false
		return fmt.Errorf("metal dspark prepare: incoming seed is not target argmax")
	}
	d.seed, d.incoming = seed, true
	return nil
}

func (d *metalDSparkDrafter) PreparedSeed() (int, int, error) {
	if d == nil || !d.prepared || d.forward == nil || d.forward.closed || d.forward.failed || d.forward.target.closed || len(d.history) < 2 || len(d.hash) != 64 {
		return 0, 0, fmt.Errorf("metal dspark: no prepared frontier")
	}
	r := d.forward.target.runtime
	if r.pos != len(d.history) || r.speculativeEpoch != d.epoch {
		return 0, 0, fmt.Errorf("metal dspark: stale prepared frontier")
	}
	prefix, err := metalDSparkHistoryPrefix(d.history)
	if err != nil || r.dsparkFeatures == nil || r.dsparkFeatures.journal.prefix != prefix {
		return 0, 0, fmt.Errorf("metal dspark: prepared history no longer matches")
	}
	return d.seed, len(d.history), nil
}

func (d *metalDSparkDrafter) Draft(token, position, limit int) ([]int, error) {
	seed, next, err := d.PreparedSeed()
	if err != nil || seed != token || next != position || limit < 1 || d.drafted {
		return nil, fmt.Errorf("metal dspark draft: invalid prepared seed or limit")
	}
	if d.ready {
		d.pending = append([]int(nil), d.pending[:min(limit, len(d.pending))]...)
		d.ready, d.drafted = false, true
		return append([]int(nil), d.pending...), nil
	}
	proposal, err := d.forward.proposal(seed, min(limit, d.forward.weights.model.block), d.threshold, d.incoming)
	if err != nil {
		return nil, err
	}
	d.pending, d.drafted = append([]int(nil), proposal...), true
	return append([]int(nil), proposal...), nil
}

func (d *metalDSparkDrafter) CommitDrafts(n int) error {
	if d == nil || !d.prepared || !d.drafted || n < 1 || n > len(d.pending) || d.forward == nil || d.verifier == nil || d.verifier.state != d.forward.target {
		return fmt.Errorf("metal dspark commit: invalid accepted prefix")
	}
	r := d.forward.target.runtime
	steps := len(d.pending) + 1 + n
	if d.verifier.lastGrouped {
		steps = 2 + n
	}
	if uint64(steps) > ^uint64(0)-d.epoch || r.speculativeEpoch != d.epoch+uint64(steps) || r.pos != len(d.history)+n {
		return fmt.Errorf("metal dspark commit: foreign verifier lifecycle")
	}
	history := append(append([]int(nil), d.history...), d.pending[:n]...)
	prefix, err := metalDSparkHistoryPrefix(history)
	if err != nil || r.dsparkFeatures == nil || r.dsparkFeatures.journal.prefix != prefix {
		return fmt.Errorf("metal dspark commit: accepted target prefix mismatch")
	}
	return d.DiscardDrafts()
}

func (d *metalDSparkDrafter) DiscardDrafts() error {
	if d == nil {
		return nil
	}
	d.pending, d.drafted, d.ready = nil, false, false
	if d.prepared {
		if _, _, err := d.PreparedSeed(); err != nil {
			d.history, d.hash, d.prepared = nil, "", false
		}
	}
	return nil
}

func (d *metalDSparkDrafter) Close() error {
	if d == nil {
		return nil
	}
	d.prepared, d.drafted, d.ready = false, false, false
	d.history, d.pending, d.hash = nil, nil, ""
	if d.speculator != nil {
		if err := d.speculator.Close(); err != nil {
			return err
		}
		d.speculator = nil
	}
	if d.verifier != nil {
		if err := d.verifier.Close(); err != nil {
			return err
		}
	}
	d.verifier = nil
	if d.forward != nil {
		if err := d.forward.Close(); err != nil {
			return err
		}
		d.forward = nil
	}
	return nil
}
