//go:build darwin && arm64

package main

import (
	"errors"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"io"
	"testing"
)

type metalDSparkCleanupTestOwner struct {
	calls int
	fail  bool
	order *[]string
	name  string
}

func TestMetalDSparkConstructorFailure(t *testing.T) {
	var order []string
	partial := &metalDSparkCleanupTestOwner{fail: true, order: &order, name: "partial-forward"}
	backend := &metalStateAdapter{}
	w := &metalDSparkWeights{model: &dsparkModel{block: 4}, leases: 1}
	target := &metalRequestState{owner: backend, runtime: &metalRuntime{}}
	cause := errors.New("injected forward allocation failure")
	factory := metalDSparkConstructorFactory{
		features: func(t *metalRequestState, w *metalDSparkWeights) (*metalDSparkFeatures, error) {
			f := &metalDSparkFeatures{request: t, weights: w}
			t.runtime.dsparkFeatures = f
			return f, nil
		},
		forward: func(*metalRequestState, *metalDSparkWeights) (*metalDSparkNative, error) {
			return nil, finishBackendCleanup(cause, partial)
		},
		verifier: func(*metalRequestState) (dsv4.SpeculativeVerifier, error) {
			t.Fatal("verifier ran after failed forward allocation")
			return nil, nil
		},
	}
	d, s, err := newMetalDSparkSpeculatorWithFactory(backend, target, w, .5, factory)
	var owner io.Closer
	if d != nil || s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
		t.Fatalf("lost partial constructor: %v", err)
	}
	if partial.calls != 1 || w.leases != 1 || target.runtime.dsparkFeatures == nil {
		t.Fatal("failed forward freed feature owner")
	}
	if err := owner.Close(); err != nil || partial.calls != 2 || w.leases != 0 || target.runtime.dsparkFeatures != nil {
		t.Fatalf("retained constructor close: %v", err)
	}
	if err := owner.Close(); err != nil || partial.calls != 2 || w.leases != 0 {
		t.Fatal("constructor released owners twice")
	}
}

func TestMetalDSparkEmptyOwners(t *testing.T) {
	for _, kind := range []string{"features", "dense", "attention", "native"} {
		t.Run(kind, func(t *testing.T) {
			w := &metalDSparkWeights{leases: 1}
			var owner io.Closer
			switch kind {
			case "features":
				owner = &metalDSparkFeatures{weights: w}
			case "dense":
				owner = &metalDSparkDense{weights: w}
			case "attention":
				owner = &metalDSparkAttention{weights: w}
			case "native":
				owner = &metalDSparkNative{}
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if kind != "native" && w.leases != 0 {
				t.Fatal("empty owner retained lease")
			}
		})
	}
}

func TestMetalDSparkConstructorAcquisitions(t *testing.T) {
	for _, fail := range []string{"features", "forward", "verifier"} {
		t.Run(fail, func(t *testing.T) {
			backend := &metalStateAdapter{}
			w := &metalDSparkWeights{model: &dsparkModel{block: 4}}
			target := &metalRequestState{owner: backend, runtime: &metalRuntime{}}
			cause := errors.New("injected " + fail)
			factory := metalDSparkConstructorFactory{
				features: func(t *metalRequestState, w *metalDSparkWeights) (*metalDSparkFeatures, error) {
					if fail == "features" {
						return nil, cause
					}
					w.leases++
					f := &metalDSparkFeatures{request: t, weights: w}
					t.runtime.dsparkFeatures = f
					return f, nil
				},
				forward: func(t *metalRequestState, w *metalDSparkWeights) (*metalDSparkNative, error) {
					if fail == "forward" {
						return nil, cause
					}
					s := &metalDSparkNative{target: t, weights: w}
					t.runtime.dsparkFeatures.support = s
					return s, nil
				},
				verifier: func(*metalRequestState) (dsv4.SpeculativeVerifier, error) { return nil, cause },
			}
			d, s, err := newMetalDSparkSpeculatorWithFactory(backend, target, w, .5, factory)
			if d != nil || s != nil || !errors.Is(err, cause) {
				t.Fatalf("stage %s %v", fail, err)
			}
			var owner io.Closer
			if errors.As(err, &owner) {
				if err := owner.Close(); err != nil {
					t.Fatal(err)
				}
				if err := owner.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if w.leases != 0 || target.runtime.dsparkFeatures != nil {
				t.Fatal("constructor retained completed feature lease")
			}
		})
	}
}

func (o *metalDSparkCleanupTestOwner) Close() error {
	o.calls++
	*o.order = append(*o.order, o.name)
	if o.fail {
		o.fail = false
		return errors.New("injected close failure")
	}
	return nil
}

func TestMetalDSparkConstructorCleanup(t *testing.T) {
	var order []string
	consumer := &metalDSparkCleanupTestOwner{fail: true, order: &order, name: "consumer"}
	features := &metalDSparkCleanupTestOwner{order: &order, name: "features"}
	cause := errors.New("injected constructor failure")
	err := finishMetalDSparkConstructor(cause, consumer, features)
	var owner io.Closer
	if !errors.Is(err, cause) || !errors.As(err, &owner) {
		t.Fatalf("constructor lost cause or owner: %v", err)
	}
	if err := owner.Close(); err == nil || features.calls != 0 {
		t.Fatal("failed consumer released features")
	}
	if err := owner.Close(); err != nil || consumer.calls != 2 || features.calls != 1 {
		t.Fatalf("retry lost owner: %v calls=%d/%d", err, consumer.calls, features.calls)
	}
	if err := owner.Close(); err != nil || consumer.calls != 2 || features.calls != 1 {
		t.Fatal("completed cleanup repeated release")
	}
	if len(order) != 3 || order[0] != "consumer" || order[1] != "consumer" || order[2] != "features" {
		t.Fatalf("release order %v", order)
	}
}

// Model-free lifecycle controls exercise the production Drafter methods but
// do not execute support kernels, construct a Metal device, or qualify drafts.
func metalDSparkDraftFixture(t *testing.T) (*metalDSparkDrafter, *metalDSparkFeatures) {
	t.Helper()
	w := &metalDSparkWeights{model: &dsparkModel{block: 4}}
	r := &metalRuntime{pos: 3, speculativeEpoch: 9}
	target := &metalRequestState{runtime: r}
	j, err := newDSparkFeatureJournal(16, 1)
	if err != nil {
		t.Fatal(err)
	}
	for pos, token := range []int{7, 8, 9} {
		if err := j.append(pos, token, 1); err != nil {
			t.Fatal(err)
		}
	}
	f := &metalDSparkFeatures{request: target, weights: w, journal: j, epoch: 1}
	r.dsparkFeatures = f
	forward := &metalDSparkNative{target: target, weights: w}
	return &metalDSparkDrafter{forward: forward, verifier: &metalSpeculativeVerifier{state: target}, threshold: .5}, f
}

func TestMetalDSparkDraftLifecycle(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*metalDSparkDrafter, *metalDSparkFeatures)
		fail   bool
	}{
		{"exact", func(d *metalDSparkDrafter, f *metalDSparkFeatures) {}, false},
		{"position", func(d *metalDSparkDrafter, f *metalDSparkFeatures) { d.forward.target.runtime.pos++ }, true},
		{"epoch", func(d *metalDSparkDrafter, f *metalDSparkFeatures) { d.forward.target.runtime.speculativeEpoch++ }, true},
		{"prefix", func(d *metalDSparkDrafter, f *metalDSparkFeatures) { f.journal.prefix[0] ^= 1 }, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			d, f := metalDSparkDraftFixture(t)
			if err := d.prepare([]int{7, 8, 9}); err != nil {
				t.Fatal(err)
			}
			test.mutate(d, f)
			seed, pos, err := d.PreparedSeed()
			if (err != nil) != test.fail || err == nil && (seed != 9 || pos != 3) {
				t.Fatalf("seed=%d pos=%d err=%v", seed, pos, err)
			}
		})
	}
	for _, grouped := range []bool{false, true} {
		for _, bad := range []string{"", "epoch", "prefix", "position"} {
			t.Run(bad, func(t *testing.T) {
				d, f := metalDSparkDraftFixture(t)
				if err := d.prepare([]int{7, 8, 9}); err != nil {
					t.Fatal(err)
				}
				d.pending, d.drafted = []int{10, 11}, true
				d.verifier.lastGrouped = grouped
				r := d.forward.target.runtime
				r.pos = 4
				steps := 4
				if grouped {
					steps = 3
				}
				r.speculativeEpoch += uint64(steps)
				if err := f.journal.append(3, 10, 1); err != nil {
					t.Fatal(err)
				}
				switch bad {
				case "epoch":
					r.speculativeEpoch++
				case "prefix":
					f.journal.prefix[0] ^= 1
				case "position":
					r.pos++
				}
				err := d.CommitDrafts(1)
				if (err != nil) != (bad != "") {
					t.Fatalf("grouped=%v commit err=%v", grouped, err)
				}
				if bad == "" && (d.prepared || d.drafted || len(d.pending) != 0) {
					t.Fatal("committed lookahead retained stale preparation")
				}
			})
		}
	}
	// A decline preserves an unchanged exact prepared frontier.
	d, _ := metalDSparkDraftFixture(t)
	if err := d.prepare([]int{7, 8, 9}); err != nil {
		t.Fatal(err)
	}
	d.drafted = true
	if err := d.DiscardDrafts(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.PreparedSeed(); err != nil {
		t.Fatalf("free decline lost seed: %v", err)
	}
}

func TestMetalDSparkDraftCachedProposal(t *testing.T) {
	d, _ := metalDSparkDraftFixture(t)
	if err := d.prepare([]int{7, 8, 9}); err != nil {
		t.Fatal(err)
	}
	d.pending, d.ready = []int{10, 11, 12, 13}, true
	proposal, err := d.Draft(9, 3, 2)
	if err != nil || len(proposal) != 2 || proposal[0] != 10 || proposal[1] != 11 || !d.drafted || d.ready {
		t.Fatalf("cached proposal %v err=%v", proposal, err)
	}
	proposal[0] = 99
	if d.pending[0] != 10 {
		t.Fatal("caller aliases retained proposal")
	}
	if _, err := d.Draft(9, 3, 2); err == nil {
		t.Fatal("same support proposal consumed twice")
	}
}
