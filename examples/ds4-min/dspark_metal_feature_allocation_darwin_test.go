//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalDSparkFeatureAllocationFailure(t *testing.T) {
	for _, cleanupFault := range []bool{false, true} {
		dev := &libmetal.Device{}
		r := &metalRequestState{runtime: &metalRuntime{dev: dev, ctx: 16}}
		w := &metalDSparkWeights{device: dev, model: &dsparkModel{block: 1, targets: []int{0}}}
		var owner *metalDSparkFeatures
		var allocationErr error
		compiled := false
		f, err := newMetalDSparkFeaturesWithFactory(r, w, 1, metalDSparkFeatureFactory{
			allocate: func(f *metalDSparkFeatures, count int) {
				owner = f
				f.arena = metalmem.NewArena(nil)
				f.data = metalmem.AllocIn[float32](f.arena, "injected-data", count)
				f.status = metalmem.AllocIn[uint32](f.arena, "injected-status", 1)
				allocationErr = f.arena.Err()
				if cleanupFault {
					f.batch = &metalBatch{}
				}
			},
			compile: func(*metalDSparkFeatures) error {
				compiled = true
				return errors.New("compile must not follow failed allocation")
			},
		})
		if allocationErr == nil || f != nil || compiled || !errors.Is(err, allocationErr) || r.runtime.dsparkFeatures != nil {
			t.Errorf("allocation failure escaped admission: f=%v compiled=%v err=%v", f, compiled, err)
		}
		if cleanupFault {
			owners := backendErrorOwners(err)
			if len(owners) == 0 || owner.closed || w.leases != 1 {
				t.Fatalf("cleanup failure lost retry owner: %v", err)
			}
			owner.batch = nil
			for _, retained := range owners {
				if err := retained.Close(); err != nil {
					t.Fatalf("cleanup retry: %v", err)
				}
			}
		}
		if !owner.closed || w.leases != 0 {
			t.Errorf("feature owner/lease leaked: closed=%v leases=%d", owner.closed, w.leases)
		}
	}
}
