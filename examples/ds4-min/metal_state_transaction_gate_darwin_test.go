//go:build darwin && arm64

package main

import (
	"bytes"
	"errors"
	"os"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// TestMetalStateSharedBufferTransactionGate qualifies native Load's actual
// shared-buffer copies, not model inference or a GPU command barrier. Its
// synthetic model identity cannot satisfy TestMetalIndependentStateGate.
func TestMetalStateSharedBufferTransactionGate(t *testing.T) {
	if os.Getenv("DS4_METAL_TRANSACTION_GATE") != "1" {
		t.Skip("opt-in shared-buffer gate required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	fixture := func(seed float32) *metalRuntime {
		r := &metalRuntime{ctx: 16, pos: 3, arena: metalmem.NewArena(dev)}
		t.Cleanup(r.arena.Free)
		metalStateFixtureIdentity(t, r)
		r.state.hc[0] = metalmem.UploadIn(r.arena, "hc-a", []float32{seed, seed + 1})
		r.state.hc[1] = metalmem.UploadIn(r.arena, "hc-b", []float32{seed + 2, seed + 3})
		r.state.logits = metalmem.UploadIn(r.arena, "logits", []float32{seed + 4, seed + 5})
		if err := r.arena.Err(); err != nil {
			t.Fatal(err)
		}
		r.ops = &metalPinnedOps{hc: &metalHC{}}
		r.ops.hc.key.Tokens = 1
		r.lastBatch.Encoded = 9
		return r
	}
	save := func(r *metalRuntime) []byte {
		var b bytes.Buffer
		if err := saveMetalState(r, &b); err != nil {
			t.Fatal(err)
		}
		return b.Bytes()
	}
	incoming := save(fixture(20))
	for _, failedRollback := range []bool{false, true} {
		t.Run(map[bool]string{false: "rollback", true: "poison"}[failedRollback], func(t *testing.T) {
			r := fixture(1)
			before := save(r)
			key, batch := r.ops.hc.key, r.lastBatch
			primary, rollback := errors.New("after two shared writes"), errors.New("after first rollback write")
			hooks := &metalStateLoadHooks{afterWrite: func(i int, restoring bool) error {
				if restoring && failedRollback && i == 0 {
					return rollback
				}
				if !restoring && i == 1 {
					if r.state.hc[0].HostSlice()[0] != 20 || r.state.hc[1].HostSlice()[0] != 22 {
						t.Fatal("failure injection preceded actual shared-buffer writes")
					}
					return primary
				}
				return nil
			}}
			err := loadMetalStateWithHooks(r, bytes.NewReader(incoming), hooks)
			if !errors.Is(err, primary) {
				t.Fatalf("primary error lost: %v", err)
			}
			if failedRollback {
				if !errors.Is(err, rollback) || !errors.Is(r.failed, dsv4.ErrDesynchronized) || r.Logits() != nil {
					t.Fatalf("failed compensation did not poison: %v", err)
				}
				return
			}
			if !bytes.Equal(before, save(r)) {
				t.Fatal("native shared-buffer rollback changed saved state")
			}
			if r.ops.hc.key != key || r.lastBatch != batch {
				t.Fatal("recoverable Load changed cache metadata")
			}
		})
	}
}
