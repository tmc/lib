package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

// This model covers only full/empty generation dependencies. It assumes an
// issued WGMMA can retire; it does not model device memory visibility, fences,
// instruction convergence or the correctness of the MMA itself.
type nvfp4TMACycleEvent struct {
	kind byte // f fetches a completed full stage; r releases a previous stage
	step int
}

type nvfp4TMACycleSlot struct {
	step              int
	fetched, released uint8 // one bit for each consumer warpgroup
}

func nvfp4TMACycleFetchFirst(source string) (bool, error) {
	fetch := "fetch(sn, pn);"
	release := "if (k > 0 && lt == 0) mbar_arrive(empty0 + 8 * sprev);"
	if strings.Count(source, fetch) != 1 || strings.Count(source, release) != 1 {
		return false, fmt.Errorf("source dependency order is missing or ambiguous")
	}
	return strings.Index(source, fetch) < strings.Index(source, release), nil
}

func nvfp4TMACycleProgram(kblocks, tiles int, fetchFirst bool) []nvfp4TMACycleEvent {
	var events []nvfp4TMACycleEvent
	for tile := 0; tile < tiles; tile++ {
		base := tile * kblocks
		events = append(events, nvfp4TMACycleEvent{'f', base})
		for k := 0; k < kblocks; k++ {
			// Issue of k is followed by fetch(k+1), wait_group<1>, then
			// release(k-1) in the retained source. At k=0 there is no
			// previous stage to release. The final drain releases the last
			// two stages only after wait_group<0>.
			if k+1 == kblocks {
				continue
			}
			if k > 0 && !fetchFirst {
				events = append(events, nvfp4TMACycleEvent{'r', base + k - 1})
			}
			events = append(events, nvfp4TMACycleEvent{'f', base + k + 1})
			if k > 0 && fetchFirst {
				events = append(events, nvfp4TMACycleEvent{'r', base + k - 1})
			}
		}
		if kblocks > 1 {
			events = append(events, nvfp4TMACycleEvent{'r', base + kblocks - 2})
		}
		events = append(events, nvfp4TMACycleEvent{'r', base + kblocks - 1})
	}
	return events
}

// nvfp4TMACycleCompletes advances one producer and two consumer warpgroups.
// A stage is reusable only after both consumers release its current step.
// If every actor is blocked in one sweep, the dependency graph is cyclic.
func nvfp4TMACycleCompletes(stages, kblocks, tiles int, fetchFirst bool, seed uint64) bool {
	program := nvfp4TMACycleProgram(kblocks, tiles, fetchFirst)
	slots := make([]nvfp4TMACycleSlot, stages)
	for i := range slots {
		slots[i] = nvfp4TMACycleSlot{step: -1, released: 3}
	}
	producer := 0
	consumers := [2]int{}
	r := rand.New(rand.NewPCG(seed, seed+1))
	for sweep := 0; sweep < 4*(len(program)+kblocks*tiles+1); sweep++ {
		if producer == kblocks*tiles && consumers[0] == len(program) && consumers[1] == len(program) {
			return true
		}
		actors := []int{0, 1, 2}
		r.Shuffle(len(actors), func(i, j int) { actors[i], actors[j] = actors[j], actors[i] })
		progress := false
		for _, actor := range actors {
			if actor == 0 {
				if producer == kblocks*tiles {
					continue
				}
				slot := &slots[producer%stages]
				if slot.released == 3 {
					*slot = nvfp4TMACycleSlot{step: producer}
					producer++
					progress = true
				}
				continue
			}
			consumer := actor - 1
			if consumers[consumer] == len(program) {
				continue
			}
			event := program[consumers[consumer]]
			slot := &slots[event.step%stages]
			if slot.step != event.step {
				continue
			}
			bit := uint8(1 << consumer)
			switch event.kind {
			case 'f':
				slot.fetched |= bit
			case 'r':
				if slot.fetched&bit == 0 {
					return false
				}
				slot.released |= bit
			default:
				return false
			}
			consumers[consumer]++
			progress = true
		}
		if !progress {
			return false
		}
	}
	return false
}

func TestNVFP4TMARetainedRingCycle(t *testing.T) {
	fetchFirst, err := nvfp4TMACycleFetchFirst(nvfp4TMADevice)
	if err != nil || !fetchFirst {
		t.Fatalf("retained source dependency order = %v, %v", fetchFirst, err)
	}
	for seed := uint64(0); seed < 8; seed++ {
		// K256 means four K64 steps. The consumer fetch of step2 needs
		// producer overwrite of stage0; that overwrite waits for the
		// release of step0 which is after the same blocked fetch.
		if nvfp4TMACycleCompletes(2, 4, 1, fetchFirst, seed) {
			t.Fatal("two-stage K256 cycle was not detected")
		}
		// The short two-step case can drain, but the public API rejects
		// all Stages2 configurations rather than creating a narrow exception.
		if !nvfp4TMACycleCompletes(2, 2, 1, fetchFirst, seed) {
			t.Fatal("model failed to distinguish the short drainable case")
		}
		for _, stages := range []int{3, 4, 6, 10, 16} {
			for _, kb := range []int{2, 4, 6, 64} {
				for _, tiles := range []int{1, 3} {
					if !nvfp4TMACycleCompletes(stages, kb, tiles, fetchFirst, seed) {
						t.Fatalf("abstract ring blocked with stages%d KB%d tiles%d seed%d", stages, kb, tiles, seed)
					}
				}
			}
		}
	}
}

func TestNVFP4TMARingOrderNegativeControl(t *testing.T) {
	const old = "fetch(sn, pn);\n        wg_wait<1>();\n        if (k > 0 && lt == 0) mbar_arrive(empty0 + 8 * sprev);"
	const changed = "wg_wait<1>();\n        if (k > 0 && lt == 0) mbar_arrive(empty0 + 8 * sprev);\n        fetch(sn, pn);"
	if strings.Count(nvfp4TMADevice, old) != 1 {
		t.Fatal("negative control no longer targets the retained dependency sequence")
	}
	mutant := strings.Replace(nvfp4TMADevice, old, changed, 1)
	fetchFirst, err := nvfp4TMACycleFetchFirst(mutant)
	if err != nil || fetchFirst {
		t.Fatalf("source-order mutation was not distinguished: %v, %v", fetchFirst, err)
	}
	if !nvfp4TMACycleCompletes(2, 4, 3, fetchFirst, 23) {
		t.Fatal("model did not respond to release-before-fetch control")
	}
	if _, err := nvfp4TMACycleFetchFirst(strings.Replace(nvfp4TMADevice, "fetch(sn, pn);", "", 1)); err == nil {
		t.Fatal("source-order oracle accepted missing dependency evidence")
	}
}
