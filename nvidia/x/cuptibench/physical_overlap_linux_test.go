//go:build linux

package cuptibench_test

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Tickets delimit kernel-body intervals. They do not establish simultaneous
// instruction issue, unchanged scheduling, or a performance estimate.
const overlapSource = `
__device__ unsigned long long overlap_timer() {
 unsigned long long t;
 asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
 return t;
}
extern "C" __global__ void overlap_work(unsigned *records, unsigned op, unsigned side, unsigned dependent) {
 unsigned *r = records + 7 * op;
 if (threadIdx.x == 0) r[1 + 2 * side] = atomicAdd(r, 1);
 __syncthreads();
 unsigned long long start = overlap_timer();
 while (overlap_timer() - start < 20000000ULL) {}
 if (threadIdx.x == 0) r[5 + side] = side == 0 ? 17 : (dependent ? 2 * r[5] : 23);
 __syncthreads();
 if (threadIdx.x == 0) r[2 + 2 * side] = atomicAdd(r, 1);
}
`

// TestPhysicalWholeOverlap qualifies direct launches under Whole/UserReplay.
// Serialization is a failed overlap gate, not a zero counter or a timing claim.
func TestPhysicalWholeOverlap(t *testing.T) {
	owner := &overlapCleanup{}
	t.Cleanup(func() {
		if owner.retain {
			retainedOverlapResources = append(retainedOverlapResources, owner.resources...)
			t.Log("native cleanup incomplete: retaining all overlap resources until process exit")
		}
	})
	tb := overlapTB{TB: t, owner: owner}
	p := newPhysicalPipeline(tb, 1)
	owner.resources = append(owner.resources, p)
	cm, err := nvrtcx.Compile(overlapSource, nvrtcx.WithDeviceArch(p.dev), nvrtcx.WithNoCache())
	physicalCheck(t, err)
	module, err := cm.Load(p.dev)
	physicalCheck(t, err)
	physicalClose(tb, func() error { return p.dev.WithContext(module.Unload) })
	owner.resources = append(owner.resources, module)
	var kernel driver.Function
	physicalCheck(t, p.dev.WithContext(func() error { var err error; kernel, err = module.Function("overlap_work"); return err }))
	records, err := cudamem.Alloc[uint32](p.dev, 21)
	physicalCheck(t, err)
	physicalClose(tb, records.Free)
	owner.resources = append(owner.resources, records)
	t.Logf("source_sha256=%x streams=nonblocking distinct=true bounded_work_ns=20000000; atomic body-interval witness, not simultaneous instruction issue", sha256.Sum256([]byte(overlapSource)))
	for _, tc := range []struct {
		name               string
		profile, dependent bool
		operations         int
	}{
		{"baseline", false, false, 3},
		{"profile-1", true, false, 1},
		{"profile-3", true, false, 3},
		{"dependent-baseline", false, true, 3},
		{"dependent-profile", true, true, 3},
	} {
		if !t.Run(tc.name, func(t *testing.T) {
			var plan *cuptibench.Plan
			var syncFailure error
			dirty := false
			synchronize := func() error {
				if syncFailure != nil {
					return syncFailure
				}
				if !dirty {
					return nil
				}
				syncFailure = p.synchronize()
				if syncFailure != nil {
					owner.retain = true
				} else {
					dirty = false
				}
				return syncFailure
			}
			t.Cleanup(func() {
				if err := synchronize(); err != nil {
					t.Error(err)
					return
				}
				if plan != nil {
					if err := plan.Close(); err != nil {
						owner.retain = true
						t.Error(err)
					}
				}
			})
			prepares, runs, checked := 0, 0, 0
			pending := false
			validate := func() error {
				if !pending {
					return nil
				}
				values := make([]uint32, 21)
				if err := records.CopyToHost(values); err != nil {
					return err
				}
				t.Logf("pass=%d operations=%d dependent=%t records=%v", checked, tc.operations, tc.dependent, values)
				for i := 0; i < tc.operations; i++ {
					row := values[i*7 : (i+1)*7]
					if err := validateOverlap(row, tc.dependent); err != nil {
						return fmt.Errorf("operation %d: %w", i, err)
					}
					wantB := uint32(23)
					if tc.dependent {
						wantB = 34
					}
					if row[5] != 17 || row[6] != wantB {
						return fmt.Errorf("operation %d: output=%v want=[17 %d]", i, row[5:], wantB)
					}
					if tc.dependent && validateOverlap(row, false) == nil {
						return errors.New("serialized negative accepted as overlap")
					}
				}
				for _, v := range values[tc.operations*7:] {
					if v != ^uint32(0) {
						return errors.New("unused record overwritten")
					}
				}
				checked++
				pending = false
				return nil
			}
			w := cuptibench.Workload{View: "full", Streams: p.streams[:], Synchronize: synchronize, Preparation: "validate previous pass; poison all records; zero per-operation atomic counters; reset kernel outside capture; no cross-kernel spin dependency"}
			w.Prepare = func() error {
				dirty = true
				if err := validate(); err != nil {
					return err
				}
				values := make([]uint32, 21)
				for i := range values {
					values[i] = ^uint32(0)
				}
				for i := 0; i < tc.operations; i++ {
					values[i*7] = 0
				}
				if err := records.CopyFromHost(values); err != nil {
					return err
				}
				prepares++
				return p.prepare() // Eight preparation CTAs must not enter the Whole total.
			}
			w.Run = func(*cuptibench.Scope) error {
				dirty = true
				op := runs % tc.operations
				dependent := uint32(0)
				if tc.dependent {
					dependent = 1
				}
				if err := cudalaunch.Kernel(kernel).Block(physicalBlock).On(p.streams[0]).Launch(records, uint32(op), uint32(0), dependent); err != nil {
					return err
				}
				if err := p.projected.Record(p.streams[0]); err != nil {
					return err
				}
				if tc.dependent {
					if err := p.streams[1].WaitEvent(p.projected); err != nil {
						return err
					}
				}
				if err := cudalaunch.Kernel(kernel).Block(physicalBlock).On(p.streams[1]).Launch(records, uint32(op), uint32(1), dependent); err != nil {
					return err
				}
				if err := p.decoded.Record(p.streams[1]); err != nil {
					return err
				}
				// Join both sides before either stream can begin the next operation.
				if err := p.streams[0].WaitEvent(p.decoded); err != nil {
					return err
				}
				if err := p.streams[1].WaitEvent(p.projected); err != nil {
					return err
				}
				runs++
				pending = true
				return nil
			}
			if !tc.profile {
				physicalCheck(t, w.Prepare())
				physicalCheck(t, synchronize())
				for i := 0; i < tc.operations; i++ {
					physicalCheck(t, w.Run(nil))
				}
				physicalCheck(t, synchronize())
				physicalCheck(t, validate())
				return
			}
			var err error
			plan, err = cuptibench.New(p.dev, cuptibench.Config{
				Mode: cuptibench.Whole, ReplayMode: cuptirange.UserReplay,
				CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric},
				Operations: tc.operations, MaxConfigBytes: 16 << 20,
				Limits: cuptirange.Limits{MaxRanges: 128, MaxRangeTreeNodes: 256, MaxCounterBytes: 32 << 20, MaxPasses: 128},
			})
			if plan != nil {
				owner.resources = append(owner.resources, plan)
			}
			physicalCheck(t, err)
			result, collectErr := plan.Collect(w)
			if result != nil {
				raw, _ := json.Marshal(result)
				t.Logf("physical result=%s", raw)
			}
			// Read output evidence even when native collection fails, without replaying.
			syncErr := synchronize()
			if syncErr == nil {
				syncErr = validate()
			}
			physicalCheck(t, errors.Join(collectErr, syncErr))
			if result == nil || result.Operations != tc.operations || len(result.Ranges) != 1 || result.Ranges[0].Kind != "operation" || result.Aggregate != nil {
				t.Fatalf("invalid Whole result: %+v", result)
			}
			q := result.Quality
			if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped || prepares != int(q.PassesStarted) || checked != prepares || runs != prepares*tc.operations {
				t.Fatalf("pass accounting prepares=%d runs=%d checked=%d quality=%+v", prepares, runs, checked, q)
			}
			if result.Metadata["replay_mode"] != "user" {
				t.Fatalf("replay mode=%q", result.Metadata["replay_mode"])
			}
			raw := physicalValue(t, result.Ranges[0])
			if raw != float64(2*tc.operations) || raw/float64(result.Operations) != 2 {
				t.Fatalf("whole CTA raw=%g operations=%d want=%d and 2/op", raw, result.Operations, 2*tc.operations)
			}
		}) {
			return
		}
	}
}
