package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// TestContextParallelLoadBalance is the load-balance oracle at 1M tokens on
// 16 ring ranks: the striped layout gives every rank the same causal tile
// count at every step, and the contiguous layout does not.
func TestContextParallelLoadBalance(t *testing.T) {
	tests := []struct {
		name     string
		mode     ContextParallelMode
		ring, u  int
		maxRatio float64
		minRatio float64
	}{
		{"striped", CPStripedRing, 16, 1, 1.0, 1.0},
		{"hybrid_4x16", CPHybrid2D, 16, 4, 1.0, 1.0},
		{"ulysses_8", CPUlyssesAll2All, 1, 8, 1.0, 1.0},
		{"contiguous", CPContiguousRing, 16, 1, 2, 1.8},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := ContextParallelSchedule{
				Mode: tc.mode, SeqLen: 1 << 20, TileM: 128, TileN: 128, HeadDim: 128,
				NumQHeads: 64, NumKVHeads: 8, UlyssesDeg: tc.u, RingDeg: tc.ring,
			}
			plan, err := PlanContextParallel(s)
			if err != nil {
				t.Fatal(err)
			}
			if len(plan) != tc.ring*tc.u || len(plan[0]) != tc.ring {
				t.Fatalf("plan is %dx%d, want %dx%d", len(plan), len(plan[0]), tc.ring*tc.u, tc.ring)
			}
			r := ImbalanceRatio(plan)
			t.Logf("%s: ImbalanceRatio %.4f", tc.name, r)
			if r < tc.minRatio || r > tc.maxRatio {
				t.Fatalf("ImbalanceRatio %.4f, want in [%v, %v]", r, tc.minRatio, tc.maxRatio)
			}
			if tc.mode != CPContiguousRing {
				for step, x := range StepImbalance(plan) {
					if x != 1 {
						t.Fatalf("step %d: max/min active tiles %.4f, want 1", step, x)
					}
				}
			}
			// Every (query tile, KV tile) pair under the causal mask is
			// computed exactly once per local head, summed over ranks.
			total := 0
			for _, steps := range plan {
				for _, st := range steps {
					total += st.ActiveTiles
				}
			}
			n := s.SeqLen / 128
			if want := n * (n + 1) / 2 * s.NumQHeads; total != want {
				t.Fatalf("total active tiles %d, want %d", total, want)
			}
		})
	}
}

func TestPlanContextParallelRing(t *testing.T) {
	s := ContextParallelSchedule{Mode: CPHybrid2D, SeqLen: 64, TileM: 4, TileN: 4, HeadDim: 8, NumQHeads: 8, NumKVHeads: 2, UlyssesDeg: 4, RingDeg: 4}
	plan, err := PlanContextParallel(s)
	if err != nil {
		t.Fatal(err)
	}
	for rank, steps := range plan {
		seen := map[int]bool{}
		for i, st := range steps {
			if st.KVSourceRank%4 != st.UlyssesRank {
				t.Fatalf("rank %d step %d reads KV from Ulysses group %d", rank, i, st.KVSourceRank%4)
			}
			seen[st.KVSourceRank] = true
			if i+1 < len(steps) && st.PrefetchRank != steps[i+1].KVSourceRank {
				t.Fatalf("rank %d step %d prefetches %d, next step reads %d", rank, i, st.PrefetchRank, steps[i+1].KVSourceRank)
			}
		}
		if len(seen) != 4 || steps[len(steps)-1].PrefetchRank != -1 {
			t.Fatalf("rank %d visits %d sources, last prefetch %d", rank, len(seen), steps[len(steps)-1].PrefetchRank)
		}
		if got := plan[rank][0].QHeads; got != [2]int{2 * (rank % 4), 2*(rank%4) + 2} {
			t.Fatalf("rank %d QHeads %v", rank, got)
		}
	}
	// UlyssesDeg 4 > NumKVHeads 2: KV heads are replicated, two ranks each.
	if plan[0][0].KVHeads != plan[1][0].KVHeads || plan[0][0].KVHeads == plan[2][0].KVHeads {
		t.Fatalf("KV heads %v %v %v, want ranks 0 and 1 to share", plan[0][0].KVHeads, plan[1][0].KVHeads, plan[2][0].KVHeads)
	}
}

func TestPlanContextParallelErrors(t *testing.T) {
	ok := ContextParallelSchedule{Mode: CPStripedRing, SeqLen: 1024, TileM: 64, TileN: 64, HeadDim: 64, NumQHeads: 8, NumKVHeads: 8, RingDeg: 4}
	tests := []struct {
		name string
		edit func(*ContextParallelSchedule)
	}{
		{"chunk_not_tile_multiple", func(s *ContextParallelSchedule) { s.SeqLen = 1000 }},
		{"ulysses_in_ring_mode", func(s *ContextParallelSchedule) { s.UlyssesDeg = 2 }},
		{"ulysses_not_dividing_heads", func(s *ContextParallelSchedule) { s.Mode, s.UlyssesDeg = CPHybrid2D, 3 }},
		{"gqa_mismatch", func(s *ContextParallelSchedule) { s.NumKVHeads = 3 }},
		{"bad_target", func(s *ContextParallelSchedule) { s.Target = "sm_90a" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := ok
			tc.edit(&s)
			if _, err := PlanContextParallel(s); err == nil {
				t.Fatal("PlanContextParallel succeeded, want error")
			}
		})
	}
}

func randomTensor(rng *rand.Rand, heads, n int) [][]float32 {
	x := make([][]float32, heads)
	for h := range x {
		x[h] = make([]float32, n)
		for i := range x[h] {
			x[h][i] = float32(rng.NormFloat64())
		}
	}
	return x
}

// causalAttention64 is monolithic causal attention in float64.
func causalAttention64(q, k, v [][]float32, L, D int) [][]float64 {
	group := len(q) / len(k)
	out := make([][]float64, len(q))
	scale := 1 / math.Sqrt(float64(D))
	for h := range q {
		out[h] = make([]float64, L*D)
		kh, vh := k[h/group], v[h/group]
		for i := range L {
			s := make([]float64, i+1)
			m := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var dot float64
				for d := range D {
					dot += float64(q[h][i*D+d]) * float64(kh[j*D+d])
				}
				s[j] = dot * scale
				m = max(m, s[j])
			}
			var l float64
			for j := range s {
				s[j] = math.Exp(s[j] - m)
				l += s[j]
			}
			for j := range s {
				for d := range D {
					out[h][i*D+d] += s[j] / l * float64(vh[j*D+d])
				}
			}
		}
	}
	return out
}

// TestSimulateContextParallelEquivalence is the numerical oracle: every
// context-parallel layout, merged with MergeOnlineSoftmaxTiles, matches
// monolithic float64 causal attention.
func TestSimulateContextParallelEquivalence(t *testing.T) {
	const L, D = 128, 16
	rng := rand.New(rand.NewPCG(21, 22))
	q, k, v := randomTensor(rng, 8, L*D), randomTensor(rng, 2, L*D), randomTensor(rng, 2, L*D)
	want := causalAttention64(q, k, v, L, D)
	tests := []ContextParallelSchedule{
		{Mode: CPStripedRing, RingDeg: 4},
		{Mode: CPContiguousRing, RingDeg: 4},
		{Mode: CPUlyssesAll2All, UlyssesDeg: 8},
		{Mode: CPHybrid2D, UlyssesDeg: 4, RingDeg: 2},
	}
	for _, s := range tests {
		t.Run(s.Mode.String(), func(t *testing.T) {
			s.SeqLen, s.TileM, s.TileN, s.HeadDim, s.NumQHeads, s.NumKVHeads = L, 8, 8, D, 8, 2
			got, err := SimulateContextParallel(s, q, k, v)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for h := range got {
				for i := range got[h] {
					worst = max(worst, math.Abs(float64(got[h][i])-want[h][i]))
				}
			}
			if !(worst <= 1e-5) {
				t.Fatalf("max abs error %g, want <= 1e-5", worst)
			}
		})
	}
}

func TestMergeOnlineSoftmaxTilesAssociative(t *testing.T) {
	const rows, dim, n = 4, 8, 24
	rng := rand.New(rand.NewPCG(23, 24))
	q, k, v := randomTensor(rng, 1, rows*dim)[0], randomTensor(rng, 1, n*dim)[0], randomTensor(rng, 1, n*dim)[0]
	for i := range k {
		k[i] *= 4 // large logits exercise the max shift
	}
	qpos := []int{n, n + 1, n + 2, 5}
	kpos := func(lo, hi int) []int {
		var p []int
		for i := lo; i < hi; i++ {
			p = append(p, i)
		}
		return p
	}
	part := func(lo, hi int) OnlineSoftmaxTile {
		return attendTile(q, k[lo*dim:hi*dim], v[lo*dim:hi*dim], qpos, kpos(lo, hi), dim)
	}
	a, b, c := part(0, 8), part(8, 16), part(16, 24)
	merge := func(ts ...OnlineSoftmaxTile) OnlineSoftmaxTile {
		m, err := MergeOnlineSoftmaxTiles(ts)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	whole := part(0, 24).Normalize()
	for name, got := range map[string][]float32{
		"(ab)c":  merge(merge(a, b), c).Normalize(),
		"a(bc)":  merge(a, merge(b, c)).Normalize(),
		"abc":    merge(a, b, c).Normalize(),
		"c(ba)":  merge(c, merge(b, a)).Normalize(),
		"single": merge(part(0, 24)).Normalize(),
	} {
		for i := range got {
			if d := math.Abs(float64(got[i] - whole[i])); !(d <= 1e-5) {
				t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], whole[i])
			}
		}
	}
	// Row 3 (position 5) sees no key in b or c; they must not perturb it.
	if !math.IsInf(float64(b.M[3]), -1) || b.L[3] != 0 {
		t.Fatalf("fully masked row has M=%v L=%v", b.M[3], b.L[3])
	}
	if _, err := MergeOnlineSoftmaxTiles([]OnlineSoftmaxTile{a, newOnlineSoftmaxTile(rows, dim+1)}); err == nil {
		t.Fatal("merging mismatched tiles succeeded")
	}
}

// TestContextParallelTMEMTrace checks that the output accumulator stays
// allocated across every ring step and is released after the loop, and
// that releasing it inside the loop is rejected.
func TestContextParallelTMEMTrace(t *testing.T) {
	s := ContextParallelSchedule{Mode: CPStripedRing, SeqLen: 4096, HeadDim: 128, NumQHeads: 8, NumKVHeads: 8, RingDeg: 4}
	tr := s.TMEMTrace()
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, tr)
	if err != nil {
		t.Fatalf("VerifyTrace: %v", err)
	}
	if sum.Allocations != sum.Deallocations || sum.MMAs != 2*s.RingDeg {
		t.Fatalf("summary %+v: want balanced alloc/dealloc and %d MMAs", sum, 2*s.RingDeg)
	}
	// The last ring step's P*V MMA must target the O columns allocated at
	// the start, with nothing freed in between.
	firstFree := slices.IndexFunc(tr, func(e cudatmem.TraceEvent) bool { return e.Kind == cudatmem.EventDealloc })
	lastMMA := -1
	for i, e := range tr {
		if e.Kind == cudatmem.EventMMA && e.Offset == cpTMEMOutAcc {
			lastMMA = i
		}
	}
	if firstFree < lastMMA {
		t.Fatalf("first dealloc at event %d precedes last O accumulation at %d", firstFree, lastMMA)
	}

	t.Run("mutant_free_inside_loop", func(t *testing.T) {
		var m []cudatmem.TraceEvent
		freed := false
		for _, e := range tr {
			m = append(m, e)
			if !freed && e.Kind == cudatmem.EventWait && e.Offset == cpTMEMOutAcc {
				d := e
				d.Kind = cudatmem.EventDealloc
				m = append(m, d)
				freed = true
			}
		}
		if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, m); err == nil {
			t.Fatal("VerifyTrace accepted O freed after the first ring step")
		}
	})
	t.Run("mutant_never_freed", func(t *testing.T) {
		if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, tr[:len(tr)-1]); err == nil {
			t.Fatal("VerifyTrace accepted a trace that never frees O")
		}
	})
}

func TestContextParallelRingPTX(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_103a", "sm_107a"} {
		t.Run(target, func(t *testing.T) {
			s := ContextParallelSchedule{Target: target, Mode: CPStripedRing, SeqLen: 1 << 20, HeadDim: 128, NumQHeads: 8, NumKVHeads: 8, RingDeg: 16}
			ptx, err := s.RingPTX("cp_ring")
			if err != nil {
				t.Fatal(err)
			}
			prefetch := "cp.async.bulk.shared::cluster.global"
			if target == "sm_107a" {
				prefetch = "fabric.try_get.async.tensor.2d"
			}
			redMax := strings.Contains(ptx, "tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32")
			if redMax != (target != "sm_100a") {
				t.Fatalf("tcgen05.ld.red present = %v on %s", redMax, target)
			}
			// The next step's prefetch is issued before this step's MMAs.
			ip, im := strings.Index(ptx, prefetch), strings.Index(ptx, "tcgen05.mma")
			if ip < 0 || im < 0 || ip > im {
				t.Fatalf("prefetch at %d, first MMA at %d", ip, im)
			}
			if !strings.Contains(ptx, "[%r_tmem_o], [%r_tmem_p], %rd_vdesc, %r_idesc, %p_acc") {
				t.Fatal("P*V MMA does not accumulate into O with enable_input_d")
			}
			if strings.Index(ptx, "tcgen05.dealloc") < strings.LastIndex(ptx, "bra $ring_step") {
				t.Fatal("TMEM released inside the ring loop")
			}
		})
	}
}

// TestContextParallelRingKernelHostEmu runs the emitted ring kernel on 4
// emulated ranks and compares with float64 monolithic causal attention.
func TestContextParallelRingKernelHostEmu(t *testing.T) {
	for _, mode := range []ContextParallelMode{CPStripedRing, CPContiguousRing} {
		t.Run(mode.String(), func(t *testing.T) {
			const P, L, HQ, HKV, D = 4, 64, 4, 2, 8
			s := ContextParallelSchedule{Mode: mode, SeqLen: L, TileM: 8, TileN: 8, HeadDim: D, NumQHeads: HQ, NumKVHeads: HKV, RingDeg: P}
			kern := ContextParallelRingKernel{Name: "cp_ring_" + mode.String(), Schedule: s}
			src, err := kern.Source()
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(25, 26))
			q, k, v := randomTensor(rng, HQ, L*D), randomTensor(rng, HKV, L*D), randomTensor(rng, HKV, L*D)
			want := causalAttention64(q, k, v, L, D)
			plan, err := PlanContextParallel(s)
			if err != nil {
				t.Fatal(err)
			}
			// Lay out each rank's owned tokens as [token][head][D].
			var positions [][]int
			for r := range P {
				var pos []int
				for _, iv := range plan[r][0].QIntervals {
					for p := iv[0]; p < iv[1]; p++ {
						pos = append(pos, p)
					}
				}
				positions = append(positions, pos)
			}
			pack := func(x [][]float32) []float32 {
				var out []float32
				for r := range P {
					for _, p := range positions[r] {
						for h := range x {
							out = append(out, x[h][p*D:(p+1)*D]...)
						}
					}
				}
				return out
			}
			outBytes := cudahostemu.Poison(4 * L * HQ * D)
			grid, block := kern.Grid()
			err = cudahostemu.RunMultiRank(context.Background(), src, P, cudahostemu.Launch{
				Kernel: kern.Name,
				Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
				Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			},
				cudahostemu.BufferArg("q", "float", f32Bytes(pack(q))),
				cudahostemu.BufferArg("kv_k", "float", f32Bytes(pack(k))),
				cudahostemu.BufferArg("kv_v", "float", f32Bytes(pack(v))),
				cudahostemu.BufferArg("out", "float", outBytes),
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank: %v", err)
			}
			got := bytesF32(outBytes)
			i := 0
			worst := 0.0
			for r := range P {
				for _, p := range positions[r] {
					for h := range HQ {
						for d := range D {
							worst = max(worst, math.Abs(float64(got[i])-want[h][p*D+d]))
							i++
						}
					}
				}
			}
			if !(worst <= 1e-5) {
				t.Fatalf("max abs error %g vs monolithic attention, want <= 1e-5", worst)
			}
		})
	}
}

func ExamplePlanContextParallel() {
	for _, mode := range []ContextParallelMode{CPStripedRing, CPContiguousRing} {
		plan, err := PlanContextParallel(ContextParallelSchedule{
			Mode: mode, SeqLen: 1 << 20, HeadDim: 128,
			NumQHeads: 8, NumKVHeads: 8, RingDeg: 16,
		})
		if err != nil {
			panic(err)
		}
		fmt.Printf("%v: imbalance %.3f\n", mode, ImbalanceRatio(plan))
	}
	// Output:
	// striped_ring: imbalance 1.000
	// contiguous_ring: imbalance 1.937
}
