package nvl72

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// ContextParallelMode selects how a long sequence is distributed across
// ranks for causal attention.
type ContextParallelMode uint8

const (
	// CPStripedRing splits the sequence into 2P chunks and gives ring rank
	// r chunks r and 2P-1-r, so every rank does the same causal work at
	// every ring step.
	CPStripedRing ContextParallelMode = iota

	// CPUlyssesAll2All shards attention heads: after an all-to-all each of
	// P ranks holds the whole sequence for NumQHeads/P query heads.
	CPUlyssesAll2All

	// CPHybrid2D shards heads over UlyssesDeg ranks (intra-tray) and runs a
	// striped ring over RingDeg ranks (inter-tray); rank = ring*Ulysses + u.
	CPHybrid2D

	// CPContiguousRing gives ring rank r the contiguous chunk r. It is the
	// unbalanced baseline: under a causal mask rank 0 attends to one chunk
	// and rank P-1 to all P.
	CPContiguousRing
)

func (m ContextParallelMode) String() string {
	switch m {
	case CPStripedRing:
		return "striped_ring"
	case CPUlyssesAll2All:
		return "ulysses"
	case CPHybrid2D:
		return "hybrid_2d"
	case CPContiguousRing:
		return "contiguous_ring"
	}
	return fmt.Sprintf("ContextParallelMode(%d)", uint8(m))
}

// ContextParallelSchedule describes a multi-rank causal attention schedule.
type ContextParallelSchedule struct {
	Target     string // "sm_100a", "sm_103a" or "sm_107a" for the emitted PTX
	Mode       ContextParallelMode
	SeqLen     int // total tokens
	TileM      int // query rows per tile (default 128)
	TileN      int // KV columns per tile (default 128)
	HeadDim    int
	NumQHeads  int
	NumKVHeads int
	UlyssesDeg int // head-sharding degree (1 for pure ring modes)
	RingDeg    int // ring length (1 for pure Ulysses)
}

func (s ContextParallelSchedule) norm() (ContextParallelSchedule, error) {
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if s.TileM == 0 {
		s.TileM = 128
	}
	if s.TileN == 0 {
		s.TileN = 128
	}
	if s.UlyssesDeg == 0 {
		s.UlyssesDeg = 1
	}
	if s.RingDeg == 0 {
		s.RingDeg = 1
	}
	switch s.Mode {
	case CPStripedRing, CPContiguousRing:
		if s.UlyssesDeg != 1 {
			return s, fmt.Errorf("context parallel: %v needs UlyssesDeg 1, have %d", s.Mode, s.UlyssesDeg)
		}
	case CPUlyssesAll2All:
		if s.RingDeg != 1 {
			return s, fmt.Errorf("context parallel: %v needs RingDeg 1, have %d", s.Mode, s.RingDeg)
		}
	case CPHybrid2D:
	default:
		return s, fmt.Errorf("context parallel: unknown mode %v", s.Mode)
	}
	switch s.Target {
	case "sm_100a", "sm_103a", "sm_107a":
	default:
		return s, fmt.Errorf("context parallel: target %q unsupported (want sm_100a, sm_103a or sm_107a)", s.Target)
	}
	switch {
	case s.SeqLen < 1:
		return s, fmt.Errorf("context parallel: SeqLen %d must be positive", s.SeqLen)
	case s.TileM < 1 || s.TileN < 1:
		return s, fmt.Errorf("context parallel: tile %dx%d must be positive", s.TileM, s.TileN)
	case s.HeadDim < 1:
		return s, fmt.Errorf("context parallel: HeadDim %d must be positive", s.HeadDim)
	case s.NumQHeads < 1 || s.NumKVHeads < 1 || s.NumQHeads%s.NumKVHeads != 0:
		return s, fmt.Errorf("context parallel: NumQHeads %d must be a positive multiple of NumKVHeads %d", s.NumQHeads, s.NumKVHeads)
	case s.UlyssesDeg < 1 || s.NumQHeads%s.UlyssesDeg != 0:
		return s, fmt.Errorf("context parallel: UlyssesDeg %d must divide NumQHeads %d", s.UlyssesDeg, s.NumQHeads)
	case s.RingDeg < 1:
		return s, fmt.Errorf("context parallel: RingDeg %d must be positive", s.RingDeg)
	}
	if c := s.chunk(); c == 0 || s.SeqLen%(s.chunks()*c) != 0 || c%s.TileM != 0 || c%s.TileN != 0 {
		return s, fmt.Errorf("context parallel: SeqLen %d must split into %d chunks that are multiples of tiles %dx%d",
			s.SeqLen, s.chunks(), s.TileM, s.TileN)
	}
	return s, nil
}

func (s ContextParallelSchedule) striped() bool {
	return s.Mode == CPStripedRing || s.Mode == CPHybrid2D
}

func (s ContextParallelSchedule) chunks() int {
	if s.striped() {
		return 2 * s.RingDeg
	}
	return s.RingDeg
}

func (s ContextParallelSchedule) chunk() int { return s.SeqLen / s.chunks() }

// NumRanks returns UlyssesDeg * RingDeg.
func (s ContextParallelSchedule) NumRanks() int { return s.UlyssesDeg * s.RingDeg }

// ringIntervals returns the token intervals ring rank r owns.
func (s ContextParallelSchedule) ringIntervals(r int) [][2]int {
	c := s.chunk()
	if !s.striped() {
		return [][2]int{{r * c, (r + 1) * c}}
	}
	j := 2*s.RingDeg - 1 - r
	return [][2]int{{r * c, (r + 1) * c}, {j * c, (j + 1) * c}}
}

// RingStepPlan is one rank's work at one ring step.
type RingStepPlan struct {
	Step         int
	Rank         int // global rank
	RingRank     int
	UlyssesRank  int
	KVSourceRank int // global rank whose KV this step consumes
	PrefetchRank int // global rank whose KV is prefetched for the next step, or -1
	ActiveTiles  int // causal TileM x TileN tiles with any unmasked entry, over local query heads

	QIntervals     [][2]int // query tokens this rank owns
	TokenIntervals [][2]int // KV tokens owned by KVSourceRank
	QHeads         [2]int   // [lo, hi) query heads computed here
	KVHeads        [2]int   // [lo, hi) KV heads needed; shared by ranks when UlyssesDeg > NumKVHeads
}

// PlanContextParallel returns the schedule indexed [rank][step].
func PlanContextParallel(s ContextParallelSchedule) ([][]RingStepPlan, error) {
	s, err := s.norm()
	if err != nil {
		return nil, err
	}
	U, R := s.UlyssesDeg, s.RingDeg
	qPer := s.NumQHeads / U
	group := s.NumQHeads / s.NumKVHeads
	plan := make([][]RingStepPlan, U*R)
	for rank := range plan {
		r, u := rank/U, rank%U
		qh := [2]int{u * qPer, (u + 1) * qPer}
		kvh := [2]int{qh[0] / group, (qh[1] + group - 1) / group}
		q := s.ringIntervals(r)
		for step := range R {
			src := (r - step + R) % R
			next := -1
			if step+1 < R {
				next = ((r-step-1+R)%R)*U + u
			}
			kv := s.ringIntervals(src)
			tiles := 0
			for _, qi := range q {
				for _, ki := range kv {
					tiles += causalTiles(qi, ki, s.TileM, s.TileN)
				}
			}
			plan[rank] = append(plan[rank], RingStepPlan{
				Step: step, Rank: rank, RingRank: r, UlyssesRank: u,
				KVSourceRank: src*U + u, PrefetchRank: next,
				ActiveTiles: tiles * qPer, QIntervals: q, TokenIntervals: kv,
				QHeads: qh, KVHeads: kvh,
			})
		}
	}
	return plan, nil
}

// causalTiles counts the tiles of q x kv holding at least one pair with
// key position <= query position.
func causalTiles(q, kv [2]int, tm, tn int) int {
	ntiles := (kv[1] - kv[0]) / tn
	n := 0
	for lo := q[0]; lo < q[1]; lo += tm {
		last := min(lo+tm, q[1]) - 1
		if last < kv[0] {
			continue
		}
		n += min((last-kv[0])/tn+1, ntiles)
	}
	return n
}

// ImbalanceRatio returns the largest rank's total active tiles over the mean.
func ImbalanceRatio(plan [][]RingStepPlan) float64 {
	top, total := 0, 0
	for _, steps := range plan {
		n := 0
		for _, st := range steps {
			n += st.ActiveTiles
		}
		top = max(top, n)
		total += n
	}
	if total == 0 {
		return 1
	}
	return float64(top) * float64(len(plan)) / float64(total)
}

// StepImbalance returns, for each ring step, the largest rank's active tiles
// over the smallest.
func StepImbalance(plan [][]RingStepPlan) []float64 {
	if len(plan) == 0 {
		return nil
	}
	out := make([]float64, len(plan[0]))
	for step := range out {
		lo, hi := math.MaxInt, 0
		for _, steps := range plan {
			lo = min(lo, steps[step].ActiveTiles)
			hi = max(hi, steps[step].ActiveTiles)
		}
		out[step] = math.Inf(1)
		if lo > 0 {
			out[step] = float64(hi) / float64(lo)
		}
	}
	return out
}

// OnlineSoftmaxTile is a partial attention result for Rows query rows:
// row maxima M, row sums L of exp(s-M), and unnormalized O[Rows*Dim] =
// sum exp(s-M) v. A row that saw no unmasked key has M = -Inf, L = 0, O = 0.
type OnlineSoftmaxTile struct {
	Rows, Dim int
	M, L      []float32
	O         []float32
}

func newOnlineSoftmaxTile(rows, dim int) OnlineSoftmaxTile {
	t := OnlineSoftmaxTile{Rows: rows, Dim: dim, M: make([]float32, rows), L: make([]float32, rows), O: make([]float32, rows*dim)}
	for i := range t.M {
		t.M[i] = float32(math.Inf(-1))
	}
	return t
}

// MergeOnlineSoftmaxTiles combines partial tiles for the same query rows
// computed over disjoint key sets with the log-sum-exp rule: M = max M_i,
// L = sum L_i exp(M_i - M), O = sum O_i exp(M_i - M).
func MergeOnlineSoftmaxTiles(partials []OnlineSoftmaxTile) (OnlineSoftmaxTile, error) {
	if len(partials) == 0 {
		return OnlineSoftmaxTile{}, fmt.Errorf("merge online softmax: no tiles")
	}
	rows, dim := partials[0].Rows, partials[0].Dim
	for i, p := range partials {
		if p.Rows != rows || p.Dim != dim || len(p.M) != rows || len(p.L) != rows || len(p.O) != rows*dim {
			return OnlineSoftmaxTile{}, fmt.Errorf("merge online softmax: tile %d is %dx%d, want %dx%d", i, p.Rows, p.Dim, rows, dim)
		}
	}
	out := newOnlineSoftmaxTile(rows, dim)
	for r := range rows {
		m := out.M[r]
		for _, p := range partials {
			m = max(m, p.M[r])
		}
		if math.IsInf(float64(m), -1) {
			continue
		}
		out.M[r] = m
		for _, p := range partials {
			if math.IsInf(float64(p.M[r]), -1) {
				continue
			}
			a := expf(p.M[r] - m)
			out.L[r] += float32(a * p.L[r])
			for d := range dim {
				out.O[r*dim+d] += float32(a * p.O[r*dim+d])
			}
		}
	}
	return out, nil
}

// Normalize returns O/L, with zeros for rows that saw no key.
func (t OnlineSoftmaxTile) Normalize() []float32 {
	out := make([]float32, len(t.O))
	for r := range t.Rows {
		if t.L[r] == 0 {
			continue
		}
		for d := range t.Dim {
			out[r*t.Dim+d] = t.O[r*t.Dim+d] / t.L[r]
		}
	}
	return out
}

// attendTile computes the causal partial attention of query rows at
// positions qpos against keys at kpos. q is [len(qpos)*dim], k and v are
// [len(kpos)*dim].
func attendTile(q, k, v []float32, qpos, kpos []int, dim int) OnlineSoftmaxTile {
	t := newOnlineSoftmaxTile(len(qpos), dim)
	scale := float32(1 / math.Sqrt(float64(dim)))
	s := make([]float32, len(kpos))
	for i, qp := range qpos {
		m := float32(math.Inf(-1))
		for j, kp := range kpos {
			if kp > qp {
				continue
			}
			var dot float32
			for d := range dim {
				dot += float32(q[i*dim+d] * k[j*dim+d])
			}
			s[j] = float32(dot * scale)
			m = max(m, s[j])
		}
		if math.IsInf(float64(m), -1) {
			continue
		}
		t.M[i] = m
		for j, kp := range kpos {
			if kp > qp {
				continue
			}
			p := expf(s[j] - m)
			t.L[i] += p
			for d := range dim {
				t.O[i*dim+d] += float32(p * v[j*dim+d])
			}
		}
	}
	return t
}

// SimulateContextParallel executes the schedule on the host. q is
// [NumQHeads][SeqLen*HeadDim]; k and v are [NumKVHeads][SeqLen*HeadDim].
// Each rank attends its query intervals to the source rank's KV intervals
// at every ring step and folds the step into its running state with
// [MergeOnlineSoftmaxTiles]. It returns the normalized output in q's layout.
func SimulateContextParallel(s ContextParallelSchedule, q, k, v [][]float32) ([][]float32, error) {
	plan, err := PlanContextParallel(s)
	if err != nil {
		return nil, err
	}
	s, _ = s.norm()
	L, D := s.SeqLen, s.HeadDim
	if len(q) != s.NumQHeads || len(k) != s.NumKVHeads || len(v) != s.NumKVHeads {
		return nil, fmt.Errorf("context parallel simulate: have %d/%d/%d heads, want %d/%d/%d",
			len(q), len(k), len(v), s.NumQHeads, s.NumKVHeads, s.NumKVHeads)
	}
	for _, x := range [][][]float32{q, k, v} {
		for h, buf := range x {
			if len(buf) != L*D {
				return nil, fmt.Errorf("context parallel simulate: head %d has %d values, want %d", h, len(buf), L*D)
			}
		}
	}
	group := s.NumQHeads / s.NumKVHeads
	gather := func(buf []float32, iv [][2]int) ([]float32, []int) {
		var vals []float32
		var pos []int
		for _, in := range iv {
			vals = append(vals, buf[in[0]*D:in[1]*D]...)
			for p := in[0]; p < in[1]; p++ {
				pos = append(pos, p)
			}
		}
		return vals, pos
	}
	out := make([][]float32, s.NumQHeads)
	for h := range out {
		out[h] = make([]float32, L*D)
	}
	for _, steps := range plan {
		for h := steps[0].QHeads[0]; h < steps[0].QHeads[1]; h++ {
			if kvh := h / group; kvh < steps[0].KVHeads[0] || kvh >= steps[0].KVHeads[1] {
				return nil, fmt.Errorf("context parallel simulate: rank %d head %d needs KV head %d outside %v", steps[0].Rank, h, kvh, steps[0].KVHeads)
			}
			qv, qpos := gather(q[h], steps[0].QIntervals)
			state := newOnlineSoftmaxTile(len(qpos), D)
			for _, st := range steps {
				kv, kpos := gather(k[h/group], st.TokenIntervals)
				vv, _ := gather(v[h/group], st.TokenIntervals)
				part := attendTile(qv, kv, vv, qpos, kpos, D)
				if state, err = MergeOnlineSoftmaxTiles([]OnlineSoftmaxTile{state, part}); err != nil {
					return nil, err
				}
			}
			o := state.Normalize()
			for i, p := range qpos {
				copy(out[h][p*D:(p+1)*D], o[i*D:(i+1)*D])
			}
		}
	}
	return out, nil
}

// TMEM columns of the ring kernel: two score buffers S (tmem[0..127] and
// tmem[128..255], alternating by step), each overwritten in place by the
// probabilities P as in FlashAttention-4, and the output accumulator O in
// tmem[256..383], which stays allocated across every ring step.
const (
	cpTMEMScores = 0
	cpTMEMOutAcc = 256
	cpTMEMCols   = 128
)

// TMEMTrace returns the tcgen05 Tensor Memory lifecycle of one CTA of the
// emitted ring kernel: S, P and O allocated once, per ring step an S = Q K^T
// MMA, row-max load and P store, a P V MMA accumulating into O, and an O
// rescale, then O read out and everything released after the ring loop.
// [ContextParallelSchedule.RingPTX] allocates 512 columns in one
// tcgen05.alloc (the count must be a power of two); the trace records the
// three 128-column slices of it that the loop uses.
// P overwrites S in place, and the two S buffers alternate by step.
func (s ContextParallelSchedule) TMEMTrace() []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	ev := func(kind cudatmem.EventKind, threads int, off uint32) cudatmem.TraceEvent {
		return cudatmem.TraceEvent{Kind: kind, ActiveThreads: threads, CTAGroup: 1, Offset: off, Columns: cpTMEMCols, Rows: 128}
	}
	warps := func(kind cudatmem.EventKind, off uint32) []cudatmem.TraceEvent {
		var e []cudatmem.TraceEvent
		for w := range 4 {
			x := ev(kind, 32, off)
			x.Warp, x.RowStart, x.Rows = w, uint32(32*w), 32
			e = append(e, x)
		}
		return e
	}
	var t []cudatmem.TraceEvent
	regions := []uint32{cpTMEMScores, cpTMEMScores + cpTMEMCols, cpTMEMOutAcc}
	for _, off := range regions {
		t = append(t, ev(cudatmem.EventAlloc, 32, off))
	}
	for step := range s.RingDeg {
		sb := uint32(cpTMEMScores + cpTMEMCols*(step%2))
		t = append(t, ev(cudatmem.EventMMA, 1, sb), ev(cudatmem.EventCommit, 1, sb), ev(cudatmem.EventWait, 32, sb))
		t = append(t, warps(cudatmem.EventLd, sb)...)
		t = append(t, warps(cudatmem.EventSt, sb)...)
		t = append(t, warps(cudatmem.EventLd, cpTMEMOutAcc)...)
		t = append(t, warps(cudatmem.EventSt, cpTMEMOutAcc)...)
		pv := ev(cudatmem.EventMMA, 1, cpTMEMOutAcc)
		pv.SourceAFromTMEM, pv.SourceAOffset = true, sb
		t = append(t, pv, ev(cudatmem.EventCommit, 1, cpTMEMOutAcc), ev(cudatmem.EventWait, 32, cpTMEMOutAcc))
	}
	t = append(t, warps(cudatmem.EventLd, cpTMEMOutAcc)...)
	for _, off := range regions {
		t = append(t, ev(cudatmem.EventDealloc, 32, off))
	}
	return t
}

// RingPTX returns the PTX instruction sequence of the persistent ring
// kernel's main loop for s.Target: the next step's KV is prefetched into the
// other shared-memory buffer before this step's MMAs, so the transfer
// overlaps compute. On sm_107a the prefetch is fabric.try_get.async.tensor;
// on sm_100a and sm_103a it is cp.async.bulk from the peer's NVLink address.
// Row maxima use tcgen05.ld.red on sm_103a and sm_107a and tcgen05.ld plus
// max.f32 on sm_100a. O accumulates in TMEM columns 256..383 with
// enable_input_d set after the first step.
//
// The listing is the instruction schedule with symbolic operands. It is not
// assembled by this package, and the numerical semantics are those of
// [SimulateContextParallel] and [ContextParallelRingKernel].
func (s ContextParallelSchedule) RingPTX(kernelName string) (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	if !cIdent(kernelName) {
		return "", fmt.Errorf("context parallel: bad kernel name %q", kernelName)
	}
	prefetch := "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r_kv_buf_next], [%rd_peer_kv], %r_kv_bytes, [%r_mbar_kv_next];"
	if s.Target == "sm_107a" {
		inst, err := RubinFabricAsyncTensorKernel{Op: FabricTensorTryGet, Target: "sm_107a"}.PTXInstruction()
		if err != nil {
			return "", err
		}
		prefetch = inst + " %r_status, [%r_kv_buf_next], [%rd_kv_le_id, %rd_kv_off, {%r_kv_col, %r_kv_row}], [%r_mbar_kv_next];"
	}
	rowMax := "tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32 {%f_s0, %f_s1, %f_s2, %f_s3}, %f_rowmax, [%r_tmem_s];"
	if s.Target == "sm_100a" {
		rowMax = "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%f_s0, %f_s1, %f_s2, %f_s3}, [%r_tmem_s];\n\tmax.f32 %f_rowmax, %f_s0, %f_s1; // ...over the row"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %v ring of %d steps, Q tile %d x %d, KV tile %d (%s).\n", kernelName, s.Mode, s.RingDeg, s.TileM, s.HeadDim, s.TileN, s.Target)
	fmt.Fprintf(&b, ".visible .entry %s(.param .u64 q, .param .u64 kv_sym, .param .u64 out)\n{\n", kernelName)
	fmt.Fprintf(&b, "\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%%r_tmem_slot], 512;\n")
	fmt.Fprintf(&b, "\t// S and P (in place), two buffers: tmem[%d..%d], O: tmem[%d..%d]\n",
		cpTMEMScores, cpTMEMScores+2*cpTMEMCols-1, cpTMEMOutAcc, cpTMEMOutAcc+cpTMEMCols-1)
	b.WriteString("\tmov.u32 %r_step, 0;\n\tsetp.ne.u32 %p_acc, 0, 0;\n")
	b.WriteString("$ring_step:\n")
	fmt.Fprintf(&b, "\tand.b32 %%r_sbuf, %%r_step, 1;\n\tmad.lo.u32 %%r_tmem_s, %%r_sbuf, %d, %%r_tmem_base;\n\tmov.u32 %%r_tmem_p, %%r_tmem_s;\n", cpTMEMCols)
	b.WriteString("\t// Prefetch step+1 KV into the other buffer before computing on this one.\n")
	b.WriteString("\tsetp.lt.u32 %p_more, %r_step, %r_last_step;\n")
	b.WriteString("\t@%p_more " + prefetch + "\n")
	b.WriteString("\tmbarrier.try_wait.parity.shared::cta.b64 %p_ready, [%r_mbar_kv], %r_phase;\n")
	b.WriteString("\ttcgen05.mma.cta_group::1.kind::f16 [%r_tmem_s], %rd_qdesc, %rd_kdesc, %r_idesc, 0;\n")
	b.WriteString("\ttcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r_mbar_s];\n")
	b.WriteString("\tmbarrier.try_wait.parity.shared::cta.b64 %p_s, [%r_mbar_s], %r_phase;\n")
	b.WriteString("\t" + rowMax + "\n")
	b.WriteString("\t// alpha = exp(m_old - m_new); P = exp(S - m_new) -> tmem P; O *= alpha in tmem.\n")
	b.WriteString("\ttcgen05.st.sync.aligned.32x32b.x4.b32 [%r_tmem_p], {%f_p0, %f_p1, %f_p2, %f_p3};\n")
	b.WriteString("\ttcgen05.ld.sync.aligned.32x32b.x4.b32 {%f_o0, %f_o1, %f_o2, %f_o3}, [%r_tmem_o];\n")
	b.WriteString("\ttcgen05.st.sync.aligned.32x32b.x4.b32 [%r_tmem_o], {%f_o0, %f_o1, %f_o2, %f_o3};\n")
	b.WriteString("\ttcgen05.mma.cta_group::1.kind::f16 [%r_tmem_o], [%r_tmem_p], %rd_vdesc, %r_idesc, %p_acc;\n")
	b.WriteString("\ttcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r_mbar_o];\n")
	b.WriteString("\tmbarrier.try_wait.parity.shared::cta.b64 %p_o, [%r_mbar_o], %r_phase;\n")
	b.WriteString("\tsetp.ne.u32 %p_acc, 1, 0;\n")
	b.WriteString("\txor.b32 %r_phase, %r_phase, 1;\n\tadd.u32 %r_step, %r_step, 1;\n")
	fmt.Fprintf(&b, "\tsetp.lt.u32 %%p_loop, %%r_step, %d;\n\t@%%p_loop bra $ring_step;\n", s.RingDeg)
	b.WriteString("\ttcgen05.ld.sync.aligned.32x32b.x4.b32 {%f_o0, %f_o1, %f_o2, %f_o3}, [%r_tmem_o];\n")
	b.WriteString("\ttcgen05.dealloc.cta_group::1.sync.aligned.b32 %r_tmem_base, 512;\n")
	b.WriteString("\ttcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n\tret;\n}\n")
	return b.String(), nil
}

// ContextParallelRingKernel emits a multi-rank CUDA kernel that computes
// striped- or contiguous-ring causal attention from symmetric-memory KV:
// each rank reads its peers' KV slices over NVLink peer pointers in ring
// order and keeps a per-row online softmax across the steps. It is the
// executable reference for the ring schedule (scalar FP32, one query row
// per thread), checked under cudahostemu.RunMultiRank; the tensor-core
// version of the loop is [ContextParallelSchedule.RingPTX].
//
// Buffers are symmetric: q, k, v and out hold, per rank, that rank's
// owned tokens in interval order, laid out [token][head][HeadDim].
type ContextParallelRingKernel struct {
	Name     string
	Schedule ContextParallelSchedule
	Threads  int // default 64
}

func (k ContextParallelRingKernel) norm() (ContextParallelRingKernel, error) {
	if k.Name == "" {
		k.Name = "nvl72_cp_ring_attention"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("context parallel: bad kernel name %q", k.Name)
	}
	if k.Threads == 0 {
		k.Threads = 64
	}
	s, err := k.Schedule.norm()
	if err != nil {
		return k, err
	}
	if s.Mode != CPStripedRing && s.Mode != CPContiguousRing {
		return k, fmt.Errorf("context parallel: kernel emits ring modes only, not %v", s.Mode)
	}
	if s.HeadDim > 256 {
		return k, fmt.Errorf("context parallel: kernel HeadDim %d exceeds 256", s.HeadDim)
	}
	k.Schedule = s
	return k, nil
}

// Grid returns one CTA per rank.
func (k ContextParallelRingKernel) Grid() (grid, block [3]int) {
	t := k.Threads
	if t == 0 {
		t = 64
	}
	return [3]int{1, 1, 1}, [3]int{t, 1, 1}
}

// LocalTokens returns the tokens each rank owns.
func (k ContextParallelRingKernel) LocalTokens() int {
	s, err := k.Schedule.norm()
	if err != nil {
		return 0
	}
	return s.SeqLen / s.RingDeg
}

// CUDA returns the kernel definition without the preludes.
func (k ContextParallelRingKernel) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	s := k.Schedule
	striped := 0
	if s.Mode == CPStripedRing {
		striped = 1
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank %v causal ring attention, %d tokens, %d/%d heads, dim %d.\n",
		k.Name, s.RingDeg, s.Mode, s.SeqLen, s.NumQHeads, s.NumKVHeads, s.HeadDim)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n", k.Threads, k.Name)
	b.WriteString("    const float* __restrict__ q, const float* __restrict__ kv_k, const float* __restrict__ kv_v, float* __restrict__ out) {\n")
	fmt.Fprintf(&b, "    constexpr int P = %d, L = %d, HQ = %d, HKV = %d, D = %d, STRIPED = %d;\n",
		s.RingDeg, s.SeqLen, s.NumQHeads, s.NumKVHeads, s.HeadDim, striped)
	b.WriteString(`    constexpr int NLOC = L / P, CHUNK = STRIPED ? L / (2 * P) : L / P, GROUP = HQ / HKV;
    const int rank = eg_nvlink_rank();
    const float* lq = eg_nvlink_local_ptr(q, (size_t)NLOC * HQ * D);
    float* lout = eg_nvlink_local_ptr(out, (size_t)NLOC * HQ * D);
    const float scale = 1.0f / sqrtf((float)D);
    // pos maps a rank's local token index to its global position.
    auto pos = [](int r, int i) {
        if (!STRIPED) return r * CHUNK + i;
        return i < CHUNK ? r * CHUNK + i : (2 * P - 1 - r) * CHUNK + (i - CHUNK);
    };
    for (int w = threadIdx.x; w < NLOC * HQ; w += blockDim.x) {
        const int i = w / HQ, h = w % HQ, kvh = h / GROUP;
        const int qp = pos(rank, i);
        const float* qrow = lq + ((size_t)i * HQ + h) * D;
        float m = -INFINITY, l = 0.0f, acc[D];
        for (int d = 0; d < D; ++d) acc[d] = 0.0f;
        for (int step = 0; step < P; ++step) {
            const int src = (rank - step + P) % P;
            const float* pk = eg_nvlink_peer_ptr(kv_k, src, rank, (size_t)NLOC * HKV * D * sizeof(float));
            const float* pv = eg_nvlink_peer_ptr(kv_v, src, rank, (size_t)NLOC * HKV * D * sizeof(float));
            for (int j = 0; j < NLOC; ++j) {
                if (pos(src, j) > qp) continue;
                const float* krow = pk + ((size_t)j * HKV + kvh) * D;
                const float* vrow = pv + ((size_t)j * HKV + kvh) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) { const float x = qrow[d] * krow[d]; dot += x; }
                const float sc = dot * scale;
                const float mn = fmaxf(m, sc);
                const float alpha = expf(m - mn), p = expf(sc - mn);
                l = l * alpha;
                l += p;
                for (int d = 0; d < D; ++d) { const float a = acc[d] * alpha; const float pvd = p * vrow[d]; acc[d] = a + pvd; }
                m = mn;
            }
        }
        float* orow = lout + ((size_t)i * HQ + h) * D;
        for (int d = 0; d < D; ++d) orow[d] = acc[d] / l;
    }
}
`)
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude] and the kernel.
func (k ContextParallelRingKernel) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + src, nil
}

// EmitContextParallelRing returns the executable ring kernel source and the
// tensor-core PTX schedule of its main loop for s.
func EmitContextParallelRing(s ContextParallelSchedule, kernelName string) (cuda, ptx string, err error) {
	if cuda, err = (ContextParallelRingKernel{Name: kernelName, Schedule: s}).Source(); err != nil {
		return "", "", err
	}
	if ptx, err = s.RingPTX(kernelName + "_tc"); err != nil {
		return "", "", err
	}
	return cuda, ptx, nil
}
