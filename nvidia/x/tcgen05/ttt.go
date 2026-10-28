package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// A TTTVariant selects the fast-weight update of a [TTTSchedule].
type TTTVariant uint8

const (
	// TTTLinearSGD is TTT-Linear: W = (1 - alpha) W - eta G.
	TTTLinearSGD TTTVariant = iota
	// TTTTitansMomentum is the Titans memory update:
	// M = mu M - eta G; W = (1 - alpha) W + M.
	TTTTitansMomentum
)

func (v TTTVariant) String() string {
	switch v {
	case TTTLinearSGD:
		return "ttt_linear"
	case TTTTitansMomentum:
		return "titans"
	}
	return fmt.Sprintf("TTTVariant(%d)", uint8(v))
}

// A TTTSchedule configures a test-time-training kernel that keeps the
// d x d fast weights W (and the Titans momentum M) in TMEM across all chunks
// of a sequence, one head per CTA iteration of a persistent loop.
//
// For each chunk of L tokens with keys k_t, values v_t and queries q_t:
//
//	z_t = W k_t                                   (tcgen05.mma, W the TMEM A operand)
//	ℓ_t = ½‖LN(z_t) - v_t‖²                       (LN without affine parameters)
//	g_t = ∂ℓ_t/∂z_t                               (exact LayerNorm VJP, one token per thread)
//	G = Σ_t g_t k_tᵀ
//	W = (1-α) W - η G   or   M = μ M - η G; W = (1-α) W + M
//	o_t = W q_t                                   (with the updated W)
//
// The decay is applied in TMEM with tcgen05.ld/st and -ηG is accumulated
// onto W (or M) with enable_input_d = 1. W_0 enters TMEM through one MMA of
// the staged W_0 against an identity tile, so the fast-weight region is
// MMA-produced before its first use as an A operand.
//
// TMEM layout (lanes are output rows of W): W at [0, d), z at [d, d+L), o
// at [d+L, d+2L), M at [d+2L, 2d+2L).
//
// The sm_100a/sm_103a/sm_107a path of the emitted source has not been
// executed; the EG_EMU path runs in the tests. Stochastic rounding of the
// staged gradient (cvt.rs) is not implemented.
type TTTSchedule struct {
	Name        string
	Target      string // sm_100a, sm_103a, sm_107a or an f variant (default sm_100a)
	Variant     TTTVariant
	HeadDim     int     // d: 32, 64, 96 or 128
	ChunkLen    int     // L: 32 or 64 tokens per gradient step
	NumChunks   int     // chunks per sequence
	LR          float32 // inner learning rate eta
	WeightDecay float32 // forgetting gate alpha in [0, 1)
	Momentum    float32 // Titans momentum mu in [0, 1)
	Eps         float32 // LayerNorm epsilon (default 1e-5)
}

func (s TTTSchedule) norm() (TTTSchedule, error) {
	if s.Name == "" {
		s.Name = s.Variant.String()
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if s.Eps == 0 {
		s.Eps = 1e-5
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("ttt schedule: bad name %q", s.Name)
	}
	if err := checkTMTarget("ttt schedule", s.Target); err != nil {
		return s, err
	}
	if s.Variant != TTTLinearSGD && s.Variant != TTTTitansMomentum {
		return s, fmt.Errorf("ttt schedule: unknown variant %v", s.Variant)
	}
	if s.HeadDim < 32 || s.HeadDim > 128 || s.HeadDim%32 != 0 {
		return s, fmt.Errorf("ttt schedule: head dim %d must be a multiple of 32 in [32, 128]", s.HeadDim)
	}
	if s.ChunkLen != 32 && s.ChunkLen != 64 {
		return s, fmt.Errorf("ttt schedule: chunk length %d must be 32 or 64", s.ChunkLen)
	}
	if s.NumChunks < 1 {
		return s, fmt.Errorf("ttt schedule: %d chunks", s.NumChunks)
	}
	if !(s.LR > 0) || !(s.WeightDecay >= 0 && s.WeightDecay < 1) || !(s.Momentum >= 0 && s.Momentum < 1) || !(s.Eps > 0) {
		return s, fmt.Errorf("ttt schedule: lr %g, decay %g, momentum %g, eps %g out of range", s.LR, s.WeightDecay, s.Momentum, s.Eps)
	}
	if b := s.sharedBytes(); b > maxDynamicShared {
		return s, fmt.Errorf("ttt schedule: %d bytes of shared memory exceed %d", b, maxDynamicShared)
	}
	return s, nil
}

// Validate reports whether s describes a kernel this package can emit.
func (s TTTSchedule) Validate() error {
	_, err := s.norm()
	return err
}

func (s TTTSchedule) columns() int {
	c := s.HeadDim + 2*s.ChunkLen
	if s.Variant == TTTTitansMomentum {
		c += s.HeadDim
	}
	return c
}

// TMEMColumns returns the TMEM allocation width of the kernel.
func (s TTTSchedule) TMEMColumns() uint32 { return allocColumns(s.columns()) }

func (s TTTSchedule) sharedBytes() int {
	d, l := s.HeadDim, s.ChunkLen
	loop := 3*l*d + 128*l
	return 4 * max(loop, 128*d+d*d)
}

// SharedBytes returns the dynamic shared memory the kernel needs, or 0 if s
// is invalid.
func (s TTTSchedule) SharedBytes() int {
	s, err := s.norm()
	if err != nil {
		return 0
	}
	return s.sharedBytes()
}

// SeqLen returns ChunkLen * NumChunks.
func (s TTTSchedule) SeqLen() int { return s.ChunkLen * s.NumChunks }

// LayerNormL2VJP returns ℓ = ½‖LN(z) - v‖² and g = ∂ℓ/∂z for LN without
// affine parameters: with ẑ = (z - μ)/σ, σ = √(var(z) + eps) and r = ẑ - v,
//
//	g = (r - mean(r) - ẑ mean(r ⊙ ẑ)) / σ.
func LayerNormL2VJP(z, v []float64, eps float64) (loss float64, g []float64) {
	d := float64(len(z))
	var mu float64
	for _, x := range z {
		mu += x
	}
	mu /= d
	var vr float64
	for _, x := range z {
		vr += (x - mu) * (x - mu)
	}
	sigma := math.Sqrt(vr/d + eps)
	zh := make([]float64, len(z))
	r := make([]float64, len(z))
	var mr, mrz float64
	for i, x := range z {
		zh[i] = (x - mu) / sigma
		r[i] = zh[i] - v[i]
		loss += r[i] * r[i] / 2
		mr += r[i]
		mrz += r[i] * zh[i]
	}
	mr /= d
	mrz /= d
	g = make([]float64, len(z))
	for i := range g {
		g[i] = (r[i] - mr - zh[i]*mrz) / sigma
	}
	return loss, g
}

// Reference runs one head in float64. w0 is [d][d], q, k, v are [T][d].
// It returns the outputs [T][d], the final W, and each chunk's mean token
// loss computed before that chunk's update.
func (s TTTSchedule) Reference(w0, q, k, v [][]float32) (out, w [][]float32, chunkLoss []float64, err error) {
	if s, err = s.norm(); err != nil {
		return nil, nil, nil, err
	}
	d, l, t := s.HeadDim, s.ChunkLen, s.SeqLen()
	for _, c := range []struct {
		name string
		x    [][]float32
		rows int
	}{{"w0", w0, d}, {"q", q, t}, {"k", k, t}, {"v", v, t}} {
		if len(c.x) != c.rows {
			return nil, nil, nil, fmt.Errorf("ttt reference: %s has %d rows, want %d", c.name, len(c.x), c.rows)
		}
		for i, r := range c.x {
			if len(r) != d {
				return nil, nil, nil, fmt.Errorf("ttt reference: %s[%d] has %d values, want %d", c.name, i, len(r), d)
			}
		}
	}
	W := make([][]float64, d)
	M := make([][]float64, d)
	for i := range W {
		W[i] = make([]float64, d)
		M[i] = make([]float64, d)
		for j := range W[i] {
			W[i][j] = float64(w0[i][j])
		}
	}
	mul := func(x []float32) []float64 {
		y := make([]float64, d)
		for i := range y {
			for j := range x {
				y[i] += W[i][j] * float64(x[j])
			}
		}
		return y
	}
	eta, alpha, mu := float64(s.LR), float64(s.WeightDecay), float64(s.Momentum)
	out = make([][]float32, t)
	G := make([][]float64, d)
	for i := range G {
		G[i] = make([]float64, d)
	}
	vt := make([]float64, d)
	for c := range s.NumChunks {
		for i := range G {
			clear(G[i])
		}
		var loss float64
		for tt := c * l; tt < (c+1)*l; tt++ {
			for i := range vt {
				vt[i] = float64(v[tt][i])
			}
			lt, g := LayerNormL2VJP(mul(k[tt]), vt, float64(s.Eps))
			loss += lt
			for i := range G {
				for j := range G[i] {
					G[i][j] += g[i] * float64(k[tt][j])
				}
			}
		}
		chunkLoss = append(chunkLoss, loss/float64(l))
		for i := range W {
			for j := range W[i] {
				switch s.Variant {
				case TTTLinearSGD:
					W[i][j] = (1-alpha)*W[i][j] - eta*G[i][j]
				case TTTTitansMomentum:
					M[i][j] = mu*M[i][j] - eta*G[i][j]
					W[i][j] = (1-alpha)*W[i][j] + M[i][j]
				}
			}
		}
		for tt := c * l; tt < (c+1)*l; tt++ {
			o := mul(q[tt])
			out[tt] = make([]float32, d)
			for i := range o {
				out[tt][i] = float32(o[i])
			}
		}
	}
	return out, to32(W), chunkLoss, nil
}

// TraceEvents returns the tcgen05 operation sequence of one head, for
// cudatmem.VerifyTrace.
func (s TTTSchedule) TraceEvents() []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	d, l := uint32(s.HeadDim), uint32(s.ChunkLen)
	cW, cZ, cO, cM := uint32(0), d, d+l, d+2*l
	titans := s.Variant == TTTTitansMomentum
	tr := newTMTrace(s.TMEMColumns())
	tr.mma(cW, d, false, 0)
	tr.commitWait()
	for c := range s.NumChunks {
		tr.mma(cZ, l, true, cW)
		tr.commitWait()
		tr.rows(cudatmem.EventLd, cZ, l, int(d))
		if titans {
			if c > 0 {
				tr.rows(cudatmem.EventLd, cM, d, int(d))
				tr.rows(cudatmem.EventSt, cM, d, int(d))
			}
			tr.mma(cM, d, false, 0)
			tr.commitWait()
			tr.rows(cudatmem.EventLd, cM, d, int(d))
			tr.rows(cudatmem.EventLd, cW, d, int(d))
			tr.rows(cudatmem.EventSt, cW, d, int(d))
		} else {
			tr.rows(cudatmem.EventLd, cW, d, int(d))
			tr.rows(cudatmem.EventSt, cW, d, int(d))
			tr.mma(cW, d, false, 0)
			tr.commitWait()
		}
		tr.mma(cO, l, true, cW)
		tr.commitWait()
		tr.rows(cudatmem.EventLd, cO, l, int(d))
	}
	tr.rows(cudatmem.EventLd, cW, d, int(d))
	return tr.done()
}

// Grid returns the launch shape for heads heads on at most ctas resident
// CTAs (ctas <= 0 means one CTA per head).
func (s TTTSchedule) Grid(heads, ctas int) (grid, block [3]int) {
	if ctas <= 0 || ctas > heads {
		ctas = heads
	}
	return [3]int{max(ctas, 1), 1, 1}, [3]int{128, 1, 1}
}

// CUDA returns the kernel source without [Prelude]. The kernel signature is
//
//	name(int heads, const float* w0, const float* q, const float* k,
//	     const float* v, float* out, float* w, float* loss)
//
// with w0 and w [heads][d][d], q, k, v, out [heads][T][d], and loss
// [heads][T] receiving each token's ℓ_t before its chunk's update.
func (s TTTSchedule) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	titans := 0
	if s.Variant == TTTTitansMomentum {
		titans = 1
	}
	var b strings.Builder
	b.WriteString(tmPrelude)
	fmt.Fprintf(&b, `
// %[1]s: %[2]s test-time training with fast weights resident in TMEM
// (d=%[3]d, L=%[4]d, %[5]d chunks, target %[6]s).
__global__ void __launch_bounds__(128) %[1]s(
    int heads, const float* __restrict__ w0,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, float* __restrict__ w, float* __restrict__ loss) {
    constexpr int D = %[3]d, L = %[4]d, NC = %[5]d, TITANS = %[7]d;
    constexpr int T = L * NC;
    constexpr uint32_t COLS = %[8]d, C_W = 0, C_Z = D, C_O = D + L, C_M = D + 2 * L;
    constexpr float ETA = %[9]s, KEEP = %[10]s, MU = %[11]s, EPS = %[12]s;
    const int tid = threadIdx.x, warp = tid >> 5;
    EG_TM_BUF(tm);
    EG_TM_SHARED(uint32_t, slot, [1]);
    EG_TM_SHARED(uint64_t, mbar, [1]);
    EG_DYNAMIC_SMEM(float, sm);
    constexpr int SM_FLOATS = %[13]d;
    float* const kq = sm;           // K [L][D]: B of z = W k
    float* const qs = kq + L * D;   // Q [L][D]: B of o = W q
    float* const kt = qs + L * D;   // Kᵀ [D][L]: B of the update
    float* const gt = kt + D * L;   // z, then -η Gᵀ [128][L], rows >= D zero: A of the update
    uint32_t phase = 0;
    if (tid == 0) eg_tm_mbar_init(&mbar[0]);
    if (warp == 0) eg_tm_alloc(tm, &slot[0], COLS);
    eg_tm_sync();
    const uint32_t base = slot[0];

    for (int h = blockIdx.x; h < heads; h += gridDim.x) {
        // W = W_0 I: one MMA puts the initial fast weights in TMEM.
        float* const w0p = sm;          // [128][D], rows >= D zero
        float* const eye = sm + 128 * D; // [D][D]
        for (int i = tid; i < SM_FLOATS; i += 128) sm[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < D * D; i += 128) {
            w0p[eg_cm_idx(i / D, i %% D, D)] = w0[(size_t)h * D * D + i];
            if (i / D == i %% D) eye[eg_cm_idx(i / D, i %% D, D)] = 1.0f;
        }
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base + C_W, w0p, eye, D, D, 0);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        __syncthreads();
        for (int i = tid; i < SM_FLOATS; i += 128) sm[i] = 0.0f;
        __syncthreads();

        for (int c = 0; c < NC; ++c) {
            const size_t r0 = (size_t)h * T + (size_t)c * L;
            if (tid < L) {
                for (int i = 0; i < D; ++i) {
                    const float kv = k[(r0 + tid) * D + i];
                    kq[eg_cm_idx(tid, i, D)] = kv;
                    kt[eg_cm_idx(i, tid, L)] = kv;
                    qs[eg_cm_idx(tid, i, D)] = q[(r0 + tid) * D + i];
                }
            }
            eg_tm_sync();
            // 1. zᵀ = W Kᵀ.
            if (tid == 0) {
                eg_tm_mma_ts(tm, base + C_Z, base + C_W, kq, L, D, 0);
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);
            if (tid < D) {
                for (int t = 0; t < L; ++t) gt[eg_cm_idx(tid, t, L)] = eg_tm_ld(tm, tid, base + C_Z + t);
            }
            __syncthreads();
            // 2. LayerNorm + L2 loss and its exact VJP, one token per thread.
            if (tid < L) {
                const float* const vt = v + (r0 + tid) * D;
                float mu = 0.0f, var = 0.0f;
                for (int j = 0; j < D; ++j) mu += gt[eg_cm_idx(j, tid, L)];
                mu /= D;
                for (int j = 0; j < D; ++j) {
                    const float x = gt[eg_cm_idx(j, tid, L)] - mu;
                    var += x * x;
                }
                const float rsig = 1.0f / sqrtf(var / D + EPS);
                float mr = 0.0f, mrz = 0.0f, l2 = 0.0f;
                for (int j = 0; j < D; ++j) {
                    const float zh = (gt[eg_cm_idx(j, tid, L)] - mu) * rsig;
                    const float r = zh - vt[j];
                    mr += r;
                    mrz += r * zh;
                    l2 += r * r;
                }
                mr /= D;
                mrz /= D;
                loss[r0 + tid] = 0.5f * l2;
                for (int j = 0; j < D; ++j) {
                    const float zh = (gt[eg_cm_idx(j, tid, L)] - mu) * rsig;
                    const float g = (zh - vt[j] - mr - zh * mrz) * rsig;
                    gt[eg_cm_idx(j, tid, L)] = -ETA * g;
                }
            }
            // 3. Decay the resident state in TMEM.
            if (tid < D) {
                if (TITANS) {
                    if (c > 0) for (int i = 0; i < D; ++i) eg_tm_st(tm, tid, base + C_M + i, MU * eg_tm_ld(tm, tid, base + C_M + i));
                } else {
                    for (int i = 0; i < D; ++i) eg_tm_st(tm, tid, base + C_W + i, KEEP * eg_tm_ld(tm, tid, base + C_W + i));
                }
            }
            eg_tm_sync();
            // 4. Accumulate -η G onto W (or M) with enable_input_d = 1.
            if (tid == 0) {
                if (TITANS) eg_tm_mma_ss(tm, base + C_M, gt, kt, D, L, c > 0);
                else eg_tm_mma_ss(tm, base + C_W, gt, kt, D, L, 1);
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);
            if (TITANS) {
                if (tid < D) {
                    for (int i = 0; i < D; ++i) {
                        eg_tm_st(tm, tid, base + C_W + i, KEEP * eg_tm_ld(tm, tid, base + C_W + i) + eg_tm_ld(tm, tid, base + C_M + i));
                    }
                }
                eg_tm_sync();
            }
            // 5. oᵀ = W Qᵀ with the updated weights.
            if (tid == 0) {
                eg_tm_mma_ts(tm, base + C_O, base + C_W, qs, L, D, 0);
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);
            if (tid < D) {
                for (int t = 0; t < L; ++t) out[(r0 + t) * D + tid] = eg_tm_ld(tm, tid, base + C_O + t);
            }
            __syncthreads();
        }
        if (tid < D) {
            for (int i = 0; i < D; ++i) w[((size_t)h * D + tid) * D + i] = eg_tm_ld(tm, tid, base + C_W + i);
        }
        eg_tm_sync();
    }
    if (warp == 0) eg_tm_dealloc(base, COLS);
}
`, s.Name, s.Variant, s.HeadDim, s.ChunkLen, s.NumChunks, s.Target, titans, s.TMEMColumns(),
		floatLiteral(s.LR), floatLiteral(1-s.WeightDecay), floatLiteral(s.Momentum), floatLiteral(s.Eps), s.sharedBytes()/4)
	return b.String(), nil
}

// CUDAFile returns [Prelude] followed by [TTTSchedule.CUDA].
func (s TTTSchedule) CUDAFile() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}
