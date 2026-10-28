package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// An SSMVariant selects the linear recurrence an [SSMChunkSchedule] runs.
// S_t is the d_v x d_k state; o_t = S_t q_t.
type SSMVariant uint8

const (
	// SSMMamba2SSD is the Mamba-2 state-space dual recurrence
	// S_t = alpha_t S_{t-1} + v_t k_tᵀ.
	SSMMamba2SSD SSMVariant = iota
	// SSMGatedDeltaNet is the gated delta rule
	// S_t = alpha_t S_{t-1} (I - beta_t k_t k_tᵀ) + beta_t v_t k_tᵀ.
	SSMGatedDeltaNet
)

func (v SSMVariant) String() string {
	switch v {
	case SSMMamba2SSD:
		return "mamba2"
	case SSMGatedDeltaNet:
		return "gated_deltanet"
	}
	return fmt.Sprintf("SSMVariant(%d)", uint8(v))
}

// An SSMChunkSchedule configures a chunkwise linear-recurrence kernel that
// keeps the FP32 state S in TMEM across all chunks of a sequence, one head
// per CTA iteration of a persistent loop.
//
// Per chunk of L tokens the kernel stages Q, K and V (after an optional
// causal depthwise convolution computed with tcgen05.shift.down) in shared
// memory, and with cumulative in-chunk decays g_t computes
//
//	O = diag(g) Q Sᵀ + (Γ ⊙ Q Kᵀ) U          (inter + intra chunk)
//	S = g_L S + Uᵀ diag(g_L / g) K            (state update, enable_input_d = 1)
//
// where U = V for Mamba-2 and, for the gated delta rule, U solves the
// unit lower-triangular system (I + diag(β) Γ⁻ ⊙ K Kᵀ) U = diag(β)(V - diag(g) K Sᵀ)
// by forward substitution. S is the TMEM A operand of the Q Sᵀ and K Sᵀ
// MMAs; the decays scale S and O in place with tcgen05.ld/st.
//
// TMEM layout (lanes are state rows d_v): S at columns [0, HeadDimK), the
// K Kᵀ / Q Kᵀ scratch at [HeadDimK, +L), K Sᵀ at [HeadDimK+L, +L), O at
// [HeadDimK+2L, +L). The convolution reuses the two scratch ranges.
//
// The sm_100a/sm_103a/sm_107a path of the emitted source is compiled by the
// native qualification test but has not been executed; the EG_EMU path is
// what the tests run (cudahostemu on the host, and sm_121a under
// CUDA_TCGEN05_NATIVE=1).
type SSMChunkSchedule struct {
	Name       string
	Target     string // sm_100a, sm_103a, sm_107a or an f variant (default sm_100a)
	Variant    SSMVariant
	HeadDimK   int // d_k: 32, 64, 96 or 128
	HeadDimV   int // d_v: 32, 64, 96 or 128
	ChunkLen   int // L: 32, 64 or 128 tokens per chunk
	NumChunks  int // chunks per sequence
	ConvKernel int // depthwise causal conv1d width over q, k, v (0 disables it, at most 4)
}

func (s SSMChunkSchedule) norm() (SSMChunkSchedule, error) {
	if s.Name == "" {
		s.Name = "ssm_" + s.Variant.String()
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("ssm schedule: bad name %q", s.Name)
	}
	if err := checkTMTarget("ssm schedule", s.Target); err != nil {
		return s, err
	}
	if s.Variant != SSMMamba2SSD && s.Variant != SSMGatedDeltaNet {
		return s, fmt.Errorf("ssm schedule: unknown variant %v", s.Variant)
	}
	for _, d := range []int{s.HeadDimK, s.HeadDimV} {
		if d < 32 || d > 128 || d%32 != 0 {
			return s, fmt.Errorf("ssm schedule: head dims %d, %d must be multiples of 32 in [32, 128]", s.HeadDimK, s.HeadDimV)
		}
	}
	if s.ChunkLen != 32 && s.ChunkLen != 64 && s.ChunkLen != 128 {
		return s, fmt.Errorf("ssm schedule: chunk length %d must be 32, 64 or 128", s.ChunkLen)
	}
	if s.NumChunks < 1 {
		return s, fmt.Errorf("ssm schedule: %d chunks", s.NumChunks)
	}
	if s.ConvKernel < 0 || s.ConvKernel > 4 {
		return s, fmt.Errorf("ssm schedule: conv width %d not in [0, 4]", s.ConvKernel)
	}
	if s.ConvKernel > 0 && max(s.HeadDimK, s.HeadDimV) > 2*s.ChunkLen {
		return s, fmt.Errorf("ssm schedule: conv staging needs head dims <= 2*chunk length")
	}
	if c := s.columns(); c > 512 {
		return s, fmt.Errorf("ssm schedule: %d TMEM columns exceed 512", c)
	}
	if b := s.sharedBytes(); b > maxDynamicShared {
		return s, fmt.Errorf("ssm schedule: %d bytes of shared memory exceed %d", b, maxDynamicShared)
	}
	return s, nil
}

// Validate reports whether s describes a kernel this package can emit.
func (s SSMChunkSchedule) Validate() error {
	_, err := s.norm()
	return err
}

func (s SSMChunkSchedule) columns() int { return s.HeadDimK + 3*s.ChunkLen }

func (s SSMChunkSchedule) sharedBytes() int {
	l, dk := s.ChunkLen, s.HeadDimK
	return 4 * (128*dk + l*dk + l*l + 128*l)
}

// TMEMColumns returns the TMEM allocation width of the kernel.
func (s SSMChunkSchedule) TMEMColumns() uint32 { return allocColumns(s.columns()) }

// SharedBytes returns the dynamic shared memory the kernel needs, or 0 if s
// is invalid.
func (s SSMChunkSchedule) SharedBytes() int {
	s, err := s.norm()
	if err != nil {
		return 0
	}
	return s.sharedBytes()
}

// SeqLen returns ChunkLen * NumChunks.
func (s SSMChunkSchedule) SeqLen() int { return s.ChunkLen * s.NumChunks }

// An SSMInput holds one head's sequence for [SSMChunkSchedule.Reference].
type SSMInput struct {
	Q, K         [][]float32 // [T][HeadDimK]
	V            [][]float32 // [T][HeadDimV]
	Alpha        []float32   // [T] decay in (0, 1]
	Beta         []float32   // [T] delta-rule write strength; unused by SSMMamba2SSD
	ConvQ, ConvK [][]float32 // [ConvKernel][HeadDimK] depthwise taps; tap j weights x_{t-j}
	ConvV        [][]float32 // [ConvKernel][HeadDimV]
}

func (s SSMChunkSchedule) check(in SSMInput) error {
	t := s.SeqLen()
	shape := func(name string, x [][]float32, rows, cols int) error {
		if len(x) != rows {
			return fmt.Errorf("ssm reference: %s has %d rows, want %d", name, len(x), rows)
		}
		for i, r := range x {
			if len(r) != cols {
				return fmt.Errorf("ssm reference: %s[%d] has %d values, want %d", name, i, len(r), cols)
			}
		}
		return nil
	}
	for _, c := range []struct {
		name       string
		x          [][]float32
		rows, cols int
	}{
		{"q", in.Q, t, s.HeadDimK}, {"k", in.K, t, s.HeadDimK}, {"v", in.V, t, s.HeadDimV},
	} {
		if err := shape(c.name, c.x, c.rows, c.cols); err != nil {
			return err
		}
	}
	if s.ConvKernel > 0 {
		for _, c := range []struct {
			name string
			x    [][]float32
			cols int
		}{{"conv q", in.ConvQ, s.HeadDimK}, {"conv k", in.ConvK, s.HeadDimK}, {"conv v", in.ConvV, s.HeadDimV}} {
			if err := shape(c.name, c.x, s.ConvKernel, c.cols); err != nil {
				return err
			}
		}
	}
	if len(in.Alpha) != t {
		return fmt.Errorf("ssm reference: %d decays, want %d", len(in.Alpha), t)
	}
	for i, a := range in.Alpha {
		if !(a > 0 && a <= 1) {
			return fmt.Errorf("ssm reference: alpha[%d] = %g not in (0, 1]", i, a)
		}
	}
	if s.Variant == SSMGatedDeltaNet && len(in.Beta) != t {
		return fmt.Errorf("ssm reference: %d betas, want %d", len(in.Beta), t)
	}
	return nil
}

// conv returns the causal depthwise convolution of x with taps w in float64.
func conv(x, w [][]float32) [][]float64 {
	out := make([][]float64, len(x))
	for t := range x {
		out[t] = make([]float64, len(x[t]))
		for c := range x[t] {
			if len(w) == 0 {
				out[t][c] = float64(x[t][c])
				continue
			}
			var acc float64
			for j := range w {
				if t-j >= 0 {
					acc += float64(w[j][c]) * float64(x[t-j][c])
				}
			}
			out[t][c] = acc
		}
	}
	return out
}

// Reference runs the token-by-token recurrence in float64 and returns the
// outputs o_t [T][HeadDimV] and the final state S [HeadDimV][HeadDimK].
func (s SSMChunkSchedule) Reference(in SSMInput) (out, state [][]float32, err error) {
	if s, err = s.norm(); err != nil {
		return nil, nil, err
	}
	if err := s.check(in); err != nil {
		return nil, nil, err
	}
	q, k, v := conv(in.Q, in.ConvQ), conv(in.K, in.ConvK), conv(in.V, in.ConvV)
	dk, dv := s.HeadDimK, s.HeadDimV
	S := make([][]float64, dv)
	for i := range S {
		S[i] = make([]float64, dk)
	}
	out = make([][]float32, s.SeqLen())
	sk := make([]float64, dv)
	for t := range out {
		a := float64(in.Alpha[t])
		switch s.Variant {
		case SSMMamba2SSD:
			for i := range S {
				for j := range S[i] {
					S[i][j] = a*S[i][j] + v[t][i]*k[t][j]
				}
			}
		case SSMGatedDeltaNet:
			b := float64(in.Beta[t])
			for i := range S {
				sk[i] = 0
				for j := range S[i] {
					sk[i] += S[i][j] * k[t][j]
				}
			}
			for i := range S {
				for j := range S[i] {
					S[i][j] = a*(S[i][j]-b*sk[i]*k[t][j]) + b*v[t][i]*k[t][j]
				}
			}
		}
		out[t] = make([]float32, dv)
		for i := range S {
			var o float64
			for j := range S[i] {
				o += S[i][j] * q[t][j]
			}
			out[t][i] = float32(o)
		}
	}
	return out, to32(S), nil
}

func to32(x [][]float64) [][]float32 {
	out := make([][]float32, len(x))
	for i := range x {
		out[i] = make([]float32, len(x[i]))
		for j, v := range x[i] {
			out[i][j] = float32(v)
		}
	}
	return out
}

// ReferenceChunked computes the same result as [SSMChunkSchedule.Reference]
// in float64 with the chunkwise algorithm the kernel uses: per chunk an
// inter-chunk term from the carried state, a masked intra-chunk term, the
// delta-rule triangular solve, and one rank-L state update.
func (s SSMChunkSchedule) ReferenceChunked(in SSMInput) (out, state [][]float32, err error) {
	if s, err = s.norm(); err != nil {
		return nil, nil, err
	}
	if err := s.check(in); err != nil {
		return nil, nil, err
	}
	q, k, v := conv(in.Q, in.ConvQ), conv(in.K, in.ConvK), conv(in.V, in.ConvV)
	dk, dv, l := s.HeadDimK, s.HeadDimV, s.ChunkLen
	S := make([][]float64, dv)
	for i := range S {
		S[i] = make([]float64, dk)
	}
	dot := func(a, b []float64) float64 {
		var r float64
		for i := range a {
			r += a[i] * b[i]
		}
		return r
	}
	out = make([][]float32, s.SeqLen())
	lg := make([]float64, l)
	u := make([][]float64, l)
	for c := range s.NumChunks {
		t0 := c * l
		var acc float64
		for t := range l {
			acc += math.Log(float64(in.Alpha[t0+t]))
			lg[t] = acc
		}
		// U = V, or the delta-rule solve against the incoming state.
		for t := range l {
			u[t] = make([]float64, dv)
			copy(u[t], v[t0+t])
			if s.Variant != SSMGatedDeltaNet {
				continue
			}
			b := float64(in.Beta[t0+t])
			g := math.Exp(lg[t])
			for i := range u[t] {
				u[t][i] = b * (u[t][i] - g*dot(S[i], k[t0+t]))
			}
			for p := range t {
				a := b * math.Exp(lg[t]-lg[p]) * dot(k[t0+t], k[t0+p])
				for i := range u[t] {
					u[t][i] -= a * u[p][i]
				}
			}
		}
		for t := range l {
			o := make([]float64, dv)
			g := math.Exp(lg[t])
			for i := range o {
				o[i] = g * dot(S[i], q[t0+t])
			}
			for p := 0; p <= t; p++ {
				w := math.Exp(lg[t]-lg[p]) * dot(q[t0+t], k[t0+p])
				for i := range o {
					o[i] += w * u[p][i]
				}
			}
			out[t0+t] = make([]float32, dv)
			for i := range o {
				out[t0+t][i] = float32(o[i])
			}
		}
		gl := math.Exp(lg[l-1])
		for i := range S {
			for j := range S[i] {
				x := gl * S[i][j]
				for p := range l {
					x += u[p][i] * math.Exp(lg[l-1]-lg[p]) * k[t0+p][j]
				}
				S[i][j] = x
			}
		}
	}
	return out, to32(S), nil
}

// TraceEvents returns the tcgen05 operation sequence of one head, for
// cudatmem.VerifyTrace.
func (s SSMChunkSchedule) TraceEvents() []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	dk, dv, l := uint32(s.HeadDimK), uint32(s.HeadDimV), uint32(s.ChunkLen)
	cS, cA, cY, cO := uint32(0), dk, dk+l, dk+2*l
	delta := s.Variant == SSMGatedDeltaNet
	tr := newTMTrace(s.TMEMColumns())
	for c := range s.NumChunks {
		if s.ConvKernel > 0 {
			for _, d := range []uint32{dk, dk, dv} {
				d8 := (d + 7) &^ 7
				tr.rows(cudatmem.EventSt, cA, d, int(l))
				tr.rows(cudatmem.EventLd, cA, d, int(l))
				for range s.ConvKernel - 1 {
					tr.shift(cA, d8)
					tr.commitWait()
					tr.rows(cudatmem.EventLd, cA, d, int(l))
					tr.rows(cudatmem.EventSt, cA, d, 32)
				}
			}
		}
		if delta {
			tr.mma(cA, l, false, 0)
			if c > 0 {
				tr.mma(cO, l, true, cS)
				tr.mma(cY, l, true, cS)
			}
			tr.commitWait()
			tr.rows(cudatmem.EventLd, cA, l, int(l))
			tr.mma(cA, l, false, 0)
			if c > 0 {
				tr.rows(cudatmem.EventLd, cY, l, int(dv))
			}
			tr.commitWait()
		} else {
			tr.mma(cA, l, false, 0)
			if c > 0 {
				tr.mma(cO, l, true, cS)
			}
			tr.commitWait()
		}
		tr.rows(cudatmem.EventLd, cA, l, int(l))
		if c > 0 {
			tr.rows(cudatmem.EventLd, cO, l, int(dv))
			tr.rows(cudatmem.EventSt, cO, l, int(dv))
			tr.rows(cudatmem.EventLd, cS, dk, int(dv))
			tr.rows(cudatmem.EventSt, cS, dk, int(dv))
		}
		tr.mma(cO, l, false, 0)
		tr.mma(cS, dk, false, 0)
		tr.commitWait()
		tr.rows(cudatmem.EventLd, cO, l, int(dv))
	}
	tr.rows(cudatmem.EventLd, cS, dk, int(dv))
	return tr.done()
}

// Grid returns the launch shape for heads heads on at most ctas resident
// CTAs (ctas <= 0 means one CTA per head).
func (s SSMChunkSchedule) Grid(heads, ctas int) (grid, block [3]int) {
	if ctas <= 0 || ctas > heads {
		ctas = heads
	}
	return [3]int{max(ctas, 1), 1, 1}, [3]int{128, 1, 1}
}

// CUDA returns the kernel source without [Prelude]. The kernel signature is
//
//	name(int heads, const float* q, const float* k, const float* v,
//	     const float* alpha, const float* beta,
//	     const float* conv_q, const float* conv_k, const float* conv_v,
//	     float* out, float* state)
//
// with q, k [heads][T][d_k], v and out [heads][T][d_v], alpha and beta
// [heads][T], conv_x [heads][ConvKernel][d_x], and state [heads][d_v][d_k].
// beta and conv_x may be null when unused.
func (s SSMChunkSchedule) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	delta := 0
	if s.Variant == SSMGatedDeltaNet {
		delta = 1
	}
	var b strings.Builder
	b.WriteString(tmPrelude)
	fmt.Fprintf(&b, `
// %[1]s: %[2]s chunkwise recurrence with the FP32 state resident in TMEM
// (d_k=%[3]d, d_v=%[4]d, L=%[5]d, %[6]d chunks, conv width %[7]d, target %[8]s).
__global__ void __launch_bounds__(128) %[1]s(
    int heads,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    const float* __restrict__ conv_q, const float* __restrict__ conv_k, const float* __restrict__ conv_v,
    float* __restrict__ out, float* __restrict__ state) {
    constexpr int DK = %[3]d, DV = %[4]d, L = %[5]d, NC = %[6]d, CW = %[7]d, DELTA = %[9]d;
    constexpr int T = L * NC;
    constexpr uint32_t COLS = %[10]d, C_S = 0, C_A = DK, C_Y = DK + L, C_O = DK + 2 * L;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    EG_TM_BUF(tm);
    EG_TM_SHARED(uint32_t, slot, [1]);
    EG_TM_SHARED(uint64_t, mbar, [1]);
    EG_TM_SHARED(float, lg, [L]);
    EG_TM_SHARED(float, bt, [L]);
    EG_DYNAMIC_SMEM(float, sm);
    float* const kp = sm;            // K [128][DK], rows >= L zero: A of K Kᵀ and P; B of K Kᵀ and K Sᵀ
    float* const qs = kp + 128 * DK; // Q [L][DK]: B of P and Q Sᵀ; then K~ [DK][L]: B of the state update
    float* const ws = qs + L * DK;   // delta-rule solve matrix [L][L] row-major; then W [L][L]
    float* const ut = ws + L * L;    // Vᵀ, then Uᵀ [128][L], rows >= DV zero: A of the O and S updates
    uint32_t phase = 0;
    for (int i = tid; i < 128 * DK + L * DK + L * L + 128 * L; i += 128) sm[i] = 0.0f;
    if (tid == 0) eg_tm_mbar_init(&mbar[0]);
    if (warp == 0) eg_tm_alloc(tm, &slot[0], COLS);
    eg_tm_sync();
    const uint32_t base = slot[0];

    // stage writes the causally convolved rows [t0, t0 + L) of x [T][D]
    // through put(t, ch, tap product, tap). With CW > 0 the rows sit in TMEM lanes and
    // each tap shifts them down one lane; lane 0 takes the halo row t0 - j.
    auto stage = [&](const float* x, const float* w, int D, int t0, auto put) {
        if (CW == 0) {
            if (tid < L) for (int ch = 0; ch < D; ++ch) put(tid, ch, x[(t0 + tid) * D + ch], 0);
            return;
        }
        if (tid < L) for (int ch = 0; ch < D; ++ch) eg_tm_st(tm, tid, base + C_A + ch, x[(t0 + tid) * D + ch]);
        for (int j = 0; j < CW; ++j) {
            if (j > 0) {
                eg_tm_sync();
                if (tid == 0) {
                    eg_tm_shift_down(tm, base + C_A, (D + 7) & ~7);
                    eg_tm_commit(&mbar[0]);
                }
                eg_tm_wait(&mbar[0], &phase);
            }
            if (tid < L) {
                for (int ch = 0; ch < D; ++ch) {
                    float xv = eg_tm_ld(tm, tid, base + C_A + ch);
                    if (j > 0 && warp == 0) {
                        if (lane == 0) xv = t0 - j >= 0 ? x[(t0 - j) * D + ch] : 0.0f;
                        eg_tm_st(tm, tid, base + C_A + ch, xv);
                    }
                    put(tid, ch, w[j * D + ch] * xv, j);
                }
            }
        }
    };

    for (int h = blockIdx.x; h < heads; h += gridDim.x) {
        const float* const qh = q + (size_t)h * T * DK;
        const float* const kh = k + (size_t)h * T * DK;
        const float* const vh = v + (size_t)h * T * DV;
        const float* const ah = alpha + (size_t)h * T;
        for (int c = 0; c < NC; ++c) {
            const int t0 = c * L;
            // 1. Q, K, V operands.
            stage(kh, conv_k + (size_t)h * CW * DK, DK, t0, [&](int t, int ch, float y, int j) {
                float& d = kp[eg_cm_idx(t, ch, DK)];
                d = j == 0 ? y : d + y;
            });
            stage(qh, conv_q + (size_t)h * CW * DK, DK, t0, [&](int t, int ch, float y, int j) {
                float& d = qs[eg_cm_idx(t, ch, DK)];
                d = j == 0 ? y : d + y;
            });
            stage(vh, conv_v + (size_t)h * CW * DV, DV, t0, [&](int t, int ch, float y, int j) {
                float& d = ut[eg_cm_idx(ch, t, L)];
                d = j == 0 ? y : d + y;
            });
            if (tid < L) {
                float acc = 0.0f;
                for (int i = 0; i <= tid; ++i) acc += logf(ah[t0 + i]);
                lg[tid] = acc;
                bt[tid] = DELTA ? beta[(size_t)h * T + t0 + tid] : 1.0f;
            }
            eg_tm_sync();

            // 2. K Kᵀ (delta rule) or Q Kᵀ, and the inter-chunk products of the carried state.
            if (tid == 0) {
                if (DELTA) eg_tm_mma_ss(tm, base + C_A, kp, kp, L, DK, 0);
                else eg_tm_mma_ss(tm, base + C_A, kp, qs, L, DK, 0);
                if (c > 0) {
                    eg_tm_mma_ts(tm, base + C_O, base + C_S, qs, L, DK, 0);
                    if (DELTA) eg_tm_mma_ts(tm, base + C_Y, base + C_S, kp, L, DK, 0);
                }
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);

            if (DELTA) {
                // 3. Solve (I + A) U = diag(β)(V - diag(g) K Sᵀ) with
                // A[t][s] = β_t g_t/g_s k_t·k_s for s < t, one state row per thread.
                if (tid < L) {
                    for (int s = 0; s < L; ++s) {
                        const float kk = eg_tm_ld(tm, tid, base + C_A + s);
                        ws[tid * L + s] = s < tid ? bt[tid] * expf(lg[tid] - lg[s]) * kk : 0.0f;
                    }
                }
                eg_tm_sync();
                if (tid == 0) {
                    eg_tm_mma_ss(tm, base + C_A, kp, qs, L, DK, 0);
                    eg_tm_commit(&mbar[0]);
                }
                if (tid < DV) {
                    for (int t = 0; t < L; ++t) {
                        float r = ut[eg_cm_idx(tid, t, L)];
                        if (c > 0) r -= expf(lg[t]) * eg_tm_ld(tm, tid, base + C_Y + t);
                        r *= bt[t];
                        for (int s = 0; s < t; ++s) r -= ws[t * L + s] * ut[eg_cm_idx(tid, s, L)];
                        ut[eg_cm_idx(tid, t, L)] = r;
                    }
                }
                eg_tm_wait(&mbar[0], &phase);
                __syncthreads();
            }

            // 4. W[t][s] = g_t/g_s q_t·k_s (s <= t) from P[s][t] = k_s·q_t, and
            // K~[i][s] = g_L/g_s k_s[i]; decay O and S in TMEM.
            if (tid < L) {
                for (int t = 0; t < L; ++t) {
                    const float p = eg_tm_ld(tm, tid, base + C_A + t);
                    ws[eg_cm_idx(t, tid, L)] = t >= tid ? expf(lg[t] - lg[tid]) * p : 0.0f;
                }
            }
            __syncthreads();
            if (tid < L) {
                const float gk = expf(lg[L - 1] - lg[tid]);
                for (int i = 0; i < DK; ++i) qs[eg_cm_idx(i, tid, L)] = gk * kp[eg_cm_idx(tid, i, DK)];
            }
            if (c > 0 && tid < DV) {
                for (int t = 0; t < L; ++t) eg_tm_st(tm, tid, base + C_O + t, expf(lg[t]) * eg_tm_ld(tm, tid, base + C_O + t));
                const float gl = expf(lg[L - 1]);
                for (int i = 0; i < DK; ++i) eg_tm_st(tm, tid, base + C_S + i, gl * eg_tm_ld(tm, tid, base + C_S + i));
            }
            eg_tm_sync();

            // 5. O += Uᵀ Wᵀ; S = g_L S + Uᵀ K~ᵀ accumulated onto the resident state.
            if (tid == 0) {
                eg_tm_mma_ss(tm, base + C_O, ut, ws, L, L, c > 0);
                eg_tm_mma_ss(tm, base + C_S, ut, qs, DK, L, c > 0);
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);
            if (tid < DV) {
                for (int t = 0; t < L; ++t) out[((size_t)h * T + t0 + t) * DV + tid] = eg_tm_ld(tm, tid, base + C_O + t);
            }
            __syncthreads();
        }
        if (tid < DV) {
            for (int i = 0; i < DK; ++i) state[((size_t)h * DV + tid) * DK + i] = eg_tm_ld(tm, tid, base + C_S + i);
        }
        eg_tm_sync();
    }
    if (warp == 0) eg_tm_dealloc(base, COLS);
}
`, s.Name, s.Variant, s.HeadDimK, s.HeadDimV, s.ChunkLen, s.NumChunks, s.ConvKernel, s.Target, delta, s.TMEMColumns())
	return b.String(), nil
}

// CUDAFile returns [Prelude] followed by [SSMChunkSchedule.CUDA].
func (s SSMChunkSchedule) CUDAFile() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}
