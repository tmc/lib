package tcgen05

import (
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// Philox4x32_10 returns the Philox4x32 counter-based random block for
// counter ctr and key key after 10 rounds (Salmon et al., SC 2011; the
// Random123 philox4x32_10 function).
func Philox4x32_10(ctr [4]uint32, key [2]uint32) [4]uint32 {
	const (
		m0, m1 = 0xD2511F53, 0xCD9E8D57
		w0, w1 = 0x9E3779B9, 0xBB67AE85
	)
	c, k := ctr, key
	for i := range 10 {
		if i > 0 {
			k[0] += w0
			k[1] += w1
		}
		hi0, lo0 := bits.Mul32(m0, c[0])
		hi1, lo1 := bits.Mul32(m1, c[2])
		c = [4]uint32{hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0}
	}
	return c
}

// FlashLogitsGumbel returns the Gumbel(0, 1) variate that the flash-logits
// kernel adds to the scaled logit of (row, token): -log(-log(u)) with
// u = (x + 0.5) / 2^32 and x the first word of
// Philox4x32_10({token, row, 0, 0}, {seed, seed >> 32}).
func FlashLogitsGumbel(seed uint64, row, token int) float64 {
	x := Philox4x32_10([4]uint32{uint32(token), uint32(row), 0, 0}, [2]uint32{uint32(seed), uint32(seed >> 32)})[0]
	u := (float64(x) + 0.5) / (1 << 32)
	return -math.Log(-math.Log(u))
}

// FlashLogitsSchedule configures a single-CTA LM head that streams the
// vocabulary through TMEM in TileV-column tiles and reduces each tile into
// per-row registers, so no [rows, VocabSize] logit array is ever written:
// for each of the TileM token rows it keeps the running max m and partition
// sum l of z/T (rescaling l by exp(m_old - m_new) when a tile raises the
// max), the Gumbel-max winner of z/T + G (G from [FlashLogitsGumbel]), and
// the scaled logit of an optional draft token.
//
// The emitted kernel computes each tile with tcgen05.mma.cta_group::1.kind::tf32
// over HiddenDim in 32-wide chunks, waiting on an mbarrier per chunk, into
// buffer t%2 of two TileV-column TMEM buffers when DoubleBuffer is set (one
// otherwise). It does not overlap the MMA of tile t+1 with the reduction of
// tile t: that needs a dedicated MMA warp and per-buffer barriers, which this
// emitter does not produce. The device path is emitted for sm_100a, sm_103a
// and sm_107a and has not been executed on tcgen05 hardware; under EG_EMU
// (cudahostemu and sm_12x devices) an exact C++ fallback runs instead.
// Speculative residual resampling is not implemented.
type FlashLogitsSchedule struct {
	Name         string
	Target       string  // "sm_100a", "sm_103a", "sm_107a" or "sm_107f" (default "sm_100a")
	TileM        int     // token rows per CTA; must be 128
	TileV        int     // vocabulary columns per TMEM tile; 32, 64 or 128 (default 64)
	VocabSize    int     // V >= 1
	HiddenDim    int     // D, a positive multiple of 32
	Temperature  float32 // T > 0
	Seed         uint64
	DoubleBuffer bool
}

func (s FlashLogitsSchedule) norm() (FlashLogitsSchedule, error) {
	if s.Name == "" {
		s.Name = "tcgen05_flash_logits"
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("tcgen05 flash logits: bad kernel name %q", s.Name)
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if s.TileM == 0 {
		s.TileM = 128
	}
	if s.TileV == 0 {
		s.TileV = 64
	}
	switch s.Target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_107f":
	default:
		return s, fmt.Errorf("tcgen05 flash logits %s: unsupported target %q", s.Name, s.Target)
	}
	if s.TileM != 128 {
		return s, fmt.Errorf("tcgen05 flash logits %s: TileM=%d, want 128", s.Name, s.TileM)
	}
	if s.TileV != 32 && s.TileV != 64 && s.TileV != 128 {
		return s, fmt.Errorf("tcgen05 flash logits %s: TileV=%d, want 32, 64 or 128", s.Name, s.TileV)
	}
	if s.VocabSize < 1 || s.HiddenDim < 32 || s.HiddenDim%32 != 0 {
		return s, fmt.Errorf("tcgen05 flash logits %s: VocabSize=%d HiddenDim=%d (want V >= 1, D a positive multiple of 32)", s.Name, s.VocabSize, s.HiddenDim)
	}
	if !(s.Temperature > 0) || math.IsInf(float64(s.Temperature), 0) {
		return s, fmt.Errorf("tcgen05 flash logits %s: Temperature=%v must be positive and finite", s.Name, s.Temperature)
	}
	return s, nil
}

// Validate reports whether s describes a supported kernel.
func (s FlashLogitsSchedule) Validate() error {
	_, err := s.norm()
	return err
}

// Tiles returns the number of vocabulary tiles, ceil(VocabSize/TileV).
func (s FlashLogitsSchedule) Tiles() int {
	tv := s.TileV
	if tv == 0 {
		tv = 64
	}
	return (s.VocabSize + tv - 1) / tv
}

func (s FlashLogitsSchedule) buffers() int {
	if s.DoubleBuffer {
		return 2
	}
	return 1
}

// FlashLogitsOutput holds the per-row results of the vocabulary stream.
type FlashLogitsOutput struct {
	SampledTokenID []int32   // Gumbel-max winner argmax_v z_v/T + G_v
	SampledLogProb []float32 // z_winner/T - LSE
	LSE            []float32 // log sum_v exp(z_v/T)
	DraftTokenProb []float32 // exp(z_draft/T - LSE), 0 when the draft token is out of range
}

// flashRow is the online state of one row: running max and partition sum
// of z/T, Gumbel-max winner, and the draft token's scaled logit.
type flashRow struct {
	m, l      float32
	bestKey   float64
	best      int32
	bestS     float32
	draftS    float32
	haveDraft bool
}

func newFlashRow() flashRow {
	return flashRow{m: float32(math.Inf(-1)), bestKey: math.Inf(-1), best: -1}
}

// addTile folds the scaled logits s of tokens [v0, v0+len(s)) into r.
func (r *flashRow) addTile(s []float32, v0 int, seed uint64, row int, draft int32) {
	mt := float32(math.Inf(-1))
	for _, x := range s {
		mt = max(mt, x)
	}
	mNew := max(r.m, mt)
	r.l *= float32(math.Exp(float64(r.m - mNew))) // rescale for the new max
	for c, x := range s {
		v := v0 + c
		r.l += float32(math.Exp(float64(x - mNew)))
		if key := float64(x) + FlashLogitsGumbel(seed, row, v); key > r.bestKey {
			r.bestKey, r.best, r.bestS = key, int32(v), x
		}
		if int32(v) == draft {
			r.draftS, r.haveDraft = x, true
		}
	}
	r.m = mNew
}

func (r *flashRow) finish() (tok int32, logProb, lse, draftProb float32) {
	lse = r.m + float32(math.Log(float64(r.l)))
	if r.haveDraft {
		draftProb = float32(math.Exp(float64(r.draftS - lse)))
	}
	return r.best, r.bestS - lse, lse, draftProb
}

// flashLogit returns h_row . w_v accumulated in float32 in index order.
func flashLogit(h, w []float32) float32 {
	var acc float32
	for i := range h {
		acc += h[i] * w[i]
	}
	return acc
}

// ExecuteReference runs the tile-streamed algorithm on the host for the
// hidden states h [rows, HiddenDim] (rows = len(h)/HiddenDim, any count),
// vocabulary matrix wVocab [VocabSize, HiddenDim] and per-row draft tokens
// (nil or len rows; -1 for none). It holds one TileV-wide tile of logits at
// a time, never a [rows, VocabSize] array.
func (s FlashLogitsSchedule) ExecuteReference(h, wVocab []float32, draftTokens []int32) (FlashLogitsOutput, error) {
	s, err := s.norm()
	if err != nil {
		return FlashLogitsOutput{}, err
	}
	d, v := s.HiddenDim, s.VocabSize
	if len(h) == 0 || len(h)%d != 0 || len(wVocab) != v*d {
		return FlashLogitsOutput{}, fmt.Errorf("tcgen05 flash logits %s reference: len(h)=%d len(wVocab)=%d, want a multiple of %d and %d", s.Name, len(h), len(wVocab), d, v*d)
	}
	rows := len(h) / d
	if draftTokens != nil && len(draftTokens) != rows {
		return FlashLogitsOutput{}, fmt.Errorf("tcgen05 flash logits %s reference: len(draftTokens)=%d, want %d", s.Name, len(draftTokens), rows)
	}
	out := FlashLogitsOutput{
		SampledTokenID: make([]int32, rows),
		SampledLogProb: make([]float32, rows),
		LSE:            make([]float32, rows),
		DraftTokenProb: make([]float32, rows),
	}
	tile := make([]float32, 0, s.TileV)
	for r := range rows {
		hr := h[r*d : (r+1)*d]
		draft := int32(-1)
		if draftTokens != nil {
			draft = draftTokens[r]
		}
		st := newFlashRow()
		for v0 := 0; v0 < v; v0 += s.TileV {
			tile = tile[:0]
			for vv := v0; vv < min(v0+s.TileV, v); vv++ {
				tile = append(tile, flashLogit(hr, wVocab[vv*d:(vv+1)*d])/s.Temperature)
			}
			st.addTile(tile, v0, s.Seed, r, draft)
		}
		out.SampledTokenID[r], out.SampledLogProb[r], out.LSE[r], out.DraftTokenProb[r] = st.finish()
	}
	return out, nil
}

func (s FlashLogitsSchedule) instDesc() uint32 {
	w, _ := tf32InstDesc(s.TileM, s.TileV)
	return w
}

// TraceEvents returns the tcgen05 event sequence of the kernel for
// cudatmem.VerifyTrace.
func (s FlashLogitsSchedule) TraceEvents() []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	tv := uint32(s.TileV)
	cols := tv * uint32(s.buffers())
	ev := []cudatmem.TraceEvent{{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Columns: cols, Rows: 128}}
	for t := range s.Tiles() {
		buf := uint32(t%s.buffers()) * tv
		for range s.HiddenDim / 32 {
			ev = append(ev,
				cudatmem.TraceEvent{Kind: cudatmem.EventMMA, ActiveThreads: 1, CTAGroup: 1, Offset: buf, Columns: tv, Rows: 128},
				cudatmem.TraceEvent{Kind: cudatmem.EventCommit, ActiveThreads: 1, CTAGroup: 1, Offset: buf, Columns: tv, Rows: 128},
				cudatmem.TraceEvent{Kind: cudatmem.EventWait, ActiveThreads: 32, CTAGroup: 1, Offset: buf, Columns: tv, Rows: 128},
			)
		}
		for w := range 4 {
			ev = append(ev, cudatmem.TraceEvent{Kind: cudatmem.EventLd, Warp: w, ActiveThreads: 32, CTAGroup: 1, Offset: buf, Columns: tv, RowStart: uint32(32 * w), Rows: 32})
		}
	}
	return append(ev, cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Columns: cols, Rows: 128})
}

// CUDA returns the kernel source without Prelude. The kernel signature is
//
//	NAME(const float* h, const float* w_vocab, const int* draft,
//	     int* out_token, float* out_logprob, float* out_lse, float* out_draft_prob)
//
// with h [TileM, HiddenDim], w_vocab [VocabSize, HiddenDim], draft [TileM]
// and the four outputs [TileM]. Global memory holds only these arrays.
func (s FlashLogitsSchedule) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(tcgen05MbarHelpers)
	b.WriteString(flashLogitsDeviceHelpers)
	fmt.Fprintf(&b, `
// %[1]s: %[2]s TMEM-streamed LM head (M=%[3]d, V=%[4]d, D=%[5]d, TileV=%[6]d, buffers=%[7]d).
__global__ void __launch_bounds__(128) %[1]s(
    const float* __restrict__ h,
    const float* __restrict__ w_vocab,
    const int* __restrict__ draft,
    int* __restrict__ out_token,
    float* __restrict__ out_logprob,
    float* __restrict__ out_lse,
    float* __restrict__ out_draft_prob) {
    constexpr int M = %[3]d, V = %[4]d, D = %[5]d, TV = %[6]d, NBUF = %[7]d, BK = 32;
    constexpr int NT = (V + TV - 1) / TV;
    constexpr float TEMP = %[8]s;
    constexpr unsigned long long SEED = %[9]dULL;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = 32 * warp + lane;

    EG_TCGEN05_SHARED(float, smem_h, [M * BK]);
    EG_TCGEN05_SHARED(float, smem_w, [TV * BK]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot[0], 0u, (uint32_t)(NBUF * TV));
    }
    if (tid == 0) eg_tcgen05_mbar_init(&mbar[0]);
    __syncthreads();
    const uint32_t base = slot[0];

    // Per-row online state, held in registers across all vocabulary tiles.
    float m = -INFINITY, l = 0.0f, best_s = 0.0f, draft_s = 0.0f;
    double best_key = -INFINITY;
    int best = -1;
    bool have_draft = false;
    const int my_draft = draft[row];
    uint32_t phase = 0u;

    for (int t = 0; t < NT; ++t) {
        const uint32_t buf = base + (uint32_t)((t %% NBUF) * TV);
        const int v0 = t * TV;
        // Z_t = H W_t^T into the TMEM buffer, D in BK-wide chunks.
        for (int kc = 0; kc < D / BK; ++kc) {
            for (int i = tid; i < M * BK; i += 128) smem_h[i] = h[(i / BK) * D + kc * BK + i %% BK];
            for (int i = tid; i < TV * BK; i += 128) {
                const int v = v0 + i / BK;
                smem_w[i] = v < V ? w_vocab[v * D + kc * BK + i %% BK] : 0.0f;
            }
            __syncthreads();
            if (warp == 0 && eg_elect_one_sync()) {
                eg_tcgen05_mma_tf32_ss(tmem, buf, smem_h, smem_w, M, TV, BK, 0x%08[10]xu, kc > 0);
                eg_tcgen05_commit_cta1(&mbar[0]);
            }
            eg_tcgen05_mbar_wait(&mbar[0], phase);
            phase ^= 1u;
            __syncthreads();
        }

        // Reduce the tile into the row state: tile max, rescaled partition
        // sum, Gumbel-max winner, draft logit.
        const int nv = (V - v0) < TV ? (V - v0) : TV;
        float mt = -INFINITY;
        for (int c = 0; c < nv; ++c) {
            mt = fmaxf(mt, eg_tcgen05_ld_row(tmem, warp, lane, buf + (uint32_t)c) / TEMP);
        }
        const float m_new = fmaxf(m, mt);
        l *= expf(m - m_new);
        for (int c = 0; c < nv; ++c) {
            const float sc = eg_tcgen05_ld_row(tmem, warp, lane, buf + (uint32_t)c) / TEMP;
            const int v = v0 + c;
            l += expf(sc - m_new);
            const double key = (double)sc + eg_flash_gumbel(SEED, (uint32_t)row, (uint32_t)v);
            if (key > best_key) {
                best_key = key;
                best = v;
                best_s = sc;
            }
            if (v == my_draft) {
                draft_s = sc;
                have_draft = true;
            }
        }
        m = m_new;
        eg_tcgen05_wait_ld();
        __syncthreads();
    }

    const float lse = m + logf(l);
    out_token[row] = best;
    out_logprob[row] = best_s - lse;
    out_lse[row] = lse;
    out_draft_prob[row] = have_draft ? expf(draft_s - lse) : 0.0f;

    __syncthreads();
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(base, (uint32_t)(NBUF * TV));
    }
}
`, s.Name, s.Target, s.TileM, s.VocabSize, s.HiddenDim, s.TileV, s.buffers(),
		formatFloat32Literal(s.Temperature), s.Seed, s.instDesc())
	return b.String(), nil
}

// formatFloat32Literal returns a C++ float literal that round-trips v.
func formatFloat32Literal(v float32) string {
	return strconv.FormatFloat(float64(v), 'e', -1, 32) + "f"
}

// flashLogitsDeviceHelpers is the device Philox4x32-10 and Gumbel variate,
// matching Philox4x32_10 and FlashLogitsGumbel.
const flashLogitsDeviceHelpers = `#ifndef EG_FLASH_LOGITS_HELPERS
#define EG_FLASH_LOGITS_HELPERS
EG_DEV uint32_t eg_philox4x32_10_x0(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
    for (int i = 0; i < 10; ++i) {
        if (i > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        const unsigned long long p0 = (unsigned long long)0xD2511F53u * c0;
        const unsigned long long p1 = (unsigned long long)0xCD9E8D57u * c2;
        const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
        const uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
    }
    return c0;
}

EG_DEV double eg_flash_gumbel(unsigned long long seed, uint32_t row, uint32_t token) {
    const uint32_t x = eg_philox4x32_10_x0(token, row, 0u, 0u, (uint32_t)seed, (uint32_t)(seed >> 32));
    const double u = ((double)x + 0.5) / 4294967296.0;
    return -log(-log(u));
}
#endif // EG_FLASH_LOGITS_HELPERS
`

// CUDAFile returns the self-contained translation unit (Prelude plus CUDA).
func (s FlashLogitsSchedule) CUDAFile() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}
