package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// A TubeletPatchSpec describes a spatiotemporal patch embedding: a video
// [Frames][Height][Width][Channels] of FP32 values is cut into
// PatchT x PatchH x PatchW tubelets (zero padded at the far edges) and each
// flattened tubelet (dt, dh, dw, c order) is projected by W [EmbedDim][Kp],
// Kp = PatchT*PatchH*PatchW*Channels.
type TubeletPatchSpec struct {
	Name                   string
	Target                 string // sm_100a, sm_103a, sm_107a or an f variant (default sm_100a)
	Frames, Height, Width  int
	Channels               int // 4 to 32 in steps of 4 (pad RGB to 4)
	PatchT, PatchH, PatchW int
	EmbedDim               int // a multiple of 16; the CTA tile is min(EmbedDim, 128) wide
}

func (s TubeletPatchSpec) norm() (TubeletPatchSpec, error) {
	if s.Name == "" {
		s.Name = "tubelet_patch_embed"
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("tubelet patch: bad name %q", s.Name)
	}
	if err := checkTMTarget("tubelet patch", s.Target); err != nil {
		return s, err
	}
	if s.Channels < 4 || s.Channels > 32 || s.Channels%4 != 0 {
		return s, fmt.Errorf("tubelet patch: %d channels: a TMA box row must be 16 to 128 bytes of FP32", s.Channels)
	}
	if s.Frames < 1 || s.Height < 1 || s.Width < 1 {
		return s, fmt.Errorf("tubelet patch: empty video %dx%dx%d", s.Frames, s.Height, s.Width)
	}
	for _, p := range [][2]int{{s.PatchT, s.Frames}, {s.PatchH, s.Height}, {s.PatchW, s.Width}} {
		if p[0] < 1 || p[0] > p[1] || p[0] > 256 {
			return s, fmt.Errorf("tubelet patch: patch %dx%dx%d does not fit the video", s.PatchT, s.PatchH, s.PatchW)
		}
	}
	if s.EmbedDim < 16 || s.EmbedDim%16 != 0 || s.EmbedDim%s.tileN() != 0 {
		return s, fmt.Errorf("tubelet patch: embed dim %d must be a multiple of 16 (and of 128 above 128)", s.EmbedDim)
	}
	if s.bk() == 0 {
		return s, fmt.Errorf("tubelet patch: patch length %d is not a multiple of 8", s.PatchLen())
	}
	return s, nil
}

// Validate reports whether s describes a kernel this package can emit.
func (s TubeletPatchSpec) Validate() error {
	_, err := s.norm()
	return err
}

func (s TubeletPatchSpec) tileN() int { return min(s.EmbedDim, 128) }

func (s TubeletPatchSpec) bk() int {
	for _, b := range []int{64, 32, 16, 8} {
		if s.PatchLen()%b == 0 {
			return b
		}
	}
	return 0
}

// PatchLen returns Kp, the flattened tubelet length.
func (s TubeletPatchSpec) PatchLen() int { return s.PatchT * s.PatchH * s.PatchW * s.Channels }

// GridTHW returns the number of tubelets along time, height and width.
func (s TubeletPatchSpec) GridTHW() (t, h, w int) {
	return (s.Frames + s.PatchT - 1) / s.PatchT, (s.Height + s.PatchH - 1) / s.PatchH, (s.Width + s.PatchW - 1) / s.PatchW
}

// Patches returns the number of tubelets.
func (s TubeletPatchSpec) Patches() int {
	t, h, w := s.GridTHW()
	return t * h * w
}

// BuildTubeletTMADescriptor returns the rank-4 tensor map that loads one
// (PatchT, PatchH, PatchW, Channels) tubelet of the FP32 video at baseAddr
// per TMA box. Dimensions are listed innermost first: channels, width,
// height, frames. Out-of-bounds boxes at the far edges are zero filled.
func BuildTubeletTMADescriptor(spec TubeletPatchSpec, baseAddr uint64) (cudatma.Descriptor, error) {
	spec, err := spec.norm()
	if err != nil {
		return cudatma.Descriptor{}, err
	}
	c := uint64(spec.Channels)
	d := cudatma.Descriptor{
		Target:       spec.Target,
		Rank:         4,
		Base:         baseAddr,
		Dims:         [cudatma.MaxRank]uint64{c, uint64(spec.Width), uint64(spec.Height), uint64(spec.Frames)},
		Strides:      [cudatma.MaxRank]uint64{4 * c, 4 * c * uint64(spec.Width), 4 * c * uint64(spec.Width*spec.Height)},
		Box:          [cudatma.MaxRank]uint32{uint32(spec.Channels), uint32(spec.PatchW), uint32(spec.PatchH), uint32(spec.PatchT)},
		ElementBytes: 4,
		ElementType:  cudatma.ElementTypeFloat32,
	}
	if err := d.Validate(); err != nil {
		return cudatma.Descriptor{}, fmt.Errorf("tubelet tma descriptor: %w", err)
	}
	return d, nil
}

// Reference computes the patch embedding [Patches][EmbedDim] in float64;
// patch p is (pt, ph, pw) in row-major order of [TubeletPatchSpec.GridTHW].
func (s TubeletPatchSpec) Reference(video []float32, w [][]float32) ([][]float32, error) {
	s, err := s.norm()
	if err != nil {
		return nil, err
	}
	if len(video) != s.Frames*s.Height*s.Width*s.Channels {
		return nil, fmt.Errorf("tubelet reference: video has %d values, want %d", len(video), s.Frames*s.Height*s.Width*s.Channels)
	}
	kp := s.PatchLen()
	if len(w) != s.EmbedDim {
		return nil, fmt.Errorf("tubelet reference: w has %d rows, want %d", len(w), s.EmbedDim)
	}
	nt, nh, nw := s.GridTHW()
	x := make([]float64, kp)
	out := make([][]float32, 0, s.Patches())
	for pt := range nt {
		for ph := range nh {
			for pw := range nw {
				for k := range x {
					c := k % s.Channels
					dw := k / s.Channels % s.PatchW
					dh := k / (s.Channels * s.PatchW) % s.PatchH
					dt := k / (s.Channels * s.PatchW * s.PatchH)
					t, h, ww := pt*s.PatchT+dt, ph*s.PatchH+dh, pw*s.PatchW+dw
					x[k] = 0
					if t < s.Frames && h < s.Height && ww < s.Width {
						x[k] = float64(video[((t*s.Height+h)*s.Width+ww)*s.Channels+c])
					}
				}
				row := make([]float32, s.EmbedDim)
				for e := range row {
					if len(w[e]) != kp {
						return nil, fmt.Errorf("tubelet reference: w[%d] has %d values, want %d", e, len(w[e]), kp)
					}
					var acc float64
					for k, xv := range x {
						acc += float64(w[e][k]) * xv
					}
					row[e] = float32(acc)
				}
				out = append(out, row)
			}
		}
	}
	return out, nil
}

// TraceEvents returns the tcgen05 operation sequence of one CTA.
func (s TubeletPatchSpec) TraceEvents() []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	tn := uint32(s.tileN())
	tr := newTMTrace(allocColumns(s.tileN()))
	for range s.PatchLen() / s.bk() {
		tr.mma(0, tn, false, 0)
		tr.commitWait()
	}
	tr.rows(cudatmem.EventLd, 0, tn, 128)
	return tr.done()
}

// SharedBytes returns the dynamic shared memory the kernel needs, or 0 if s
// is invalid.
func (s TubeletPatchSpec) SharedBytes() int {
	s, err := s.norm()
	if err != nil {
		return 0
	}
	return 4 * (128 + s.tileN()) * s.bk()
}

// Grid returns the launch shape: EmbedDim tiles by 128-tubelet tiles.
func (s TubeletPatchSpec) Grid() (grid, block [3]int) {
	s, err := s.norm()
	if err != nil {
		return
	}
	return [3]int{s.EmbedDim / s.tileN(), (s.Patches() + 127) / 128, 1}, [3]int{128, 1, 1}
}

// CUDA returns the patch-embedding kernel source without [Prelude]:
//
//	name(const float* video, const float* w, float* out)
//
// Each thread gathers one tubelet row per K step into shared memory in the
// order the TMA box of [BuildTubeletTMADescriptor] delivers it, with zeros
// past the far edges, and the tile is projected by tcgen05.mma into TMEM.
// The kernel does not issue the TMA load itself. The sm_100a path has not
// been executed; the EG_EMU path runs in the tests.
func (s TubeletPatchSpec) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	nt, nh, nw := s.GridTHW()
	var b strings.Builder
	b.WriteString(tmPrelude)
	fmt.Fprintf(&b, `
// %[1]s: %[2]dx%[3]dx%[4]d tubelet patch embedding of a %[5]dx%[6]dx%[7]dx%[8]d video into %[9]d dims (target %[10]s).
__global__ void __launch_bounds__(128) %[1]s(const float* __restrict__ video, const float* __restrict__ w, float* __restrict__ out) {
    constexpr int T = %[5]d, H = %[6]d, W = %[7]d, C = %[8]d, PT = %[2]d, PH = %[3]d, PW = %[4]d;
    constexpr int NT = %[11]d, NH = %[12]d, NW = %[13]d, P = NT * NH * NW, KP = PT * PH * PW * C;
    constexpr int E = %[9]d, TN = %[14]d, BK = %[15]d;
    constexpr uint32_t COLS = %[16]d;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int p = blockIdx.y * 128 + tid, n0 = blockIdx.x * TN;
    const int pt = p / (NH * NW), ph = p / NW %% NH, pw = p %% NW;
    EG_TM_BUF(tm);
    EG_TM_SHARED(uint32_t, slot, [1]);
    EG_TM_SHARED(uint64_t, mbar, [1]);
    EG_DYNAMIC_SMEM(float, sm);
    float* const xs = sm;            // tubelet rows [128][BK]
    float* const wb = xs + 128 * BK; // W [TN][BK]
    uint32_t phase = 0;
    if (tid == 0) eg_tm_mbar_init(&mbar[0]);
    if (warp == 0) eg_tm_alloc(tm, &slot[0], COLS);
    eg_tm_sync();
    const uint32_t base = slot[0];
    for (int k0 = 0; k0 < KP; k0 += BK) {
        for (int kk = 0; kk < BK; ++kk) {
            const int k = k0 + kk, c = k %% C, dw = k / C %% PW, dh = k / (C * PW) %% PH, dt = k / (C * PW * PH);
            const int t = pt * PT + dt, h = ph * PH + dh, x = pw * PW + dw;
            xs[eg_cm_idx(tid, kk, BK)] = p < P && t < T && h < H && x < W ? video[((t * H + h) * W + x) * C + c] : 0.0f;
        }
        for (int i = tid; i < TN * BK; i += 128) wb[eg_cm_idx(i / BK, i %% BK, BK)] = w[(size_t)(n0 + i / BK) * KP + k0 + i %% BK];
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base, xs, wb, TN, BK, k0 > 0);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        __syncthreads();
    }
    for (int n = 0; n < TN; ++n) {
        const float v = eg_tm_ld(tm, tid, base + n);
        if (p < P) out[(size_t)p * E + n0 + n] = v;
    }
    eg_tm_sync();
    if (warp == 0) eg_tm_dealloc(base, COLS);
}
`, s.Name, s.PatchT, s.PatchH, s.PatchW, s.Frames, s.Height, s.Width, s.Channels, s.EmbedDim, s.Target,
		nt, nh, nw, s.tileN(), s.bk(), allocColumns(s.tileN()))
	return b.String(), nil
}

// An MRoPESpec configures multimodal rotary position embedding: the first
// PairsT rotary pairs of a head turn by the temporal position, the next
// PairsH by the height position, and the next PairsW by the width
// position. Pair i rotates channels (i, i+R) with R = PairsT+PairsH+PairsW
// (rotate-half layout) at frequency BaseTheta^(-i/R); channels [2R,
// HeadDim) are not rotated.
type MRoPESpec struct {
	HeadDim   int
	PairsT    int
	PairsH    int
	PairsW    int
	BaseTheta float64
}

func (s MRoPESpec) pairs() int { return s.PairsT + s.PairsH + s.PairsW }

// Validate checks 2*(PairsT+PairsH+PairsW) <= HeadDim and a positive base.
func (s MRoPESpec) Validate() error {
	if s.HeadDim < 2 || s.PairsT < 0 || s.PairsH < 0 || s.PairsW < 0 || s.pairs() == 0 || 2*s.pairs() > s.HeadDim {
		return fmt.Errorf("mrope: sections %d+%d+%d pairs do not fit head dim %d", s.PairsT, s.PairsH, s.PairsW, s.HeadDim)
	}
	if !(s.BaseTheta > 1) {
		return fmt.Errorf("mrope: base %g must exceed 1", s.BaseTheta)
	}
	return nil
}

// InvFreq returns the FP32 inverse frequency of each rotary pair. Kernels
// take this table as input so device and host rotate by the same angles.
func (s MRoPESpec) InvFreq() []float32 {
	r := s.pairs()
	f := make([]float32, r)
	for i := range f {
		f[i] = float32(math.Pow(s.BaseTheta, -float64(i)/float64(r)))
	}
	return f
}

// rotate applies the rotation with per-pair positions pos(i) to x in FP32.
func (s MRoPESpec) rotate(x []float32, invFreq []float32, pos func(i int) int32) []float32 {
	r := len(invFreq)
	y := append([]float32(nil), x...)
	for i, f := range invFreq {
		a := float32(pos(i)) * f
		c, sn := float32(math.Cos(float64(a))), float32(math.Sin(float64(a)))
		y[i] = x[i]*c - x[i+r]*sn
		y[i+r] = x[i+r]*c + x[i]*sn
	}
	return y
}

func (s MRoPESpec) section(i int) int {
	switch {
	case i < s.PairsT:
		return 0
	case i < s.PairsT+s.PairsH:
		return 1
	}
	return 2
}

// ApplyMRoPEReference rotates each row x[t] [HeadDim] by its (temporal,
// height, width) position posTHW[t] in FP32, as the emitted kernels do.
func ApplyMRoPEReference(x [][]float32, posTHW [][3]int32, spec MRoPESpec) ([][]float32, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if len(posTHW) != len(x) {
		return nil, fmt.Errorf("mrope: %d positions for %d rows", len(posTHW), len(x))
	}
	inv := spec.InvFreq()
	out := make([][]float32, len(x))
	for t, row := range x {
		if len(row) != spec.HeadDim {
			return nil, fmt.Errorf("mrope: row %d has %d values, want %d", t, len(row), spec.HeadDim)
		}
		out[t] = spec.rotate(row, inv, func(i int) int32 { return posTHW[t][spec.section(i)] })
	}
	return out, nil
}

// ApplyRoPEReference is 1D rotary embedding over the same rotary pairs as
// spec: every pair turns by pos[t]. With equal (t, h, w) positions
// [ApplyMRoPEReference] gives the same bits.
func ApplyRoPEReference(x [][]float32, pos []int32, spec MRoPESpec) ([][]float32, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if len(pos) != len(x) {
		return nil, fmt.Errorf("rope: %d positions for %d rows", len(pos), len(x))
	}
	inv := spec.InvFreq()
	out := make([][]float32, len(x))
	for t, row := range x {
		if len(row) != spec.HeadDim {
			return nil, fmt.Errorf("rope: row %d has %d values, want %d", t, len(row), spec.HeadDim)
		}
		out[t] = spec.rotate(row, inv, func(int) int32 { return pos[t] })
	}
	return out, nil
}

// A JaggedBatchSchedule maps variable-length sequences, given by their
// prefix offsets CuSeqlens [B+1], to TileM-row query tiles that never cross
// a sequence boundary.
type JaggedBatchSchedule struct {
	CuSeqlens []int32
	TileM     int
	TileMap   [][2]int // per tile: sequence index, tile index within the sequence
}

// BuildJaggedBatchSchedule tiles each sequence separately; empty sequences
// get no tiles.
func BuildJaggedBatchSchedule(cuSeqlens []int32, tileM int) (JaggedBatchSchedule, error) {
	if tileM < 1 {
		return JaggedBatchSchedule{}, fmt.Errorf("jagged schedule: tile %d", tileM)
	}
	if len(cuSeqlens) < 2 || cuSeqlens[0] != 0 {
		return JaggedBatchSchedule{}, fmt.Errorf("jagged schedule: cu_seqlens must start at 0 and hold B+1 offsets")
	}
	s := JaggedBatchSchedule{CuSeqlens: append([]int32(nil), cuSeqlens...), TileM: tileM}
	for b := range len(cuSeqlens) - 1 {
		n := int(cuSeqlens[b+1] - cuSeqlens[b])
		if n < 0 {
			return JaggedBatchSchedule{}, fmt.Errorf("jagged schedule: cu_seqlens decreases at %d", b+1)
		}
		for i := range (n + tileM - 1) / tileM {
			s.TileMap = append(s.TileMap, [2]int{b, i})
		}
	}
	return s, nil
}

// Tokens returns the total token count.
func (s JaggedBatchSchedule) Tokens() int { return int(s.CuSeqlens[len(s.CuSeqlens)-1]) }

// A JaggedAttention configures single-head-per-CTA attention over a
// [JaggedBatchSchedule]: each CTA owns one 128-row query tile, applies the
// MRoPE rotation to q and k as it stages them, and streams 64-key tiles of
// its own sequence only: S = Q Kᵀ into TMEM, online softmax one query row
// per thread, the O correction applied in TMEM with tcgen05.ld/st, and
// O += P V accumulated in TMEM. Keys stop at kv_end = cu_seqlens[seq+1]
// (or the query tile end when Causal), so no tile reads another sequence.
//
// The sm_100a/sm_103a/sm_107a path of the emitted source has not been
// executed; the EG_EMU path runs in the tests.
type JaggedAttention struct {
	Name   string
	Target string // sm_100a, sm_103a, sm_107a or an f variant (default sm_100a)
	MRoPE  MRoPESpec
	Causal bool
	Scale  float32 // softmax scale (default 1/sqrt(HeadDim))
}

// jaggedKVTile is the number of keys per inner iteration.
const jaggedKVTile = 64

func (a JaggedAttention) norm() (JaggedAttention, error) {
	if a.Name == "" {
		a.Name = "jagged_mrope_attention"
	}
	if a.Target == "" {
		a.Target = "sm_100a"
	}
	if a.Scale == 0 && a.MRoPE.HeadDim > 0 {
		a.Scale = float32(1 / math.Sqrt(float64(a.MRoPE.HeadDim)))
	}
	if !cIdent(a.Name) {
		return a, fmt.Errorf("jagged attention: bad name %q", a.Name)
	}
	if err := checkTMTarget("jagged attention", a.Target); err != nil {
		return a, err
	}
	if err := a.MRoPE.Validate(); err != nil {
		return a, fmt.Errorf("jagged attention: %w", err)
	}
	if d := a.MRoPE.HeadDim; d < 16 || d > 128 || d%16 != 0 {
		return a, fmt.Errorf("jagged attention: head dim %d must be a multiple of 16 in [16, 128]", d)
	}
	return a, nil
}

// Validate reports whether a describes a kernel this package can emit.
func (a JaggedAttention) Validate() error {
	_, err := a.norm()
	return err
}

// SharedBytes returns the dynamic shared memory the kernel needs, or 0 if a
// is invalid.
func (a JaggedAttention) SharedBytes() int {
	a, err := a.norm()
	if err != nil {
		return 0
	}
	d := a.MRoPE.HeadDim
	return 4 * (128*d + 2*jaggedKVTile*d + 128*jaggedKVTile)
}

// TMEMColumns returns the TMEM allocation width: S then O.
func (a JaggedAttention) TMEMColumns() uint32 {
	return allocColumns(jaggedKVTile + a.MRoPE.HeadDim)
}

// Grid returns the launch shape for sched and heads heads.
func (a JaggedAttention) Grid(sched JaggedBatchSchedule, heads int) (grid, block [3]int) {
	return [3]int{len(sched.TileMap), heads, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the tcgen05 operation sequence of a CTA that streams
// kvTiles key tiles.
func (a JaggedAttention) TraceEvents(kvTiles int) []cudatmem.TraceEvent {
	a, err := a.norm()
	if err != nil {
		return nil
	}
	d, kt := uint32(a.MRoPE.HeadDim), uint32(jaggedKVTile)
	tr := newTMTrace(a.TMEMColumns())
	for i := range kvTiles {
		tr.mma(0, kt, false, 0)
		tr.commitWait()
		tr.rows(cudatmem.EventLd, 0, kt, 128)
		if i > 0 {
			tr.rows(cudatmem.EventLd, kt, d, 128)
			tr.rows(cudatmem.EventSt, kt, d, 128)
		}
		tr.mma(kt, d, false, 0)
		tr.commitWait()
	}
	tr.rows(cudatmem.EventLd, kt, d, 128)
	return tr.done()
}

// Reference computes attention for one head in float64 after rotating q and
// k with [ApplyMRoPEReference]. q, k, v and the result are [tokens][HeadDim].
func (a JaggedAttention) Reference(sched JaggedBatchSchedule, q, k, v [][]float32, posTHW [][3]int32) ([][]float32, error) {
	a, err := a.norm()
	if err != nil {
		return nil, err
	}
	n := sched.Tokens()
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("jagged attention reference: q, k, v rows %d, %d, %d, want %d", len(q), len(k), len(v), n)
	}
	qr, err := ApplyMRoPEReference(q, posTHW, a.MRoPE)
	if err != nil {
		return nil, err
	}
	kr, err := ApplyMRoPEReference(k, posTHW, a.MRoPE)
	if err != nil {
		return nil, err
	}
	d := a.MRoPE.HeadDim
	out := make([][]float32, n)
	for b := range len(sched.CuSeqlens) - 1 {
		lo, hi := int(sched.CuSeqlens[b]), int(sched.CuSeqlens[b+1])
		for i := lo; i < hi; i++ {
			end := hi
			if a.Causal {
				end = i + 1
			}
			s := make([]float64, end-lo)
			mx := math.Inf(-1)
			for j := lo; j < end; j++ {
				var dot float64
				for c := range d {
					dot += float64(qr[i][c]) * float64(kr[j][c])
				}
				s[j-lo] = dot * float64(a.Scale)
				mx = max(mx, s[j-lo])
			}
			var sum float64
			o := make([]float64, d)
			for j := lo; j < end; j++ {
				p := math.Exp(s[j-lo] - mx)
				sum += p
				for c := range o {
					o[c] += p * float64(v[j][c])
				}
			}
			out[i] = make([]float32, d)
			for c := range o {
				out[i][c] = float32(o[c] / sum)
			}
		}
	}
	return out, nil
}

// CUDA returns the kernel source without [Prelude]:
//
//	name(int tokens, const int* tile_seq, const int* tile_idx,
//	     const int* cu_seqlens, const float* inv_freq, const int* pos,
//	     const float* q, const float* k, const float* v, float* out)
//
// tile_seq and tile_idx are the columns of a [JaggedBatchSchedule]'s
// TileMap (TileM = 128), inv_freq is [MRoPESpec.InvFreq], pos is
// [tokens][3], and q, k, v, out are [heads][tokens][HeadDim] with the head
// in blockIdx.y.
func (a JaggedAttention) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	m := a.MRoPE
	causal := 0
	if a.Causal {
		causal = 1
	}
	var b strings.Builder
	b.WriteString(tmPrelude)
	fmt.Fprintf(&b, `
// %[1]s: jagged cu_seqlens attention with fused MRoPE (T/H/W pairs %[2]d/%[3]d/%[4]d,
// head dim %[5]d, causal %[6]d, target %[7]s).
__global__ void __launch_bounds__(128) %[1]s(
    int tokens, const int* __restrict__ tile_seq, const int* __restrict__ tile_idx,
    const int* __restrict__ cu_seqlens, const float* __restrict__ inv_freq, const int* __restrict__ pos,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out) {
    constexpr int D = %[5]d, R = %[8]d, PT = %[2]d, PH = %[3]d, KT = %[9]d, CAUSAL = %[6]d;
    constexpr uint32_t COLS = %[10]d, C_S = 0, C_O = KT;
    constexpr float SCALE = %[11]s;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int seq = tile_seq[blockIdx.x];
    const int seq0 = cu_seqlens[seq], seq1 = cu_seqlens[seq + 1];
    const int q0 = seq0 + tile_idx[blockIdx.x] * 128;
    const int qrows = seq1 - q0 < 128 ? seq1 - q0 : 128;
    const int kv_end = CAUSAL ? (q0 + qrows < seq1 ? q0 + qrows : seq1) : seq1;
    const size_t hoff = (size_t)blockIdx.y * tokens * D;
    EG_TM_BUF(tm);
    EG_TM_SHARED(uint32_t, slot, [1]);
    EG_TM_SHARED(uint64_t, mbar, [1]);
    EG_DYNAMIC_SMEM(float, sm);
    float* const qs = sm;           // rotated Q [128][D], rows >= qrows zero
    float* const ks = qs + 128 * D; // rotated K [KT][D]
    float* const vt = ks + KT * D;  // Vᵀ [D][KT]
    float* const ps = vt + D * KT;  // P [128][KT]
    uint32_t phase = 0;
    if (tid == 0) eg_tm_mbar_init(&mbar[0]);
    if (warp == 0) eg_tm_alloc(tm, &slot[0], COLS);
    eg_tm_sync();
    const uint32_t base = slot[0];

    // rope writes row i of x (head offset applied) rotated by token i's
    // MRoPE position through put(channel, value).
    auto rope = [&](const float* x, int i, auto put) {
        for (int c = 2 * R; c < D; ++c) put(c, x[(size_t)i * D + c]);
        for (int p = 0; p < R; ++p) {
            const int sec = p < PT ? 0 : (p < PT + PH ? 1 : 2);
            const float ang = (float)pos[i * 3 + sec] * inv_freq[p];
            const float cs = cosf(ang), sn = sinf(ang);
            const float x0 = x[(size_t)i * D + p], x1 = x[(size_t)i * D + p + R];
            put(p, x0 * cs - x1 * sn);
            put(p + R, x1 * cs + x0 * sn);
        }
    };

    const int qi = q0 + tid;
    const bool qvalid = tid < qrows;
    if (qvalid) rope(q + hoff, qi, [&](int c, float y) { qs[eg_cm_idx(tid, c, D)] = y; });
    else for (int c = 0; c < D; ++c) qs[eg_cm_idx(tid, c, D)] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int kv0 = seq0; kv0 < kv_end; kv0 += KT) {
        if (tid < KT) {
            const int j = kv0 + tid;
            if (j < kv_end) {
                rope(k + hoff, j, [&](int c, float y) { ks[eg_cm_idx(tid, c, D)] = y; });
                for (int c = 0; c < D; ++c) vt[eg_cm_idx(c, tid, KT)] = v[hoff + (size_t)j * D + c];
            } else {
                for (int c = 0; c < D; ++c) {
                    ks[eg_cm_idx(tid, c, D)] = 0.0f;
                    vt[eg_cm_idx(c, tid, KT)] = 0.0f;
                }
            }
        }
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base + C_S, qs, ks, KT, D, 0);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        float s[KT];
        float mx = -INFINITY;
        for (int j = 0; j < KT; ++j) {
            const float x = SCALE * eg_tm_ld(tm, tid, base + C_S + j);
            const bool live = kv0 + j < kv_end && (!CAUSAL || kv0 + j <= qi);
            s[j] = live ? x : -INFINITY;
            mx = s[j] > mx ? s[j] : mx;
        }
        const float mn = m > mx ? m : mx;
        const float corr = m == -INFINITY ? 0.0f : expf(m - mn);
        float sum = 0.0f;
        for (int j = 0; j < KT; ++j) {
            const float p = s[j] == -INFINITY ? 0.0f : expf(s[j] - mn);
            sum += p;
            ps[eg_cm_idx(tid, j, KT)] = p;
        }
        l = l * corr + sum;
        m = mn;
        if (kv0 > seq0) {
            for (int c = 0; c < D; ++c) eg_tm_st(tm, tid, base + C_O + c, corr * eg_tm_ld(tm, tid, base + C_O + c));
        }
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base + C_O, ps, vt, D, KT, kv0 > seq0);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        __syncthreads();
    }
    for (int c = 0; c < D; ++c) {
        const float o = eg_tm_ld(tm, tid, base + C_O + c);
        if (qvalid) out[hoff + (size_t)qi * D + c] = o / l;
    }
    eg_tm_sync();
    if (warp == 0) eg_tm_dealloc(base, COLS);
}
`, a.Name, m.PairsT, m.PairsH, m.PairsW, m.HeadDim, causal, a.Target, m.pairs(), jaggedKVTile, a.TMEMColumns(), floatLiteral(a.Scale))
	return b.String(), nil
}
