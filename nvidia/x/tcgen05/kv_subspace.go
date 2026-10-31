package tcgen05

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// SubspaceKVProjector projects GQA key and value rows of width
// NumKVHeads*HeadDim onto Rank-dimensional subspaces shared by all KV heads.
//
// BasisK and BasisV are row-major [NumKVHeads*HeadDim, Rank] matrices with
// orthonormal columns (U^T U = I). A key row k is stored as c = k U_K and
// read back as c U_K^T. The basis is meant for keys before RoPE (or layers
// without RoPE): rotary embeddings applied after projection do not commute
// with U_K, and this package does not model them.
type SubspaceKVProjector struct {
	NumKVHeads int
	HeadDim    int
	Rank       int
	BasisK     []float32
	BasisV     []float32

	ExplainedVarK float64 // fraction of sum ||k||^2 in the calibration sample captured by BasisK
	ExplainedVarV float64
}

// Dim returns NumKVHeads*HeadDim, the width of a full KV row.
func (p SubspaceKVProjector) Dim() int { return p.NumKVHeads * p.HeadDim }

// CalibrateSubspaceBasis returns the projector whose bases are the top-rank
// eigenvectors of the uncentered second-moment matrices K^T K and V^T V of
// the calibration rows sampleK, sampleV [tokens, numKVHeads*headDim],
// found by one-sided Jacobi on the sample itself (tokens <= D) or on the
// D x D second-moment matrix. No gradients or training are involved.
func CalibrateSubspaceBasis(sampleK, sampleV []float32, tokens, numKVHeads, headDim, rank int) (SubspaceKVProjector, error) {
	d := numKVHeads * headDim
	if tokens < 1 || numKVHeads < 1 || headDim < 1 || rank < 1 || rank > d {
		return SubspaceKVProjector{}, fmt.Errorf("tcgen05 kv subspace calibrate: invalid tokens=%d heads=%d headDim=%d rank=%d", tokens, numKVHeads, headDim, rank)
	}
	if len(sampleK) != tokens*d || len(sampleV) != tokens*d {
		return SubspaceKVProjector{}, fmt.Errorf("tcgen05 kv subspace calibrate: len(sampleK)=%d len(sampleV)=%d, want %d", len(sampleK), len(sampleV), tokens*d)
	}
	p := SubspaceKVProjector{NumKVHeads: numKVHeads, HeadDim: headDim, Rank: rank}
	p.BasisK, p.ExplainedVarK = principalBasis(sampleK, tokens, d, rank)
	p.BasisV, p.ExplainedVarV = principalBasis(sampleV, tokens, d, rank)
	return p, nil
}

// principalBasis returns the top-rank right singular vectors of x [t, d] as
// a [d, rank] row-major matrix and the fraction of ||x||_F^2 they capture.
//
// It orthogonalizes the rows of M by one-sided (Hestenes) Jacobi rotations,
// where M = x when t <= d and M = x^T x otherwise. Left rotations keep
// M^T M, so the nonzero rows of the result are the right singular
// directions of M, which are those of x, ordered by row norm.
func principalBasis(x []float32, t, d, rank int) ([]float32, float64) {
	var m []float64
	rows := t
	if t <= d {
		m = make([]float64, t*d)
		for i, v := range x {
			m[i] = float64(v)
		}
	} else {
		rows = d
		m = make([]float64, d*d)
		for r := range t {
			row := x[r*d : (r+1)*d]
			for i := range d {
				xi := float64(row[i])
				for j := i; j < d; j++ {
					m[i*d+j] += xi * float64(row[j])
				}
			}
		}
		for i := range d {
			for j := range i {
				m[i*d+j] = m[j*d+i]
			}
		}
	}
	rowOrthogonalize(m, rows, d)

	type dir struct {
		norm float64
		row  int
	}
	dirs := make([]dir, rows)
	var total, maxNorm float64
	for i := range rows {
		var n float64
		for _, v := range m[i*d : (i+1)*d] {
			n += v * v
		}
		n = math.Sqrt(n)
		dirs[i] = dir{n, i}
		maxNorm = max(maxNorm, n)
	}
	slices.SortStableFunc(dirs, func(a, b dir) int { return cmp.Compare(b.norm, a.norm) })
	// Variance along direction i: ||row_i||^2 for M = x, ||row_i|| (an
	// eigenvalue of the PSD matrix x^T x) for M = x^T x.
	variance := func(n float64) float64 {
		if t <= d {
			return n * n
		}
		return n
	}
	var top float64
	var cols [][]float64
	for k, dr := range dirs {
		v := variance(dr.norm)
		total += v
		if k >= rank || dr.norm <= 1e-9*maxNorm {
			continue
		}
		top += v
		u := make([]float64, d)
		for i := range d {
			u[i] = m[dr.row*d+i] / dr.norm
		}
		cols = append(cols, u)
	}
	// Complete the basis with unit vectors when the sample spans fewer
	// than rank directions.
	for e := 0; len(cols) < rank && e < d; e++ {
		u := make([]float64, d)
		u[e] = 1
		for range 2 {
			for _, c := range cols {
				var dot float64
				for i := range d {
					dot += c[i] * u[i]
				}
				for i := range d {
					u[i] -= dot * c[i]
				}
			}
		}
		var n float64
		for _, v := range u {
			n += v * v
		}
		if n < 1e-6 {
			continue
		}
		n = 1 / math.Sqrt(n)
		for i := range u {
			u[i] *= n
		}
		cols = append(cols, u)
	}
	frac := 1.0
	if total > 0 {
		frac = top / total
	}
	basis := make([]float32, d*rank)
	for k, c := range cols {
		for i := range d {
			basis[i*rank+k] = float32(c[i])
		}
	}
	return basis, frac
}

// rowOrthogonalize applies one-sided Jacobi rotations to pairs of rows of
// the row-major [rows, cols] matrix m until every pair of rows is orthogonal
// to within 1e-12 relative, or 60 sweeps have run.
func rowOrthogonalize(m []float64, rows, cols int) {
	norm := make([]float64, rows) // squared row norms, refreshed each sweep
	for range 60 {
		var maxSq float64
		for i := range rows {
			r := m[i*cols : (i+1)*cols]
			norm[i] = dotF64(r, r)
			maxSq = max(maxSq, norm[i])
		}
		tiny := 1e-30 * maxSq
		rotated := false
		for i := range rows - 1 {
			ri := m[i*cols : (i+1)*cols]
			for j := i + 1; j < rows; j++ {
				a, b := norm[i], norm[j]
				if a <= tiny || b <= tiny {
					continue
				}
				rj := m[j*cols : (j+1)*cols]
				g := dotF64(ri, rj)
				if math.Abs(g) <= 1e-12*math.Sqrt(a*b) {
					continue
				}
				rotated = true
				zeta := (b - a) / (2 * g)
				t := 1 / (math.Abs(zeta) + math.Sqrt(1+zeta*zeta))
				if zeta < 0 {
					t = -t
				}
				c := 1 / math.Sqrt(1+t*t)
				s := c * t
				for k := range cols {
					x, y := ri[k], rj[k]
					ri[k], rj[k] = c*x-s*y, s*x+c*y
				}
				// The rotation zeroes g; row norms move by t*g.
				norm[i], norm[j] = a-t*g, b+t*g
			}
		}
		if !rotated {
			return
		}
	}
}

func dotF64(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// OrthonormalityError returns ||U^T U - I||_F for the row-major [dim, rank]
// matrix u.
func OrthonormalityError(u []float32, dim, rank int) float64 {
	var e float64
	for a := range rank {
		for b := range rank {
			var s float64
			for i := range dim {
				s += float64(u[i*rank+a]) * float64(u[i*rank+b])
			}
			if a == b {
				s--
			}
			e += s * s
		}
	}
	return math.Sqrt(e)
}

// Validate checks the projector's shape and that both bases are
// orthonormal to within 1e-5.
func (p SubspaceKVProjector) Validate() error {
	d := p.Dim()
	if p.NumKVHeads < 1 || p.HeadDim < 1 || p.Rank < 1 || p.Rank > d {
		return fmt.Errorf("tcgen05 kv subspace: invalid heads=%d headDim=%d rank=%d", p.NumKVHeads, p.HeadDim, p.Rank)
	}
	for _, b := range []struct {
		name string
		u    []float32
	}{{"BasisK", p.BasisK}, {"BasisV", p.BasisV}} {
		if len(b.u) != d*p.Rank {
			return fmt.Errorf("tcgen05 kv subspace: len(%s)=%d, want %d", b.name, len(b.u), d*p.Rank)
		}
		if e := OrthonormalityError(b.u, d, p.Rank); !(e < 1e-5) {
			return fmt.Errorf("tcgen05 kv subspace: %s is not orthonormal (||U^T U - I||_F = %.3g)", b.name, e)
		}
	}
	return nil
}

// CompressedKVPage is a page of KV rows stored in the projector's latent
// subspace.
type CompressedKVPage struct {
	PageID int
	Tokens int
	Rank   int

	LatentK []float32 // [Tokens, Rank]; dequantized values when Use3BitLUT
	LatentV []float32

	// Use3BitLUT reports that the latents are stored as 32-element
	// LearnedLUTBlocks. Each block runs along the reduction dimension of
	// the MMA that reads it: BlocksK along Rank for S = q~ C_K^T, in
	// row-major order; BlocksV along Tokens for o~ = P C_V, ordered by
	// (32-token group, latent column).
	Use3BitLUT bool
	BlocksK    []LearnedLUTBlock
	BlocksV    []LearnedLUTBlock
}

// Bytes returns the storage size of the page: 2 bytes per latent value in
// BF16, or 21 bytes per 32-value LUT block (12 bytes of 3-bit indices, an
// 8-byte E4M3 codebook and a UE8M0 scale).
func (c CompressedKVPage) Bytes() int64 {
	if c.Use3BitLUT {
		return int64(len(c.BlocksK)+len(c.BlocksV)) * (12 + 8 + 1)
	}
	return int64(len(c.LatentK)+len(c.LatentV)) * 2
}

// CompressionRatio returns the BF16 size of the uncompressed K and V rows
// divided by c.Bytes().
func (p SubspaceKVProjector) CompressionRatio(c CompressedKVPage) float64 {
	return float64(2*c.Tokens*p.Dim()*2) / float64(c.Bytes())
}

// CompressKVPage projects the rows k, v [tokens, Dim()] onto the latent
// subspace. With use3BitLUT, each 32-value block of the latents (see
// [CompressedKVPage]) is fitted with [FitBlock32LearnedLUT] with unit
// weights, which needs Rank and tokens to be multiples of 32.
func (p SubspaceKVProjector) CompressKVPage(k, v []float32, tokens, pageID int, use3BitLUT bool) (CompressedKVPage, error) {
	if err := p.Validate(); err != nil {
		return CompressedKVPage{}, err
	}
	d, r := p.Dim(), p.Rank
	if tokens < 1 || len(k) != tokens*d || len(v) != tokens*d {
		return CompressedKVPage{}, fmt.Errorf("tcgen05 kv subspace compress: tokens=%d len(k)=%d len(v)=%d, want tokens*%d", tokens, len(k), len(v), d)
	}
	if use3BitLUT && (r%32 != 0 || tokens%32 != 0) {
		return CompressedKVPage{}, fmt.Errorf("tcgen05 kv subspace compress: rank %d and tokens %d must be multiples of 32 for 3-bit LUT blocks", r, tokens)
	}
	c := CompressedKVPage{
		PageID:     pageID,
		Tokens:     tokens,
		Rank:       r,
		LatentK:    projectRows(k, p.BasisK, tokens, d, r),
		LatentV:    projectRows(v, p.BasisV, tokens, d, r),
		Use3BitLUT: use3BitLUT,
	}
	if use3BitLUT {
		var err error
		if c.BlocksK, err = lutQuantizeInPlace(c.LatentK, tokens, r, true); err != nil {
			return CompressedKVPage{}, err
		}
		if c.BlocksV, err = lutQuantizeInPlace(c.LatentV, tokens, r, false); err != nil {
			return CompressedKVPage{}, err
		}
	}
	return c, nil
}

// lutQuantizeInPlace fits a LearnedLUTBlock to each 32-value run of the
// row-major [rows, cols] matrix x and overwrites x with the dequantized
// values. Runs follow rows (32 consecutive columns) when alongRows is true
// and columns (32 consecutive rows) otherwise.
func lutQuantizeInPlace(x []float32, rows, cols int, alongRows bool) ([]LearnedLUTBlock, error) {
	blocks := make([]LearnedLUTBlock, len(x)/32)
	pos := func(blk, i int) int {
		if alongRows {
			return 32*blk + i
		}
		c, r0 := blk%cols, 32*(blk/cols)
		return (r0+i)*cols + c
	}
	for blk := range blocks {
		var w [32]float32
		for i := range w {
			w[i] = x[pos(blk, i)]
		}
		b, err := FitBlock32LearnedLUT(w, ones32f(), 50)
		if err != nil {
			return nil, fmt.Errorf("tcgen05 kv subspace compress: block %d: %w", blk, err)
		}
		blocks[blk] = b
		for i, v := range b.Dequantize() {
			x[pos(blk, i)] = v
		}
	}
	return blocks, nil
}

func ones32f() [32]float32 {
	var h [32]float32
	for i := range h {
		h[i] = 1
	}
	return h
}

// projectRows returns x [t, d] times u [d, r], accumulated in float64.
func projectRows(x, u []float32, t, d, r int) []float32 {
	out := make([]float32, t*r)
	acc := make([]float64, r)
	for a := range t {
		clear(acc)
		for i := range d {
			xi := float64(x[a*d+i])
			for j := range r {
				acc[j] += xi * float64(u[i*r+j])
			}
		}
		for j := range r {
			out[a*r+j] = float32(acc[j])
		}
	}
	return out
}

// DecompressKVPage returns the reconstructed rows k = C_K U_K^T and
// v = C_V U_V^T, each [Tokens, Dim()].
func (p SubspaceKVProjector) DecompressKVPage(c CompressedKVPage) (k, v []float32) {
	d, r := p.Dim(), p.Rank
	back := func(lat, u []float32) []float32 {
		out := make([]float32, c.Tokens*d)
		for a := range c.Tokens {
			for i := range d {
				var s float64
				for j := range r {
					s += float64(lat[a*r+j]) * float64(u[i*r+j])
				}
				out[a*d+i] = float32(s)
			}
		}
		return out
	}
	return back(c.LatentK, p.BasisK), back(c.LatentV, p.BasisV)
}

// AbsorbedLatentAttention computes single-position GQA attention for the
// queries q [numQHeads, HeadDim] against page without reconstructing K or
// V. For query head h in KV group g = h / (numQHeads/NumKVHeads), with
// U_{K,g} the HeadDim rows of BasisK belonging to head g, it forms
// q~ = q_h U_{K,g} in R^Rank, scores s_t = q~ . C_K[t] / sqrt(HeadDim),
// o~ = softmax(s) C_V and o_h = U_{V,g} o~. This equals attention over the
// reconstructed rows of [SubspaceKVProjector.DecompressKVPage] up to
// rounding. It returns the pre-softmax scores [numQHeads, Tokens] and the
// outputs [numQHeads, HeadDim].
func (p SubspaceKVProjector) AbsorbedLatentAttention(q []float32, numQHeads int, c CompressedKVPage) (scores, out []float32, err error) {
	hd, r := p.HeadDim, p.Rank
	if numQHeads < 1 || numQHeads%p.NumKVHeads != 0 || len(q) != numQHeads*hd {
		return nil, nil, fmt.Errorf("tcgen05 kv subspace attention: numQHeads=%d len(q)=%d with %d KV heads of dim %d", numQHeads, len(q), p.NumKVHeads, hd)
	}
	if c.Rank != r || len(c.LatentK) != c.Tokens*r || len(c.LatentV) != c.Tokens*r {
		return nil, nil, fmt.Errorf("tcgen05 kv subspace attention: page rank %d tokens %d does not match projector rank %d", c.Rank, c.Tokens, r)
	}
	group := numQHeads / p.NumKVHeads
	scale := 1 / math.Sqrt(float64(hd))
	scores = make([]float32, numQHeads*c.Tokens)
	out = make([]float32, numQHeads*hd)
	qt := make([]float64, r)
	ot := make([]float64, r)
	s := make([]float64, c.Tokens)
	for h := range numQHeads {
		g := h / group
		clear(qt)
		for i := range hd {
			qi := float64(q[h*hd+i])
			row := (g*hd + i) * r
			for j := range r {
				qt[j] += qi * float64(p.BasisK[row+j])
			}
		}
		for t := range c.Tokens {
			var dot float64
			for j := range r {
				dot += qt[j] * float64(c.LatentK[t*r+j])
			}
			s[t] = dot * scale
			scores[h*c.Tokens+t] = float32(s[t])
		}
		softmaxF64(s)
		clear(ot)
		for t := range c.Tokens {
			for j := range r {
				ot[j] += s[t] * float64(c.LatentV[t*r+j])
			}
		}
		for i := range hd {
			var o float64
			row := (g*hd + i) * r
			for j := range r {
				o += float64(p.BasisV[row+j]) * ot[j]
			}
			out[h*hd+i] = float32(o)
		}
	}
	return scores, out, nil
}

func softmaxF64(s []float64) {
	m := math.Inf(-1)
	for _, v := range s {
		m = max(m, v)
	}
	var z float64
	for i, v := range s {
		s[i] = math.Exp(v - m)
		z += s[i]
	}
	for i := range s {
		s[i] /= z
	}
}

// GQAAttention is the uncompressed reference for [SubspaceKVProjector.AbsorbedLatentAttention]:
// single-position grouped-query attention of q [numQHeads, headDim] over
// k, v [tokens, numKVHeads*headDim], accumulated in float64.
func GQAAttention(q, k, v []float32, tokens, numQHeads, numKVHeads, headDim int) (scores, out []float32, err error) {
	d := numKVHeads * headDim
	if numKVHeads < 1 || numQHeads%numKVHeads != 0 || len(q) != numQHeads*headDim || len(k) != tokens*d || len(v) != tokens*d {
		return nil, nil, fmt.Errorf("tcgen05 gqa attention: invalid shapes q=%d k=%d v=%d for tokens=%d heads=%d/%d dim=%d", len(q), len(k), len(v), tokens, numQHeads, numKVHeads, headDim)
	}
	group := numQHeads / numKVHeads
	scale := 1 / math.Sqrt(float64(headDim))
	scores = make([]float32, numQHeads*tokens)
	out = make([]float32, numQHeads*headDim)
	s := make([]float64, tokens)
	for h := range numQHeads {
		off := (h / group) * headDim
		for t := range tokens {
			var dot float64
			for i := range headDim {
				dot += float64(q[h*headDim+i]) * float64(k[t*d+off+i])
			}
			s[t] = dot * scale
			scores[h*tokens+t] = float32(s[t])
		}
		softmaxF64(s)
		for i := range headDim {
			var o float64
			for t := range tokens {
				o += s[t] * float64(v[t*d+off+i])
			}
			out[h*headDim+i] = float32(o)
		}
	}
	return scores, out, nil
}

// KVCompressorPartition is the SM partition a background KV compressor
// requests: BackgroundSMs of TotalSMs run the compressor in their own green
// context while the rest serve decode. SMGranularity is the split unit the
// driver reports for the device; BackgroundSMs must be a multiple of it.
// This type only describes the request; creating the green context is left
// to the caller.
type KVCompressorPartition struct {
	TotalSMs      int
	BackgroundSMs int
	SMGranularity int
}

// Validate checks that the partition leaves SMs on both sides and respects
// the split granularity.
func (p KVCompressorPartition) Validate() error {
	if p.SMGranularity < 1 {
		return fmt.Errorf("tcgen05 kv compressor partition: SMGranularity=%d must be >= 1", p.SMGranularity)
	}
	if p.BackgroundSMs < 1 || p.BackgroundSMs >= p.TotalSMs {
		return fmt.Errorf("tcgen05 kv compressor partition: BackgroundSMs=%d must be in [1, TotalSMs=%d)", p.BackgroundSMs, p.TotalSMs)
	}
	if p.BackgroundSMs%p.SMGranularity != 0 || (p.TotalSMs-p.BackgroundSMs) < p.SMGranularity {
		return fmt.Errorf("tcgen05 kv compressor partition: %d background of %d SMs not splittable at granularity %d", p.BackgroundSMs, p.TotalSMs, p.SMGranularity)
	}
	return nil
}

// DecodeSMs returns the SMs left for the decode partition.
func (p KVCompressorPartition) DecodeSMs() int { return p.TotalSMs - p.BackgroundSMs }

// KVCompressSchedule decides which KV pages the background compressor
// handles: the most recent HotWindowTokens tokens stay uncompressed, and
// each completed page of PageTokens tokens that has left the hot window is
// compressed once.
type KVCompressSchedule struct {
	PageTokens      int
	HotWindowTokens int
	Partition       KVCompressorPartition
}

// KVCompressTask assigns one page to one background SM (an index in
// [0, Partition.BackgroundSMs)). Pages map to SMs by PageID modulo the
// background SM count, so a page keeps its SM across calls.
type KVCompressTask struct {
	PageID     int
	FirstToken int
	SM         int
}

// Plan returns the pages to compress for a sequence of seqLen tokens when
// pages [0, compressedPages) are already compressed.
func (s KVCompressSchedule) Plan(seqLen, compressedPages int) ([]KVCompressTask, error) {
	if err := s.Partition.Validate(); err != nil {
		return nil, err
	}
	if s.PageTokens < 1 || s.HotWindowTokens < 0 || seqLen < 0 || compressedPages < 0 {
		return nil, fmt.Errorf("tcgen05 kv compress schedule: PageTokens=%d HotWindowTokens=%d seqLen=%d compressedPages=%d", s.PageTokens, s.HotWindowTokens, seqLen, compressedPages)
	}
	cold := (seqLen - s.HotWindowTokens) / s.PageTokens // pages entirely outside the hot window
	var tasks []KVCompressTask
	for pg := compressedPages; pg < cold; pg++ {
		tasks = append(tasks, KVCompressTask{PageID: pg, FirstToken: pg * s.PageTokens, SM: pg % s.Partition.BackgroundSMs})
	}
	return tasks, nil
}

// tcgen05MbarHelpers declares mbarrier init/wait helpers shared by kernels
// that restage shared memory between tcgen05.mma issues.
const tcgen05MbarHelpers = `#ifndef EG_TCGEN05_MBAR_HELPERS
#define EG_TCGEN05_MBAR_HELPERS
EG_DEV void eg_tcgen05_mbar_init(uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(s) : "memory");
#else
    *mbar = 0ULL;
#endif
}

// eg_tcgen05_mbar_wait spins until the phase with the given parity of mbar
// completes (the arrive comes from tcgen05.commit).
EG_DEV void eg_tcgen05_mbar_wait(uint64_t* mbar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t done = 0;
    do {
        asm volatile("{\n"
                     ".reg .pred P1;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, P1;\n"
                     "}" : "=r"(done) : "r"(s), "r"(parity) : "memory");
    } while (done == 0);
#else
    (void)mbar;
    (void)parity;
#endif
}

// eg_tcgen05_mma_tf32_ss issues tcgen05.mma.cta_group::1.kind::tf32 with A
// [m_rows, k_dim] and B^T [n_cols, k_dim] in shared memory. accumulate
// selects D += A B^T over D = A B^T.
EG_DEV void eg_tcgen05_mma_tf32_ss(
    float* tmem, uint32_t d_taddr, const float* smem_a, const float* smem_bt,
    int m_rows, int n_cols, int k_dim, uint32_t idesc, bool accumulate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    const uint64_t desc_a = eg_wgmma_desc(smem_a, (uint32_t)(k_dim * 4), (uint32_t)(8 * k_dim * 4));
    const uint64_t desc_b = eg_wgmma_desc(smem_bt, (uint32_t)(k_dim * 4), (uint32_t)(8 * k_dim * 4));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"((uint32_t)accumulate) : "memory");
#else
    (void)idesc;
    for (int r = 0; r < m_rows; ++r) {
        for (int n = 0; n < n_cols; ++n) {
            float acc = accumulate ? tmem[r * 512 + d_taddr + n] : 0.0f;
            for (int k = 0; k < k_dim; ++k) {
                acc += smem_a[r * k_dim + k] * smem_bt[n * k_dim + k];
            }
            tmem[r * 512 + d_taddr + n] = acc;
        }
    }
#endif
}
#endif // EG_TCGEN05_MBAR_HELPERS
`

// TCGen05KVSubspaceProject configures the background compressor's
// projection kernel: one CTA computes C = X U for a tile of Tokens KV rows
// X [Tokens, Dim] and a basis U [Dim, Rank], streaming Dim in 32-wide chunks
// through shared memory into a Rank-column TMEM accumulator with
// tcgen05.mma.cta_group::1.kind::tf32 and an mbarrier wait per chunk.
//
// The device path is emitted for sm_100a, sm_103a and sm_107a and has not
// been executed on tcgen05 hardware. Under EG_EMU (cudahostemu and sm_12x
// devices) an exact C++ fallback runs instead and is what the tests check.
// Quantizing the latents and swapping page pointers stay on the host side
// ([SubspaceKVProjector.CompressKVPage], [KVCompressSchedule]).
type TCGen05KVSubspaceProject struct {
	Name   string
	Tokens int    // rows per CTA; must be 128
	Dim    int    // full KV width; a positive multiple of 32
	Rank   int    // latent rank; 32, 64 or 128 (default 128)
	Target string // "sm_100a", "sm_103a", "sm_107a" or "sm_107f" (default "sm_100a")
}

func (k TCGen05KVSubspaceProject) norm() (TCGen05KVSubspaceProject, error) {
	if k.Name == "" {
		k.Name = "tcgen05_kv_subspace_project"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 kv subspace project: bad kernel name %q", k.Name)
	}
	if k.Tokens == 0 {
		k.Tokens = 128
	}
	if k.Rank == 0 {
		k.Rank = 128
	}
	if k.Target == "" {
		k.Target = "sm_100a"
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 kv subspace project %s: unsupported target %q", k.Name, k.Target)
	}
	if k.Tokens != 128 {
		return k, fmt.Errorf("tcgen05 kv subspace project %s: Tokens=%d, want 128", k.Name, k.Tokens)
	}
	if k.Dim < 32 || k.Dim%32 != 0 {
		return k, fmt.Errorf("tcgen05 kv subspace project %s: Dim=%d, want a positive multiple of 32", k.Name, k.Dim)
	}
	if k.Rank != 32 && k.Rank != 64 && k.Rank != 128 {
		return k, fmt.Errorf("tcgen05 kv subspace project %s: Rank=%d, want 32, 64 or 128", k.Name, k.Rank)
	}
	return k, nil
}

// Validate reports whether k describes a supported kernel.
func (k TCGen05KVSubspaceProject) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 128-thread block.
func (k TCGen05KVSubspaceProject) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

func (k TCGen05KVSubspaceProject) instDesc() uint32 {
	w, _ := tf32InstDesc(k.Tokens, k.Rank)
	return w
}

// TraceEvents returns the tcgen05 event sequence of the kernel for
// cudatmem.VerifyTrace.
func (k TCGen05KVSubspaceProject) TraceEvents() []cudatmem.TraceEvent {
	k, err := k.norm()
	if err != nil {
		return nil
	}
	r := uint32(k.Rank)
	ev := []cudatmem.TraceEvent{{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Columns: r, Rows: 128}}
	for range k.Dim / 32 {
		ev = append(ev,
			cudatmem.TraceEvent{Kind: cudatmem.EventMMA, ActiveThreads: 1, CTAGroup: 1, Columns: r, Rows: 128},
			cudatmem.TraceEvent{Kind: cudatmem.EventCommit, ActiveThreads: 1, CTAGroup: 1, Columns: r, Rows: 128},
			cudatmem.TraceEvent{Kind: cudatmem.EventWait, ActiveThreads: 32, CTAGroup: 1, Columns: r, Rows: 128},
		)
	}
	for w := range 4 {
		ev = append(ev, cudatmem.TraceEvent{Kind: cudatmem.EventLd, Warp: w, ActiveThreads: 32, CTAGroup: 1, Columns: r, RowStart: uint32(32 * w), Rows: 32})
	}
	return append(ev, cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Columns: r, Rows: 128})
}

// CUDA returns the kernel source without Prelude. The kernel signature is
//
//	NAME(const float* x, const float* basis, float* out)
//
// with x [Tokens, Dim], basis [Dim, Rank] and out [Tokens, Rank].
func (k TCGen05KVSubspaceProject) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(tcgen05MbarHelpers)
	fmt.Fprintf(&b, `
// %[1]s: %[2]s KV subspace projection C = X U (T=%[3]d, D=%[4]d, r=%[5]d).
__global__ void __launch_bounds__(128) %[1]s(
    const float* __restrict__ x,
    const float* __restrict__ basis,
    float* __restrict__ out) {
    constexpr int T = %[3]d, D = %[4]d, R = %[5]d, BK = 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = 32 * warp + lane;

    EG_TCGEN05_SHARED(float, smem_x, [T * BK]);
    EG_TCGEN05_SHARED(float, smem_ut, [R * BK]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot[0], 0u, (uint32_t)R);
    }
    if (tid == 0) eg_tcgen05_mbar_init(&mbar[0]);
    __syncthreads();
    const uint32_t d_taddr = slot[0];

    // Stream D in BK-wide chunks: stage X[:, chunk] and U[chunk, :]^T, issue
    // one MMA, and wait for it before the chunk's shared memory is reused.
    for (int kc = 0; kc < D / BK; ++kc) {
        for (int i = tid; i < T * BK; i += 128) smem_x[i] = x[(i / BK) * D + kc * BK + i %% BK];
        for (int i = tid; i < R * BK; i += 128) smem_ut[i] = basis[(kc * BK + i %% BK) * R + i / BK];
        __syncthreads();
        if (warp == 0 && eg_elect_one_sync()) {
            eg_tcgen05_mma_tf32_ss(tmem, d_taddr, smem_x, smem_ut, T, R, BK, 0x%08[6]xu, kc > 0);
            eg_tcgen05_commit_cta1(&mbar[0]);
        }
        eg_tcgen05_mbar_wait(&mbar[0], (uint32_t)(kc & 1));
        __syncthreads();
    }

    for (int n = 0; n < R; ++n) {
        out[row * R + n] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)n);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, (uint32_t)R);
    }
}
`, k.Name, k.Target, k.Tokens, k.Dim, k.Rank, k.instDesc())
	return b.String(), nil
}

// CUDAFile returns the self-contained translation unit (Prelude plus CUDA).
func (k TCGen05KVSubspaceProject) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference returns x [Tokens, Dim] times basis [Dim, Rank] accumulated in
// float32 in chunk order, as the EG_EMU fallback computes it.
func (k TCGen05KVSubspaceProject) Reference(x, basis []float32) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	if len(x) != k.Tokens*k.Dim || len(basis) != k.Dim*k.Rank {
		return nil, fmt.Errorf("tcgen05 kv subspace project %s reference: len(x)=%d len(basis)=%d, want %d and %d", k.Name, len(x), len(basis), k.Tokens*k.Dim, k.Dim*k.Rank)
	}
	out := make([]float32, k.Tokens*k.Rank)
	for t := range k.Tokens {
		for n := range k.Rank {
			var acc float32
			for i := range k.Dim {
				acc += x[t*k.Dim+i] * basis[i*k.Rank+n]
			}
			out[t*k.Rank+n] = acc
		}
	}
	return out, nil
}
