package tcgen05

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// LearnedLUTBlock is one 32-element weight block quantized to 3-bit indices
// into its own 8-entry E4M3 codebook with a UE8M0 block scale, the operand
// form of PTX ISA 9.4 tcgen05.mma ... .decompress::lut::b.block32. Weight i
// decodes as FromE4M3Bits(Codebook[Indices[i]]) * DecodeUE8M0(ScaleUE8M0).
type LearnedLUTBlock struct {
	ScaleUE8M0 uint8
	Codebook   [8]uint8  // E4M3 centroid codes
	Indices    [32]uint8 // each in 0..7

	WeightedMSE float64 // sum_i h_i (w_i - wHat_i)^2 / sum_i h_i
	Iterations  int     // Lloyd-Max iterations run
}

// Dequantize returns the 32 weights that b encodes.
func (b LearnedLUTBlock) Dequantize() [32]float32 {
	var out [32]float32
	s := DecodeUE8M0(b.ScaleUE8M0)
	for i, q := range b.Indices {
		out[i] = FromE4M3Bits(b.Codebook[q&7]) * s
	}
	return out
}

// PackIndices packs the 32 3-bit indices into a 96-bit little-endian stream:
// index i occupies bits 3i..3i+2.
func (b LearnedLUTBlock) PackIndices() [12]byte {
	var p [12]byte
	for i, q := range b.Indices {
		bit := 3 * i
		v := uint16(q&7) << (bit % 8)
		p[bit/8] |= byte(v)
		if bit/8+1 < len(p) {
			p[bit/8+1] |= byte(v >> 8)
		}
	}
	return p
}

// UnpackIndices3Bit is the inverse of [LearnedLUTBlock.PackIndices].
func UnpackIndices3Bit(p [12]byte) [32]uint8 {
	var idx [32]uint8
	for i := range idx {
		bit := 3 * i
		v := uint16(p[bit/8])
		if bit/8+1 < len(p) {
			v |= uint16(p[bit/8+1]) << 8
		}
		idx[i] = uint8(v>>(bit%8)) & 7
	}
	return idx
}

// CodebookWords returns the codebook as the two little-endian 32-bit words
// copied into a 2-column TMEM LUT slot by tcgen05.cp.
func (b LearnedLUTBlock) CodebookWords() [2]uint32 {
	var w [2]uint32
	for i, c := range b.Codebook {
		w[i/4] |= uint32(c) << (8 * (i % 4))
	}
	return w
}

// ValidateLUTColumn reports whether col can hold a 2-column TMEM LUT for
// .decompress::lut::b: it must be even and leave room for both columns.
func ValidateLUTColumn(col uint32) error {
	if col%2 != 0 {
		return fmt.Errorf("tcgen05 lut column %d is not 2-column aligned", col)
	}
	if col > 510 {
		return fmt.Errorf("tcgen05 lut column %d leaves no room for 2 columns below 512", col)
	}
	return nil
}

// FitBlock32LearnedLUT fits an 8-entry E4M3 codebook to the weight block w
// by Hessian-weighted Lloyd-Max, minimizing
//
//	sum_i h_i (w_i - 2^(s-127) c_{q_i})^2
//
// over codebooks c on the E4M3 grid and indices q. The scale s is the
// smallest UE8M0 power of two with max|w| / 2^(s-127) <= 448. Centroids start
// at weighted quantiles with the largest h_i w_i^2 element pinned to an end
// centroid; each M-step rounds the weighted cluster mean to the nearest E4M3
// value, which is the best grid point for that cluster, so the objective
// never increases. Fitting stops when the assignment is stable or after
// maxIters iterations. hessianDiag entries must be positive and finite; pass
// all ones for unweighted MSE.
func FitBlock32LearnedLUT(w, hessianDiag [32]float32, maxIters int) (LearnedLUTBlock, error) {
	var amax float32
	for i := range w {
		if math.IsNaN(float64(w[i])) || math.IsInf(float64(w[i]), 0) {
			return LearnedLUTBlock{}, fmt.Errorf("tcgen05 learned lut: w[%d]=%v is not finite", i, w[i])
		}
		h := float64(hessianDiag[i])
		if !(h > 0) || math.IsInf(h, 0) {
			return LearnedLUTBlock{}, fmt.Errorf("tcgen05 learned lut: hessianDiag[%d]=%v must be positive and finite", i, hessianDiag[i])
		}
		amax = max(amax, float32(math.Abs(float64(w[i]))))
	}
	if maxIters < 1 {
		return LearnedLUTBlock{}, fmt.Errorf("tcgen05 learned lut: maxIters=%d must be >= 1", maxIters)
	}
	var b LearnedLUTBlock
	b.ScaleUE8M0 = UE8M0Bias
	if amax > 0 {
		b.ScaleUE8M0 = EncodeUE8M0(amax / 448)
	}
	s := float64(DecodeUE8M0(b.ScaleUE8M0))
	vals := make([]float64, 32)
	wts := make([]float64, 32)
	for i := range w {
		vals[i] = float64(w[i]) / s
		wts[i] = float64(hessianDiag[i])
	}
	fit := fitLUT8(vals, wts, maxIters)
	b.Codebook = fit.codebook
	copy(b.Indices[:], fit.indices)
	b.Iterations = fit.iters
	b.WeightedMSE = fit.mse * s * s
	return b, nil
}

// lutFit is the result of fitLUT8. history[i] is the weighted MSE after
// iteration i (history[0] is the initial codebook).
type lutFit struct {
	codebook [8]uint8
	indices  []uint8
	mse      float64
	iters    int
	history  []float64
}

// fitLUT8 runs weighted Lloyd-Max with E4M3-projected centroids on vals.
func fitLUT8(vals, wts []float64, maxIters int) lutFit {
	n := len(vals)
	var wsum float64
	for _, h := range wts {
		wsum += h
	}

	// Weighted-quantile initialization, with the dominant h*v^2 element pinned.
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(vals[a], vals[b]) })
	var cb [8]uint8
	var cum float64
	k := 0
	for _, i := range order {
		cum += wts[i]
		for k < 8 && cum >= (float64(k)+0.5)/8*wsum {
			cb[k] = E4M3Bits(float32(vals[i]))
			k++
		}
	}
	for ; k < 8; k++ {
		cb[k] = E4M3Bits(float32(vals[order[n-1]]))
	}
	top := 0
	for i := range vals {
		if wts[i]*vals[i]*vals[i] > wts[top]*vals[top]*vals[top] {
			top = i
		}
	}
	if vals[top] >= 0 {
		cb[7] = E4M3Bits(float32(vals[top]))
	} else {
		cb[0] = E4M3Bits(float32(vals[top]))
	}

	idx := make([]uint8, n)
	assign := func() (changed bool) {
		for i, v := range vals {
			best, bestD := uint8(0), math.Inf(1)
			for c := range uint8(8) {
				if d := math.Abs(v - float64(FromE4M3Bits(cb[c]))); d < bestD {
					best, bestD = c, d
				}
			}
			if idx[i] != best {
				changed = true
			}
			idx[i] = best
		}
		return changed
	}
	objective := func() float64 {
		var e float64
		for i, v := range vals {
			d := v - float64(FromE4M3Bits(cb[idx[i]]))
			e += wts[i] * d * d
		}
		return e / wsum
	}

	assign()
	fit := lutFit{history: []float64{objective()}}
	for fit.iters < maxIters {
		fit.iters++
		var num, den [8]float64
		for i, v := range vals {
			num[idx[i]] += wts[i] * v
			den[idx[i]] += wts[i]
		}
		for c := range 8 {
			if den[c] > 0 {
				cb[c] = E4M3Bits(float32(num[c] / den[c]))
				continue
			}
			// Re-seed an empty centroid at the worst-served element.
			worst, worstE := -1, 0.0
			for i, v := range vals {
				d := v - float64(FromE4M3Bits(cb[idx[i]]))
				if e := wts[i] * d * d; e > worstE {
					worst, worstE = i, e
				}
			}
			if worst >= 0 {
				cb[c] = E4M3Bits(float32(vals[worst]))
			}
		}
		changed := assign()
		fit.history = append(fit.history, objective())
		if !changed {
			break
		}
	}
	fit.codebook = cb
	fit.indices = idx
	fit.mse = fit.history[len(fit.history)-1]
	return fit
}

// UniformLUTBlock32 quantizes w with a fixed symmetric codebook of 8
// uniformly spaced levels spanning [-max|w|, max|w|], rounded to E4M3, using
// the same scale rule as [FitBlock32LearnedLUT]. It is the baseline the
// learned codebook is measured against; WeightedMSE is unweighted.
func UniformLUTBlock32(w [32]float32) LearnedLUTBlock {
	var amax float32
	for _, v := range w {
		amax = max(amax, float32(math.Abs(float64(v))))
	}
	var b LearnedLUTBlock
	b.ScaleUE8M0 = UE8M0Bias
	if amax > 0 {
		b.ScaleUE8M0 = EncodeUE8M0(amax / 448)
	}
	s := DecodeUE8M0(b.ScaleUE8M0)
	for c := range 8 {
		b.Codebook[c] = E4M3Bits(amax / s * float32(2*c-7) / 7)
	}
	var mse float64
	for i, v := range w {
		best, bestD := uint8(0), math.Inf(1)
		for c := range uint8(8) {
			if d := math.Abs(float64(v - FromE4M3Bits(b.Codebook[c])*s)); d < bestD {
				best, bestD = c, d
			}
		}
		b.Indices[i] = best
		mse += bestD * bestD
	}
	b.WeightedMSE = mse / 32
	return b
}

// QuantizeLearnedLUTB quantizes B [K, N] into the operand layout that
// [Quantize3BitLUTB] produces and [TCGen05LUTDecompressGEMM] consumes (one
// UE8M0 scale per column and 32-row block, one 8-entry E4M3 LUT per 8-column
// slice, a 3-bit index per element), but fits each slice's LUT by
// Hessian-weighted Lloyd-Max ([FitBlock32LearnedLUT]'s algorithm) instead of
// four unweighted refinement steps. hessianDiag [K] weights input channel k;
// nil means all ones.
func QuantizeLearnedLUTB(b, hessianDiag []float32, K, N, maxIters int) (indices, lutE4M3, bScales []uint8, dequantized []float32, err error) {
	if K <= 0 || K%MXBlock != 0 || N <= 0 || N%8 != 0 {
		return nil, nil, nil, nil, fmt.Errorf("quantize learned lut b: K=%d N=%d (want K a positive multiple of 32, N a positive multiple of 8)", K, N)
	}
	if len(b) != K*N {
		return nil, nil, nil, nil, fmt.Errorf("quantize learned lut b: len(b)=%d, want %d", len(b), K*N)
	}
	if hessianDiag != nil && len(hessianDiag) != K {
		return nil, nil, nil, nil, fmt.Errorf("quantize learned lut b: len(hessianDiag)=%d, want %d", len(hessianDiag), K)
	}
	if maxIters < 1 {
		return nil, nil, nil, nil, fmt.Errorf("quantize learned lut b: maxIters=%d must be >= 1", maxIters)
	}
	h := func(k int) float64 {
		if hessianDiag == nil {
			return 1
		}
		return float64(hessianDiag[k])
	}
	for k := range K {
		if v := h(k); !(v > 0) || math.IsInf(v, 0) {
			return nil, nil, nil, nil, fmt.Errorf("quantize learned lut b: hessianDiag[%d]=%v must be positive and finite", k, v)
		}
	}
	kBlocks := K / MXBlock
	indices = make([]uint8, K*N)
	lutE4M3 = make([]uint8, N)
	bScales = make([]uint8, N*kBlocks)
	dequantized = make([]float32, K*N)
	norm := make([]float64, K*N)
	for n := range N {
		for kb := range kBlocks {
			var amax float32
			for j := range MXBlock {
				amax = max(amax, float32(math.Abs(float64(b[(kb*MXBlock+j)*N+n]))))
			}
			sc := uint8(UE8M0Bias)
			if amax > 0 {
				sc = EncodeUE8M0(amax / 4) // same normalization as Quantize3BitLUTB
			}
			bScales[n*kBlocks+kb] = sc
			s := FromUE8M0Bits(sc)
			for j := range MXBlock {
				k := kb*MXBlock + j
				norm[k*N+n] = float64(b[k*N+n] / s)
			}
		}
	}
	vals := make([]float64, 8*K)
	wts := make([]float64, 8*K)
	for g := range N / 8 {
		for k := range K {
			for c := range 8 {
				vals[k*8+c] = norm[k*N+8*g+c]
				wts[k*8+c] = h(k)
			}
		}
		fit := fitLUT8(vals, wts, maxIters)
		copy(lutE4M3[8*g:], fit.codebook[:])
		for k := range K {
			for c := range 8 {
				n := 8*g + c
				q := fit.indices[k*8+c]
				indices[k*N+n] = q
				dequantized[k*N+n] = FromE4M3Bits(fit.codebook[q]) * FromUE8M0Bits(bScales[n*kBlocks+k/MXBlock])
			}
		}
	}
	return indices, lutE4M3, bScales, dequantized, nil
}
