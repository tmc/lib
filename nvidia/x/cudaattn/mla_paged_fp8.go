package cudaattn

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// PagedMLAConfig configures FlashMLA V4.1 paged decode attention.
type PagedMLAConfig struct {
	NumHeads        int
	HeadDimCKV      int
	HeadDimKPE      int
	PageBlockSize   int
	TopK            int
	WindowSize      int
	SoftmaxScale    float32
	HasAttnSink     bool
	HasExtraKVCache bool
	BatchInvariant  bool
	EmitFP8Output   bool
	ReturnLSE       bool
}

func (c PagedMLAConfig) normalized() PagedMLAConfig {
	if c.HeadDimCKV == 0 {
		c.HeadDimCKV = 512
	}
	if c.HeadDimKPE == 0 {
		c.HeadDimKPE = 64
	}
	if c.PageBlockSize == 0 {
		c.PageBlockSize = 64
	}
	if c.SoftmaxScale == 0 && (c.HeadDimCKV+c.HeadDimKPE) > 0 {
		c.SoftmaxScale = float32(1.0 / math.Sqrt(float64(c.HeadDimCKV+c.HeadDimKPE)))
	}
	return c
}

// Validate checks that c has supported dimensions.
func (c PagedMLAConfig) Validate() error {
	c = c.normalized()
	if c.NumHeads <= 0 || c.NumHeads > 128 {
		return fmt.Errorf("cudaattn: numHeads %d out of range [1, 128]: %w", c.NumHeads, cudaerrors.ErrUnsupported)
	}
	if c.HeadDimCKV <= 0 || c.HeadDimKPE < 0 || c.PageBlockSize <= 0 {
		return fmt.Errorf("cudaattn: invalid MLA dimensions ckv=%d kpe=%d block=%d: %w", c.HeadDimCKV, c.HeadDimKPE, c.PageBlockSize, cudaerrors.ErrUnsupported)
	}
	if !(c.SoftmaxScale > 0) || math.IsNaN(float64(c.SoftmaxScale)) || math.IsInf(float64(c.SoftmaxScale), 0) {
		return fmt.Errorf("cudaattn: invalid softmax scale %v: %w", c.SoftmaxScale, cudaerrors.ErrUnsupported)
	}
	return nil
}

// PadAttentionHeads snaps numHeads up to hardware-supported FlashMLA head counts ({64, 128}),
// returning cudaerrors.ErrUnsupported if numHeads <= 0 || numHeads > 128.
func PadAttentionHeads(numHeads int) (paddedHeads int, padCount int, err error) {
	if numHeads <= 0 || numHeads > 128 {
		return 0, 0, fmt.Errorf("cudaattn: attention head count %d out of FlashMLA range [1, 128]: %w", numHeads, cudaerrors.ErrUnsupported)
	}
	if numHeads <= 64 {
		return 64, 64 - numHeads, nil
	}
	return 128, 128 - numHeads, nil
}

// PermuteQBProj permutes q_b_proj weights by splitting the RoPE tail (qRopeDim)
// from [numHeads, qRopeDim/2, 2, inDim] and transposing pairs (dim 1, dim 2) to
// [numHeads, 2, qRopeDim/2, inDim] so FlashMLA's interleaved RoPE layout matches
// non-interleaved weights (MLA.weight_rearrange).
func PermuteQBProj(weights []float32, numHeads, qNopeDim, qRopeDim, inDim int) ([]float32, error) {
	if numHeads <= 0 || qNopeDim < 0 || qRopeDim <= 0 || qRopeDim%2 != 0 || inDim <= 0 {
		return nil, fmt.Errorf("cudaattn: invalid PermuteQBProj shape heads=%d nope=%d rope=%d in=%d: %w",
			numHeads, qNopeDim, qRopeDim, inDim, cudaerrors.ErrUnsupported)
	}
	headStride := (qNopeDim + qRopeDim) * inDim
	wantLen := numHeads * headStride
	if len(weights) != wantLen {
		return nil, fmt.Errorf("cudaattn: PermuteQBProj weights length %d != %d: %w", len(weights), wantLen, cudaerrors.ErrUnsupported)
	}

	out := make([]float32, wantLen)
	halfRope := qRopeDim / 2
	nopeElems := qNopeDim * inDim

	for h := 0; h < numHeads; h++ {
		headBase := h * headStride
		copy(out[headBase:headBase+nopeElems], weights[headBase:headBase+nopeElems])

		ropeSrc := weights[headBase+nopeElems : headBase+headStride]
		ropeDst := out[headBase+nopeElems : headBase+headStride]
		for r := 0; r < halfRope; r++ {
			for p := 0; p < 2; p++ {
				srcOff := (r*2 + p) * inDim
				dstOff := (p*halfRope + r) * inDim
				copy(ropeDst[dstOff:dstOff+inDim], ropeSrc[srcOff:srcOff+inDim])
			}
		}
	}
	return out, nil
}

// PermuteWVProj transposes w_v from [numHeads, vHeadDim, kvLoraRank] to
// [padHeads, kvLoraRank, vHeadDim] and zero-pads the head dimension up to padHeads ({64, 128}).
func PermuteWVProj(weights []float32, numHeads, kvLoraRank, vHeadDim int, padHeads int) ([]float32, error) {
	if numHeads <= 0 || kvLoraRank <= 0 || vHeadDim <= 0 {
		return nil, fmt.Errorf("cudaattn: invalid PermuteWVProj shape heads=%d rank=%d vDim=%d: %w",
			numHeads, kvLoraRank, vHeadDim, cudaerrors.ErrUnsupported)
	}
	if padHeads == 0 {
		var err error
		padHeads, _, err = PadAttentionHeads(numHeads)
		if err != nil {
			return nil, err
		}
	}
	if (padHeads != 64 && padHeads != 128) || padHeads < numHeads {
		return nil, fmt.Errorf("cudaattn: invalid padHeads %d for numHeads %d (want 64 or 128 >= numHeads): %w",
			padHeads, numHeads, cudaerrors.ErrUnsupported)
	}
	wantLen := numHeads * vHeadDim * kvLoraRank
	if len(weights) != wantLen {
		return nil, fmt.Errorf("cudaattn: PermuteWVProj weights length %d != %d: %w", len(weights), wantLen, cudaerrors.ErrUnsupported)
	}

	out := make([]float32, padHeads*kvLoraRank*vHeadDim)
	for h := 0; h < numHeads; h++ {
		srcHead := weights[h*vHeadDim*kvLoraRank : (h+1)*vHeadDim*kvLoraRank]
		dstHead := out[h*kvLoraRank*vHeadDim : (h+1)*kvLoraRank*vHeadDim]
		for v := 0; v < vHeadDim; v++ {
			for k := 0; k < kvLoraRank; k++ {
				dstHead[k*vHeadDim+v] = srcHead[v*kvLoraRank+k]
			}
		}
	}
	return out, nil
}

// PagedMLAResult holds the outputs of PagedMLAForwardCPU.
type PagedMLAResult struct {
	Output    []float32
	FP8Output []uint8
	FP8Scales []float32
	LSE       []float32
}

// PagedMLAForwardCPU executes the deterministic CPU reference for FlashMLA V4.1,
// supporting attention sinks, LSE output, extra KV cache LSE-weighted online softmax
// merge, and optional fused per-head/block FP8 output quantization.
func PagedMLAForwardCPU(
	qNope []float32,
	qPe []float32,
	kvNoPE []float32,
	kPE []float32,
	attnSink []float32,
	extraKVNoPE []float32,
	extraKPE []float32,
	extraLSE []float32,
	cfg PagedMLAConfig,
) (PagedMLAResult, error) {
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return PagedMLAResult{}, err
	}
	if len(qNope) != cfg.NumHeads*cfg.HeadDimCKV {
		return PagedMLAResult{}, fmt.Errorf("cudaattn: qNope length %d != %d: %w", len(qNope), cfg.NumHeads*cfg.HeadDimCKV, cudaerrors.ErrUnsupported)
	}
	if len(qPe) != cfg.NumHeads*cfg.HeadDimKPE {
		return PagedMLAResult{}, fmt.Errorf("cudaattn: qPe length %d != %d: %w", len(qPe), cfg.NumHeads*cfg.HeadDimKPE, cudaerrors.ErrUnsupported)
	}
	if len(kvNoPE) == 0 || len(kvNoPE)%cfg.HeadDimCKV != 0 {
		return PagedMLAResult{}, fmt.Errorf("cudaattn: kvNoPE length %d must be a positive multiple of %d: %w", len(kvNoPE), cfg.HeadDimCKV, cudaerrors.ErrUnsupported)
	}
	numKV := len(kvNoPE) / cfg.HeadDimCKV
	if len(kPE) != numKV*cfg.HeadDimKPE {
		return PagedMLAResult{}, fmt.Errorf("cudaattn: kPE length %d != %d: %w", len(kPE), numKV*cfg.HeadDimKPE, cudaerrors.ErrUnsupported)
	}
	if cfg.HasAttnSink && len(attnSink) != cfg.NumHeads {
		return PagedMLAResult{}, fmt.Errorf("cudaattn: attnSink length %d != numHeads %d: %w", len(attnSink), cfg.NumHeads, cudaerrors.ErrUnsupported)
	}

	var numExtraKV int
	if cfg.HasExtraKVCache {
		if len(extraKVNoPE) == 0 || len(extraKVNoPE)%cfg.HeadDimCKV != 0 {
			return PagedMLAResult{}, fmt.Errorf("cudaattn: extraKVNoPE length %d must be a positive multiple of %d: %w", len(extraKVNoPE), cfg.HeadDimCKV, cudaerrors.ErrUnsupported)
		}
		numExtraKV = len(extraKVNoPE) / cfg.HeadDimCKV
		if len(extraKPE) != 0 && len(extraKPE) != numExtraKV*cfg.HeadDimKPE {
			return PagedMLAResult{}, fmt.Errorf("cudaattn: extraKPE length %d != %d: %w", len(extraKPE), numExtraKV*cfg.HeadDimKPE, cudaerrors.ErrUnsupported)
		}
		if len(extraLSE) != 0 && len(extraLSE) != cfg.NumHeads && len(extraLSE) != numExtraKV {
			return PagedMLAResult{}, fmt.Errorf("cudaattn: extraLSE length %d invalid: %w", len(extraLSE), cudaerrors.ErrUnsupported)
		}
	}

	startKV := 0
	if cfg.WindowSize > 0 && numKV > cfg.WindowSize {
		startKV = numKV - cfg.WindowSize
	}
	if cfg.TopK > 0 && (numKV-startKV) > cfg.TopK {
		startKV = numKV - cfg.TopK
	}

	output := make([]float32, cfg.NumHeads*cfg.HeadDimCKV)
	lseOut := make([]float32, cfg.NumHeads)
	scale := float64(cfg.SoftmaxScale)

	for h := 0; h < cfg.NumHeads; h++ {
		qN := qNope[h*cfg.HeadDimCKV : (h+1)*cfg.HeadDimCKV]
		qP := qPe[h*cfg.HeadDimKPE : (h+1)*cfg.HeadDimKPE]

		activeKV := numKV - startKV
		scores := make([]float64, activeKV)
		maxLogit := math.Inf(-1)
		for idx := 0; idx < activeKV; idx++ {
			j := startKV + idx
			kvN := kvNoPE[j*cfg.HeadDimCKV : (j+1)*cfg.HeadDimCKV]
			kP := kPE[j*cfg.HeadDimKPE : (j+1)*cfg.HeadDimKPE]
			var dot float64
			for d := 0; d < cfg.HeadDimCKV; d++ {
				dot += float64(qN[d]) * float64(kvN[d])
			}
			for d := 0; d < cfg.HeadDimKPE; d++ {
				dot += float64(qP[d]) * float64(kP[d])
			}
			s := dot * scale
			scores[idx] = s
			if s > maxLogit {
				maxLogit = s
			}
		}

		m := maxLogit
		if cfg.HasAttnSink {
			sinkVal := float64(attnSink[h])
			if sinkVal > m {
				m = sinkVal
			}
		}

		var sumExp float64
		acc := make([]float64, cfg.HeadDimCKV)
		for idx := 0; idx < activeKV; idx++ {
			j := startKV + idx
			w := math.Exp(scores[idx] - m)
			sumExp += w
			kvN := kvNoPE[j*cfg.HeadDimCKV : (j+1)*cfg.HeadDimCKV]
			for d := 0; d < cfg.HeadDimCKV; d++ {
				acc[d] += w * float64(kvN[d])
			}
		}
		denom := sumExp
		if cfg.HasAttnSink {
			denom += math.Exp(float64(attnSink[h]) - m)
		}

		mainLSE := m + math.Log(denom)
		mainOut := make([]float64, cfg.HeadDimCKV)
		if denom > 0 {
			for d := 0; d < cfg.HeadDimCKV; d++ {
				mainOut[d] = acc[d] / denom
			}
		}

		if cfg.HasExtraKVCache {
			if len(extraKPE) == 0 && numExtraKV == cfg.NumHeads && len(extraLSE) == cfg.NumHeads {
				// extraKVNoPE is pre-aggregated per-head partial output [numHeads, HeadDimCKV] with extraLSE [numHeads].
				extLSE := float64(extraLSE[h])
				extOut := extraKVNoPE[h*cfg.HeadDimCKV : (h+1)*cfg.HeadDimCKV]
				mComb := math.Max(mainLSE, extLSE)
				wMain := math.Exp(mainLSE - mComb)
				wExt := math.Exp(extLSE - mComb)
				wSum := wMain + wExt
				mainLSE = mComb + math.Log(wSum)
				for d := 0; d < cfg.HeadDimCKV; d++ {
					mainOut[d] = (wMain*mainOut[d] + wExt*float64(extOut[d])) / wSum
				}
			} else {
				// extraKVNoPE and extraKPE are extra KV tokens [numExtraKV, ...].
				extScores := make([]float64, numExtraKV)
				extMax := math.Inf(-1)
				for j := 0; j < numExtraKV; j++ {
					kvN := extraKVNoPE[j*cfg.HeadDimCKV : (j+1)*cfg.HeadDimCKV]
					var dot float64
					for d := 0; d < cfg.HeadDimCKV; d++ {
						dot += float64(qN[d]) * float64(kvN[d])
					}
					if len(extraKPE) > 0 {
						kP := extraKPE[j*cfg.HeadDimKPE : (j+1)*cfg.HeadDimKPE]
						for d := 0; d < cfg.HeadDimKPE; d++ {
							dot += float64(qP[d]) * float64(kP[d])
						}
					}
					s := dot * scale
					extScores[j] = s
					if s > extMax {
						extMax = s
					}
				}
				var extSum float64
				extAcc := make([]float64, cfg.HeadDimCKV)
				for j := 0; j < numExtraKV; j++ {
					w := math.Exp(extScores[j] - extMax)
					extSum += w
					kvN := extraKVNoPE[j*cfg.HeadDimCKV : (j+1)*cfg.HeadDimCKV]
					for d := 0; d < cfg.HeadDimCKV; d++ {
						extAcc[d] += w * float64(kvN[d])
					}
				}
				extLSE := extMax + math.Log(extSum)
				if len(extraLSE) == cfg.NumHeads {
					extLSE = float64(extraLSE[h])
				}
				extOut := make([]float64, cfg.HeadDimCKV)
				if extSum > 0 {
					for d := 0; d < cfg.HeadDimCKV; d++ {
						extOut[d] = extAcc[d] / extSum
					}
				}
				mComb := math.Max(mainLSE, extLSE)
				wMain := math.Exp(mainLSE - mComb)
				wExt := math.Exp(extLSE - mComb)
				wSum := wMain + wExt
				mainLSE = mComb + math.Log(wSum)
				for d := 0; d < cfg.HeadDimCKV; d++ {
					mainOut[d] = (wMain*mainOut[d] + wExt*extOut[d]) / wSum
				}
			}
		}

		dst := output[h*cfg.HeadDimCKV : (h+1)*cfg.HeadDimCKV]
		for d := 0; d < cfg.HeadDimCKV; d++ {
			dst[d] = float32(mainOut[d])
		}
		lseOut[h] = float32(mainLSE)
	}

	res := PagedMLAResult{
		Output: output,
	}
	if cfg.ReturnLSE || cfg.HasAttnSink || cfg.HasExtraKVCache {
		res.LSE = lseOut
	}
	if cfg.EmitFP8Output {
		fp8Bytes, fp8Scales := quantizeFP8Blocks(output, 32)
		res.FP8Output = fp8Bytes
		res.FP8Scales = fp8Scales
	}
	return res, nil
}

// NormRoPEHadamardKVConfig configures fused RMSNorm, RoPE, Walsh-Hadamard transform,
// and block quantization for KV cache storage.
type NormRoPEHadamardKVConfig struct {
	Tokens        int
	NopeDim       int
	RopeDim       int
	HadamardBlock int
	Eps           float32
	StoreFP4      bool
}

func (c NormRoPEHadamardKVConfig) normalized() NormRoPEHadamardKVConfig {
	if c.NopeDim == 0 {
		c.NopeDim = 512
	}
	if c.RopeDim == 0 {
		c.RopeDim = 64
	}
	if c.HadamardBlock == 0 {
		c.HadamardBlock = 64
	}
	if c.Eps == 0 {
		c.Eps = 1e-6
	}
	return c
}

// Validate checks that c has valid dimensions and a power-of-two HadamardBlock.
func (c NormRoPEHadamardKVConfig) Validate() error {
	c = c.normalized()
	if c.Tokens <= 0 {
		return fmt.Errorf("cudaattn: tokens %d must be positive: %w", c.Tokens, cudaerrors.ErrUnsupported)
	}
	if c.NopeDim <= 0 || c.RopeDim <= 0 || c.RopeDim%2 != 0 {
		return fmt.Errorf("cudaattn: invalid nopeDim=%d ropeDim=%d: %w", c.NopeDim, c.RopeDim, cudaerrors.ErrUnsupported)
	}
	if c.HadamardBlock <= 0 || (c.HadamardBlock&(c.HadamardBlock-1)) != 0 {
		return fmt.Errorf("cudaattn: hadamardBlock %d must be a power of two: %w", c.HadamardBlock, cudaerrors.ErrUnsupported)
	}
	if c.NopeDim%c.HadamardBlock != 0 || c.RopeDim%c.HadamardBlock != 0 {
		return fmt.Errorf("cudaattn: nopeDim=%d and ropeDim=%d must be multiples of hadamardBlock=%d: %w",
			c.NopeDim, c.RopeDim, c.HadamardBlock, cudaerrors.ErrUnsupported)
	}
	if !(c.Eps > 0) || math.IsNaN(float64(c.Eps)) || math.IsInf(float64(c.Eps), 0) {
		return fmt.Errorf("cudaattn: invalid eps %v: %w", c.Eps, cudaerrors.ErrUnsupported)
	}
	return nil
}

// FusedNormRoPEHadamardQuantStoreKVCPU fuses RMSNorm on ckv, RoPE rotation on kpe,
// normalized Walsh-Hadamard butterfly transform across both ckv and kpe, and block
// quantization into FP8/FP4 cache slots.
func FusedNormRoPEHadamardQuantStoreKVCPU(
	ckv []float32,
	kpe []float32,
	rmsWeight []float32,
	cosSin []float32,
	cfg NormRoPEHadamardKVConfig,
) (quantNoPE []byte, scales []float32, rotKPE []float32, err error) {
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	if len(ckv) != cfg.Tokens*cfg.NopeDim {
		return nil, nil, nil, fmt.Errorf("cudaattn: ckv length %d != %d: %w", len(ckv), cfg.Tokens*cfg.NopeDim, cudaerrors.ErrUnsupported)
	}
	if len(kpe) != cfg.Tokens*cfg.RopeDim {
		return nil, nil, nil, fmt.Errorf("cudaattn: kpe length %d != %d: %w", len(kpe), cfg.Tokens*cfg.RopeDim, cudaerrors.ErrUnsupported)
	}
	if len(rmsWeight) != cfg.NopeDim {
		return nil, nil, nil, fmt.Errorf("cudaattn: rmsWeight length %d != %d: %w", len(rmsWeight), cfg.NopeDim, cudaerrors.ErrUnsupported)
	}
	if len(cosSin) != cfg.Tokens*cfg.RopeDim && len(cosSin) != cfg.RopeDim {
		return nil, nil, nil, fmt.Errorf("cudaattn: cosSin length %d must be %d or %d: %w", len(cosSin), cfg.Tokens*cfg.RopeDim, cfg.RopeDim, cudaerrors.ErrUnsupported)
	}

	transformedNoPE := make([]float32, cfg.Tokens*cfg.NopeDim)
	rotKPE = make([]float32, cfg.Tokens*cfg.RopeDim)
	invSqrtHad := float64(1.0 / math.Sqrt(float64(cfg.HadamardBlock)))

	for t := 0; t < cfg.Tokens; t++ {
		ckvRow := ckv[t*cfg.NopeDim : (t+1)*cfg.NopeDim]
		outNoPERow := transformedNoPE[t*cfg.NopeDim : (t+1)*cfg.NopeDim]

		// 1. RMSNorm on ckv.
		var sumSq float64
		for d := 0; d < cfg.NopeDim; d++ {
			v := float64(ckvRow[d])
			sumSq += v * v
		}
		invRMS := 1.0 / math.Sqrt(sumSq/float64(cfg.NopeDim)+float64(cfg.Eps))
		for d := 0; d < cfg.NopeDim; d++ {
			outNoPERow[d] = float32(float64(ckvRow[d]) * invRMS * float64(rmsWeight[d]))
		}

		// 2. Normalized Walsh-Hadamard transform on ckv in blocks of HadamardBlock.
		for b := 0; b < cfg.NopeDim; b += cfg.HadamardBlock {
			walshHadamardInPlace(outNoPERow[b:b+cfg.HadamardBlock], invSqrtHad)
		}

		// 3. Interleaved RoPE rotation on kpe.
		kpeRow := kpe[t*cfg.RopeDim : (t+1)*cfg.RopeDim]
		outKPERow := rotKPE[t*cfg.RopeDim : (t+1)*cfg.RopeDim]
		csRow := cosSin
		if len(cosSin) == cfg.Tokens*cfg.RopeDim {
			csRow = cosSin[t*cfg.RopeDim : (t+1)*cfg.RopeDim]
		}
		for i := 0; i < cfg.RopeDim; i += 2 {
			x0 := float64(kpeRow[i])
			x1 := float64(kpeRow[i+1])
			c := float64(csRow[i])
			s := float64(csRow[i+1])
			outKPERow[i] = float32(x0*c - x1*s)
			outKPERow[i+1] = float32(x1*c + x0*s)
		}

		// 4. Normalized Walsh-Hadamard transform on rotated kpe in blocks of HadamardBlock.
		for b := 0; b < cfg.RopeDim; b += cfg.HadamardBlock {
			walshHadamardInPlace(outKPERow[b:b+cfg.HadamardBlock], invSqrtHad)
		}
	}

	// 5. Block quantization of transformedNoPE into FP8 or FP4.
	if cfg.StoreFP4 {
		quantNoPE, scales = quantizeFP4Blocks(transformedNoPE, cfg.HadamardBlock)
	} else {
		quantNoPE, scales = quantizeFP8Blocks(transformedNoPE, cfg.HadamardBlock)
	}
	return quantNoPE, scales, rotKPE, nil
}

func walshHadamardInPlace(block []float32, scale float64) {
	n := len(block)
	buf := make([]float64, n)
	for i, v := range block {
		buf[i] = float64(v)
	}
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				a := buf[j]
				b := buf[j+h]
				buf[j] = a + b
				buf[j+h] = a - b
			}
		}
	}
	for i := range block {
		block[i] = float32(buf[i] * scale)
	}
}

func quantizeFP8Blocks(src []float32, blockSize int) ([]uint8, []float32) {
	if blockSize <= 0 {
		blockSize = 32
	}
	numBlocks := (len(src) + blockSize - 1) / blockSize
	out := make([]uint8, len(src))
	scales := make([]float32, numBlocks)
	const fp8Max = 448.0

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(src) {
			end = len(src)
		}
		var amax float64
		for i := start; i < end; i++ {
			v := math.Abs(float64(src[i]))
			if v > amax {
				amax = v
			}
		}
		scale := float32(amax / fp8Max)
		if scale == 0 {
			scale = 1.0
		}
		scales[b] = scale
		invScale := 127.0 / (float64(scale) * fp8Max)
		for i := start; i < end; i++ {
			q := math.Round(float64(src[i]) * invScale)
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			out[i] = uint8(int8(q))
		}
	}
	return out, scales
}

func quantizeFP4Blocks(src []float32, blockSize int) ([]uint8, []float32) {
	if blockSize <= 0 {
		blockSize = 64
	}
	numBlocks := (len(src) + blockSize - 1) / blockSize
	out := make([]uint8, (len(src)+1)/2)
	scales := make([]float32, numBlocks)
	const fp4Max = 6.0

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(src) {
			end = len(src)
		}
		var amax float64
		for i := start; i < end; i++ {
			v := math.Abs(float64(src[i]))
			if v > amax {
				amax = v
			}
		}
		scale := float32(amax / fp4Max)
		if scale == 0 {
			scale = 1.0
		}
		scales[b] = scale
		for i := start; i < end; i += 2 {
			n0 := encodeFP4Nibble(src[i] / scale)
			var n1 uint8
			if i+1 < end {
				n1 = encodeFP4Nibble(src[i+1] / scale)
			}
			out[i/2] = (n1 << 4) | (n0 & 0x0F)
		}
	}
	return out, scales
}

func encodeFP4Nibble(v float32) uint8 {
	var sign uint8
	if v < 0 {
		sign = 0x8
		v = -v
	}
	// E2M1 positive grid: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
	grid := [8]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
	bestIdx := uint8(0)
	bestDiff := float32(math.MaxFloat32)
	for idx, g := range grid {
		diff := float32(math.Abs(float64(v - g)))
		if diff < bestDiff {
			bestDiff = diff
			bestIdx = uint8(idx)
		}
	}
	if bestIdx == 0 {
		return 0
	}
	return sign | bestIdx
}

// FusedNormRoPEHadamardCastStoreKVSource generates NVRTC-compilable CUDA C++ kernel source
// with warp-butterfly Walsh-Hadamard transform (__shfl_xor_sync), RMSNorm, RoPE, and
// quantized KV page write.
func FusedNormRoPEHadamardCastStoreKVSource(cfg NormRoPEHadamardKVConfig) (string, error) {
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return "", err
	}
	quantMode := "FP8"
	if cfg.StoreFP4 {
		quantMode = "FP4"
	}
	src := fmt.Sprintf(`// Generated by github.com/tmc/lib/nvidia/x/cudaattn.FusedNormRoPEHadamardCastStoreKVSource
// Mode=%s NopeDim=%d RopeDim=%d HadamardBlock=%d Eps=%.9g
extern "C" __global__ void fused_norm_rope_hadamard_cast_store_kv_kernel(
    const float* __restrict__ ckv,
    const float* __restrict__ kpe,
    const float* __restrict__ rms_weight,
    const float* __restrict__ cos_sin,
    unsigned char* __restrict__ quant_nope,
    float* __restrict__ scales,
    float* __restrict__ rot_kpe,
    int tokens
) {
    int token = blockIdx.x;
    int tid = threadIdx.x;
    if (token >= tokens) return;

    const int nope_dim = %d;
    const int rope_dim = %d;
    const int had_block = %d;
    const float eps = %.9ff;

    extern __shared__ float smem[];
    float* s_work = smem;

    // 1. RMSNorm over ckv[token, 0..nope_dim)
    float sum_sq = 0.0f;
    for (int d = tid; d < nope_dim; d += blockDim.x) {
        float v = ckv[token * nope_dim + d];
        sum_sq += v * v;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, mask);
    }
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float wsum = (tid < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            wsum += __shfl_xor_sync(0xffffffffu, wsum, mask);
        }
        if (lane == 0) warp_sums[0] = rsqrtf(wsum / (float)nope_dim + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];

    for (int d = tid; d < nope_dim; d += blockDim.x) {
        s_work[d] = ckv[token * nope_dim + d] * inv_rms * rms_weight[d];
    }
    __syncthreads();

    // 2. Warp-butterfly Walsh-Hadamard transform over each had_block
    const float inv_sqrt_had = rsqrtf((float)had_block);
    for (int base = 0; base < nope_dim; base += had_block) {
        for (int stride = 1; stride < had_block; stride <<= 1) {
            for (int idx = tid; idx < (had_block >> 1); idx += blockDim.x) {
                int j = ((idx / stride) * (stride << 1)) + (idx %% stride);
                float a = s_work[base + j];
                float b = s_work[base + j + stride];
                s_work[base + j] = a + b;
                s_work[base + j + stride] = a - b;
            }
            __syncthreads();
        }
        for (int d = tid; d < had_block; d += blockDim.x) {
            s_work[base + d] *= inv_sqrt_had;
        }
        __syncthreads();
    }

    // 3. Interleaved RoPE on kpe + Hadamard
    for (int i = tid * 2; i < rope_dim; i += blockDim.x * 2) {
        float x0 = kpe[token * rope_dim + i];
        float x1 = kpe[token * rope_dim + i + 1];
        float c = cos_sin[token * rope_dim + i];
        float s = cos_sin[token * rope_dim + i + 1];
        rot_kpe[token * rope_dim + i] = (x0 * c - x1 * s) * inv_sqrt_had;
        rot_kpe[token * rope_dim + i + 1] = (x1 * c + x0 * s) * inv_sqrt_had;
    }

    // 4. Block quantization store
    int num_blocks = nope_dim / had_block;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        float amax = 0.0f;
        for (int d = 0; d < had_block; d++) {
            amax = fmaxf(amax, fabsf(s_work[b * had_block + d]));
        }
        float sc = amax > 0.0f ? (amax / 448.0f) : 1.0f;
        scales[token * num_blocks + b] = sc;
        for (int d = 0; d < had_block; d++) {
            float q = rintf(s_work[b * had_block + d] * (127.0f / (sc * 448.0f)));
            q = fminf(127.0f, fmaxf(-127.0f, q));
            quant_nope[token * nope_dim + b * had_block + d] = (unsigned char)((signed char)q);
        }
    }
}
`,
		quantMode,
		cfg.NopeDim,
		cfg.RopeDim,
		cfg.HadamardBlock,
		cfg.Eps,
		cfg.NopeDim,
		cfg.RopeDim,
		cfg.HadamardBlock,
		cfg.Eps,
	)
	return src, nil
}
