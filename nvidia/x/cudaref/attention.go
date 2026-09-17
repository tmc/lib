package cudaref

import (
	"fmt"
	"math"
)

// OnlineAttentionRow computes the online softmax attention for a single query row across
// sequence key/value tiles according to the authoritative online attention contract.
//
// Masking policy:
// A true value in mask indicates that the key position is masked out (invalid).
// If a row is completely masked (all key positions invalid), the output policy produces
// an exact zero vector without NaNs or infinities.
func OnlineAttentionRow(scores, values []float32, mask []bool, seqLen, headDim int, tileSize int) ([]float32, error) {
	if len(scores) != seqLen {
		return nil, fmt.Errorf("cudaref: scores length %d != seqLen %d", len(scores), seqLen)
	}
	if len(values) != seqLen*headDim {
		return nil, fmt.Errorf("cudaref: values length %d != seqLen*headDim %d", len(values), seqLen*headDim)
	}
	if len(mask) > 0 && len(mask) != seqLen {
		return nil, fmt.Errorf("cudaref: mask length %d != seqLen %d", len(mask), seqLen)
	}
	if tileSize <= 0 {
		tileSize = seqLen
	}

	// Output policy for completely masked row: return all zeros
	allMasked := true
	for i := 0; i < seqLen; i++ {
		if len(mask) == 0 || !mask[i] {
			allMasked = false
			break
		}
	}
	if allMasked {
		return make([]float32, headDim), nil
	}

	// State variables
	hasPrior := false
	var m float64 // Running maximum
	var l float64 // Running normalizer
	U := make([]float64, headDim) // Running unnormalized output accumulator

	for start := 0; start < seqLen; start += tileSize {
		end := start + tileSize
		if end > seqLen {
			end = seqLen
		}

		// Find maximum in current tile among unmasked entries
		tileHasValid := false
		var tileMax float64 = -math.MaxFloat64
		for i := start; i < end; i++ {
			if len(mask) > 0 && mask[i] {
				continue
			}
			s := float64(scores[i])
			if !tileHasValid || s > tileMax {
				tileMax = s
				tileHasValid = true
			}
		}

		if !tileHasValid {
			// Entire tile is masked; contributes nothing to accumulator
			continue
		}

		var mNew float64
		var alpha float64
		if !hasPrior {
			mNew = tileMax
			alpha = 1.0
			hasPrior = true
		} else {
			mNew = math.Max(m, tileMax)
			alpha = math.Exp(m - mNew)
		}

		// Compute P = exp(S - mNew), masked entries contribute zero
		var pSum float64
		tileP := make([]float64, end-start)
		for i := start; i < end; i++ {
			idx := i - start
			if len(mask) > 0 && mask[i] {
				tileP[idx] = 0.0
			} else {
				p := math.Exp(float64(scores[i]) - mNew)
				tileP[idx] = p
				pSum += p
			}
		}

		// lNew = alpha * l + sum(P)
		lNew := alpha*l + pSum

		// UNew = alpha * U + P * V
		for d := 0; d < headDim; d++ {
			var pv float64
			for i := start; i < end; i++ {
				p := tileP[i-start]
				if p != 0 {
					pv += p * float64(values[i*headDim+d])
				}
			}
			U[d] = alpha*U[d] + pv
		}

		m = mNew
		l = lNew
	}

	if l == 0 {
		return make([]float32, headDim), nil
	}

	out := make([]float32, headDim)
	for d := 0; d < headDim; d++ {
		out[d] = float32(U[d] / l)
	}
	return out, nil
}

// OnlineAttentionRowMutantRescale implements a deliberate mutation: applying inverse rescaling (1/alpha).
// This serves as an oracle mutant control to ensure verification tests reject buggy rescaling.
func OnlineAttentionRowMutantRescale(scores, values []float32, mask []bool, seqLen, headDim int, tileSize int) ([]float32, error) {
	if tileSize <= 0 {
		tileSize = seqLen
	}
	hasPrior := false
	var m, l float64
	U := make([]float64, headDim)

	for start := 0; start < seqLen; start += tileSize {
		end := start + tileSize
		if end > seqLen {
			end = seqLen
		}
		tileHasValid := false
		var tileMax float64 = -math.MaxFloat64
		for i := start; i < end; i++ {
			if len(mask) > 0 && mask[i] {
				continue
			}
			s := float64(scores[i])
			if !tileHasValid || s > tileMax {
				tileMax = s
				tileHasValid = true
			}
		}
		if !tileHasValid {
			continue
		}

		var mNew, alpha float64
		if !hasPrior {
			mNew = tileMax
			alpha = 1.0
			hasPrior = true
		} else {
			mNew = math.Max(m, tileMax)
			// Buggy mutant: inverse rescaling 1.0 / alpha
			realAlpha := math.Exp(m - mNew)
			if realAlpha != 0 {
				alpha = 1.0 / realAlpha
			}
		}

		var pSum float64
		tileP := make([]float64, end-start)
		for i := start; i < end; i++ {
			idx := i - start
			if len(mask) > 0 && mask[i] {
				tileP[idx] = 0.0
			} else {
				p := math.Exp(float64(scores[i]) - mNew)
				tileP[idx] = p
				pSum += p
			}
		}

		lNew := alpha*l + pSum
		for d := 0; d < headDim; d++ {
			var pv float64
			for i := start; i < end; i++ {
				p := tileP[i-start]
				if p != 0 {
					pv += p * float64(values[i*headDim+d])
				}
			}
			U[d] = alpha*U[d] + pv
		}
		m = mNew
		l = lNew
	}

	if l == 0 {
		return make([]float32, headDim), nil
	}
	out := make([]float32, headDim)
	for d := 0; d < headDim; d++ {
		out[d] = float32(U[d] / l)
	}
	return out, nil
}

// OnlineAttentionRowMutantNoNormalizer implements a deliberate mutation: omitting normalizer accumulation.
// This serves as an oracle mutant control to ensure verification tests catch unaccumulated normalizers.
func OnlineAttentionRowMutantNoNormalizer(scores, values []float32, mask []bool, seqLen, headDim int, tileSize int) ([]float32, error) {
	if tileSize <= 0 {
		tileSize = seqLen
	}
	hasPrior := false
	var m, l float64
	U := make([]float64, headDim)

	for start := 0; start < seqLen; start += tileSize {
		end := start + tileSize
		if end > seqLen {
			end = seqLen
		}
		tileHasValid := false
		var tileMax float64 = -math.MaxFloat64
		for i := start; i < end; i++ {
			if len(mask) > 0 && mask[i] {
				continue
			}
			s := float64(scores[i])
			if !tileHasValid || s > tileMax {
				tileMax = s
				tileHasValid = true
			}
		}
		if !tileHasValid {
			continue
		}

		var mNew, alpha float64
		if !hasPrior {
			mNew = tileMax
			alpha = 1.0
			hasPrior = true
		} else {
			mNew = math.Max(m, tileMax)
			alpha = math.Exp(m - mNew)
		}

		var pSum float64
		tileP := make([]float64, end-start)
		for i := start; i < end; i++ {
			idx := i - start
			if len(mask) > 0 && mask[i] {
				tileP[idx] = 0.0
			} else {
				p := math.Exp(float64(scores[i]) - mNew)
				tileP[idx] = p
				pSum += p
			}
		}

		// Buggy mutant: omits alpha*l accumulation!
		lNew := pSum
		for d := 0; d < headDim; d++ {
			var pv float64
			for i := start; i < end; i++ {
				p := tileP[i-start]
				if p != 0 {
					pv += p * float64(values[i*headDim+d])
				}
			}
			U[d] = alpha*U[d] + pv
		}
		m = mNew
		l = lNew
	}

	if l == 0 {
		return make([]float32, headDim), nil
	}
	out := make([]float32, headDim)
	for d := 0; d < headDim; d++ {
		out[d] = float32(U[d] / l)
	}
	return out, nil
}
