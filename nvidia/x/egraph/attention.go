package egraph

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrAttentionDimensionMismatch is returned when attention query, key, value, or mask dimensions do not align.
	ErrAttentionDimensionMismatch = errors.New("egraph: attention dimension mismatch")

	// ErrInvalidTilePartition is returned when tile size does not divide sequence length or is non-positive.
	ErrInvalidTilePartition = errors.New("egraph: invalid attention tile partition")
	ErrInvalidMaskIndex     = errors.New("egraph: invalid attention mask index")
)

// OnlineAttentionState represents the explicit running reduction state of online attention for a row.
type OnlineAttentionState struct {
	HasPrior bool
	M        float64   // Running maximum m
	D        float64   // Running normalizer d
	O        []float64 // Running unnormalized output accumulator O
}

// NewOnlineAttentionState initializes an online attention running state for a given head dimension.
func NewOnlineAttentionState(headDim int) *OnlineAttentionState {
	return &OnlineAttentionState{
		HasPrior: false,
		M:        0,
		D:        0,
		O:        make([]float64, headDim),
	}
}

// StepTile updates the running reduction state with a score tile and corresponding value tile.
//
// Recurrence:
//
//	m_new = max(m_prev, rowmax(S_ij))
//	alpha = exp(m_prev - m_new)
//	P_ij = exp(S_ij - m_new)
//	d_new = d_prev * alpha + sum(P_ij)
//	O_new = O_prev * alpha + P_ij * V_j
//
// Masking:
//
//	tileMask specifies invalid key entries (true = masked out).
//	Masked entries contribute zero to P_ij.
//	If all entries in the tile are masked, the state remains unchanged (no -Inf - -Inf evaluation).
func (s *OnlineAttentionState) StepTile(scoreTile []float32, valTile []float32, tileMask []bool, headDim int) error {
	if s == nil || headDim < 0 || len(s.O) != headDim {
		return fmt.Errorf("%w: invalid head dimension %d", ErrAttentionDimensionMismatch, headDim)
	}
	tileSize := len(scoreTile)
	if len(valTile) != tileSize*headDim {
		return fmt.Errorf("%w: valTile length %d != tileSize*headDim (%d*%d)",
			ErrAttentionDimensionMismatch, len(valTile), tileSize, headDim)
	}
	if len(tileMask) > 0 && len(tileMask) != tileSize {
		return fmt.Errorf("%w: tileMask length %d != tileSize %d",
			ErrAttentionDimensionMismatch, len(tileMask), tileSize)
	}

	// 1. Find tile maximum among valid (unmasked) elements
	tileHasValid := false
	var tileMax float64 = -math.MaxFloat64
	for i := 0; i < tileSize; i++ {
		if len(tileMask) > 0 && tileMask[i] {
			continue
		}
		sc := float64(scoreTile[i])
		if !tileHasValid || sc > tileMax {
			tileMax = sc
			tileHasValid = true
		}
	}

	if !tileHasValid {
		// All entries in this tile are masked; contributes zero to accumulation.
		return nil
	}

	// 2. Compute new maximum and scaling factor
	var mNew float64
	var alpha float64
	if !s.HasPrior {
		mNew = tileMax
		alpha = 1.0
		s.HasPrior = true
	} else {
		mNew = math.Max(s.M, tileMax)
		alpha = math.Exp(s.M - mNew)
	}

	// 3. Compute P_ij = exp(S_ij - m_new) and sum(P_ij)
	pVals := make([]float64, tileSize)
	var pSum float64
	for i := 0; i < tileSize; i++ {
		if len(tileMask) > 0 && tileMask[i] {
			pVals[i] = 0.0
		} else {
			p := math.Exp(float64(scoreTile[i]) - mNew)
			pVals[i] = p
			pSum += p
		}
	}

	// 4. Update normalizer d_new = d_prev * alpha + sum(P_ij)
	dNew := s.D*alpha + pSum

	// 5. Update unnormalized output O_new = O_prev * alpha + P_ij * V_j
	for d := 0; d < headDim; d++ {
		var pv float64
		for i := 0; i < tileSize; i++ {
			p := pVals[i]
			if p != 0 {
				pv += p * float64(valTile[i*headDim+d])
			}
		}
		s.O[d] = s.O[d]*alpha + pv
	}

	s.M = mNew
	s.D = dNew
	return nil
}

// Finalize produces the final normalized output vector O_final = O / d_final.
// If the row was completely masked (d == 0), returns an exact zero vector according to policy.
func (s *OnlineAttentionState) Finalize() []float32 {
	headDim := len(s.O)
	out := make([]float32, headDim)
	if s.D == 0 {
		// Completely masked row policy: return exact zeros without NaNs or Infinities.
		return out
	}
	for d := 0; d < headDim; d++ {
		out[d] = float32(s.O[d] / s.D)
	}
	return out
}

// OnlineAttentionRow runs the online attention recurrence for a query row over key/value tiles.
func OnlineAttentionRow(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) ([]float32, error) {
	if err := validateAttentionInputs(scores, values, mask, seqLen, headDim, tileSize); err != nil {
		return nil, err
	}
	if len(scores) != seqLen {
		return nil, fmt.Errorf("%w: scores len %d != seqLen %d", ErrAttentionDimensionMismatch, len(scores), seqLen)
	}
	if len(values) != seqLen*headDim {
		return nil, fmt.Errorf("%w: values len %d != seqLen*headDim %d", ErrAttentionDimensionMismatch, len(values), seqLen*headDim)
	}
	if len(mask) > 0 && len(mask) != seqLen {
		return nil, fmt.Errorf("%w: mask len %d != seqLen %d", ErrAttentionDimensionMismatch, len(mask), seqLen)
	}
	if tileSize <= 0 {
		tileSize = seqLen
	}
	if tileSize > seqLen {
		tileSize = seqLen
	}

	state := NewOnlineAttentionState(headDim)

	for start := 0; start < seqLen; start += tileSize {
		end := start + tileSize
		if end > seqLen {
			end = seqLen
		}
		tileScores := scores[start:end]
		tileValues := values[start*headDim : end*headDim]
		var tileMask []bool
		if len(mask) > 0 {
			tileMask = mask[start:end]
		}
		if err := state.StepTile(tileScores, tileValues, tileMask, headDim); err != nil {
			return nil, err
		}
	}

	return state.Finalize(), nil
}

// MaterializedAttentionReference is the dense reference oracle that materializes
// the full softmax row before multiplying by values.
func MaterializedAttentionReference(scores, values []float32, mask []bool, seqLen, headDim int) ([]float32, error) {
	if err := validateAttentionInputs(scores, values, mask, seqLen, headDim, 0); err != nil {
		return nil, err
	}
	if len(scores) != seqLen || len(values) != seqLen*headDim {
		return nil, ErrAttentionDimensionMismatch
	}
	if len(mask) > 0 && len(mask) != seqLen {
		return nil, ErrAttentionDimensionMismatch
	}

	// Output policy for completely masked row: exact zeros
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

	// 1. Compute row max over unmasked scores
	var maxScore float64 = -math.MaxFloat64
	for i := 0; i < seqLen; i++ {
		if len(mask) > 0 && mask[i] {
			continue
		}
		s := float64(scores[i])
		if s > maxScore {
			maxScore = s
		}
	}

	// 2. Compute softmax weights
	var sumExp float64
	p := make([]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		if len(mask) > 0 && mask[i] {
			p[i] = 0.0
		} else {
			expVal := math.Exp(float64(scores[i]) - maxScore)
			p[i] = expVal
			sumExp += expVal
		}
	}

	if sumExp == 0 {
		return make([]float32, headDim), nil
	}

	// 3. Matrix multiply P * V
	out := make([]float32, headDim)
	for d := 0; d < headDim; d++ {
		var acc float64
		for i := 0; i < seqLen; i++ {
			if p[i] != 0 {
				acc += (p[i] / sumExp) * float64(values[i*headDim+d])
			}
		}
		out[d] = float32(acc)
	}

	return out, nil
}

func validateAttentionInputs(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) error {
	if seqLen < 0 || headDim < 0 || tileSize < 0 {
		return fmt.Errorf("%w: negative dimension", ErrAttentionDimensionMismatch)
	}
	if headDim != 0 && seqLen > int(^uint(0)>>1)/headDim {
		return fmt.Errorf("%w: value extent overflow", ErrAttentionDimensionMismatch)
	}
	if len(scores) != seqLen || len(values) != seqLen*headDim {
		return ErrAttentionDimensionMismatch
	}
	if len(mask) > 0 && len(mask) != seqLen {
		return ErrAttentionDimensionMismatch
	}
	return nil
}

// CausalAttentionMask generates a causal mask for query index qIdx across key sequence length seqLen.
// True indicates masked out (key position > qIdx).
func CausalAttentionMask(qIdx, seqLen int) []bool {
	if seqLen < 0 {
		return nil
	}
	mask := make([]bool, seqLen)
	for k := 0; k < seqLen; k++ {
		if k > qIdx {
			mask[k] = true
		}
	}
	return mask
}

// CausalAttentionMaskChecked is the validating form for callers that need
// malformed query indices to be rejected rather than silently clamped.
func CausalAttentionMaskChecked(qIdx, seqLen int) ([]bool, error) {
	if seqLen < 0 || qIdx < 0 || qIdx >= seqLen {
		return nil, ErrInvalidMaskIndex
	}
	return CausalAttentionMask(qIdx, seqLen), nil
}

// SparseTreeMask generates a sparse tree/prefix mask given allowed key indices for a query.
// Positions not in allowedKeys are masked out (set to true).
func SparseTreeMask(allowedKeys []int, seqLen int) []bool {
	mask := make([]bool, seqLen)
	for i := range mask {
		mask[i] = true
	}
	for _, k := range allowedKeys {
		if k >= 0 && k < seqLen {
			mask[k] = false
		}
	}
	return mask
}

// SparseTreeMaskChecked rejects out-of-range key positions before constructing
// a sparse predicate. SparseTreeMask is retained as a convenience constructor
// for trusted callers.
func SparseTreeMaskChecked(allowedKeys []int, seqLen int) ([]bool, error) {
	if seqLen < 0 {
		return nil, ErrInvalidMaskIndex
	}
	for _, k := range allowedKeys {
		if k < 0 || k >= seqLen {
			return nil, ErrInvalidMaskIndex
		}
	}
	return SparseTreeMask(allowedKeys, seqLen), nil
}

// Mutant controls for verification:

// OnlineAttentionRowMutantInverseRescale simulates the inverse rescaling error:
// alpha = 1.0 / exp(m_prev - m_new)
func OnlineAttentionRowMutantInverseRescale(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) ([]float32, error) {
	if err := validateAttentionInputs(scores, values, mask, seqLen, headDim, tileSize); err != nil {
		return nil, err
	}
	if tileSize <= 0 {
		tileSize = seqLen
	}
	hasPrior := false
	var m, d float64
	O := make([]float64, headDim)

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
			// Buggy: inverted rescaling!
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

		dNew := d*alpha + pSum
		for dim := 0; dim < headDim; dim++ {
			var pv float64
			for i := start; i < end; i++ {
				p := tileP[i-start]
				if p != 0 {
					pv += p * float64(values[i*headDim+dim])
				}
			}
			O[dim] = O[dim]*alpha + pv
		}
		m = mNew
		d = dNew
	}

	if d == 0 {
		return make([]float32, headDim), nil
	}
	out := make([]float32, headDim)
	for dim := 0; dim < headDim; dim++ {
		out[dim] = float32(O[dim] / d)
	}
	return out, nil
}

// OnlineAttentionRowMutantOmitNormalizer simulates the un-rescaled accumulator / omitted normalizer error.
func OnlineAttentionRowMutantOmitNormalizer(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) ([]float32, error) {
	if err := validateAttentionInputs(scores, values, mask, seqLen, headDim, tileSize); err != nil {
		return nil, err
	}
	if tileSize <= 0 {
		tileSize = seqLen
	}
	hasPrior := false
	var m, d float64
	O := make([]float64, headDim)

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

		// Buggy: omit alpha*d accumulation!
		dNew := pSum
		for dim := 0; dim < headDim; dim++ {
			var pv float64
			for i := start; i < end; i++ {
				p := tileP[i-start]
				if p != 0 {
					pv += p * float64(values[i*headDim+dim])
				}
			}
			O[dim] = O[dim]*alpha + pv
		}
		m = mNew
		d = dNew
	}

	if d == 0 {
		return make([]float32, headDim), nil
	}
	out := make([]float32, headDim)
	for dim := 0; dim < headDim; dim++ {
		out[dim] = float32(O[dim] / d)
	}
	return out, nil
}
