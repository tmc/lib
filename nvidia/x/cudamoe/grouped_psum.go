// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"fmt"
	"math"
	"sort"
)

// GroupedPsumGEMMConfig configures per-expert grouped matrix multiplication
// with optional chained partial-sum (psum) accumulation.
type GroupedPsumGEMMConfig struct {
	Groups         int
	M              []int
	N              int
	K              int
	AccumulatePsum bool
	ScaleD         float32
}

// GroupedGEMMPsumCPU computes per-expert grouped matrix multiplication
// D_g = ScaleD * (A_g * B_g) + psum_g, simulating Blackwell UMMA
// AccumulateInPsumLayout chained accumulation without intermediate zero-fill.
func GroupedGEMMPsumCPU(a [][]float32, b [][]float32, psum [][]float32, cfg GroupedPsumGEMMConfig) ([][]float32, error) {
	if cfg.Groups <= 0 {
		return nil, fmt.Errorf("cudamoe: Groups %d must be positive", cfg.Groups)
	}
	if len(cfg.M) != cfg.Groups {
		return nil, fmt.Errorf("cudamoe: len(M) %d must equal Groups %d", len(cfg.M), cfg.Groups)
	}
	if cfg.N <= 0 || cfg.K <= 0 {
		return nil, fmt.Errorf("cudamoe: N %d and K %d must be positive", cfg.N, cfg.K)
	}
	if len(a) != cfg.Groups || len(b) != cfg.Groups {
		return nil, fmt.Errorf("cudamoe: len(a)=%d and len(b)=%d must equal Groups=%d", len(a), len(b), cfg.Groups)
	}
	if cfg.AccumulatePsum && len(psum) != cfg.Groups {
		return nil, fmt.Errorf("cudamoe: len(psum)=%d must equal Groups=%d when AccumulatePsum is true", len(psum), cfg.Groups)
	}
	if !finite(cfg.ScaleD) {
		return nil, fmt.Errorf("cudamoe: ScaleD %v must be finite", cfg.ScaleD)
	}
	scale := cfg.ScaleD
	if scale == 0 {
		scale = 1.0
	}

	out := make([][]float32, cfg.Groups)
	for g := 0; g < cfg.Groups; g++ {
		m := cfg.M[g]
		if m < 0 {
			return nil, fmt.Errorf("cudamoe: group %d has negative M=%d", g, m)
		}
		if len(a[g]) != m*cfg.K {
			return nil, fmt.Errorf("cudamoe: group %d len(a)=%d, want %d*%d=%d", g, len(a[g]), m, cfg.K, m*cfg.K)
		}
		if len(b[g]) != cfg.K*cfg.N {
			return nil, fmt.Errorf("cudamoe: group %d len(b)=%d, want %d*%d=%d", g, len(b[g]), cfg.K, cfg.N, cfg.K*cfg.N)
		}
		if cfg.AccumulatePsum && len(psum[g]) != m*cfg.N {
			return nil, fmt.Errorf("cudamoe: group %d len(psum)=%d, want %d*%d=%d", g, len(psum[g]), m, cfg.N, m*cfg.N)
		}

		dg := make([]float32, m*cfg.N)
		ag := a[g]
		bg := b[g]
		var pg []float32
		if cfg.AccumulatePsum {
			pg = psum[g]
		}
		for i := 0; i < m; i++ {
			aRow := ag[i*cfg.K : (i+1)*cfg.K]
			dRow := dg[i*cfg.N : (i+1)*cfg.N]
			for j := 0; j < cfg.N; j++ {
				var sum float32
				for k := 0; k < cfg.K; k++ {
					sum += aRow[k] * bg[k*cfg.N+j]
				}
				val := scale * sum
				if cfg.AccumulatePsum {
					val += pg[i*cfg.N+j]
				}
				dRow[j] = val
			}
		}
		out[g] = dg
	}
	return out, nil
}

// ComputeCommSMPartition partitions totalSMs into overlap communication SMs
// and compute SMs (totalSMs - commSMs), clamping commSMs to even CTA pairs in
// [minCommSMs, totalSMs-2].
func ComputeCommSMPartition(totalSMs int, commRatio float64, minCommSMs int) (commSMs int, computeSMs int, err error) {
	if totalSMs < 4 {
		return 0, 0, fmt.Errorf("cudamoe: totalSMs %d must be at least 4", totalSMs)
	}
	if math.IsNaN(commRatio) || math.IsInf(commRatio, 0) || commRatio < 0 || commRatio > 1 {
		return 0, 0, fmt.Errorf("cudamoe: commRatio %v must be in [0, 1]", commRatio)
	}
	if minCommSMs < 2 {
		return 0, 0, fmt.Errorf("cudamoe: minCommSMs %d must be at least 2", minCommSMs)
	}
	minEven := (minCommSMs + 1) &^ 1
	maxEven := (totalSMs - 2) &^ 1
	if minEven > maxEven {
		return 0, 0, fmt.Errorf("cudamoe: minCommSMs %d exceeds available even CTA pair limit %d for totalSMs %d", minCommSMs, maxEven, totalSMs)
	}

	raw := int(math.Round(float64(totalSMs) * commRatio))
	comm := raw &^ 1
	if comm < minEven {
		comm = minEven
	}
	if comm > maxEven {
		comm = maxEven
	}
	return comm, totalSMs - comm, nil
}

// NoAuxTCGateConfig configures the auxiliary-loss-free token-choice MoE gate.
type NoAuxTCGateConfig struct {
	NumExperts          int
	NumGroups           int
	TopKGroup           int
	TopK                int
	ScoringFunc         string
	RoutedScalingFactor float32
	UseBiasVL           bool
}

// SqrtSoftplus computes sqrt(log(1 + exp(x))) with numerical stability for
// large positive inputs (returning sqrt(x) when x > 20).
func SqrtSoftplus(x float32) float32 {
	if x > 20 {
		return float32(math.Sqrt(float64(x)))
	}
	return float32(math.Sqrt(math.Log1p(math.Exp(float64(x)))))
}

// MoEGateNoAuxTCCPU implements DeepInfer's noaux_tc MoE routing gate supporting
// "sigmoid" and "sqrtsoftplus" scoring, optional biasVL adjustment for vision
// tokens, group top-2 score summation to pick TopKGroup groups, TopK expert
// selection, and weight normalization scaled by RoutedScalingFactor.
func MoEGateNoAuxTCCPU(
	logits []float32,
	correctionBias []float32,
	biasVL []float32,
	tokenVisionMask []bool,
	cfg NoAuxTCGateConfig,
) (topkIds [][]int32, topkWeights [][]float32, err error) {
	if cfg.NumExperts <= 0 {
		return nil, nil, fmt.Errorf("cudamoe: NumExperts %d must be positive", cfg.NumExperts)
	}
	if cfg.NumGroups <= 0 || cfg.NumExperts%cfg.NumGroups != 0 {
		return nil, nil, fmt.Errorf("cudamoe: NumExperts %d must be divisible by NumGroups %d", cfg.NumExperts, cfg.NumGroups)
	}
	if cfg.TopKGroup <= 0 || cfg.TopKGroup > cfg.NumGroups {
		return nil, nil, fmt.Errorf("cudamoe: TopKGroup %d outside [1, %d]", cfg.TopKGroup, cfg.NumGroups)
	}
	expertsPerGroup := cfg.NumExperts / cfg.NumGroups
	if cfg.TopK <= 0 || cfg.TopK > cfg.TopKGroup*expertsPerGroup {
		return nil, nil, fmt.Errorf("cudamoe: TopK %d outside [1, %d]", cfg.TopK, cfg.TopKGroup*expertsPerGroup)
	}
	scoringFunc := cfg.ScoringFunc
	if scoringFunc == "" {
		scoringFunc = "sqrtsoftplus"
	}
	if scoringFunc != "sigmoid" && scoringFunc != "sqrtsoftplus" {
		return nil, nil, fmt.Errorf("cudamoe: unsupported ScoringFunc %q", cfg.ScoringFunc)
	}
	if len(logits) == 0 || len(logits)%cfg.NumExperts != 0 {
		return nil, nil, fmt.Errorf("cudamoe: len(logits) %d must be a positive multiple of NumExperts %d", len(logits), cfg.NumExperts)
	}
	numTokens := len(logits) / cfg.NumExperts
	if len(correctionBias) != 0 && len(correctionBias) != cfg.NumExperts {
		return nil, nil, fmt.Errorf("cudamoe: len(correctionBias) %d must be 0 or NumExperts %d", len(correctionBias), cfg.NumExperts)
	}
	if cfg.UseBiasVL {
		if len(biasVL) != cfg.NumExperts {
			return nil, nil, fmt.Errorf("cudamoe: len(biasVL) %d must equal NumExperts %d when UseBiasVL is true", len(biasVL), cfg.NumExperts)
		}
		if len(tokenVisionMask) != numTokens {
			return nil, nil, fmt.Errorf("cudamoe: len(tokenVisionMask) %d must equal numTokens %d when UseBiasVL is true", len(tokenVisionMask), numTokens)
		}
	}

	scaleFactor := cfg.RoutedScalingFactor
	if scaleFactor == 0 {
		scaleFactor = 1.0
	}

	topkIds = make([][]int32, numTokens)
	topkWeights = make([][]float32, numTokens)

	rawScores := make([]float32, cfg.NumExperts)
	choiceScores := make([]float32, cfg.NumExperts)
	type groupRank struct {
		group int
		score float32
	}
	groups := make([]groupRank, cfg.NumGroups)
	selectedGroup := make([]bool, cfg.NumGroups)

	type candidateExpert struct {
		expert      int32
		choiceScore float32
		rawScore    float32
	}
	candidates := make([]candidateExpert, 0, cfg.TopKGroup*expertsPerGroup)

	for t := 0; t < numTokens; t++ {
		tLogits := logits[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
		activeBias := correctionBias
		if cfg.UseBiasVL && tokenVisionMask[t] {
			activeBias = biasVL
		}

		for e := 0; e < cfg.NumExperts; e++ {
			x := tLogits[e]
			if !finite(x) {
				return nil, nil, fmt.Errorf("cudamoe: non-finite logit at token %d expert %d", t, e)
			}
			var s float32
			if scoringFunc == "sigmoid" {
				s = float32(1.0 / (1.0 + math.Exp(float64(-x))))
			} else {
				s = SqrtSoftplus(x)
			}
			rawScores[e] = s
			if len(activeBias) > 0 {
				choiceScores[e] = s + activeBias[e]
			} else {
				choiceScores[e] = s
			}
		}

		// Compute each group's score as the sum of its top-2 choiceScores.
		for g := 0; g < cfg.NumGroups; g++ {
			gSlice := choiceScores[g*expertsPerGroup : (g+1)*expertsPerGroup]
			top1 := float32(math.Inf(-1))
			top2 := float32(math.Inf(-1))
			for _, v := range gSlice {
				if v > top1 {
					top2 = top1
					top1 = v
				} else if v > top2 {
					top2 = v
				}
			}
			gScore := top1
			if expertsPerGroup >= 2 {
				gScore += top2
			}
			groups[g] = groupRank{group: g, score: gScore}
			selectedGroup[g] = false
		}

		sort.Slice(groups, func(i, j int) bool {
			if groups[i].score != groups[j].score {
				return groups[i].score > groups[j].score
			}
			return groups[i].group < groups[j].group
		})
		for i := 0; i < cfg.TopKGroup; i++ {
			selectedGroup[groups[i].group] = true
		}

		candidates = candidates[:0]
		for g := 0; g < cfg.NumGroups; g++ {
			if !selectedGroup[g] {
				continue
			}
			base := g * expertsPerGroup
			for j := 0; j < expertsPerGroup; j++ {
				e := base + j
				candidates = append(candidates, candidateExpert{
					expert:      int32(e),
					choiceScore: choiceScores[e],
					rawScore:    rawScores[e],
				})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].choiceScore != candidates[j].choiceScore {
				return candidates[i].choiceScore > candidates[j].choiceScore
			}
			return candidates[i].expert < candidates[j].expert
		})

		ids := make([]int32, cfg.TopK)
		weights := make([]float32, cfg.TopK)
		var sumRaw float32
		for k := 0; k < cfg.TopK; k++ {
			ids[k] = candidates[k].expert
			weights[k] = candidates[k].rawScore
			sumRaw += candidates[k].rawScore
		}
		if sumRaw > 0 {
			inv := scaleFactor / sumRaw
			for k := 0; k < cfg.TopK; k++ {
				weights[k] *= inv
			}
		}
		topkIds[t] = ids
		topkWeights[t] = weights
	}

	return topkIds, topkWeights, nil
}
