// SPDX-License-Identifier: Apache-2.0

package cudasample

import (
	"fmt"
	"math"
)

// SequenceSampleParams holds per-sequence deterministic sampling hyperparameters
// and optional packed grammar bitmask constraints.
type SequenceSampleParams struct {
	SeqSeed        uint64
	StepOffset     uint64
	Temperature    float32
	TopK           int
	TopP           float32
	MinP           float32
	AllowedBitmask []uint32
}

// SeededSampleConfig configures NVRTC kernel generation for batch-invariant
// per-sequence Philox seeded sampling.
type SeededSampleConfig struct {
	VocabSize         int
	BlockThreads      int
	EnableGrammarMask bool
}

// KernelSource emits the NVRTC CUDA C++ source for c.
func (c SeededSampleConfig) KernelSource() (string, error) {
	return SeededSampleKernelSource(c.VocabSize)
}

const (
	philoxM0 = uint64(0xD2511F53)
	philoxM1 = uint64(0xCD9E8D57)
	philoxW0 = uint32(0x9E3779B9)
	philoxW1 = uint32(0xBB67AE85)
)

// Philox4x32_10 executes 10 rounds of the counter-based Philox-4x32 PRNG
// matching CUDA's curand_philox4x32_10. Output words depend strictly on
// (seed, offset, subseq) and are independent of batch size or row index.
func Philox4x32_10(seed uint64, offset uint64, subseq uint64) [4]uint32 {
	ctr := [4]uint32{
		uint32(offset),
		uint32(offset >> 32),
		uint32(subseq),
		uint32(subseq >> 32),
	}
	key0 := uint32(seed)
	key1 := uint32(seed >> 32)

	for round := 0; round < 10; round++ {
		prod0 := philoxM0 * uint64(ctr[0])
		prod1 := philoxM1 * uint64(ctr[2])
		hi0, lo0 := uint32(prod0>>32), uint32(prod0)
		hi1, lo1 := uint32(prod1>>32), uint32(prod1)

		ctr[0] = hi1 ^ ctr[1] ^ key0
		ctr[1] = lo1
		ctr[2] = hi0 ^ ctr[3] ^ key1
		ctr[3] = lo0

		key0 += philoxW0
		key1 += philoxW1
	}
	return ctr
}

// PhiloxUniformFloat32 maps (seed, offset, vocabIdx) to a deterministic uniform
// float32 strictly inside the open interval (0, 1) using [Philox4x32_10].
func PhiloxUniformFloat32(seed uint64, offset uint64, vocabIdx int) float32 {
	uIdx := uint64(uint32(vocabIdx))
	subseq := uIdx >> 2
	lane := uIdx & 3
	words := Philox4x32_10(seed, offset, subseq)
	u24 := words[lane] >> 8
	return (float32(u24) + 0.5) * (1.0 / 16777216.0)
}

// SumSmallerProbsCPU computes for each token v the cumulative probability mass
// of all tokens u with strictly smaller probability (p_u < p_v, or
// p_u == p_v && u > v for deterministic tie-breaking). This matches DeepInfer's
// sorting-free top-p mass kernel where 1 - sumSmaller[v] is the inclusive
// cumulative probability of v in descending probability order.
func SumSmallerProbsCPU(probs []float32) ([]float32, error) {
	if len(probs) == 0 {
		return nil, fmt.Errorf("cudasample: empty probabilities slice")
	}
	for i, p := range probs {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || p < 0 {
			return nil, fmt.Errorf("cudasample: invalid probability at index %d: %v", i, p)
		}
	}

	out := make([]float32, len(probs))
	for v, pv := range probs {
		var sum float64
		for u, pu := range probs {
			if pu < pv || (pu == pv && u > v) {
				sum += float64(pu)
			}
		}
		out[v] = float32(sum)
	}
	return out, nil
}

// SampleSeededCPU performs deterministic per-sequence sampling keyed strictly
// by (params.SeqSeed, params.StepOffset). It applies params.AllowedBitmask
// (if non-nil), takes the greedy argmax when params.Temperature <= 0, and
// otherwise applies temperature scaling, softmax, TopK + TopP (via
// [SumSmallerProbsCPU]) + MinP filtering, renormalization, and inverse-CDF
// sampling using [PhiloxUniformFloat32].
func SampleSeededCPU(logits []float32, params SequenceSampleParams) (tokenID int32, filteredProbs []float32, err error) {
	vocabSize := len(logits)
	if vocabSize == 0 {
		return -1, nil, fmt.Errorf("cudasample: empty logits")
	}
	if math.IsNaN(float64(params.Temperature)) || math.IsInf(float64(params.Temperature), 1) {
		return -1, nil, fmt.Errorf("cudasample: invalid temperature %v", params.Temperature)
	}
	if math.IsNaN(float64(params.TopP)) || params.TopP < 0 || params.TopP > 1 {
		return -1, nil, fmt.Errorf("cudasample: top_p %v out of [0, 1]", params.TopP)
	}
	if math.IsNaN(float64(params.MinP)) || params.MinP < 0 || params.MinP > 1 {
		return -1, nil, fmt.Errorf("cudasample: min_p %v out of [0, 1]", params.MinP)
	}

	needWords := (vocabSize + 31) / 32
	if params.AllowedBitmask != nil && len(params.AllowedBitmask) < needWords {
		return -1, nil, fmt.Errorf("cudasample: allowed bitmask length %d < required %d words for vocab size %d",
			len(params.AllowedBitmask), needWords, vocabSize)
	}

	negInf := float32(math.Inf(-1))
	work := make([]float32, vocabSize)
	maxLogit := negInf
	maxIdx := -1

	for i, l := range logits {
		if math.IsNaN(float64(l)) {
			return -1, nil, fmt.Errorf("cudasample: NaN logit at index %d", i)
		}
		if params.AllowedBitmask != nil {
			word := params.AllowedBitmask[i/32]
			if (word & (1 << uint(i%32))) == 0 {
				l = negInf
			}
		}
		work[i] = l
		if l > maxLogit {
			maxLogit = l
			maxIdx = i
		}
	}

	if maxIdx < 0 || math.IsInf(float64(maxLogit), -1) {
		return -1, nil, fmt.Errorf("cudasample: all tokens masked or -inf")
	}

	filteredProbs = make([]float32, vocabSize)
	if params.Temperature <= 0 {
		filteredProbs[maxIdx] = 1.0
		return int32(maxIdx), filteredProbs, nil
	}

	invTemp := 1.0 / float64(params.Temperature)
	scaledMax := float64(maxLogit) * invTemp
	probs := make([]float32, vocabSize)
	var sumExp float64
	for i, l := range work {
		if math.IsInf(float64(l), -1) {
			continue
		}
		e := math.Exp(float64(l)*invTemp - scaledMax)
		probs[i] = float32(e)
		sumExp += e
	}
	if sumExp <= 0 || math.IsNaN(sumExp) {
		filteredProbs[maxIdx] = 1.0
		return int32(maxIdx), filteredProbs, nil
	}

	var maxProb float32
	for i := range probs {
		probs[i] = float32(float64(probs[i]) / sumExp)
		if probs[i] > maxProb {
			maxProb = probs[i]
		}
	}

	sumSmaller, err := SumSmallerProbsCPU(probs)
	if err != nil {
		return -1, nil, err
	}

	topK := params.TopK
	topP := params.TopP
	if topP == 0 {
		topP = 1.0
	}
	minP := params.MinP

	var keptSum float64
	for v, pv := range probs {
		if pv <= 0 {
			continue
		}
		if v == maxIdx {
			filteredProbs[v] = pv
			keptSum += float64(pv)
			continue
		}
		if topK > 0 && topK < vocabSize {
			greaterCount := 0
			for u, pu := range probs {
				if pu > pv || (pu == pv && u < v) {
					greaterCount++
				}
			}
			if greaterCount >= topK {
				continue
			}
		}
		if topP < 1.0 {
			inclusiveMass := 1.0 - sumSmaller[v]
			if inclusiveMass > topP+1e-6 {
				continue
			}
		}
		if minP > 0 {
			if pv < minP*maxProb {
				continue
			}
		}
		filteredProbs[v] = pv
		keptSum += float64(pv)
	}

	if keptSum <= 0 {
		filteredProbs[maxIdx] = 1.0
		return int32(maxIdx), filteredProbs, nil
	}
	for v := range filteredProbs {
		if filteredProbs[v] > 0 {
			filteredProbs[v] = float32(float64(filteredProbs[v]) / keptSum)
		}
	}

	u := float64(PhiloxUniformFloat32(params.SeqSeed, params.StepOffset, 0))
	var cdf float64
	lastValid := int32(maxIdx)
	for v, pv := range filteredProbs {
		if pv <= 0 {
			continue
		}
		lastValid = int32(v)
		cdf += float64(pv)
		if u <= cdf {
			return int32(v), filteredProbs, nil
		}
	}
	return lastValid, filteredProbs, nil
}

// SampleBatchSeededCPU evaluates a batch of sequences with [SampleSeededCPU].
// Because each sequence draws solely from (batchParams[b].SeqSeed, batchParams[b].StepOffset),
// any permutation or subset of the batch yields identical token IDs.
func SampleBatchSeededCPU(batchLogits [][]float32, batchParams []SequenceSampleParams) ([]int32, error) {
	if len(batchLogits) == 0 {
		return nil, fmt.Errorf("cudasample: empty batch logits")
	}
	if len(batchLogits) != len(batchParams) {
		return nil, fmt.Errorf("cudasample: batch logits count %d != batch params count %d", len(batchLogits), len(batchParams))
	}
	out := make([]int32, len(batchLogits))
	for b := range batchLogits {
		tok, _, err := SampleSeededCPU(batchLogits[b], batchParams[b])
		if err != nil {
			return nil, fmt.Errorf("cudasample: batch row %d: %w", b, err)
		}
		out[b] = tok
	}
	return out, nil
}

// VerifySpeculativeSeededCPU performs deterministic speculative rejection sampling
// over draftTokens using (seqSeed, baseStepOffset + s) at draft step s.
// Draft token x_s is accepted when u_s < min(1, targetProbs[s][x_s] / draftProbs[s][x_s]).
// On the first rejection at step s, bonusToken is resampled from the normalized
// residual distribution max(0, targetProbs[s] - draftProbs[s]) keyed by
// (seqSeed, baseStepOffset + s). If all draft tokens are accepted, bonusToken
// is sampled from targetProbs[len(draftTokens)] (or targetProbs[len(draftTokens)-1]
// when len(targetProbs) == len(draftTokens)).
func VerifySpeculativeSeededCPU(
	targetProbs [][]float32,
	draftProbs [][]float32,
	draftTokens []int32,
	seqSeed uint64,
	baseStepOffset uint64,
) (acceptedTokens []int32, bonusToken int32, err error) {
	numDraft := len(draftTokens)
	if len(draftProbs) != numDraft {
		return nil, -1, fmt.Errorf("cudasample: draftProbs length %d != draftTokens length %d", len(draftProbs), numDraft)
	}
	if len(targetProbs) == 0 || len(targetProbs) < numDraft || len(targetProbs) > numDraft+1 {
		return nil, -1, fmt.Errorf("cudasample: targetProbs length %d must be %d or %d", len(targetProbs), numDraft, numDraft+1)
	}
	vocabSize := len(targetProbs[0])
	if vocabSize == 0 {
		return nil, -1, fmt.Errorf("cudasample: empty vocabulary in targetProbs")
	}
	for s, row := range targetProbs {
		if len(row) != vocabSize {
			return nil, -1, fmt.Errorf("cudasample: targetProbs[%d] length %d != vocab size %d", s, len(row), vocabSize)
		}
	}
	for s, row := range draftProbs {
		if len(row) != vocabSize {
			return nil, -1, fmt.Errorf("cudasample: draftProbs[%d] length %d != vocab size %d", s, len(row), vocabSize)
		}
	}

	acceptedTokens = make([]int32, 0, numDraft)
	for s := 0; s < numDraft; s++ {
		stepOffset := baseStepOffset + uint64(s)
		tok := draftTokens[s]
		if tok < -1 || int(tok) >= vocabSize {
			return nil, -1, fmt.Errorf("cudasample: draft token %d at step %d out of range [-1, %d)", tok, s, vocabSize)
		}

		accepted := false
		if tok >= 0 {
			pTarget := targetProbs[s][tok]
			pDraft := draftProbs[s][tok]
			var acceptProb float32
			switch {
			case pTarget <= 0:
				acceptProb = 0
			case pDraft <= 0 || pTarget >= pDraft:
				acceptProb = 1.0
			default:
				acceptProb = pTarget / pDraft
			}
			uAccept := PhiloxUniformFloat32(seqSeed, stepOffset, 0)
			accepted = uAccept < acceptProb
		}

		if accepted {
			acceptedTokens = append(acceptedTokens, tok)
			continue
		}

		residual := make([]float32, vocabSize)
		var resSum float64
		for v := 0; v < vocabSize; v++ {
			diff := targetProbs[s][v] - draftProbs[s][v]
			if diff > 0 {
				residual[v] = diff
				resSum += float64(diff)
			}
		}
		uRes := PhiloxUniformFloat32(seqSeed, stepOffset, 1)
		if resSum > 0 {
			bonus, sampleErr := sampleFromNormalizedProbs(residual, resSum, uRes)
			if sampleErr != nil {
				return nil, -1, sampleErr
			}
			return acceptedTokens, bonus, nil
		}
		var targetSum float64
		for _, p := range targetProbs[s] {
			if p > 0 {
				targetSum += float64(p)
			}
		}
		bonus, sampleErr := sampleFromNormalizedProbs(targetProbs[s], targetSum, uRes)
		if sampleErr != nil {
			return nil, -1, sampleErr
		}
		return acceptedTokens, bonus, nil
	}

	bonusRow := targetProbs[len(targetProbs)-1]
	var bonusSum float64
	for _, p := range bonusRow {
		if p > 0 {
			bonusSum += float64(p)
		}
	}
	uBonus := PhiloxUniformFloat32(seqSeed, baseStepOffset+uint64(numDraft), 0)
	bonus, err := sampleFromNormalizedProbs(bonusRow, bonusSum, uBonus)
	if err != nil {
		return nil, -1, err
	}
	return acceptedTokens, bonus, nil
}

func sampleFromNormalizedProbs(probs []float32, sum float64, u float32) (int32, error) {
	if sum <= 0 || math.IsNaN(sum) {
		return -1, fmt.Errorf("cudasample: probability distribution has non-positive mass %v", sum)
	}
	target := float64(u) * sum
	var cdf float64
	lastPositive := int32(-1)
	for v, p := range probs {
		if p <= 0 {
			continue
		}
		lastPositive = int32(v)
		cdf += float64(p)
		if target <= cdf {
			return int32(v), nil
		}
	}
	if lastPositive < 0 {
		return -1, fmt.Errorf("cudasample: no positive probability entry")
	}
	return lastPositive, nil
}

// SeededSampleKernelSource generates NVRTC CUDA C++ source for
// seeded_batch_invariant_sample_kernel using counter-based Philox-4x32-10
// keyed by (seq_seed[b], step_offset[b]).
func SeededSampleKernelSource(vocabSize int) (string, error) {
	if vocabSize <= 0 {
		return "", fmt.Errorf("cudasample: vocab size must be positive, got %d", vocabSize)
	}
	src := fmt.Sprintf(`// Generated by cudasample.SeededSampleKernelSource(vocabSize=%d)
extern "C" {

struct uint4_philox {
    unsigned int x, y, z, w;
};

__device__ __forceinline__ uint4_philox philox4x32_10(
    unsigned long long seed,
    unsigned long long offset,
    unsigned long long subseq
) {
    unsigned int c0 = (unsigned int)(offset);
    unsigned int c1 = (unsigned int)(offset >> 32);
    unsigned int c2 = (unsigned int)(subseq);
    unsigned int c3 = (unsigned int)(subseq >> 32);
    unsigned int k0 = (unsigned int)(seed);
    unsigned int k1 = (unsigned int)(seed >> 32);

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        unsigned long long p0 = 0xD2511F53ull * (unsigned long long)c0;
        unsigned long long p1 = 0xCD9E8D57ull * (unsigned long long)c2;
        unsigned int hi0 = (unsigned int)(p0 >> 32);
        unsigned int lo0 = (unsigned int)(p0);
        unsigned int hi1 = (unsigned int)(p1 >> 32);
        unsigned int lo1 = (unsigned int)(p1);

        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;

        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    uint4_philox out = {c0, c1, c2, c3};
    return out;
}

__device__ __forceinline__ float philox_uniform_f32(
    unsigned long long seed,
    unsigned long long offset,
    int vocab_idx
) {
    unsigned long long u = (unsigned int)vocab_idx;
    uint4_philox w = philox4x32_10(seed, offset, u >> 2);
    unsigned int lane = (unsigned int)(u & 3ull);
    unsigned int word = (lane == 0u) ? w.x : ((lane == 1u) ? w.y : ((lane == 2u) ? w.z : w.w));
    return ((float)(word >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__global__ void seeded_batch_invariant_sample_kernel(
    const float* __restrict__ logits,
    const unsigned int* __restrict__ allowed_bitmask,
    const unsigned long long* __restrict__ seq_seed,
    const unsigned long long* __restrict__ step_offset,
    const float* __restrict__ temperatures,
    const float* __restrict__ top_ps,
    const float* __restrict__ min_ps,
    int* __restrict__ out_tokens,
    int batch_size,
    int vocab_size
) {
    const int kConfiguredVocabSize = %d;
    int V = (vocab_size > 0) ? vocab_size : kConfiguredVocabSize;
    int b = blockIdx.x;
    if (b >= batch_size || threadIdx.x != 0) {
        return;
    }

    const float* row_logits = logits + (long long)b * V;
    int mask_words = (V + 31) / 32;
    const unsigned int* row_mask = (allowed_bitmask != 0) ? (allowed_bitmask + (long long)b * mask_words) : 0;

    float temp = temperatures[b];
    float max_val = -__int_as_float(0x7f800000);
    int max_idx = -1;
    for (int i = 0; i < V; ++i) {
        if (row_mask != 0 && ((row_mask[i / 32] & (1u << (i & 31))) == 0u)) {
            continue;
        }
        float v = row_logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    if (max_idx < 0 || temp <= 0.0f) {
        out_tokens[b] = max_idx;
        return;
    }

    float inv_temp = 1.0f / temp;
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        if (row_mask != 0 && ((row_mask[i / 32] & (1u << (i & 31))) == 0u)) {
            continue;
        }
        sum_exp += __expf((row_logits[i] - max_val) * inv_temp);
    }

    float u = philox_uniform_f32(seq_seed[b], step_offset[b], 0);
    float target = u * sum_exp;
    float cdf = 0.0f;
    int chosen = max_idx;
    for (int i = 0; i < V; ++i) {
        if (row_mask != 0 && ((row_mask[i / 32] & (1u << (i & 31))) == 0u)) {
            continue;
        }
        float p = __expf((row_logits[i] - max_val) * inv_temp);
        chosen = i;
        cdf += p;
        if (target <= cdf) {
            break;
        }
    }
    out_tokens[b] = chosen;
}

} // extern "C"
`, vocabSize, vocabSize)
	return src, nil
}
