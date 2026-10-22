// SPDX-License-Identifier: Apache-2.0

package cudamtp

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrInvalidConfidenceSpec indicates an invalid ConfidenceHeadSpec or mismatched tensor dimensions.
var ErrInvalidConfidenceSpec = errors.New("cudamtp: invalid confidence head specification")

// ConfidenceHeadSpec configures the lightweight MTP confidence prediction head
// from DeepInfer (deepinfer/models/mtp.py and tilelang_ops/dflash.py).
type ConfidenceHeadSpec struct {
	HiddenDim        int
	MarkovVocabSize  int
	StepTemperatures []float32
	Eps              float32
}

func (s ConfidenceHeadSpec) normalize() (ConfidenceHeadSpec, error) {
	if s.HiddenDim <= 0 {
		return s, fmt.Errorf("%w: HiddenDim must be > 0, got %d", ErrInvalidConfidenceSpec, s.HiddenDim)
	}
	if len(s.StepTemperatures) == 0 {
		return s, fmt.Errorf("%w: StepTemperatures must not be empty", ErrInvalidConfidenceSpec)
	}
	for i, t := range s.StepTemperatures {
		if t <= 0 || math.IsNaN(float64(t)) || math.IsInf(float64(t), 0) {
			return s, fmt.Errorf("%w: StepTemperatures[%d] must be positive finite, got %v", ErrInvalidConfidenceSpec, i, t)
		}
	}
	if s.MarkovVocabSize == 0 {
		s.MarkovVocabSize = 512
	}
	if s.MarkovVocabSize < 0 {
		return s, fmt.Errorf("%w: MarkovVocabSize must be > 0, got %d", ErrInvalidConfidenceSpec, s.MarkovVocabSize)
	}
	if s.Eps == 0 {
		s.Eps = 1e-6
	}
	if s.Eps < 0 || math.IsNaN(float64(s.Eps)) {
		return s, fmt.Errorf("%w: Eps must be > 0, got %v", ErrInvalidConfidenceSpec, s.Eps)
	}
	return s, nil
}

// ConfidenceReferenceCPU computes the CPU mathematical reference for the MTP
// confidence head:
//  1. RMSNorm on hidden using rmsW and spec.Eps.
//  2. Linear projection via wProj.
//  3. Optional Markov bigram logit bias from markovMatrix[prevToken % V, curToken % V].
//  4. Per-step temperature scaling logits / StepTemperatures[s].
//  5. Sigmoid p_s = 1 / (1 + exp(-logit)) and cumulative product cumProbs[i][s] = prod_{k <= s} p_k.
func ConfidenceReferenceCPU(
	hidden []float32,
	rmsW []float32,
	wProj []float32,
	markovMatrix []float32,
	tokenIDs []int32,
	spec ConfidenceHeadSpec,
) (stepProbs [][]float32, cumProbs [][]float32, err error) {
	cfg, err := spec.normalize()
	if err != nil {
		return nil, nil, err
	}
	numSteps := len(cfg.StepTemperatures)
	dim := cfg.HiddenDim

	if len(hidden) == 0 || len(hidden)%dim != 0 {
		return nil, nil, fmt.Errorf("%w: hidden length %d is not a positive multiple of HiddenDim %d", ErrInvalidConfidenceSpec, len(hidden), dim)
	}
	if len(rmsW) != 0 && len(rmsW) != dim {
		return nil, nil, fmt.Errorf("%w: rmsW length %d != HiddenDim %d", ErrInvalidConfidenceSpec, len(rmsW), dim)
	}

	// wProj can be [numSteps * dim] (per-step projection vector) or [dim] (shared projection vector).
	var perStepWProj bool
	switch len(wProj) {
	case numSteps * dim:
		perStepWProj = true
	case dim:
		perStepWProj = false
	default:
		return nil, nil, fmt.Errorf("%w: wProj length %d must be HiddenDim (%d) or numSteps*HiddenDim (%d)", ErrInvalidConfidenceSpec, len(wProj), dim, numSteps*dim)
	}

	totalHiddenVecs := len(hidden) / dim
	var numSeqs int
	var perStepHidden bool
	if totalHiddenVecs%numSteps == 0 && len(tokenIDs) > 0 && len(tokenIDs) == (totalHiddenVecs/numSteps)*(numSteps+1) {
		numSeqs = totalHiddenVecs / numSteps
		perStepHidden = true
	} else if len(tokenIDs) > 0 && len(tokenIDs)%(numSteps+1) == 0 && totalHiddenVecs == len(tokenIDs)/(numSteps+1) {
		numSeqs = totalHiddenVecs
		perStepHidden = false
	} else if !perStepWProj && totalHiddenVecs%numSteps == 0 {
		numSeqs = totalHiddenVecs / numSteps
		perStepHidden = true
	} else {
		numSeqs = totalHiddenVecs
		perStepHidden = false
	}

	v := cfg.MarkovVocabSize
	hasMarkov := len(markovMatrix) > 0
	if hasMarkov && len(markovMatrix) != v*v {
		return nil, nil, fmt.Errorf("%w: markovMatrix length %d != MarkovVocabSize^2 (%d)", ErrInvalidConfidenceSpec, len(markovMatrix), v*v)
	}
	if hasMarkov && len(tokenIDs) > 0 {
		if len(tokenIDs) != numSeqs*(numSteps+1) && len(tokenIDs) != numSeqs*numSteps {
			return nil, nil, fmt.Errorf("%w: tokenIDs length %d must be numSeqs*(numSteps+1)=%d or numSeqs*numSteps=%d",
				ErrInvalidConfidenceSpec, len(tokenIDs), numSeqs*(numSteps+1), numSeqs*numSteps)
		}
	}

	normed := make([]float64, dim)
	stepProbs = make([][]float32, numSeqs)
	cumProbs = make([][]float32, numSeqs)

	for i := 0; i < numSeqs; i++ {
		stepProbs[i] = make([]float32, numSteps)
		cumProbs[i] = make([]float32, numSteps)
		cum := float64(1.0)

		for s := 0; s < numSteps; s++ {
			var hSlice []float32
			if perStepHidden {
				idx := (i*numSteps + s) * dim
				hSlice = hidden[idx : idx+dim]
			} else {
				idx := i * dim
				hSlice = hidden[idx : idx+dim]
			}

			var sumSq float64
			for d := 0; d < dim; d++ {
				val := float64(hSlice[d])
				sumSq += val * val
			}
			invRMS := 1.0 / math.Sqrt(sumSq/float64(dim)+float64(cfg.Eps))
			for d := 0; d < dim; d++ {
				w := 1.0
				if len(rmsW) == dim {
					w = float64(rmsW[d])
				}
				normed[d] = float64(hSlice[d]) * invRMS * w
			}

			var wSlice []float32
			if perStepWProj {
				wSlice = wProj[s*dim : (s+1)*dim]
			} else {
				wSlice = wProj[:dim]
			}

			var logit float64
			for d := 0; d < dim; d++ {
				logit += normed[d] * float64(wSlice[d])
			}

			if hasMarkov && len(tokenIDs) > 0 {
				var prevTok, curTok int32
				if len(tokenIDs) == numSeqs*(numSteps+1) {
					base := i * (numSteps + 1)
					prevTok = tokenIDs[base+s]
					curTok = tokenIDs[base+s+1]
				} else {
					base := i * numSteps
					if s == 0 {
						prevTok = tokenIDs[base]
					} else {
						prevTok = tokenIDs[base+s-1]
					}
					curTok = tokenIDs[base+s]
				}
				prevIdx := ((int(prevTok) % v) + v) % v
				curIdx := ((int(curTok) % v) + v) % v
				logit += float64(markovMatrix[prevIdx*v+curIdx])
			}

			scaled := logit / float64(cfg.StepTemperatures[s])
			if scaled < -60.0 {
				scaled = -60.0
			} else if scaled > 60.0 {
				scaled = 60.0
			}
			p := 1.0 / (1.0 + math.Exp(-scaled))
			stepProbs[i][s] = float32(p)
			cum *= p
			cumProbs[i][s] = float32(cum)
		}
	}

	return stepProbs, cumProbs, nil
}

// GenerateConfidenceCudaSource generates the NVRTC CUDA C++ source for
// mtp_confidence_head_kernel matching ConfidenceReferenceCPU.
func GenerateConfidenceCudaSource(spec ConfidenceHeadSpec) (string, error) {
	cfg, err := spec.normalize()
	if err != nil {
		return "", err
	}
	numSteps := len(cfg.StepTemperatures)

	var tempItems []string
	for _, t := range cfg.StepTemperatures {
		tempItems = append(tempItems, fmt.Sprintf("%.8ff", t))
	}
	tempArray := strings.Join(tempItems, ", ")

	src := fmt.Sprintf(`// Auto-generated MTP Confidence Head CUDA kernel by nvidia/x/cudamtp
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void mtp_confidence_head_kernel(
    const float* __restrict__ hidden,        // [num_seqs, %d]
    const float* __restrict__ rms_w,         // [%d]
    const float* __restrict__ w_proj,        // [%d, %d]
    const float* __restrict__ markov_matrix, // [%d, %d] or nullptr
    const int*   __restrict__ token_ids,     // [num_seqs, %d] or nullptr
    float*       __restrict__ step_probs,    // [num_seqs, %d]
    float*       __restrict__ cum_probs,     // [num_seqs, %d]
    int num_seqs
) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= num_seqs) return;

    const int D = %d;
    const int S = %d;
    const int V = %d;
    const float eps = %.8ff;
    const float step_temps[%d] = { %s };

    __shared__ float s_normed[%d];
    __shared__ float s_inv_rms;

    const float* h_row = hidden + seq_idx * D;

    float local_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = h_row[d];
        local_sq += v * v;
    }

    // Block-wide reduction via shared memory for thread 0
    __shared__ float s_reduce[256];
    int tid = threadIdx.x;
    s_reduce[tid] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_inv_rms = rsqrtf(s_reduce[0] / (float)D + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float rw = (rms_w != nullptr) ? rms_w[d] : 1.0f;
        s_normed[d] = h_row[d] * s_inv_rms * rw;
    }
    __syncthreads();

    if (tid == 0) {
        float cum = 1.0f;
        for (int s = 0; s < S; ++s) {
            const float* wp = w_proj + s * D;
            float logit = 0.0f;
            for (int d = 0; d < D; ++d) {
                logit += s_normed[d] * wp[d];
            }
            if (markov_matrix != nullptr && token_ids != nullptr) {
                int prev_tok = token_ids[seq_idx * (S + 1) + s];
                int cur_tok  = token_ids[seq_idx * (S + 1) + s + 1];
                int prev_idx = ((prev_tok %% V) + V) %% V;
                int cur_idx  = ((cur_tok  %% V) + V) %% V;
                logit += markov_matrix[prev_idx * V + cur_idx];
            }
            float scaled = logit / step_temps[s];
            if (scaled < -60.0f) scaled = -60.0f;
            if (scaled >  60.0f) scaled =  60.0f;
            float p = 1.0f / (1.0f + expf(-scaled));
            step_probs[seq_idx * S + s] = p;
            cum *= p;
            cum_probs[seq_idx * S + s] = cum;
        }
    }
}
`,
		cfg.HiddenDim,
		cfg.HiddenDim,
		numSteps, cfg.HiddenDim,
		cfg.MarkovVocabSize, cfg.MarkovVocabSize,
		numSteps+1,
		numSteps,
		numSteps,
		cfg.HiddenDim,
		numSteps,
		cfg.MarkovVocabSize,
		cfg.Eps,
		numSteps,
		tempArray,
		cfg.HiddenDim,
	)

	return src, nil
}
