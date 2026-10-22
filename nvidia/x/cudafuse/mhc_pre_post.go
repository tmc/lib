// SPDX-License-Identifier: Apache-2.0

package cudafuse

import (
	"fmt"
	"math"
)

// MHCPrePostConfig configures cross-layer fused Multi-Headed Residual
// Connections (mHC) pre/post mixing and Sinkhorn-Knopp normalization.
type MHCPrePostConfig struct {
	NumTokens     int
	NumStreams    int
	HiddenDim     int
	Eps           float32
	PreEnd        int
	PostEnd       int
	ResEnd        int
	SinkhornIters int
	SinkhornEps   float32
}

func (c MHCPrePostConfig) normalize() (MHCPrePostConfig, error) {
	if c.NumStreams == 0 {
		c.NumStreams = 4
	}
	if c.Eps == 0 {
		c.Eps = 1e-6
	}
	if c.SinkhornIters == 0 {
		c.SinkhornIters = 20
	}
	if c.SinkhornEps == 0 {
		c.SinkhornEps = 1e-6
	}
	if c.PreEnd == 0 {
		c.PreEnd = c.NumStreams
	}
	if c.PostEnd == 0 {
		c.PostEnd = 2 * c.NumStreams
	}
	if c.ResEnd == 0 {
		c.ResEnd = 2*c.NumStreams + c.NumStreams*c.NumStreams
	}
	if c.NumTokens <= 0 || c.NumStreams <= 0 || c.HiddenDim <= 0 {
		return c, fmt.Errorf("cudafuse: invalid mhc dimensions numTokens=%d numStreams=%d hiddenDim=%d", c.NumTokens, c.NumStreams, c.HiddenDim)
	}
	if !mhcFinite(c.Eps) || c.Eps <= 0 || !mhcFinite(c.SinkhornEps) || c.SinkhornEps <= 0 || c.SinkhornIters <= 0 {
		return c, fmt.Errorf("cudafuse: invalid mhc eps (%v), sinkhornEps (%v) or sinkhornIters (%d)", c.Eps, c.SinkhornEps, c.SinkhornIters)
	}
	wantPreEnd := c.NumStreams
	wantPostEnd := 2 * c.NumStreams
	wantResEnd := 2*c.NumStreams + c.NumStreams*c.NumStreams
	if c.PreEnd != wantPreEnd || c.PostEnd != wantPostEnd || c.ResEnd != wantResEnd {
		return c, fmt.Errorf("cudafuse: invalid split bounds [%d, %d, %d], want [%d, %d, %d]", c.PreEnd, c.PostEnd, c.ResEnd, wantPreEnd, wantPostEnd, wantResEnd)
	}
	return c, nil
}

// PreSplitMixesResult holds the split per-token hyper-connection mixing weights:
//   - PreMix:  [NumTokens, NumStreams]
//   - PostMix: [NumTokens, NumStreams]
//   - ResMix:  [NumTokens, NumStreams*NumStreams]
type PreSplitMixesResult struct {
	PreMix  []float32
	PostMix []float32
	ResMix  []float32
}

// PreSplitMixesCPU computes the fused RMSNorm + linear projection + inv_rms
// scaling + sigmoid (for PreMix), 2*sigmoid (for PostMix), and Sinkhorn-Knopp
// doubly stochastic normalization (for ResMix), matching DeepInfer's
// _hc_PreSplitMixes_kernel.
//
// Input shapes:
//   - x:           [NumTokens, NumStreams*HiddenDim]
//   - rmsWeight:   optional [NumStreams*HiddenDim] or [HiddenDim] (nil uses unit weights)
//   - phiCombined: [NumStreams*HiddenDim, ResEnd] row-major projection matrix
//   - bias:        optional [ResEnd] additive bias
func PreSplitMixesCPU(
	x []float32,
	rmsWeight []float32,
	phiCombined []float32,
	bias []float32,
	alphaPre, alphaPost, alphaRes float32,
	cfg MHCPrePostConfig,
) (PreSplitMixesResult, error) {
	var res PreSplitMixesResult
	cfg, err := cfg.normalize()
	if err != nil {
		return res, err
	}
	if !mhcFinite(alphaPre) || !mhcFinite(alphaPost) || !mhcFinite(alphaRes) {
		return res, fmt.Errorf("cudafuse: non-finite alpha scaling factor")
	}

	totalDim := cfg.NumStreams * cfg.HiddenDim
	mixDim := cfg.ResEnd
	if len(x) != cfg.NumTokens*totalDim {
		return res, fmt.Errorf("cudafuse: x length %d, want %d (numTokens=%d * totalDim=%d)", len(x), cfg.NumTokens*totalDim, cfg.NumTokens, totalDim)
	}
	if rmsWeight != nil && len(rmsWeight) != totalDim && len(rmsWeight) != cfg.HiddenDim {
		return res, fmt.Errorf("cudafuse: rmsWeight length %d, want %d or %d", len(rmsWeight), totalDim, cfg.HiddenDim)
	}
	if len(phiCombined) != totalDim*mixDim {
		return res, fmt.Errorf("cudafuse: phiCombined length %d, want %d (totalDim=%d * mixDim=%d)", len(phiCombined), totalDim*mixDim, totalDim, mixDim)
	}
	if bias != nil && len(bias) != mixDim {
		return res, fmt.Errorf("cudafuse: bias length %d, want %d", len(bias), mixDim)
	}
	for _, s := range [][]float32{x, rmsWeight, phiCombined, bias} {
		if err := mhcValidateValues(s); err != nil {
			return res, err
		}
	}

	S := cfg.NumStreams
	res.PreMix = make([]float32, cfg.NumTokens*S)
	res.PostMix = make([]float32, cfg.NumTokens*S)
	res.ResMix = make([]float32, cfg.NumTokens*S*S)

	mat := make([]float64, S*S)
	for t := 0; t < cfg.NumTokens; t++ {
		xRow := x[t*totalDim : (t+1)*totalDim]
		var sumSq float64
		for d := 0; d < totalDim; d++ {
			v := float64(xRow[d])
			sumSq += v * v
		}
		invRMS := 1.0 / math.Sqrt(sumSq/float64(totalDim)+float64(cfg.Eps))

		for k := 0; k < mixDim; k++ {
			var dot float64
			for d := 0; d < totalDim; d++ {
				w := 1.0
				if rmsWeight != nil {
					if len(rmsWeight) == totalDim {
						w = float64(rmsWeight[d])
					} else {
						w = float64(rmsWeight[d%cfg.HiddenDim])
					}
				}
				dot += (float64(xRow[d]) * w) * float64(phiCombined[d*mixDim+k])
			}
			proj := dot * invRMS
			var bVal float64
			if bias != nil {
				bVal = float64(bias[k])
			}

			switch {
			case k < cfg.PreEnd:
				logit := float64(alphaPre)*proj + bVal
				res.PreMix[t*S+k] = float32(1.0 / (1.0 + math.Exp(-logit)))
			case k < cfg.PostEnd:
				logit := float64(alphaPost)*proj + bVal
				res.PostMix[t*S+(k-cfg.PreEnd)] = float32(2.0 / (1.0 + math.Exp(-logit)))
			default:
				logit := float64(alphaRes)*proj + bVal
				mat[k-cfg.PostEnd] = logit
			}
		}

		// Sinkhorn-Knopp normalization over the [S, S] residual matrix.
		maxLogit := mat[0]
		for i := 1; i < S*S; i++ {
			if mat[i] > maxLogit {
				maxLogit = mat[i]
			}
		}
		for i := 0; i < S*S; i++ {
			mat[i] = math.Exp(mat[i] - maxLogit)
		}
		sinkEps := float64(cfg.SinkhornEps)
		for iter := 0; iter < cfg.SinkhornIters; iter++ {
			// Row normalization.
			for r := 0; r < S; r++ {
				rowSum := sinkEps
				for c := 0; c < S; c++ {
					rowSum += mat[r*S+c]
				}
				for c := 0; c < S; c++ {
					mat[r*S+c] /= rowSum
				}
			}
			// Column normalization.
			for c := 0; c < S; c++ {
				colSum := sinkEps
				for r := 0; r < S; r++ {
					colSum += mat[r*S+c]
				}
				for r := 0; r < S; r++ {
					mat[r*S+c] /= colSum
				}
			}
		}
		for i := 0; i < S*S; i++ {
			res.ResMix[t*S*S+i] = float32(mat[i])
		}
	}
	return res, nil
}

// IdentityPreMix generates uniform pre-mix weights of shape [numTokens, numStreams]
// for Layer 0 hc_shifted initialization. When scale == 0, scale defaults to
// 1.0 / float32(numStreams).
func IdentityPreMix(numTokens, numStreams int, scale float32) ([]float32, error) {
	if numTokens <= 0 || numStreams <= 0 {
		return nil, fmt.Errorf("cudafuse: invalid dimensions numTokens=%d numStreams=%d", numTokens, numStreams)
	}
	if !mhcFinite(scale) || scale < 0 {
		return nil, fmt.Errorf("cudafuse: invalid scale %v", scale)
	}
	if scale == 0 {
		scale = 1.0 / float32(numStreams)
	}
	out := make([]float32, numTokens*numStreams)
	for i := range out {
		out[i] = scale
	}
	return out, nil
}

// ExecuteMHCPrePostCPU fuses the previous sub-layer's post-mix residual update
//
//	hOut[t, s, d] = sum_j(resMixPrev[t, s, j] * hPrev[t, j, d]) + postMixPrev[t, s] * yPrev[t, d]
//
// with the next sub-layer's pre-mix stream contraction
//
//	xInNext[t, d] = sum_s(preMixNext[t, s] * hOut[t, s, d])
//
// in a single pass without an intermediate HBM round-trip, matching DeepInfer's
// _hc_fused_post_pre_kernel.
func ExecuteMHCPrePostCPU(
	yPrev []float32,
	hPrev []float32,
	postMixPrev []float32,
	resMixPrev []float32,
	preMixNext []float32,
	numTokens, numStreams, hiddenDim int,
) (xInNext []float32, hOut []float32, err error) {
	if numTokens <= 0 || numStreams <= 0 || hiddenDim <= 0 {
		return nil, nil, fmt.Errorf("cudafuse: invalid dimensions numTokens=%d numStreams=%d hiddenDim=%d", numTokens, numStreams, hiddenDim)
	}
	if len(yPrev) != numTokens*hiddenDim {
		return nil, nil, fmt.Errorf("cudafuse: yPrev length %d, want %d", len(yPrev), numTokens*hiddenDim)
	}
	if len(hPrev) != numTokens*numStreams*hiddenDim {
		return nil, nil, fmt.Errorf("cudafuse: hPrev length %d, want %d", len(hPrev), numTokens*numStreams*hiddenDim)
	}
	if len(postMixPrev) != numTokens*numStreams {
		return nil, nil, fmt.Errorf("cudafuse: postMixPrev length %d, want %d", len(postMixPrev), numTokens*numStreams)
	}
	if len(resMixPrev) != numTokens*numStreams*numStreams {
		return nil, nil, fmt.Errorf("cudafuse: resMixPrev length %d, want %d", len(resMixPrev), numTokens*numStreams*numStreams)
	}
	if len(preMixNext) != numTokens*numStreams {
		return nil, nil, fmt.Errorf("cudafuse: preMixNext length %d, want %d", len(preMixNext), numTokens*numStreams)
	}
	for _, slice := range [][]float32{yPrev, hPrev, postMixPrev, resMixPrev, preMixNext} {
		if err := mhcValidateValues(slice); err != nil {
			return nil, nil, err
		}
	}

	xInNext = make([]float32, numTokens*hiddenDim)
	hOut = make([]float32, numTokens*numStreams*hiddenDim)
	S := numStreams
	D := hiddenDim

	for t := 0; t < numTokens; t++ {
		for d := 0; d < D; d++ {
			var xAcc float64
			yVal := float64(yPrev[t*D+d])
			for s := 0; s < S; s++ {
				var hAcc float64
				for j := 0; j < S; j++ {
					rMix := float64(resMixPrev[t*S*S+s*S+j])
					hVal := float64(hPrev[(t*S+j)*D+d])
					hAcc += rMix * hVal
				}
				hAcc += float64(postMixPrev[t*S+s]) * yVal
				hOutVal := float32(hAcc)
				hOut[(t*S+s)*D+d] = hOutVal
				xAcc += float64(preMixNext[t*S+s]) * float64(hOutVal)
			}
			xInNext[t*D+d] = float32(xAcc)
		}
	}
	return xInNext, hOut, nil
}

// MHCPrePostFusedSource returns self-contained NVRTC-compilable CUDA C++ source
// implementing _hc_fused_post_pre_kernel and _hc_PreSplitMixes_kernel.
func MHCPrePostFusedSource(cfg MHCPrePostConfig) (string, error) {
	cfg, err := cfg.normalize()
	if err != nil {
		return "", err
	}
	totalDim := cfg.NumStreams * cfg.HiddenDim
	return fmt.Sprintf(`// Code generated by cudafuse.MHCPrePostFusedSource; DO NOT EDIT.
#define MHC_NUM_STREAMS %d
#define MHC_HIDDEN_DIM %d
#define MHC_TOTAL_DIM %d
#define MHC_PRE_END %d
#define MHC_POST_END %d
#define MHC_RES_END %d
#define MHC_SINKHORN_ITERS %d
#define MHC_EPS %sf
#define MHC_SINKHORN_EPS %sf

extern "C" __global__ void _hc_PreSplitMixes_kernel(
    const float* __restrict__ x,
    const float* __restrict__ rms_weight,
    const float* __restrict__ phi_combined,
    const float* __restrict__ bias,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    float* __restrict__ pre_mix,
    float* __restrict__ post_mix,
    float* __restrict__ res_mix,
    int num_tokens
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;

    const float* x_row = x + (size_t)t * MHC_TOTAL_DIM;
    float sum_sq = 0.0f;
    for (int d = 0; d < MHC_TOTAL_DIM; ++d) {
        float v = x_row[d];
        sum_sq += v * v;
    }
    float inv_rms = rsqrtf(sum_sq / (float)MHC_TOTAL_DIM + MHC_EPS);

    const int S = MHC_NUM_STREAMS;
    float mat[MHC_NUM_STREAMS * MHC_NUM_STREAMS];

    for (int k = 0; k < MHC_RES_END; ++k) {
        float dot = 0.0f;
        for (int d = 0; d < MHC_TOTAL_DIM; ++d) {
            float w = rms_weight != nullptr ? rms_weight[d] : 1.0f;
            dot += (x_row[d] * w) * phi_combined[d * MHC_RES_END + k];
        }
        float proj = dot * inv_rms;
        float b_val = bias != nullptr ? bias[k] : 0.0f;
        if (k < MHC_PRE_END) {
            float logit = alpha_pre * proj + b_val;
            pre_mix[t * S + k] = 1.0f / (1.0f + expf(-logit));
        } else if (k < MHC_POST_END) {
            float logit = alpha_post * proj + b_val;
            post_mix[t * S + (k - MHC_PRE_END)] = 2.0f / (1.0f + expf(-logit));
        } else {
            mat[k - MHC_POST_END] = alpha_res * proj + b_val;
        }
    }

    float max_logit = mat[0];
    for (int i = 1; i < S * S; ++i) {
        max_logit = fmaxf(max_logit, mat[i]);
    }
    for (int i = 0; i < S * S; ++i) {
        mat[i] = expf(mat[i] - max_logit);
    }
    for (int iter = 0; iter < MHC_SINKHORN_ITERS; ++iter) {
        for (int r = 0; r < S; ++r) {
            float rsum = MHC_SINKHORN_EPS;
            for (int c = 0; c < S; ++c) rsum += mat[r * S + c];
            for (int c = 0; c < S; ++c) mat[r * S + c] /= rsum;
        }
        for (int c = 0; c < S; ++c) {
            float csum = MHC_SINKHORN_EPS;
            for (int r = 0; r < S; ++r) csum += mat[r * S + c];
            for (int r = 0; r < S; ++r) mat[r * S + c] /= csum;
        }
    }
    for (int i = 0; i < S * S; ++i) {
        res_mix[t * S * S + i] = mat[i];
    }
}

extern "C" __global__ void _hc_fused_post_pre_kernel(
    const float* __restrict__ y_prev,
    const float* __restrict__ h_prev,
    const float* __restrict__ post_mix_prev,
    const float* __restrict__ res_mix_prev,
    const float* __restrict__ pre_mix_next,
    float* __restrict__ x_in_next,
    float* __restrict__ h_out,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * MHC_HIDDEN_DIM;
    if (idx >= total) return;

    int t = idx / MHC_HIDDEN_DIM;
    int d = idx %% MHC_HIDDEN_DIM;
    const int S = MHC_NUM_STREAMS;
    const int D = MHC_HIDDEN_DIM;

    float y_val = y_prev[t * D + d];
    float x_acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        float h_acc = 0.0f;
        for (int j = 0; j < S; ++j) {
            h_acc += res_mix_prev[t * S * S + s * S + j] * h_prev[(t * S + j) * D + d];
        }
        h_acc += post_mix_prev[t * S + s] * y_val;
        h_out[(t * S + s) * D + d] = h_acc;
        x_acc += pre_mix_next[t * S + s] * h_acc;
    }
    x_in_next[t * D + d] = x_acc;
}
`, cfg.NumStreams, cfg.HiddenDim, totalDim, cfg.PreEnd, cfg.PostEnd, cfg.ResEnd, cfg.SinkhornIters,
		fmt.Sprintf("%.9e", cfg.Eps), fmt.Sprintf("%.9e", cfg.SinkhornEps)), nil
}
