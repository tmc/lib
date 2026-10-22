// SPDX-License-Identifier: Apache-2.0

package cudaspec

import (
	"errors"
	"fmt"
)

// ErrInvalidVarlenSpec indicates invalid parameters passed to ComputeSeqlensQByConfidenceCPU
// or VarlenSeqlensKernelSource.
var ErrInvalidVarlenSpec = errors.New("cudaspec: invalid varlen confidence schedule parameters")

// ComputeSeqlensQByConfidenceCPU computes the exact CPU reference for DeepInfer's
// _compute_seqlens_q_by_conf_kernel (deepinfer/kernels/schedule_mtp_varlen.py):
//  1. Each sequence b in [0, bsz) starts with base query length 1 (the verified/bonus token),
//     and extraBudget is clamped to [0, bsz * maxStep] from targetTotalBsz - bsz.
//  2. For each sequence b, srcRow = remap[b] (or b when remap == nil; -1 yields 0.0 cumulative
//     probabilities for newly admitted sequences), and cumulative products
//     c[b, s] = prod_{k=0..s} prob[srcRow][k] are computed for s in [0, maxStep).
//  3. Across all bsz*maxStep flattened entries, global rank[i] counts how many entries j
//     satisfy (m_j > m_i) || (m_j == m_i && j < i) (deterministic flat-index tie-breaking).
//  4. Entries with rank < extraBudget are allocated to their sequence, yielding seqlensQ[b]
//     and prefix sums cuSeqlensQ of length bsz + 1 (with cuSeqlensQ[0] = 0).
func ComputeSeqlensQByConfidenceCPU(
	prob [][]float32,
	remap []int32,
	targetTotalBsz int,
	maxStep int,
) (seqlensQ []int32, cuSeqlensQ []int32, err error) {
	if maxStep < 0 {
		return nil, nil, fmt.Errorf("%w: maxStep must be >= 0, got %d", ErrInvalidVarlenSpec, maxStep)
	}
	bsz := len(prob)
	if remap != nil {
		bsz = len(remap)
	}
	if bsz == 0 {
		return nil, nil, fmt.Errorf("%w: batch size must be > 0", ErrInvalidVarlenSpec)
	}

	totalCand := bsz * maxStep
	flat := make([]float32, totalCand)

	for b := 0; b < bsz; b++ {
		srcRow := b
		if remap != nil {
			srcRow = int(remap[b])
		}
		if srcRow == -1 {
			continue
		}
		if srcRow < -1 || srcRow >= len(prob) {
			return nil, nil, fmt.Errorf("%w: remap[%d]=%d out of range [0, %d)", ErrInvalidVarlenSpec, b, srcRow, len(prob))
		}
		if len(prob[srcRow]) < maxStep {
			return nil, nil, fmt.Errorf("%w: prob[%d] length %d < maxStep %d", ErrInvalidVarlenSpec, srcRow, len(prob[srcRow]), maxStep)
		}
		acc := float32(1.0)
		for s := 0; s < maxStep; s++ {
			acc *= prob[srcRow][s]
			flat[b*maxStep+s] = acc
		}
	}

	extraBudget := targetTotalBsz - bsz
	if extraBudget < 0 {
		extraBudget = 0
	} else if extraBudget > totalCand {
		extraBudget = totalCand
	}

	seqlensQ = make([]int32, bsz)
	for b := 0; b < bsz; b++ {
		var extra int32
		for s := 0; s < maxStep; s++ {
			i := b*maxStep + s
			mi := flat[i]
			rank := 0
			for j := 0; j < totalCand; j++ {
				mj := flat[j]
				if mj > mi || (mj == mi && j < i) {
					rank++
				}
			}
			if rank < extraBudget {
				extra++
			}
		}
		seqlensQ[b] = 1 + extra
	}

	cuSeqlensQ = make([]int32, bsz+1)
	var sum int32
	for b := 0; b < bsz; b++ {
		cuSeqlensQ[b] = sum
		sum += seqlensQ[b]
	}
	cuSeqlensQ[bsz] = sum

	return seqlensQ, cuSeqlensQ, nil
}

// VarlenSeqlensKernelSource generates NVRTC CUDA C++ source implementing
// compute_seqlens_q_by_conf_kernel with shared-memory cumulative products,
// parallel rank counting with flat-index tie-breaking, and prefix scan for cu_seqlens_q.
func VarlenSeqlensKernelSource(maxBsz int, maxStep int) (string, error) {
	if maxBsz <= 0 {
		return "", fmt.Errorf("%w: maxBsz must be > 0, got %d", ErrInvalidVarlenSpec, maxBsz)
	}
	if maxStep <= 0 {
		return "", fmt.Errorf("%w: maxStep must be > 0, got %d", ErrInvalidVarlenSpec, maxStep)
	}
	maxFlat := maxBsz * maxStep

	src := fmt.Sprintf(`// Auto-generated Varlen Seqlens Confidence Kernel by nvidia/x/cudaspec
#include <cuda_runtime.h>

extern "C" __global__ void compute_seqlens_q_by_conf_kernel(
    const float* __restrict__ prob,        // [n_prev_seq, %d]
    const int*   __restrict__ remap,       // [bsz] or nullptr
    int                       bsz,
    int                       target_total_bsz,
    int*         __restrict__ seqlens_q,   // [bsz]
    int*         __restrict__ cu_seqlens_q // [bsz + 1]
) {
    const int MAX_BSZ  = %d;
    const int MAX_STEP = %d;
    if (bsz <= 0 || bsz > MAX_BSZ) return;

    __shared__ float s_cum[%d];
    __shared__ int   s_seqlens[%d];

    const int tid = threadIdx.x;
    const int total_cand = bsz * MAX_STEP;

    // Phase 1: compute per-sequence cumulative product in shared memory.
    for (int b = tid; b < bsz; b += blockDim.x) {
        int src_row = (remap != nullptr) ? remap[b] : b;
        float acc = (src_row >= 0) ? 1.0f : 0.0f;
        for (int s = 0; s < MAX_STEP; ++s) {
            if (src_row >= 0) {
                acc *= prob[src_row * MAX_STEP + s];
            }
            s_cum[b * MAX_STEP + s] = acc;
        }
    }
    __syncthreads();

    int extra_budget = target_total_bsz - bsz;
    if (extra_budget < 0) extra_budget = 0;
    if (extra_budget > total_cand) extra_budget = total_cand;

    // Phase 2: parallel rank counting with deterministic flat-index tie-breaking.
    for (int b = tid; b < bsz; b += blockDim.x) {
        int num_tokens = 0;
        for (int s = 0; s < MAX_STEP; ++s) {
            int my_idx = b * MAX_STEP + s;
            float my_val = s_cum[my_idx];
            int rank = 0;
            for (int other_idx = 0; other_idx < total_cand; ++other_idx) {
                float other_val = s_cum[other_idx];
                if (other_val > my_val || (other_val == my_val && other_idx < my_idx)) {
                    rank += 1;
                }
            }
            if (rank < extra_budget) {
                num_tokens += 1;
            }
        }
        int q = 1 + num_tokens;
        s_seqlens[b] = q;
        seqlens_q[b] = q;
    }
    __syncthreads();

    // Phase 3: prefix scan for cu_seqlens_q[0..bsz].
    if (tid == 0) {
        int prefix = 0;
        for (int b = 0; b < bsz; ++b) {
            cu_seqlens_q[b] = prefix;
            prefix += s_seqlens[b];
        }
        cu_seqlens_q[bsz] = prefix;
    }
}
`, maxStep, maxBsz, maxStep, maxFlat, maxBsz)

	return src, nil
}
