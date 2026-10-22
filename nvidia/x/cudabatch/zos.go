// SPDX-License-Identifier: Apache-2.0

package cudabatch

import (
	"errors"
	"fmt"
)

// ErrInvalidZOSPlan indicates invalid parameters or mismatched slice lengths in ZOS remap planning.
var ErrInvalidZOSPlan = errors.New("cudabatch: invalid ZOS remap parameters")

// ZOSRemapSpec configures device-side Zero-Overhead Speculation (ZOS) remap kernel generation.
type ZOSRemapSpec struct {
	MaxBatchSize int
	MaxDraftStep int
}

// ZOSRemapPlan describes the device-side Zero-Overhead Speculation (ZOS) token
// compaction and position schedule between decode step N-1 and decode step N.
type ZOSRemapPlan struct {
	Remap            []int32
	SeqlensQ         []int32
	CuSeqlensQ       []int32
	Positions        []int32
	OutToIn          []int32
	TotalDraftTokens int
}

// BuildZOSRemapPlan computes the device-side Zero-Overhead Speculation (ZOS) token
// compaction and position schedule between step N-1 and step N:
//   - Remap[b] = prevCuSeqlensQ[b] + acceptedCounts[b] - 1 (points to the last
//     accepted token slot in step N-1's flattened output so step N's hidden state
//     and logits can be gathered without a host round-trip).
//   - CuSeqlensQ is the prefix sum of nextSeqlensQ of length bsz + 1.
//   - Positions and OutToIn hold the flattened token positions and batch indices
//     for all TotalDraftTokens tokens in step N.
func BuildZOSRemapPlan(
	prevCuSeqlensQ []int32,
	acceptedCounts []int32,
	nextSeqlensQ []int32,
	basePositions []int32,
) (ZOSRemapPlan, error) {
	bsz := len(acceptedCounts)
	if bsz == 0 {
		return ZOSRemapPlan{}, fmt.Errorf("%w: empty batch", ErrInvalidZOSPlan)
	}
	if len(nextSeqlensQ) != bsz {
		return ZOSRemapPlan{}, fmt.Errorf("%w: nextSeqlensQ length %d != batch size %d", ErrInvalidZOSPlan, len(nextSeqlensQ), bsz)
	}
	if len(basePositions) != bsz {
		return ZOSRemapPlan{}, fmt.Errorf("%w: basePositions length %d != batch size %d", ErrInvalidZOSPlan, len(basePositions), bsz)
	}
	if len(prevCuSeqlensQ) != bsz && len(prevCuSeqlensQ) != bsz+1 {
		return ZOSRemapPlan{}, fmt.Errorf("%w: prevCuSeqlensQ length %d must be %d or %d", ErrInvalidZOSPlan, len(prevCuSeqlensQ), bsz, bsz+1)
	}

	remap := make([]int32, bsz)
	seqlensQ := make([]int32, bsz)
	cuSeqlensQ := make([]int32, bsz+1)

	var totalQ int32
	for b := 0; b < bsz; b++ {
		if prevCuSeqlensQ[b] < 0 {
			return ZOSRemapPlan{}, fmt.Errorf("%w: prevCuSeqlensQ[%d]=%d < 0", ErrInvalidZOSPlan, b, prevCuSeqlensQ[b])
		}
		if acceptedCounts[b] <= 0 {
			return ZOSRemapPlan{}, fmt.Errorf("%w: acceptedCounts[%d]=%d must be >= 1", ErrInvalidZOSPlan, b, acceptedCounts[b])
		}
		if len(prevCuSeqlensQ) == bsz+1 {
			prevLen := prevCuSeqlensQ[b+1] - prevCuSeqlensQ[b]
			if acceptedCounts[b] > prevLen {
				return ZOSRemapPlan{}, fmt.Errorf("%w: acceptedCounts[%d]=%d exceeds previous seqlen_q %d", ErrInvalidZOSPlan, b, acceptedCounts[b], prevLen)
			}
		}
		if nextSeqlensQ[b] <= 0 {
			return ZOSRemapPlan{}, fmt.Errorf("%w: nextSeqlensQ[%d]=%d must be >= 1", ErrInvalidZOSPlan, b, nextSeqlensQ[b])
		}
		if basePositions[b] < 0 {
			return ZOSRemapPlan{}, fmt.Errorf("%w: basePositions[%d]=%d < 0", ErrInvalidZOSPlan, b, basePositions[b])
		}

		remap[b] = prevCuSeqlensQ[b] + acceptedCounts[b] - 1
		seqlensQ[b] = nextSeqlensQ[b]
		cuSeqlensQ[b] = totalQ
		totalQ += nextSeqlensQ[b]
	}
	cuSeqlensQ[bsz] = totalQ

	positions := make([]int32, totalQ)
	outToIn := make([]int32, totalQ)
	for b := 0; b < bsz; b++ {
		qStart := cuSeqlensQ[b]
		qLen := seqlensQ[b]
		startPos := basePositions[b] + acceptedCounts[b]
		for j := int32(0); j < qLen; j++ {
			positions[qStart+j] = startPos + j
			outToIn[qStart+j] = int32(b)
		}
	}

	return ZOSRemapPlan{
		Remap:            remap,
		SeqlensQ:         seqlensQ,
		CuSeqlensQ:       cuSeqlensQ,
		Positions:        positions,
		OutToIn:          outToIn,
		TotalDraftTokens: int(totalQ),
	}, nil
}

// ApplyZOSRemapCPU gathers compacted step N input token IDs and updates per-sequence
// KV cache lengths (nextSeqlensK) according to plan without host synchronization.
func ApplyZOSRemapCPU(
	prevTokens []int32,
	bonusTokens []int32,
	prevSeqlensK []int32,
	acceptedCounts []int32,
	plan ZOSRemapPlan,
) (nextInputIDs []int32, nextSeqlensK []int32, err error) {
	bsz := len(plan.SeqlensQ)
	if bsz == 0 {
		return nil, nil, fmt.Errorf("%w: empty ZOSRemapPlan", ErrInvalidZOSPlan)
	}
	if len(plan.Remap) != bsz || len(plan.CuSeqlensQ) != bsz+1 {
		return nil, nil, fmt.Errorf("%w: inconsistent ZOSRemapPlan lengths", ErrInvalidZOSPlan)
	}
	if len(prevSeqlensK) != bsz {
		return nil, nil, fmt.Errorf("%w: prevSeqlensK length %d != batch size %d", ErrInvalidZOSPlan, len(prevSeqlensK), bsz)
	}
	if len(acceptedCounts) != bsz {
		return nil, nil, fmt.Errorf("%w: acceptedCounts length %d != batch size %d", ErrInvalidZOSPlan, len(acceptedCounts), bsz)
	}

	totalQ := int(plan.CuSeqlensQ[bsz])
	if totalQ != plan.TotalDraftTokens || totalQ < bsz {
		return nil, nil, fmt.Errorf("%w: invalid TotalDraftTokens %d (cuSeqlensQ=%d)", ErrInvalidZOSPlan, plan.TotalDraftTokens, totalQ)
	}

	maxExtra := 0
	totalExtra := 0
	for b := 0; b < bsz; b++ {
		qLen := int(plan.SeqlensQ[b])
		if qLen <= 0 {
			return nil, nil, fmt.Errorf("%w: SeqlensQ[%d]=%d <= 0", ErrInvalidZOSPlan, b, qLen)
		}
		extra := qLen - 1
		if extra > maxExtra {
			maxExtra = extra
		}
		totalExtra += extra
	}

	bonusStride := 0
	if maxExtra > 0 {
		if len(bonusTokens) == totalQ {
			bonusStride = -1 // indexed by [qStart + j]
		} else if len(bonusTokens) == totalExtra {
			bonusStride = -2 // packed extra tokens [qStart - b + (j - 1)]
		} else if len(bonusTokens)%bsz == 0 && len(bonusTokens)/bsz >= maxExtra {
			bonusStride = len(bonusTokens) / bsz
		} else {
			return nil, nil, fmt.Errorf("%w: bonusTokens length %d insufficient for %d extra draft tokens", ErrInvalidZOSPlan, len(bonusTokens), totalExtra)
		}
	}

	nextInputIDs = make([]int32, totalQ)
	nextSeqlensK = make([]int32, bsz)

	for b := 0; b < bsz; b++ {
		if prevSeqlensK[b] < 0 {
			return nil, nil, fmt.Errorf("%w: prevSeqlensK[%d]=%d < 0", ErrInvalidZOSPlan, b, prevSeqlensK[b])
		}
		if acceptedCounts[b] <= 0 {
			return nil, nil, fmt.Errorf("%w: acceptedCounts[%d]=%d <= 0", ErrInvalidZOSPlan, b, acceptedCounts[b])
		}
		srcIdx := int(plan.Remap[b])
		if srcIdx < 0 || srcIdx >= len(prevTokens) {
			return nil, nil, fmt.Errorf("%w: Remap[%d]=%d out of range [0, %d)", ErrInvalidZOSPlan, b, srcIdx, len(prevTokens))
		}

		nextSeqlensK[b] = prevSeqlensK[b] + acceptedCounts[b]
		qStart := int(plan.CuSeqlensQ[b])
		qLen := int(plan.SeqlensQ[b])

		// First query token in step N is the last accepted token from step N-1.
		nextInputIDs[qStart] = prevTokens[srcIdx]

		// Remaining query tokens in step N come from bonusTokens.
		for j := 1; j < qLen; j++ {
			var tok int32
			switch bonusStride {
			case -1:
				tok = bonusTokens[qStart+j]
			case -2:
				tok = bonusTokens[(qStart-b)+(j-1)]
			default:
				tok = bonusTokens[b*bonusStride+(j-1)]
			}
			nextInputIDs[qStart+j] = tok
		}
	}

	return nextInputIDs, nextSeqlensK, nil
}

// ZOSRemapKernelSource generates the NVRTC CUDA C++ source for
// zos_remap_and_pack_kernel, which computes device-side ZOS remap indices,
// compacted input token IDs, token positions, batch indices, and updated KV lengths.
func ZOSRemapKernelSource() (string, error) {
	src := `// Auto-generated ZOS Remap and Pack CUDA kernel by nvidia/x/cudabatch
#include <cuda_runtime.h>

extern "C" __global__ void zos_remap_and_pack_kernel(
    const int* __restrict__ prev_cu_seqlens_q, // [bsz + 1] or [bsz]
    const int* __restrict__ accepted_counts,   // [bsz]
    const int* __restrict__ next_seqlens_q,    // [bsz]
    const int* __restrict__ next_cu_seqlens_q, // [bsz + 1]
    const int* __restrict__ base_positions,    // [bsz]
    const int* __restrict__ prev_tokens,       // [total_prev_q]
    const int* __restrict__ bonus_tokens,      // [bsz, bonus_stride]
    const int* __restrict__ prev_seqlens_k,    // [bsz]
    int*       __restrict__ remap_out,         // [bsz]
    int*       __restrict__ next_input_ids,    // [total_next_q]
    int*       __restrict__ positions_out,     // [total_next_q]
    int*       __restrict__ out_to_in,         // [total_next_q]
    int*       __restrict__ next_seqlens_k,    // [bsz]
    int                     bsz,
    int                     bonus_stride
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bsz) return;

    int acc = accepted_counts[b];
    int remap_idx = prev_cu_seqlens_q[b] + acc - 1;
    remap_out[b] = remap_idx;
    next_seqlens_k[b] = prev_seqlens_k[b] + acc;

    int q_start = next_cu_seqlens_q[b];
    int q_len   = next_seqlens_q[b];
    int pos0    = base_positions[b] + acc;

    for (int j = 0; j < q_len; ++j) {
        int out_idx = q_start + j;
        positions_out[out_idx] = pos0 + j;
        out_to_in[out_idx]     = b;
        if (j == 0) {
            next_input_ids[out_idx] = prev_tokens[remap_idx];
        } else {
            next_input_ids[out_idx] = bonus_tokens[b * bonus_stride + (j - 1)];
        }
    }
}
`
	return src, nil
}
