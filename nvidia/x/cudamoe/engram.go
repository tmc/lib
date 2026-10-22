// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"fmt"
	"math"
	"strings"
	"unicode"
)

// CompressedTokenizer holds a surjective mapping from raw vocabulary token IDs
// onto canonical compressed token IDs by collapsing surface-equivalent token
// variants (compatibility width, case, and whitespace runs).
type CompressedTokenizer struct {
	lookupTable   []int32
	NumCompressed int
}

func normalizeTokenPiece(s string) string {
	if s == "" {
		return ""
	}
	if strings.ContainsRune(s, '\ufffd') {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	inSpace := false
	hasNonSpace := false
	for _, r := range s {
		// Fold common fullwidth ASCII variants (U+FF01..U+FF5E) and ideographic space (U+3000)
		// as part of NFKC compatibility normalization without external dependencies.
		if r >= 0xFF01 && r <= 0xFF5E {
			r -= 0xFEE0
		} else if r == 0x3000 || r == '\u00a0' || r == '▁' {
			r = ' '
		}
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		r = unicode.ToLower(r)
		if unicode.IsSpace(r) {
			inSpace = true
			continue
		}
		if inSpace && hasNonSpace {
			b.WriteByte(' ')
		}
		inSpace = false
		hasNonSpace = true
		b.WriteRune(r)
	}
	if !hasNonSpace {
		if inSpace {
			return " "
		}
		return s
	}
	return b.String()
}

// NewCompressedTokenizer builds a CompressedTokenizer from vocab by grouping
// normalized token representations into equivalence classes in order of first
// appearance.
func NewCompressedTokenizer(vocab []string) *CompressedTokenizer {
	lookup := make([]int32, len(vocab))
	canonToID := make(map[string]int32, len(vocab))
	var nextID int32
	for i, tok := range vocab {
		key := normalizeTokenPiece(tok)
		id, ok := canonToID[key]
		if !ok {
			id = nextID
			canonToID[key] = id
			nextID++
		}
		lookup[i] = id
	}
	return &CompressedTokenizer{
		lookupTable:   lookup,
		NumCompressed: int(nextID),
	}
}

// CompressIDs maps raw vocabulary token IDs to their compressed canonical IDs.
func (ct *CompressedTokenizer) CompressIDs(ids []int32) ([]int32, error) {
	if ct == nil {
		return nil, fmt.Errorf("cudamoe: nil CompressedTokenizer")
	}
	out := make([]int32, len(ids))
	for i, id := range ids {
		if id < 0 || int(id) >= len(ct.lookupTable) {
			return nil, fmt.Errorf("cudamoe: token ID %d at index %d out of range [0, %d)", id, i, len(ct.lookupTable))
		}
		out[i] = ct.lookupTable[id]
	}
	return out, nil
}

// EngramSpec configures multi-order causal n-gram hash lookup.
type EngramSpec struct {
	MaxNgramSize     int
	NumHeadsPerNgram int
	EmbedDimPerHead  int
	LayerID          int
	PadID            int32
	Seed             int64
	TablePrimes      [][]int64
	Multipliers      []int64
}

// ComputeEngramMultipliers returns maxNgramSize deterministic positive odd
// multipliers for a given layerID and seed, matching DeepInfer's NgramHashMapping
// odd-multiplier construction.
func ComputeEngramMultipliers(maxNgramSize int, layerID int, seed int64) []int64 {
	if maxNgramSize <= 0 {
		return nil
	}
	state := uint64(seed) + 10007*uint64(int64(layerID))
	mults := make([]int64, maxNgramSize)
	for i := 0; i < maxNgramSize; i++ {
		state += 0x9e3779b97f4a7c15
		z := state
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z = z ^ (z >> 31)
		// Keep strictly positive 62-bit odd integer.
		val := int64((z & 0x3fffffffffffffff) | 1)
		mults[i] = val
	}
	return mults
}

// EngramHashIndicesCPU computes causal n-gram XOR-multiplicative hash indices
// within each sequence boundary defined by cuSeqlens (preventing cross-sequence
// n-gram leakage by padding prefix positions < k with spec.PadID) and reduces
// each hash modulo spec.TablePrimes[n-2][h].
//
// The returned slice has length len(compressedIDs) * (MaxNgramSize-1) * NumHeadsPerNgram,
// laid out in row-major [token, ngramOrder(2..MaxNgramSize), head] order.
func EngramHashIndicesCPU(compressedIDs []int32, cuSeqlens []int32, spec EngramSpec) ([]int64, error) {
	if spec.MaxNgramSize < 2 {
		return nil, fmt.Errorf("cudamoe: MaxNgramSize %d must be at least 2", spec.MaxNgramSize)
	}
	if spec.NumHeadsPerNgram <= 0 {
		return nil, fmt.Errorf("cudamoe: NumHeadsPerNgram %d must be positive", spec.NumHeadsPerNgram)
	}
	numOrders := spec.MaxNgramSize - 1
	if len(spec.TablePrimes) != numOrders {
		return nil, fmt.Errorf("cudamoe: len(TablePrimes) %d must equal MaxNgramSize-1 (%d)", len(spec.TablePrimes), numOrders)
	}
	for nIdx, row := range spec.TablePrimes {
		if len(row) != spec.NumHeadsPerNgram {
			return nil, fmt.Errorf("cudamoe: len(TablePrimes[%d]) %d must equal NumHeadsPerNgram %d", nIdx, len(row), spec.NumHeadsPerNgram)
		}
		for h, p := range row {
			if p <= 0 {
				return nil, fmt.Errorf("cudamoe: TablePrimes[%d][%d] = %d must be positive", nIdx, h, p)
			}
		}
	}
	if len(cuSeqlens) < 2 {
		return nil, fmt.Errorf("cudamoe: cuSeqlens must have at least 2 elements, got %d", len(cuSeqlens))
	}
	if cuSeqlens[0] != 0 {
		return nil, fmt.Errorf("cudamoe: cuSeqlens[0] = %d, want 0", cuSeqlens[0])
	}
	for i := 1; i < len(cuSeqlens); i++ {
		if cuSeqlens[i] < cuSeqlens[i-1] {
			return nil, fmt.Errorf("cudamoe: cuSeqlens must be non-decreasing at index %d (%d < %d)", i, cuSeqlens[i], cuSeqlens[i-1])
		}
	}
	totalTokens := int(cuSeqlens[len(cuSeqlens)-1])
	if totalTokens != len(compressedIDs) {
		return nil, fmt.Errorf("cudamoe: cuSeqlens end %d does not match len(compressedIDs) %d", totalTokens, len(compressedIDs))
	}

	mults := spec.Multipliers
	if len(mults) == 0 {
		mults = ComputeEngramMultipliers(spec.MaxNgramSize, spec.LayerID, spec.Seed)
	} else if len(mults) < spec.MaxNgramSize {
		return nil, fmt.Errorf("cudamoe: len(Multipliers) %d must be at least MaxNgramSize %d", len(mults), spec.MaxNgramSize)
	}

	entriesPerToken := numOrders * spec.NumHeadsPerNgram
	out := make([]int64, totalTokens*entriesPerToken)

	for seq := 0; seq < len(cuSeqlens)-1; seq++ {
		start := int(cuSeqlens[seq])
		end := int(cuSeqlens[seq+1])
		for t := start; t < end; t++ {
			baseOut := t * entriesPerToken
			var xorMix uint64
			for k := 0; k < spec.MaxNgramSize; k++ {
				tok := spec.PadID
				if t-k >= start {
					tok = compressedIDs[t-k]
				}
				term := uint64(int64(tok)) * uint64(mults[k])
				xorMix ^= term
				if k >= 1 {
					nIdx := k - 1
					primes := spec.TablePrimes[nIdx]
					for h := 0; h < spec.NumHeadsPerNgram; h++ {
						p := uint64(primes[h])
						out[baseOut+nIdx*spec.NumHeadsPerNgram+h] = int64(xorMix % p)
					}
				}
			}
		}
	}
	return out, nil
}

// EngramGateCPU computes DeepInfer's engram_gate on host:
//  1. RMSNorm on query and key per token using rmsQueryW, rmsKeyW, and eps.
//  2. Dot product scaled by 1 / sqrt(hiddenDim).
//  3. Signed square-root sigmoid gate: alpha = sigmoid(sign(s) * sqrt(|s| + eps)).
//  4. Gated value output: out[t, d] = alpha * value[t, d].
func EngramGateCPU(
	query []float32,
	key []float32,
	value []float32,
	rmsQueryW []float32,
	rmsKeyW []float32,
	numTokens, hiddenDim int,
	eps float32,
) ([]float32, error) {
	if numTokens <= 0 || hiddenDim <= 0 {
		return nil, fmt.Errorf("cudamoe: numTokens %d and hiddenDim %d must be positive", numTokens, hiddenDim)
	}
	if !finite(eps) || eps <= 0 {
		return nil, fmt.Errorf("cudamoe: eps %v must be finite and positive", eps)
	}
	expected := numTokens * hiddenDim
	if len(query) != expected || len(key) != expected || len(value) != expected {
		return nil, fmt.Errorf("cudamoe: query(%d), key(%d), value(%d) must all have length %d", len(query), len(key), len(value), expected)
	}
	if len(rmsQueryW) != hiddenDim || len(rmsKeyW) != hiddenDim {
		return nil, fmt.Errorf("cudamoe: rmsQueryW(%d) and rmsKeyW(%d) must have length hiddenDim %d", len(rmsQueryW), len(rmsKeyW), hiddenDim)
	}

	out := make([]float32, expected)
	invSqrtDim := 1.0 / math.Sqrt(float64(hiddenDim))

	for t := 0; t < numTokens; t++ {
		qRow := query[t*hiddenDim : (t+1)*hiddenDim]
		kRow := key[t*hiddenDim : (t+1)*hiddenDim]
		vRow := value[t*hiddenDim : (t+1)*hiddenDim]
		oRow := out[t*hiddenDim : (t+1)*hiddenDim]

		var qSumSq, kSumSq float64
		for d := 0; d < hiddenDim; d++ {
			qv := float64(qRow[d])
			kv := float64(kRow[d])
			qSumSq += qv * qv
			kSumSq += kv * kv
		}
		qRmsInv := 1.0 / math.Sqrt(qSumSq/float64(hiddenDim)+float64(eps))
		kRmsInv := 1.0 / math.Sqrt(kSumSq/float64(hiddenDim)+float64(eps))

		var dot float64
		for d := 0; d < hiddenDim; d++ {
			qn := float64(qRow[d]) * qRmsInv * float64(rmsQueryW[d])
			kn := float64(kRow[d]) * kRmsInv * float64(rmsKeyW[d])
			dot += qn * kn
		}
		s := dot * invSqrtDim

		var sign float64
		if s > 0 {
			sign = 1.0
		} else if s < 0 {
			sign = -1.0
		}
		signedSqrt := sign * math.Sqrt(math.Abs(s)+float64(eps))
		alpha := float32(1.0 / (1.0 + math.Exp(-signedSqrt)))

		for d := 0; d < hiddenDim; d++ {
			oRow[d] = alpha * vRow[d]
		}
	}
	return out, nil
}

// FusedEngramGateSource generates the NVRTC CUDA C++ source for engram_gate_kernel.
func FusedEngramGateSource(hiddenDim int, eps float32) (string, error) {
	if hiddenDim <= 0 {
		return "", fmt.Errorf("cudamoe: hiddenDim %d must be positive", hiddenDim)
	}
	if !finite(eps) || eps <= 0 {
		return "", fmt.Errorf("cudamoe: eps %v must be finite and positive", eps)
	}
	src := fmt.Sprintf(`extern "C" __global__ void engram_gate_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ rms_q_w,
    const float* __restrict__ rms_k_w,
    float* __restrict__ out,
    int num_tokens
) {
    const int hidden_dim = %d;
    const float eps = %.9ef;
    int t = (int)blockIdx.x;
    if (t >= num_tokens) return;

    const float* q_row = query + (size_t)t * hidden_dim;
    const float* k_row = key + (size_t)t * hidden_dim;
    const float* v_row = value + (size_t)t * hidden_dim;
    float* o_row = out + (size_t)t * hidden_dim;

    float q_sq = 0.0f;
    float k_sq = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        float qv = q_row[d];
        float kv = k_row[d];
        q_sq += qv * qv;
        k_sq += kv * kv;
    }
    float q_inv = rsqrtf(q_sq / (float)hidden_dim + eps);
    float k_inv = rsqrtf(k_sq / (float)hidden_dim + eps);

    float dot = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        float qn = q_row[d] * q_inv * rms_q_w[d];
        float kn = k_row[d] * k_inv * rms_k_w[d];
        dot += qn * kn;
    }
    float s = dot * rsqrtf((float)hidden_dim);
    float s_sign = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    float abs_s = (s >= 0.0f) ? s : -s;
    float signed_sqrt = s_sign * sqrtf(abs_s + eps);
    float alpha = 1.0f / (1.0f + expf(-signed_sqrt));

    for (int d = (int)threadIdx.x; d < hidden_dim; d += (int)blockDim.x) {
        o_row[d] = alpha * v_row[d];
    }
}
`, hiddenDim, eps)
	return src, nil
}
