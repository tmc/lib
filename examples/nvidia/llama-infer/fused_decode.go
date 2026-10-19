package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
)

// fusedDecodeCUDA holds the 5-node/layer norm-folded and epilogue-fused decode
// kernels in FP32 and BF16-weight variants. It is compiled after
// ForwardKernelsCUDA and forwardBF16CUDA in the same NVRTC module.
//
//go:embed fused_decode.cu
var fusedDecodeCUDA string

const (
	// FusedDecodeKernelsPerLayer is the number of kernel launches per decoder
	// layer in the 5-node/layer fused decode schedule:
	//   1. FusedRMSNormQKVRoPEStore[BF16]
	//   2. CausalFlashAttentionDecode
	//   3. FusedOutProjResidual[BF16]
	//   4. FusedRMSNormGateUpSwiGLU[BF16]
	//   5. FusedDownProjResidual[BF16]
	FusedDecodeKernelsPerLayer = 5

	// UnfusedDecodeKernelsPerLayer is the number of kernel launches per
	// decoder layer in the unfused schedule (RMSNorm, 3x QKV GEMV, ApplyRoPE,
	// StoreKVCache, Attention, O GEMV, AddResidual, RMSNorm, 2x Gate/Up GEMV,
	// SwiGLU, Down GEMV, AddResidual).
	UnfusedDecodeKernelsPerLayer = 15

	// FusedDecodeEliminatedWritesPerLayer is the number of intermediate
	// global-memory materialization stages eliminated per decoder layer
	// compared with the unfused schedule:
	//   1. Pre-attention RMSNorm output (NormX)
	//   2. Unrotated K/V scratch before RoPE and KV-cache store (K, V)
	//   3. Pre-residual output-projection vector (AttnProj)
	//   4. Post-attention RMSNorm output (NormX)
	//   5. Pre-activation Gate and Up vectors (Gate, Up)
	//   6. Pre-residual down-projection vector (FFNOut)
	FusedDecodeEliminatedWritesPerLayer = 6
)

// FusedDecodeLayerPlan describes the validated geometry and launch/memory
// accounting for one 5-node/layer norm-folded and epilogue-fused decoder layer.
type FusedDecodeLayerPlan struct {
	hidden, inter, qHeads, kvHeads, headDim, maxSeq int
	epsilon, theta                                  float32
	qDim, kvDim                                     int
	qPairs, kvPairs                                 int
	attn                                            attnPlan
	ffn                                             ffnPlan
}

// NewFusedDecodeLayerPlan validates all layer dimensions, GQA grouping, even
// rotary head dimension, positive finite epsilon and rope theta, and integer
// overflow bounds before constructing a FusedDecodeLayerPlan.
func NewFusedDecodeLayerPlan(hidden, inter, qHeads, kvHeads, headDim, maxSeq int, epsilon, theta float32) (FusedDecodeLayerPlan, error) {
	maxBudget := int(^uint(0) >> 1)
	ap, err := newAttnPlan(hidden, qHeads, kvHeads, headDim, maxSeq, epsilon, theta, maxBudget)
	if err != nil {
		return FusedDecodeLayerPlan{}, err
	}
	fp, err := newFFNPlan(hidden, inter, epsilon, maxBudget)
	if err != nil {
		return FusedDecodeLayerPlan{}, err
	}
	half := headDim / 2
	qPairs := qHeads * half
	kvPairs := kvHeads * half
	maxWords := int(^uint(0)>>1) / 4
	if (3*hidden+2*ap.kvDim) > maxWords || 2*inter > maxWords-(3*hidden+2*ap.kvDim) {
		return FusedDecodeLayerPlan{}, errors.New("fused decode eliminated write byte size exceeds int range")
	}
	return FusedDecodeLayerPlan{
		hidden:  hidden,
		inter:   inter,
		qHeads:  qHeads,
		kvHeads: kvHeads,
		headDim: headDim,
		maxSeq:  maxSeq,
		epsilon: epsilon,
		theta:   theta,
		qDim:    ap.qDim,
		kvDim:   ap.kvDim,
		qPairs:  qPairs,
		kvPairs: kvPairs,
		attn:    ap,
		ffn:     fp,
	}, nil
}

// KernelsPerLayer returns the number of kernel launches per decoder layer (5).
func (p FusedDecodeLayerPlan) KernelsPerLayer() int {
	return FusedDecodeKernelsPerLayer
}

// UnfusedKernelsPerLayer returns the number of unfused kernel launches per
// decoder layer (15).
func (p FusedDecodeLayerPlan) UnfusedKernelsPerLayer() int {
	return UnfusedDecodeKernelsPerLayer
}

// EliminatedLaunchesPerLayer returns the number of kernel launches saved per
// decoder layer relative to the unfused schedule (10).
func (p FusedDecodeLayerPlan) EliminatedLaunchesPerLayer() int {
	return UnfusedDecodeKernelsPerLayer - FusedDecodeKernelsPerLayer
}

// EliminatedWritesPerLayer returns the number of intermediate global-memory
// materialization stages eliminated per decoder layer (6).
func (p FusedDecodeLayerPlan) EliminatedWritesPerLayer() int {
	return FusedDecodeEliminatedWritesPerLayer
}

// EliminatedWriteBytesPerLayer returns the float32 global-memory write bytes
// saved per decoder layer across the eliminated scratchpad materializations:
// pre-attn NormX (hidden), K and V scratch (2*kvDim), O-proj scratch (hidden),
// post-attn NormX (hidden), Gate and Up scratch (2*inter), and Down-proj
// scratch (hidden) -> 4 * (4*hidden + 2*kvDim + 2*inter) bytes, plus in-place
// RoPE read/write elimination.
func (p FusedDecodeLayerPlan) EliminatedWriteBytesPerLayer() int {
	return 4 * (4*p.hidden + 2*p.kvDim + 2*p.inter)
}

// QKVRoPEStoreGrid returns the 1D grid size (qPairs + kvPairs) for
// FusedRMSNormQKVRoPEStore[BF16].
func (p FusedDecodeLayerPlan) QKVRoPEStoreGrid() uint32 {
	return uint32(p.qPairs + p.kvPairs)
}

// Stages returns the ordered 5-kernel execution schedule for a decoder layer.
func (p FusedDecodeLayerPlan) Stages() []string {
	return []string{
		"FusedRMSNormQKVRoPEStore",
		"CausalFlashAttentionDecode",
		"FusedOutProjResidual",
		"FusedRMSNormGateUpSwiGLU",
		"FusedDownProjResidual",
	}
}

// fusedDecodeLayerBuffers holds the read-only host inputs for one full decoder
// layer.
type fusedDecodeLayerBuffers struct {
	x        []float32
	attnNorm []float32
	wq       []float32
	wk       []float32
	wv       []float32
	qBias    []float32
	kBias    []float32
	vBias    []float32
	wo       []float32
	ffnNorm  []float32
	wGate    []float32
	wUp      []float32
	wDown    []float32
}

// fusedDecodeLayerOutputs holds the per-stage outputs of executeFusedDecodeLayerCPU.
type fusedDecodeLayerOutputs struct {
	Q            []float32
	AttnOut      []float32
	AttnResidual []float32
	HB           []float32
	Out          []float32
}

// fusedRMSNormQKVRoPEStoreCPU is the host reference for FusedRMSNormQKVRoPEStore.
// It computes inv_rms = 1 / sqrt(mean(x^2) + eps), projects Q, K, and V with
// optional biases, applies split-half RoPE at pos to Q and K, writes qOut, and
// stores rotated K and V into row pos of kCache and vCache.
func fusedRMSNormQKVRoPEStoreCPU(
	x, rmsAttW, wq, wk, wv, bq, bk, bv []float32,
	qOut, kCache, vCache []float32,
	inDim, qHeads, kvHeads, headDim, pos int,
	eps, ropeTheta float32,
) {
	var sumSq float32
	for i := 0; i < inDim; i++ {
		v := x[i]
		sumSq += v * v
	}
	invRMS := float32(1.0 / math.Sqrt(float64(sumSq/float32(inDim)+eps)))

	half := headDim / 2
	qPairs := qHeads * half
	kvPairs := kvHeads * half
	kvDim := kvHeads * headDim

	for pairIdx := 0; pairIdx < qPairs; pairIdx++ {
		h := pairIdx / half
		d := pairIdx % half
		row0 := h*headDim + d
		row1 := row0 + half
		wq0 := wq[row0*inDim : (row0+1)*inDim]
		wq1 := wq[row1*inDim : (row1+1)*inDim]
		var sumQ0, sumQ1 float32
		for i := 0; i < inDim; i++ {
			gw := float32(1.0)
			if rmsAttW != nil {
				gw = rmsAttW[i]
			}
			nx := x[i] * invRMS * gw
			sumQ0 += nx * wq0[i]
			sumQ1 += nx * wq1[i]
		}
		if bq != nil {
			sumQ0 += bq[row0]
			sumQ1 += bq[row1]
		}
		freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*d)/float64(headDim))
		angle := float64(pos) * freq
		cosVal := float32(math.Cos(angle))
		sinVal := float32(math.Sin(angle))
		qOut[row0] = sumQ0*cosVal - sumQ1*sinVal
		qOut[row1] = sumQ0*sinVal + sumQ1*cosVal
	}

	cacheBase := pos * kvDim
	for kvPair := 0; kvPair < kvPairs; kvPair++ {
		h := kvPair / half
		d := kvPair % half
		row0 := h*headDim + d
		row1 := row0 + half
		wk0 := wk[row0*inDim : (row0+1)*inDim]
		wk1 := wk[row1*inDim : (row1+1)*inDim]
		wv0 := wv[row0*inDim : (row0+1)*inDim]
		wv1 := wv[row1*inDim : (row1+1)*inDim]
		var sumK0, sumK1, sumV0, sumV1 float32
		for i := 0; i < inDim; i++ {
			gw := float32(1.0)
			if rmsAttW != nil {
				gw = rmsAttW[i]
			}
			nx := x[i] * invRMS * gw
			sumK0 += nx * wk0[i]
			sumK1 += nx * wk1[i]
			sumV0 += nx * wv0[i]
			sumV1 += nx * wv1[i]
		}
		if bk != nil {
			sumK0 += bk[row0]
			sumK1 += bk[row1]
		}
		if bv != nil {
			sumV0 += bv[row0]
			sumV1 += bv[row1]
		}
		freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*d)/float64(headDim))
		angle := float64(pos) * freq
		cosVal := float32(math.Cos(angle))
		sinVal := float32(math.Sin(angle))
		kCache[cacheBase+row0] = sumK0*cosVal - sumK1*sinVal
		kCache[cacheBase+row1] = sumK0*sinVal + sumK1*cosVal
		vCache[cacheBase+row0] = sumV0
		vCache[cacheBase+row1] = sumV1
	}
}

// fusedOutProjResidualCPU is the host reference for FusedOutProjResidual.
func fusedOutProjResidualCPU(attnOut, residual, wo, out []float32, inDim, outDim int) {
	for row := 0; row < outDim; row++ {
		wRow := wo[row*inDim : (row+1)*inDim]
		var sum float32
		for i := 0; i < inDim; i++ {
			sum += attnOut[i] * wRow[i]
		}
		out[row] = residual[row] + sum
	}
}

// fusedRMSNormGateUpSwiGLUCPU is the host reference for FusedRMSNormGateUpSwiGLU.
func fusedRMSNormGateUpSwiGLUCPU(x, rmsFFNW, wGate, wUp, outHB []float32, hiddenDim, interDim int, eps float32) {
	var sumSq float32
	for i := 0; i < hiddenDim; i++ {
		v := x[i]
		sumSq += v * v
	}
	invRMS := float32(1.0 / math.Sqrt(float64(sumSq/float32(hiddenDim)+eps)))

	for row := 0; row < interDim; row++ {
		wgRow := wGate[row*hiddenDim : (row+1)*hiddenDim]
		wuRow := wUp[row*hiddenDim : (row+1)*hiddenDim]
		var gateSum, upSum float32
		for i := 0; i < hiddenDim; i++ {
			gw := float32(1.0)
			if rmsFFNW != nil {
				gw = rmsFFNW[i]
			}
			nx := x[i] * invRMS * gw
			gateSum += nx * wgRow[i]
			upSum += nx * wuRow[i]
		}
		outHB[row] = (gateSum / (1.0 + float32(math.Exp(float64(-gateSum))))) * upSum
	}
}

// fusedDownProjResidualCPU is the host reference for FusedDownProjResidual.
func fusedDownProjResidualCPU(hb, residual, wDown, out []float32, interDim, hiddenDim int) {
	for row := 0; row < hiddenDim; row++ {
		wRow := wDown[row*interDim : (row+1)*interDim]
		var sum float32
		for i := 0; i < interDim; i++ {
			sum += hb[i] * wRow[i]
		}
		out[row] = residual[row] + sum
	}
}

func (p FusedDecodeLayerPlan) validate(b fusedDecodeLayerBuffers, c attnCache, pos int) error {
	checked, err := NewFusedDecodeLayerPlan(p.hidden, p.inter, p.qHeads, p.kvHeads, p.headDim, p.maxSeq, p.epsilon, p.theta)
	if err != nil {
		return err
	}
	if checked != p {
		return fmt.Errorf("fused decode plan storage counts are inconsistent")
	}
	ab := attnBuffers{
		x:     b.x,
		norm:  b.attnNorm,
		wq:    b.wq,
		wk:    b.wk,
		wv:    b.wv,
		wo:    b.wo,
		qBias: b.qBias,
		kBias: b.kBias,
		vBias: b.vBias,
	}
	if err := p.attn.validate(ab, c, pos); err != nil {
		return err
	}
	fb := ffnBuffers{
		x:    b.x,
		norm: b.ffnNorm,
		gate: b.wGate,
		up:   b.wUp,
		down: b.wDown,
	}
	return p.ffn.validate(fb)
}

// executeFusedDecodeLayerCPU executes the 5-stage norm-folded and epilogue-fused
// decoder layer on the host and returns all stage outputs. It writes only row
// pos of cache.k and cache.v and leaves all inputs and other cache rows
// unchanged.
func executeFusedDecodeLayerCPU(p FusedDecodeLayerPlan, b fusedDecodeLayerBuffers, cache attnCache, pos int) (fusedDecodeLayerOutputs, error) {
	if err := p.validate(b, cache, pos); err != nil {
		return fusedDecodeLayerOutputs{}, err
	}
	q := make([]float32, p.qDim)
	fusedRMSNormQKVRoPEStoreCPU(
		b.x, b.attnNorm, b.wq, b.wk, b.wv, b.qBias, b.kBias, b.vBias,
		q, cache.k, cache.v,
		p.hidden, p.qHeads, p.kvHeads, p.headDim, pos, p.epsilon, p.theta,
	)

	attnOut := make([]float32, p.qDim)
	scores := make([]float32, pos+1)
	group := p.qHeads / p.kvHeads
	invScale := float32(1.0 / math.Sqrt(float64(p.headDim)))
	for h := 0; h < p.qHeads; h++ {
		kvh := h / group
		qh := q[h*p.headDim : (h+1)*p.headDim]
		peak := float32(math.Inf(-1))
		for t := 0; t <= pos; t++ {
			kt := cache.k[t*p.kvDim+kvh*p.headDim : t*p.kvDim+(kvh+1)*p.headDim]
			var dot float32
			for d, xv := range qh {
				dot += xv * kt[d]
			}
			scores[t] = dot * invScale
			peak = float32(math.Max(float64(peak), float64(scores[t])))
		}
		var total float32
		for t, s := range scores {
			scores[t] = float32(math.Exp(float64(s - peak)))
			total += scores[t]
		}
		headOut := attnOut[h*p.headDim : (h+1)*p.headDim]
		for t, s := range scores {
			vt := cache.v[t*p.kvDim+kvh*p.headDim : t*p.kvDim+(kvh+1)*p.headDim]
			for d := range headOut {
				headOut[d] += s * vt[d]
			}
		}
		for d := range headOut {
			headOut[d] /= total
		}
	}

	attnRes := make([]float32, p.hidden)
	fusedOutProjResidualCPU(attnOut, b.x, b.wo, attnRes, p.qDim, p.hidden)

	hb := make([]float32, p.inter)
	fusedRMSNormGateUpSwiGLUCPU(attnRes, b.ffnNorm, b.wGate, b.wUp, hb, p.hidden, p.inter, p.epsilon)

	out := make([]float32, p.hidden)
	fusedDownProjResidualCPU(hb, attnRes, b.wDown, out, p.inter, p.hidden)

	return fusedDecodeLayerOutputs{
		Q:            q,
		AttnOut:      attnOut,
		AttnResidual: attnRes,
		HB:           hb,
		Out:          out,
	}, nil
}
