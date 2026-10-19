package main

import (
	"math"
	"reflect"
	"testing"
)

// fusedDecodeDefect enumerates the 7 must-fail defect controls (with both O and
// Down residual omissions tested).
type fusedDecodeDefect int

const (
	fusedDefectNone           fusedDecodeDefect = iota
	fusedDefectNoRsqrt                          // 1. missing rsqrt in RMSNorm
	fusedDefectVTerm                            // 2. forward.cu V-term bug
	fusedDefectRoPEPrev                         // 3. RoPE angle at pos-1
	fusedDefectHeadSwap                         // 4. swapped KV head mapping (h % kvHeads)
	fusedDefectRowNext                          // 5. KV cache written to row pos+1
	fusedDefectNoOResidual                      // 6a. missing O-projection residual
	fusedDefectNoDownResidual                   // 6b. missing Down-projection residual
	fusedDefectSwapGateUp                       // 7. swapped gate and up in SwiGLU
)

var allFusedDecodeDefects = []struct {
	defect fusedDecodeDefect
	name   string
}{
	{fusedDefectNoRsqrt, "missing rsqrt"},
	{fusedDefectVTerm, "v-term bug"},
	{fusedDefectRoPEPrev, "rope at pos-1"},
	{fusedDefectHeadSwap, "swapped kv head map"},
	{fusedDefectRowNext, "wrong cache row pos+1"},
	{fusedDefectNoOResidual, "missing o-proj residual"},
	{fusedDefectNoDownResidual, "missing down-proj residual"},
	{fusedDefectSwapGateUp, "swapped gate/up in swiglu"},
}

// fusedDecodeDefectiveOracle computes one decoder layer with a specific defect
// injected so CPU and GPU parity tests can verify that each defect is rejected.
func fusedDecodeDefectiveOracle(
	p FusedDecodeLayerPlan,
	b fusedDecodeLayerBuffers,
	cache attnCache,
	pos int,
	defect fusedDecodeDefect,
) fusedDecodeLayerOutputs {
	normalized := rmsNorm(b.x, b.attnNorm, p.epsilon)
	if defect == fusedDefectNoRsqrt {
		for i := range normalized {
			normalized[i] = b.x[i] * b.attnNorm[i]
		}
	}
	q := matmulVec(normalized, b.wq, p.hidden, p.qDim)
	k := matmulVec(normalized, b.wk, p.hidden, p.kvDim)
	v := matmulVec(normalized, b.wv, p.hidden, p.kvDim)
	if defect == fusedDefectVTerm {
		for o := range v {
			var sum float32
			for i := 0; i+3 < p.hidden; i += 4 {
				wk, wv := b.wk[o*p.hidden:], b.wv[o*p.hidden:]
				sum += normalized[i]*wv[i] + normalized[i+1]*wv[i+1] + normalized[i+2]*wk[i+3]*wv[i+3]
			}
			v[o] = sum
		}
	}
	addBias(q, b.qBias)
	addBias(k, b.kBias)
	addBias(v, b.vBias)

	ropePos := pos
	if defect == fusedDefectRoPEPrev {
		ropePos = pos - 1
	}
	applyRoPE(q, ropePos, p.qHeads, p.headDim, p.theta)
	applyRoPE(k, ropePos, p.kvHeads, p.headDim, p.theta)

	writeRow := pos
	if defect == fusedDefectRowNext {
		clear(cache.k[pos*p.kvDim : (pos+1)*p.kvDim])
		clear(cache.v[pos*p.kvDim : (pos+1)*p.kvDim])
		writeRow = pos + 1
	}
	if writeRow >= 0 && writeRow < p.maxSeq {
		copy(cache.k[writeRow*p.kvDim:(writeRow+1)*p.kvDim], k)
		copy(cache.v[writeRow*p.kvDim:(writeRow+1)*p.kvDim], v)
	}

	attnOut := make([]float32, p.qDim)
	group := p.qHeads / p.kvHeads
	invScale := float32(1.0 / math.Sqrt(float64(p.headDim)))
	scores := make([]float32, pos+1)
	for h := 0; h < p.qHeads; h++ {
		kvh := h / group
		if defect == fusedDefectHeadSwap {
			kvh = h % p.kvHeads
		}
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

	oProj := matmulVec(attnOut, b.wo, p.qDim, p.hidden)
	attnRes := make([]float32, p.hidden)
	for i := range attnRes {
		if defect == fusedDefectNoOResidual {
			attnRes[i] = oProj[i]
		} else {
			attnRes[i] = b.x[i] + oProj[i]
		}
	}

	ffnNormed := rmsNorm(attnRes, b.ffnNorm, p.epsilon)
	if defect == fusedDefectNoRsqrt {
		for i := range ffnNormed {
			ffnNormed[i] = attnRes[i] * b.ffnNorm[i]
		}
	}
	gate := matmulVec(ffnNormed, b.wGate, p.hidden, p.inter)
	up := matmulVec(ffnNormed, b.wUp, p.hidden, p.inter)
	hb := make([]float32, p.inter)
	for i := range hb {
		g, u := gate[i], up[i]
		if defect == fusedDefectSwapGateUp {
			g, u = u, g
		}
		hb[i] = (g / (1.0 + float32(math.Exp(float64(-g))))) * u
	}

	down := matmulVec(hb, b.wDown, p.inter, p.hidden)
	out := make([]float32, p.hidden)
	for i := range out {
		if defect == fusedDefectNoDownResidual {
			out[i] = down[i]
		} else {
			out[i] = attnRes[i] + down[i]
		}
	}
	return fusedDecodeLayerOutputs{
		Q:            q,
		AttnOut:      attnOut,
		AttnResidual: attnRes,
		HB:           hb,
		Out:          out,
	}
}

func TestFusedDecodeLayerPlanValidationAndAccounting(t *testing.T) {
	for _, tt := range []struct {
		name                                            string
		hidden, inter, qHeads, kvHeads, headDim, maxSeq int
		eps, theta                                      float32
	}{
		{"zero hidden", 0, 16, 2, 2, 4, 8, 1e-5, 1e4},
		{"negative inter", 8, -1, 2, 2, 4, 8, 1e-5, 1e4},
		{"zero qHeads", 8, 16, 0, 2, 4, 8, 1e-5, 1e4},
		{"zero kvHeads", 8, 16, 2, 0, 4, 8, 1e-5, 1e4},
		{"zero headDim", 8, 16, 2, 2, 0, 8, 1e-5, 1e4},
		{"zero maxSeq", 8, 16, 2, 2, 4, 0, 1e-5, 1e4},
		{"ungrouped heads", 8, 16, 3, 2, 4, 8, 1e-5, 1e4},
		{"odd headDim", 8, 16, 2, 2, 3, 8, 1e-5, 1e4},
		{"zero eps", 8, 16, 2, 2, 4, 8, 0, 1e4},
		{"nan eps", 8, 16, 2, 2, 4, 8, float32(math.NaN()), 1e4},
		{"inf eps", 8, 16, 2, 2, 4, 8, float32(math.Inf(1)), 1e4},
		{"zero theta", 8, 16, 2, 2, 4, 8, 1e-5, 0},
		{"neg theta", 8, 16, 2, 2, 4, 8, 1e-5, -100},
		{"nan theta", 8, 16, 2, 2, 4, 8, 1e-5, float32(math.NaN())},
		{"inf theta", 8, 16, 2, 2, 4, 8, 1e-5, float32(math.Inf(1))},
		{"attn matrix overflow", 1 << 32, 16, 1, 1, 1 << 32, 8, 1e-5, 1e4},
		{"ffn matrix overflow", 8, 1 << 60, 2, 2, 4, 8, 1e-5, 1e4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewFusedDecodeLayerPlan(tt.hidden, tt.inter, tt.qHeads, tt.kvHeads, tt.headDim, tt.maxSeq, tt.eps, tt.theta); err == nil {
				t.Fatal("accepted invalid plan")
			}
		})
	}

	p, err := NewFusedDecodeLayerPlan(64, 128, 4, 2, 16, 16, 1e-5, 1e4)
	if err != nil {
		t.Fatal(err)
	}
	if p.KernelsPerLayer() != 5 {
		t.Fatalf("KernelsPerLayer = %d, want 5", p.KernelsPerLayer())
	}
	if p.UnfusedKernelsPerLayer() != 15 {
		t.Fatalf("UnfusedKernelsPerLayer = %d, want 15", p.UnfusedKernelsPerLayer())
	}
	if p.EliminatedLaunchesPerLayer() != 10 {
		t.Fatalf("EliminatedLaunchesPerLayer = %d, want 10", p.EliminatedLaunchesPerLayer())
	}
	if p.EliminatedWritesPerLayer() != 6 {
		t.Fatalf("EliminatedWritesPerLayer = %d, want 6", p.EliminatedWritesPerLayer())
	}
	wantBytes := 4 * (4*64 + 2*32 + 2*128)
	if p.EliminatedWriteBytesPerLayer() != wantBytes {
		t.Fatalf("EliminatedWriteBytesPerLayer = %d, want %d", p.EliminatedWriteBytesPerLayer(), wantBytes)
	}
	// qPairs = 4 * 8 = 32, kvPairs = 2 * 8 = 16 -> grid = 48.
	if p.QKVRoPEStoreGrid() != 48 {
		t.Fatalf("QKVRoPEStoreGrid = %d, want 48", p.QKVRoPEStoreGrid())
	}
	wantStages := []string{
		"FusedRMSNormQKVRoPEStore",
		"CausalFlashAttentionDecode",
		"FusedOutProjResidual",
		"FusedRMSNormGateUpSwiGLU",
		"FusedDownProjResidual",
	}
	stages := p.Stages()
	if !reflect.DeepEqual(stages, wantStages) {
		t.Fatalf("Stages = %v, want %v", stages, wantStages)
	}
	stages[0] = "mutated"
	if !reflect.DeepEqual(p.Stages(), wantStages) {
		t.Fatal("Stages aliases internal storage")
	}

	// Validate buffer length checks on executeFusedDecodeLayerCPU.
	oddPlan, err := NewFusedDecodeLayerPlan(7, 11, 2, 1, 6, 9, 1e-5, 1e4)
	if err != nil {
		t.Fatal(err)
	}
	valid := func() (fusedDecodeLayerBuffers, attnCache) {
		return fusedDecodeLayerBuffers{
			x:        make([]float32, 7),
			attnNorm: make([]float32, 7),
			wq:       make([]float32, 12*7),
			wk:       make([]float32, 6*7),
			wv:       make([]float32, 6*7),
			qBias:    make([]float32, 12),
			kBias:    make([]float32, 6),
			vBias:    make([]float32, 6),
			wo:       make([]float32, 7*12),
			ffnNorm:  make([]float32, 7),
			wGate:    make([]float32, 11*7),
			wUp:      make([]float32, 11*7),
			wDown:    make([]float32, 7*11),
		}, attnCache{
			k: make([]float32, 9*6),
			v: make([]float32, 9*6),
		}
	}
	b, c := valid()
	if _, err := executeFusedDecodeLayerCPU(oddPlan, b, c, 0); err != nil {
		t.Fatalf("valid buffers refused: %v", err)
	}
	b.qBias, b.kBias, b.vBias = nil, nil, nil
	if _, err := executeFusedDecodeLayerCPU(oddPlan, b, c, 7); err != nil {
		t.Fatalf("nil biases refused: %v", err)
	}
	for _, field := range []string{
		"x", "attnNorm", "wq", "wk", "wv", "qBias", "kBias", "vBias",
		"wo", "ffnNorm", "wGate", "wUp", "wDown", "k", "v",
	} {
		for _, delta := range []int{-1, 1} {
			b, c := valid()
			target := map[string]*[]float32{
				"x": &b.x, "attnNorm": &b.attnNorm, "wq": &b.wq, "wk": &b.wk, "wv": &b.wv,
				"qBias": &b.qBias, "kBias": &b.kBias, "vBias": &b.vBias, "wo": &b.wo,
				"ffnNorm": &b.ffnNorm, "wGate": &b.wGate, "wUp": &b.wUp, "wDown": &b.wDown,
				"k": &c.k, "v": &c.v,
			}[field]
			*target = make([]float32, len(*target)+delta)
			if _, err := executeFusedDecodeLayerCPU(oddPlan, b, c, 0); err == nil {
				t.Fatalf("field %s delta %d accepted", field, delta)
			}
		}
	}
	for _, badPos := range []int{-1, 9} {
		b, c := valid()
		if _, err := executeFusedDecodeLayerCPU(oddPlan, b, c, badPos); err == nil {
			t.Fatalf("invalid pos %d accepted", badPos)
		}
	}
	b, c = valid()
	if _, err := executeFusedDecodeLayerCPU(FusedDecodeLayerPlan{}, b, c, 0); err == nil {
		t.Fatal("zero plan accepted")
	}
	corrupt := oddPlan
	corrupt.qPairs--
	if _, err := executeFusedDecodeLayerCPU(corrupt, b, c, 0); err == nil {
		t.Fatal("corrupt plan accepted")
	}
}

var fusedDecodeTestShapes = []struct {
	name                                    string
	hidden, inter, qHeads, kvHeads, headDim int
}{
	{"mha_gqa1_aligned", 8, 12, 2, 2, 4},
	{"gqa2_default", 64, 128, 4, 2, 16},
	{"gqa4_wide_query", 12, 16, 4, 1, 4},
	{"gqa2_odd_hidden_inter", 7, 11, 2, 1, 6},
	{"gqa2_odd_multi_kv", 13, 9, 4, 2, 4},
	{"gqa4_multi_kv", 16, 20, 8, 2, 2},
}

func makeFusedDecodeLayerFixture(
	p FusedDecodeLayerPlan,
	bias bool,
	bf16Round bool,
	pos int,
) (fusedDecodeLayerBuffers, attnCache) {
	maybeRound := func(v []float32) []float32 {
		if !bf16Round {
			return v
		}
		return widenBF16(roundBF16(v))
	}
	x := parityValues(p.hidden, 1+pos*3)
	for i := range x {
		if x[i] == 0 {
			x[i] = 0.125
		}
	}
	b := fusedDecodeLayerBuffers{
		x:        x,
		attnNorm: parityValues(p.hidden, 3),
		wq:       maybeRound(parityValues(p.qDim*p.hidden, 5)),
		wk:       maybeRound(parityValues(p.kvDim*p.hidden, 9)),
		wv:       maybeRound(parityValues(p.kvDim*p.hidden, 13)),
		wo:       maybeRound(parityValues(p.hidden*p.qDim, 17)),
		ffnNorm:  parityValues(p.hidden, 21),
		wGate:    maybeRound(parityValues(p.inter*p.hidden, 2)),
		wUp:      maybeRound(parityValues(p.inter*p.hidden, 6)),
		wDown:    maybeRound(parityValues(p.hidden*p.inter, 10)),
	}
	// Scale WQ and WK so attention logits across positions are well-separated.
	for i := range b.wq {
		b.wq[i] *= 2.0
	}
	for i := range b.wk {
		b.wk[i] *= 2.0
	}
	if bf16Round {
		b.wq = maybeRound(b.wq)
		b.wk = maybeRound(b.wk)
	}
	if bias {
		b.qBias = parityValues(p.qDim, 14)
		b.kBias = parityValues(p.kvDim, 18)
		b.vBias = parityValues(p.kvDim, 22)
	}

	sentinel := math.Float32frombits(0x7fc0beef)
	cache := attnCache{
		k: make([]float32, p.maxSeq*p.kvDim),
		v: make([]float32, p.maxSeq*p.kvDim),
	}
	for i := range cache.k {
		cache.k[i] = sentinel
		cache.v[i] = sentinel
	}
	// Populate prior positions [0, pos) with distinct keys and values so causal
	// attention at pos > 0 exercises the full history.
	for t := 0; t < pos; t++ {
		kt := parityValues(p.kvDim, 31+t*5)
		vt := parityValues(p.kvDim, 47+t*7)
		copy(cache.k[t*p.kvDim:(t+1)*p.kvDim], kt)
		copy(cache.v[t*p.kvDim:(t+1)*p.kvDim], vt)
	}
	return b, cache
}

func cloneFusedCache(c attnCache) attnCache {
	return attnCache{
		k: append([]float32(nil), c.k...),
		v: append([]float32(nil), c.v...),
	}
}

func TestFusedDecodeLayerCPUParityAndControls(t *testing.T) {
	const maxSeq = 9
	positions := []int{0, 1, 7}

	for _, shape := range fusedDecodeTestShapes {
		for _, bias := range []bool{false, true} {
			for _, bf16 := range []bool{false, true} {
				p, err := NewFusedDecodeLayerPlan(
					shape.hidden, shape.inter, shape.qHeads, shape.kvHeads, shape.headDim,
					maxSeq, 1e-5, 1e4,
				)
				if err != nil {
					t.Fatal(err)
				}
				for _, pos := range positions {
					b, baseCache := makeFusedDecodeLayerFixture(p, bias, bf16, pos)
					beforeB := b
					beforeB.x = append([]float32(nil), b.x...)

					gotCache := cloneFusedCache(baseCache)
					got, err := executeFusedDecodeLayerCPU(p, b, gotCache, pos)
					if err != nil {
						t.Fatalf("%s bias=%v bf16=%v pos=%d: %v", shape.name, bias, bf16, pos, err)
					}

					// Compare against attnPlan.reference + ffnPlan.reference.
					refCache := cloneFusedCache(baseCache)
					wantAttnRes, err := p.attn.reference(attnBuffers{
						x:     b.x,
						norm:  b.attnNorm,
						wq:    b.wq,
						wk:    b.wk,
						wv:    b.wv,
						wo:    b.wo,
						qBias: b.qBias,
						kBias: b.kBias,
						vBias: b.vBias,
					}, refCache, pos)
					if err != nil {
						t.Fatal(err)
					}
					wantOut, err := p.ffn.reference(ffnBuffers{
						x:    wantAttnRes,
						norm: b.ffnNorm,
						gate: b.wGate,
						up:   b.wUp,
						down: b.wDown,
					})
					if err != nil {
						t.Fatal(err)
					}

					if i := parityMismatch(got.AttnResidual, wantAttnRes, 1e-6); i >= 0 {
						t.Fatalf("%s bias=%v bf16=%v pos=%d attnRes[%d]: got %g want %g",
							shape.name, bias, bf16, pos, i, got.AttnResidual[i], wantAttnRes[i])
					}
					if i := parityMismatch(got.Out, wantOut, 1e-6); i >= 0 {
						t.Fatalf("%s bias=%v bf16=%v pos=%d out[%d]: got %g want %g",
							shape.name, bias, bf16, pos, i, got.Out[i], wantOut[i])
					}

					row := func(s []float32, r int) []float32 {
						return s[r*p.kvDim : (r+1)*p.kvDim]
					}
					if i := parityMismatch(row(gotCache.k, pos), row(refCache.k, pos), 1e-6); i >= 0 {
						t.Fatalf("%s pos=%d k_cache[%d]: got %g want %g",
							shape.name, pos, i, row(gotCache.k, pos)[i], row(refCache.k, pos)[i])
					}
					if i := parityMismatch(row(gotCache.v, pos), row(refCache.v, pos), 1e-6); i >= 0 {
						t.Fatalf("%s pos=%d v_cache[%d]: got %g want %g",
							shape.name, pos, i, row(gotCache.v, pos)[i], row(refCache.v, pos)[i])
					}
					for r := 0; r < maxSeq; r++ {
						if r == pos {
							continue
						}
						if !sameBits(row(gotCache.k, r), row(baseCache.k, r)) || !sameBits(row(gotCache.v, r), row(baseCache.v, r)) {
							t.Fatalf("%s pos=%d mutated cache row %d", shape.name, pos, r)
						}
					}
					if !reflect.DeepEqual(b.x, beforeB.x) {
						t.Fatalf("%s pos=%d mutated input x", shape.name, pos)
					}

					// Verify all 7 defect controls are detected and discriminate beyond kernelTolerance.
					for _, d := range allFusedDecodeDefects {
						if d.defect == fusedDefectHeadSwap && (shape.kvHeads == 1 || shape.kvHeads == shape.qHeads) {
							continue
						}
						badCache := cloneFusedCache(baseCache)
						bad := fusedDecodeDefectiveOracle(p, b, badCache, pos, d.defect)
						outDisc := discriminates(got.Out, bad.Out, kernelTolerance)
						attnDisc := discriminates(got.AttnResidual, bad.AttnResidual, kernelTolerance)
						hbDisc := discriminates(got.HB, bad.HB, kernelTolerance)
						qDisc := discriminates(got.Q, bad.Q, kernelTolerance)
						kDisc := discriminates(row(gotCache.k, pos), row(badCache.k, pos), kernelTolerance)
						vDisc := discriminates(row(gotCache.v, pos), row(badCache.v, pos), kernelTolerance)
						if !outDisc && !attnDisc && !hbDisc && !qDisc && !kDisc && !vDisc {
							t.Errorf("%s bias=%v bf16=%v pos=%d: defect %q not discriminated at kernelTolerance",
								shape.name, bias, bf16, pos, d.name)
						}
					}
				}
			}
		}
	}
}
