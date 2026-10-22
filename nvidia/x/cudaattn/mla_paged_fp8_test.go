package cudaattn_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestPadAttentionHeads(t *testing.T) {
	tests := []struct {
		numHeads int
		wantPad  int
		wantDiff int
		wantErr  bool
	}{
		{numHeads: 1, wantPad: 64, wantDiff: 63},
		{numHeads: 16, wantPad: 64, wantDiff: 48},
		{numHeads: 64, wantPad: 64, wantDiff: 0},
		{numHeads: 65, wantPad: 128, wantDiff: 63},
		{numHeads: 96, wantPad: 128, wantDiff: 32},
		{numHeads: 128, wantPad: 128, wantDiff: 0},
		{numHeads: 0, wantErr: true},
		{numHeads: -4, wantErr: true},
		{numHeads: 129, wantErr: true},
	}

	for _, tc := range tests {
		padded, padCount, err := cudaattn.PadAttentionHeads(tc.numHeads)
		if tc.wantErr {
			if !errors.Is(err, cudaerrors.ErrUnsupported) {
				t.Fatalf("PadAttentionHeads(%d) error = %v, want ErrUnsupported", tc.numHeads, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("PadAttentionHeads(%d) unexpected error: %v", tc.numHeads, err)
		}
		if padded != tc.wantPad || padCount != tc.wantDiff {
			t.Fatalf("PadAttentionHeads(%d) = (%d, %d), want (%d, %d)", tc.numHeads, padded, padCount, tc.wantPad, tc.wantDiff)
		}
	}
}

func TestPermuteQBProj(t *testing.T) {
	numHeads, qNopeDim, qRopeDim, inDim := 2, 3, 4, 2
	headStride := (qNopeDim + qRopeDim) * inDim
	weights := make([]float32, numHeads*headStride)
	for i := range weights {
		weights[i] = float32(i + 1)
	}

	got, err := cudaattn.PermuteQBProj(weights, numHeads, qNopeDim, qRopeDim, inDim)
	if err != nil {
		t.Fatalf("PermuteQBProj() unexpected error: %v", err)
	}
	if len(got) != len(weights) {
		t.Fatalf("len(PermuteQBProj) = %d, want %d", len(got), len(weights))
	}

	halfRope := qRopeDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headStride
		// NoPE prefix must be untouched.
		for i := 0; i < qNopeDim*inDim; i++ {
			if got[base+i] != weights[base+i] {
				t.Fatalf("head %d nope[%d] = %v, want %v", h, i, got[base+i], weights[base+i])
			}
		}
		// RoPE tail [qRopeDim/2, 2, inDim] -> [2, qRopeDim/2, inDim].
		ropeBase := base + qNopeDim*inDim
		for r := 0; r < halfRope; r++ {
			for p := 0; p < 2; p++ {
				for d := 0; d < inDim; d++ {
					want := weights[ropeBase+(r*2+p)*inDim+d]
					actual := got[ropeBase+(p*halfRope+r)*inDim+d]
					if actual != want {
						t.Fatalf("head %d rope(p=%d,r=%d,d=%d) = %v, want %v", h, p, r, d, actual, want)
					}
				}
			}
		}
	}

	// Negative controls: odd qRopeDim or mismatched slice length.
	if _, err := cudaattn.PermuteQBProj(weights, numHeads, qNopeDim, 3, inDim); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("odd qRopeDim error = %v, want ErrUnsupported", err)
	}
	if _, err := cudaattn.PermuteQBProj(weights[:4], numHeads, qNopeDim, qRopeDim, inDim); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("short weights error = %v, want ErrUnsupported", err)
	}
}

func TestPermuteWVProj(t *testing.T) {
	numHeads, kvLoraRank, vHeadDim, padHeads := 4, 3, 2, 64
	weights := make([]float32, numHeads*vHeadDim*kvLoraRank)
	for i := range weights {
		weights[i] = float32(i + 1)
	}

	got, err := cudaattn.PermuteWVProj(weights, numHeads, kvLoraRank, vHeadDim, padHeads)
	if err != nil {
		t.Fatalf("PermuteWVProj() unexpected error: %v", err)
	}
	if len(got) != padHeads*kvLoraRank*vHeadDim {
		t.Fatalf("len(PermuteWVProj) = %d, want %d", len(got), padHeads*kvLoraRank*vHeadDim)
	}

	for h := 0; h < numHeads; h++ {
		for v := 0; v < vHeadDim; v++ {
			for k := 0; k < kvLoraRank; k++ {
				want := weights[(h*vHeadDim+v)*kvLoraRank+k]
				actual := got[(h*kvLoraRank+k)*vHeadDim+v]
				if actual != want {
					t.Fatalf("head %d (k=%d,v=%d) = %v, want %v", h, k, v, actual, want)
				}
			}
		}
	}
	for i := numHeads * kvLoraRank * vHeadDim; i < len(got); i++ {
		if got[i] != 0 {
			t.Fatalf("padded region[%d] = %v, want 0", i, got[i])
		}
	}

	// Negative controls: invalid padHeads.
	if _, err := cudaattn.PermuteWVProj(weights, numHeads, kvLoraRank, vHeadDim, 32); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("padHeads=32 error = %v, want ErrUnsupported", err)
	}
}

func TestPagedMLAForwardCPU(t *testing.T) {
	cfg := cudaattn.PagedMLAConfig{
		NumHeads:        2,
		HeadDimCKV:      4,
		HeadDimKPE:      2,
		PageBlockSize:   2,
		SoftmaxScale:    0.5,
		HasAttnSink:     true,
		HasExtraKVCache: true,
		BatchInvariant:  true,
		EmitFP8Output:   true,
		ReturnLSE:       true,
	}

	qNope := []float32{
		1.0, 0.5, -0.5, 0.25,
		-0.5, 1.0, 0.25, 0.75,
	}
	qPe := []float32{
		0.5, -0.25,
		-0.25, 0.5,
	}
	// 2 main KV tokens.
	kvNoPE := []float32{
		0.8, -0.2, 0.4, 0.1,
		-0.4, 0.6, 0.2, 0.9,
	}
	kPE := []float32{
		0.3, -0.1,
		-0.2, 0.7,
	}
	attnSink := []float32{0.2, -0.3}
	// 1 extra KV token.
	extraKVNoPE := []float32{
		0.5, 0.5, -0.5, 0.5,
	}
	extraKPE := []float32{
		0.1, 0.2,
	}

	res, err := cudaattn.PagedMLAForwardCPU(qNope, qPe, kvNoPE, kPE, attnSink, extraKVNoPE, extraKPE, nil, cfg)
	if err != nil {
		t.Fatalf("PagedMLAForwardCPU() unexpected error: %v", err)
	}
	if len(res.Output) != cfg.NumHeads*cfg.HeadDimCKV || len(res.LSE) != cfg.NumHeads {
		t.Fatalf("unexpected output/LSE lengths: out=%d lse=%d", len(res.Output), len(res.LSE))
	}
	if len(res.FP8Output) != len(res.Output) || len(res.FP8Scales) == 0 {
		t.Fatalf("expected non-empty FP8Output and FP8Scales, got %d and %d", len(res.FP8Output), len(res.FP8Scales))
	}

	// Independent float64 oracle combining main KV, sink, and extra KV token.
	allKVNoPE := append(append([]float32(nil), kvNoPE...), extraKVNoPE...)
	allKPE := append(append([]float32(nil), kPE...), extraKPE...)
	totalKV := 3
	for h := 0; h < cfg.NumHeads; h++ {
		logits := make([]float64, totalKV)
		maxVal := float64(attnSink[h])
		for j := 0; j < totalKV; j++ {
			var dot float64
			for d := 0; d < cfg.HeadDimCKV; d++ {
				dot += float64(qNope[h*cfg.HeadDimCKV+d]) * float64(allKVNoPE[j*cfg.HeadDimCKV+d])
			}
			for d := 0; d < cfg.HeadDimKPE; d++ {
				dot += float64(qPe[h*cfg.HeadDimKPE+d]) * float64(allKPE[j*cfg.HeadDimKPE+d])
			}
			logits[j] = dot * float64(cfg.SoftmaxScale)
			if logits[j] > maxVal {
				maxVal = logits[j]
			}
		}
		denom := math.Exp(float64(attnSink[h]) - maxVal)
		wantOut := make([]float64, cfg.HeadDimCKV)
		for j := 0; j < totalKV; j++ {
			w := math.Exp(logits[j] - maxVal)
			denom += w
			for d := 0; d < cfg.HeadDimCKV; d++ {
				wantOut[d] += w * float64(allKVNoPE[j*cfg.HeadDimCKV+d])
			}
		}
		wantLSE := maxVal + math.Log(denom)
		if math.Abs(float64(res.LSE[h])-wantLSE) > 1e-5 {
			t.Fatalf("head %d LSE = %v, want %v", h, res.LSE[h], wantLSE)
		}
		for d := 0; d < cfg.HeadDimCKV; d++ {
			want := wantOut[d] / denom
			got := float64(res.Output[h*cfg.HeadDimCKV+d])
			if math.Abs(got-want) > 1e-5 {
				t.Fatalf("head %d out[%d] = %v, want %v", h, d, got, want)
			}
		}
	}

	// Negative control: invalid NumHeads.
	badCfg := cfg
	badCfg.NumHeads = 0
	if _, err := cudaattn.PagedMLAForwardCPU(qNope, qPe, kvNoPE, kPE, attnSink, nil, nil, nil, badCfg); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("invalid NumHeads error = %v, want ErrUnsupported", err)
	}
}

func TestFusedNormRoPEHadamardQuantStoreKV(t *testing.T) {
	cfg := cudaattn.NormRoPEHadamardKVConfig{
		Tokens:        2,
		NopeDim:       8,
		RopeDim:       8,
		HadamardBlock: 8,
		Eps:           1e-6,
		StoreFP4:      false,
	}
	ckv := make([]float32, cfg.Tokens*cfg.NopeDim)
	kpe := make([]float32, cfg.Tokens*cfg.RopeDim)
	rmsWeight := make([]float32, cfg.NopeDim)
	cosSin := make([]float32, cfg.Tokens*cfg.RopeDim)

	for i := range ckv {
		ckv[i] = float32(i+1) * 0.25
		kpe[i] = float32(i+1) * 0.125
	}
	for i := range rmsWeight {
		rmsWeight[i] = 1.0 + float32(i)*0.05
	}
	for tIdx := 0; tIdx < cfg.Tokens; tIdx++ {
		for i := 0; i < cfg.RopeDim; i += 2 {
			angle := float64(tIdx+1) * float64(i/2+1) * 0.1
			cosSin[tIdx*cfg.RopeDim+i] = float32(math.Cos(angle))
			cosSin[tIdx*cfg.RopeDim+i+1] = float32(math.Sin(angle))
		}
	}

	quantNoPE, scales, rotKPE, err := cudaattn.FusedNormRoPEHadamardQuantStoreKVCPU(ckv, kpe, rmsWeight, cosSin, cfg)
	if err != nil {
		t.Fatalf("FusedNormRoPEHadamardQuantStoreKVCPU() unexpected error: %v", err)
	}
	if len(quantNoPE) != cfg.Tokens*cfg.NopeDim || len(scales) != cfg.Tokens*(cfg.NopeDim/cfg.HadamardBlock) || len(rotKPE) != cfg.Tokens*cfg.RopeDim {
		t.Fatalf("unexpected output lengths: quant=%d scales=%d rotKPE=%d", len(quantNoPE), len(scales), len(rotKPE))
	}

	// Independent Sylvester Hadamard matrix oracle H_8 / sqrt(8) where H_{i,j} = (-1)^popcount(i & j) / sqrt(8).
	invSqrt8 := 1.0 / math.Sqrt(8.0)
	popcount := func(x int) int {
		c := 0
		for x > 0 {
			c += x & 1
			x >>= 1
		}
		return c
	}
	for tIdx := 0; tIdx < cfg.Tokens; tIdx++ {
		// Check rotKPE against explicit RoPE + dense Sylvester Hadamard matrix multiplication.
		roped := make([]float64, 8)
		for i := 0; i < 8; i += 2 {
			x0 := float64(kpe[tIdx*8+i])
			x1 := float64(kpe[tIdx*8+i+1])
			c := float64(cosSin[tIdx*8+i])
			s := float64(cosSin[tIdx*8+i+1])
			roped[i] = x0*c - x1*s
			roped[i+1] = x1*c + x0*s
		}
		for row := 0; row < 8; row++ {
			var want float64
			for col := 0; col < 8; col++ {
				sign := 1.0
				if popcount(row&col)%2 == 1 {
					sign = -1.0
				}
				want += sign * roped[col] * invSqrt8
			}
			got := float64(rotKPE[tIdx*8+row])
			if math.Abs(got-want) > 1e-5 {
				t.Fatalf("token %d rotKPE[%d] = %v, want %v", tIdx, row, got, want)
			}
		}
	}

	// Also verify FP4 mode packs 2 elements per byte.
	cfgFP4 := cfg
	cfgFP4.StoreFP4 = true
	quantFP4, scalesFP4, _, err := cudaattn.FusedNormRoPEHadamardQuantStoreKVCPU(ckv, kpe, rmsWeight, cosSin, cfgFP4)
	if err != nil {
		t.Fatalf("FusedNormRoPEHadamardQuantStoreKVCPU(StoreFP4) unexpected error: %v", err)
	}
	if len(quantFP4) != (cfg.Tokens*cfg.NopeDim)/2 || len(scalesFP4) != len(scales) {
		t.Fatalf("unexpected FP4 output lengths: quant=%d scales=%d", len(quantFP4), len(scalesFP4))
	}

	// Verify CUDA C++ source emission.
	src, err := cudaattn.FusedNormRoPEHadamardCastStoreKVSource(cfg)
	if err != nil {
		t.Fatalf("FusedNormRoPEHadamardCastStoreKVSource() unexpected error: %v", err)
	}
	if !strings.Contains(src, "fused_norm_rope_hadamard_cast_store_kv_kernel") || !strings.Contains(src, "__shfl_xor_sync") {
		t.Fatalf("generated source missing expected kernel or butterfly shuffle:\n%s", src)
	}

	// Negative control: non-power-of-two HadamardBlock.
	badCfg := cfg
	badCfg.HadamardBlock = 6
	if _, err := cudaattn.FusedNormRoPEHadamardCastStoreKVSource(badCfg); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("non-power-of-two HadamardBlock error = %v, want ErrUnsupported", err)
	}
}
