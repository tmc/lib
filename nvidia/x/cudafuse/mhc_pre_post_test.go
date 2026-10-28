// SPDX-License-Identifier: Apache-2.0

package cudafuse

import (
	"math"
	"strings"
	"testing"
)

func TestPreSplitMixesCPU(t *testing.T) {
	cfg := MHCPrePostConfig{
		NumTokens:     3,
		NumStreams:    4,
		HiddenDim:     8,
		Eps:           1e-6,
		SinkhornIters: 20,
		SinkhornEps:   1e-6,
	}
	totalDim := cfg.NumStreams * cfg.HiddenDim
	mixDim := 2*cfg.NumStreams + cfg.NumStreams*cfg.NumStreams

	x := make([]float32, cfg.NumTokens*totalDim)
	for i := range x {
		x[i] = float32((i%9)-4) * 0.25
	}
	rmsWeight := make([]float32, totalDim)
	for i := range rmsWeight {
		rmsWeight[i] = 1.0 + float32(i%5)*0.1
	}
	phi := make([]float32, totalDim*mixDim)
	for i := range phi {
		phi[i] = float32((i%13)-6) * 0.05
	}
	bias := make([]float32, mixDim)
	for i := range bias {
		bias[i] = float32((i%7)-3) * 0.1
	}

	res, err := PreSplitMixesCPU(x, rmsWeight, phi, bias, 0.8, 1.2, 0.5, cfg)
	if err != nil {
		t.Fatalf("PreSplitMixesCPU failed: %v", err)
	}
	S := cfg.NumStreams
	if len(res.PreMix) != cfg.NumTokens*S {
		t.Fatalf("len(PreMix) = %d, want %d", len(res.PreMix), cfg.NumTokens*S)
	}
	if len(res.PostMix) != cfg.NumTokens*S {
		t.Fatalf("len(PostMix) = %d, want %d", len(res.PostMix), cfg.NumTokens*S)
	}
	if len(res.ResMix) != cfg.NumTokens*S*S {
		t.Fatalf("len(ResMix) = %d, want %d", len(res.ResMix), cfg.NumTokens*S*S)
	}

	// Verify PreMix in (0, 1), PostMix in (0, 2), and ResMix doubly stochastic (row & col sums ~ 1).
	for tok := 0; tok < cfg.NumTokens; tok++ {
		for s := 0; s < S; s++ {
			pre := res.PreMix[tok*S+s]
			post := res.PostMix[tok*S+s]
			if pre <= 0 || pre >= 1 {
				t.Fatalf("tok=%d s=%d: PreMix=%v out of (0, 1)", tok, s, pre)
			}
			if post <= 0 || post >= 2 {
				t.Fatalf("tok=%d s=%d: PostMix=%v out of (0, 2)", tok, s, post)
			}
		}
		for r := 0; r < S; r++ {
			var rowSum float64
			for c := 0; c < S; c++ {
				v := res.ResMix[tok*S*S+r*S+c]
				if v <= 0 {
					t.Fatalf("tok=%d r=%d c=%d: ResMix=%v <= 0", tok, r, c, v)
				}
				rowSum += float64(v)
			}
			if math.Abs(rowSum-1.0) > 1e-4 {
				t.Fatalf("tok=%d row=%d sum=%v, want ~1.0", tok, r, rowSum)
			}
		}
		for c := 0; c < S; c++ {
			var colSum float64
			for r := 0; r < S; r++ {
				colSum += float64(res.ResMix[tok*S*S+r*S+c])
			}
			if math.Abs(colSum-1.0) > 1e-4 {
				t.Fatalf("tok=%d col=%d sum=%v, want ~1.0", tok, c, colSum)
			}
		}
	}

	// Verify when alphaPre = alphaPost = 0 and bias = 0: PreMix == 0.5 and PostMix == 1.0 exactly.
	zeroBiasRes, err := PreSplitMixesCPU(x, nil, phi, nil, 0, 0, 0, cfg)
	if err != nil {
		t.Fatalf("PreSplitMixesCPU(zero alphas) failed: %v", err)
	}
	for i, v := range zeroBiasRes.PreMix {
		if v != 0.5 {
			t.Fatalf("zeroBiasRes.PreMix[%d] = %v, want 0.5", i, v)
		}
	}
	for i, v := range zeroBiasRes.PostMix {
		if v != 1.0 {
			t.Fatalf("zeroBiasRes.PostMix[%d] = %v, want 1.0", i, v)
		}
	}

	// Error cases.
	if _, err := PreSplitMixesCPU(x, rmsWeight, phi, bias, 1, 1, 1, MHCPrePostConfig{NumTokens: 0, HiddenDim: 8}); err == nil {
		t.Fatal("expected error for NumTokens=0")
	}
	if _, err := PreSplitMixesCPU(x[:len(x)-1], rmsWeight, phi, bias, 1, 1, 1, cfg); err == nil {
		t.Fatal("expected error for short x")
	}
}

func TestIdentityPreMix(t *testing.T) {
	mix, err := IdentityPreMix(3, 4, 0)
	if err != nil {
		t.Fatalf("IdentityPreMix failed: %v", err)
	}
	if len(mix) != 12 {
		t.Fatalf("len(mix) = %d, want 12", len(mix))
	}
	for i, v := range mix {
		if v != 0.25 {
			t.Fatalf("mix[%d] = %v, want 0.25", i, v)
		}
	}

	mixCustom, err := IdentityPreMix(2, 4, 1.0)
	if err != nil {
		t.Fatalf("IdentityPreMix(scale=1.0) failed: %v", err)
	}
	for i, v := range mixCustom {
		if v != 1.0 {
			t.Fatalf("mixCustom[%d] = %v, want 1.0", i, v)
		}
	}

	if _, err := IdentityPreMix(0, 4, 0.25); err == nil {
		t.Fatal("IdentityPreMix(0, 4) succeeded")
	}
	if _, err := IdentityPreMix(2, 4, -0.5); err == nil {
		t.Fatal("IdentityPreMix(scale=-0.5) succeeded")
	}
}

func TestExecuteMHCPrePostCPU(t *testing.T) {
	numTokens, numStreams, hiddenDim := 3, 4, 16
	yPrev := make([]float32, numTokens*hiddenDim)
	hPrev := make([]float32, numTokens*numStreams*hiddenDim)
	postMix := make([]float32, numTokens*numStreams)
	resMix := make([]float32, numTokens*numStreams*numStreams)
	preMixNext := make([]float32, numTokens*numStreams)

	for i := range yPrev {
		yPrev[i] = float32((i%11)-5) * 0.2
	}
	for i := range hPrev {
		hPrev[i] = float32((i%13)-6) * 0.15
	}
	for i := range postMix {
		postMix[i] = 0.5 + float32(i%4)*0.25
	}
	for i := range resMix {
		resMix[i] = 0.25
	}
	for i := range preMixNext {
		preMixNext[i] = 0.1 * float32((i%4)+1)
	}

	xInNext, hOut, err := ExecuteMHCPrePostCPU(yPrev, hPrev, postMix, resMix, preMixNext, numTokens, numStreams, hiddenDim)
	if err != nil {
		t.Fatalf("ExecuteMHCPrePostCPU failed: %v", err)
	}

	// Independent unfused two-stage oracle: first compute hOut, then contract xInNext.
	wantHOut := make([]float32, len(hPrev))
	for tok := 0; tok < numTokens; tok++ {
		for s := 0; s < numStreams; s++ {
			for d := 0; d < hiddenDim; d++ {
				var sum float64
				for j := 0; j < numStreams; j++ {
					sum += float64(resMix[tok*numStreams*numStreams+s*numStreams+j]) * float64(hPrev[(tok*numStreams+j)*hiddenDim+d])
				}
				sum += float64(postMix[tok*numStreams+s]) * float64(yPrev[tok*hiddenDim+d])
				wantHOut[(tok*numStreams+s)*hiddenDim+d] = float32(sum)
			}
		}
	}
	wantXInNext := make([]float32, len(yPrev))
	for tok := 0; tok < numTokens; tok++ {
		for d := 0; d < hiddenDim; d++ {
			var sum float64
			for s := 0; s < numStreams; s++ {
				sum += float64(preMixNext[tok*numStreams+s]) * float64(wantHOut[(tok*numStreams+s)*hiddenDim+d])
			}
			wantXInNext[tok*hiddenDim+d] = float32(sum)
		}
	}

	for i, want := range wantHOut {
		if math.Abs(float64(hOut[i]-want)) > 1e-6 {
			t.Fatalf("hOut[%d] = %v, want %v", i, hOut[i], want)
		}
	}
	for i, want := range wantXInNext {
		if math.Abs(float64(xInNext[i]-want)) > 1e-6 {
			t.Fatalf("xInNext[%d] = %v, want %v", i, xInNext[i], want)
		}
	}

	if _, _, err := ExecuteMHCPrePostCPU(yPrev[:1], hPrev, postMix, resMix, preMixNext, numTokens, numStreams, hiddenDim); err == nil {
		t.Fatal("expected error for short yPrev")
	}
}

func TestMHCPrePostFusedSource(t *testing.T) {
	src, err := MHCPrePostFusedSource(MHCPrePostConfig{
		NumTokens:  4,
		NumStreams: 4,
		HiddenDim:  64,
	})
	if err != nil {
		t.Fatalf("MHCPrePostFusedSource failed: %v", err)
	}
	for _, needle := range []string{
		"extern \"C\" __global__ void _hc_PreSplitMixes_kernel",
		"extern \"C\" __global__ void _hc_fused_post_pre_kernel",
		"#define MHC_NUM_STREAMS 4",
		"#define MHC_HIDDEN_DIM 64",
		"#define MHC_SINKHORN_ITERS 20",
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("generated source missing %q", needle)
		}
	}

	if _, err := MHCPrePostFusedSource(MHCPrePostConfig{NumTokens: 0, HiddenDim: 64}); err == nil {
		t.Fatal("expected error for invalid MHCPrePostConfig")
	}
}
