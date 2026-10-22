// SPDX-License-Identifier: Apache-2.0

package cudafuse

import (
	"math"
	"testing"
)

func TestHCLazyStateDeferredVsFused(t *testing.T) {
	numTokens, numStreams, hiddenDim := 4, 4, 8
	initRes := make([]float32, numTokens*numStreams*hiddenDim)
	for i := range initRes {
		initRes[i] = float32((i%9)-4) * 0.25
	}

	stateEager, err := NewHCLazyState(initRes, numTokens, numStreams, hiddenDim)
	if err != nil {
		t.Fatalf("NewHCLazyState(eager) failed: %v", err)
	}
	stateFused, err := NewHCLazyState(initRes, numTokens, numStreams, hiddenDim)
	if err != nil {
		t.Fatalf("NewHCLazyState(fused) failed: %v", err)
	}

	// Step 0: Pending == false (Layer 0 hc_shifted).
	preMix0, err := IdentityPreMix(numTokens, numStreams, 0)
	if err != nil {
		t.Fatal(err)
	}
	x0Eager, err := stateEager.FusedStepIntoNext(preMix0)
	if err != nil {
		t.Fatal(err)
	}
	x0Fused, err := stateFused.FusedStepIntoNext(preMix0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x0Eager {
		if x0Eager[i] != x0Fused[i] {
			t.Fatalf("step 0 idx %d: eager=%v, fused=%v", i, x0Eager[i], x0Fused[i])
		}
	}

	// Sub-layer 0 produces layerOut, postMix, resMix.
	layerOut := make([]float32, numTokens*hiddenDim)
	postMix := make([]float32, numTokens*numStreams)
	resMix := make([]float32, numTokens*numStreams*numStreams)
	preMix1 := make([]float32, numTokens*numStreams)
	for i := range layerOut {
		layerOut[i] = float32((i%7)-3) * 0.3
	}
	for i := range postMix {
		postMix[i] = 0.4 + float32(i%4)*0.2
	}
	for i := range resMix {
		resMix[i] = 0.25
	}
	for i := range preMix1 {
		preMix1[i] = 0.15 * float32((i%4)+1)
	}

	if err := stateEager.DeferPost(layerOut, postMix, resMix); err != nil {
		t.Fatal(err)
	}
	if err := stateFused.DeferPost(layerOut, postMix, resMix); err != nil {
		t.Fatal(err)
	}
	// Duplicate DeferPost while Pending must fail.
	if err := stateFused.DeferPost(layerOut, postMix, resMix); err == nil {
		t.Fatal("duplicate DeferPost while Pending succeeded")
	}

	// Eager path: Materialize() first, then FusedStepIntoNext(preMix1).
	matEager, err := stateEager.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	if stateEager.Pending {
		t.Fatal("stateEager.Pending remained true after Materialize")
	}
	x1Eager, err := stateEager.FusedStepIntoNext(preMix1)
	if err != nil {
		t.Fatal(err)
	}

	// Fused path: FusedStepIntoNext(preMix1) directly while Pending == true.
	x1Fused, err := stateFused.FusedStepIntoNext(preMix1)
	if err != nil {
		t.Fatal(err)
	}
	if stateFused.Pending {
		t.Fatal("stateFused.Pending remained true after FusedStepIntoNext")
	}
	matFused, err := stateFused.Materialize()
	if err != nil {
		t.Fatal(err)
	}

	for i := range matEager {
		if math.Abs(float64(matEager[i]-matFused[i])) > 1e-6 {
			t.Fatalf("residuals[%d]: eager=%v, fused=%v", i, matEager[i], matFused[i])
		}
	}
	for i := range x1Eager {
		if math.Abs(float64(x1Eager[i]-x1Fused[i])) > 1e-6 {
			t.Fatalf("x1[%d]: eager=%v, fused=%v", i, x1Eager[i], x1Fused[i])
		}
	}
}

func TestHCLazyStateCompactActiveTokens(t *testing.T) {
	numTokens, numStreams, hiddenDim := 5, 4, 6
	initRes := make([]float32, numTokens*numStreams*hiddenDim)
	for i := range initRes {
		initRes[i] = float32(i+1) * 0.05
	}
	state, err := NewHCLazyState(initRes, numTokens, numStreams, hiddenDim)
	if err != nil {
		t.Fatal(err)
	}

	layerOut := make([]float32, numTokens*hiddenDim)
	postMix := make([]float32, numTokens*numStreams)
	resMix := make([]float32, numTokens*numStreams*numStreams)
	for i := range layerOut {
		layerOut[i] = float32(i+1) * 0.1
	}
	for i := range postMix {
		postMix[i] = 0.5
	}
	for i := range resMix {
		resMix[i] = 0.25
	}
	if err := state.DeferPost(layerOut, postMix, resMix); err != nil {
		t.Fatal(err)
	}

	keep := []int{1, 3, 4}
	compacted, err := state.CompactActiveTokens(keep)
	if err != nil {
		t.Fatalf("CompactActiveTokens failed: %v", err)
	}
	if compacted.NumTokens != len(keep) || !compacted.Pending {
		t.Fatalf("compacted NumTokens=%d Pending=%v, want %d and true", compacted.NumTokens, compacted.Pending, len(keep))
	}

	// Materialize full state, then compare kept token slices against Materialize on compacted state.
	fullMat, err := state.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	compMat, err := compacted.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	stride := numStreams * hiddenDim
	for i, srcTok := range keep {
		for j := 0; j < stride; j++ {
			got := compMat[i*stride+j]
			want := fullMat[srcTok*stride+j]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("keep[%d]=%d elem=%d: got=%v, want=%v", i, srcTok, j, got, want)
			}
		}
	}

	// Error cases.
	if _, err := state.CompactActiveTokens(nil); err == nil {
		t.Fatal("CompactActiveTokens(nil) succeeded")
	}
	if _, err := state.CompactActiveTokens([]int{-1}); err == nil {
		t.Fatal("CompactActiveTokens([-1]) succeeded")
	}
	if _, err := state.CompactActiveTokens([]int{numTokens}); err == nil {
		t.Fatal("CompactActiveTokens([numTokens]) succeeded")
	}
}
