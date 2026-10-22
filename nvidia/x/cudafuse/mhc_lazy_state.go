// SPDX-License-Identifier: Apache-2.0

package cudafuse

import "fmt"

// HCLazyState manages deferred residual post-mixing across transformer layers.
// Instead of materializing hc_post immediately after each sub-layer, HCLazyState
// records the pending layer output and mixing weights and fuses the post-mix
// update into the following sub-layer's pre-mix contraction via [HCLazyState.FusedStepIntoNext].
type HCLazyState struct {
	Residuals   []float32
	PostMix     []float32
	ResMix      []float32
	LayerOutput []float32
	NumTokens   int
	NumStreams  int
	HiddenDim   int
	Pending     bool
}

// NewHCLazyState constructs an HCLazyState owning a copy of residuals with shape
// [numTokens, numStreams, hiddenDim].
func NewHCLazyState(residuals []float32, numTokens, numStreams, hiddenDim int) (*HCLazyState, error) {
	if numTokens <= 0 || numStreams <= 0 || hiddenDim <= 0 {
		return nil, fmt.Errorf("cudafuse: invalid lazy state dimensions numTokens=%d numStreams=%d hiddenDim=%d", numTokens, numStreams, hiddenDim)
	}
	want := numTokens * numStreams * hiddenDim
	if len(residuals) != want {
		return nil, fmt.Errorf("cudafuse: residuals length %d, want %d", len(residuals), want)
	}
	if err := mhcValidateValues(residuals); err != nil {
		return nil, err
	}
	resCopy := make([]float32, len(residuals))
	copy(resCopy, residuals)
	return &HCLazyState{
		Residuals:  resCopy,
		NumTokens:  numTokens,
		NumStreams: numStreams,
		HiddenDim:  hiddenDim,
		Pending:    false,
	}, nil
}

// DeferPost records a sub-layer's output [NumTokens, HiddenDim], post_mix
// [NumTokens, NumStreams], and res_mix [NumTokens, NumStreams*NumStreams]
// without materializing hc_post into Residuals.
func (s *HCLazyState) DeferPost(layerOutput, postMix, resMix []float32) error {
	if s == nil {
		return fmt.Errorf("cudafuse: nil HCLazyState")
	}
	if s.Pending {
		return fmt.Errorf("cudafuse: pending hc_post must be materialized or stepped before DeferPost")
	}
	T, S, D := s.NumTokens, s.NumStreams, s.HiddenDim
	if len(layerOutput) != T*D {
		return fmt.Errorf("cudafuse: layerOutput length %d, want %d", len(layerOutput), T*D)
	}
	if len(postMix) != T*S {
		return fmt.Errorf("cudafuse: postMix length %d, want %d", len(postMix), T*S)
	}
	if len(resMix) != T*S*S {
		return fmt.Errorf("cudafuse: resMix length %d, want %d", len(resMix), T*S*S)
	}
	for _, slice := range [][]float32{layerOutput, postMix, resMix} {
		if err := mhcValidateValues(slice); err != nil {
			return err
		}
	}
	s.LayerOutput = append([]float32(nil), layerOutput...)
	s.PostMix = append([]float32(nil), postMix...)
	s.ResMix = append([]float32(nil), resMix...)
	s.Pending = true
	return nil
}

// Materialize flushes any pending hc_post update into s.Residuals, clears the
// pending buffers, and returns a copy of the updated [NumTokens, NumStreams, HiddenDim]
// residual trunk.
func (s *HCLazyState) Materialize() ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("cudafuse: nil HCLazyState")
	}
	if s.Pending {
		T, S, D := s.NumTokens, s.NumStreams, s.HiddenDim
		hOut := make([]float32, T*S*D)
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				yVal := float64(s.LayerOutput[t*D+d])
				for st := 0; st < S; st++ {
					var hAcc float64
					for j := 0; j < S; j++ {
						rMix := float64(s.ResMix[t*S*S+st*S+j])
						hVal := float64(s.Residuals[(t*S+j)*D+d])
						hAcc += rMix * hVal
					}
					hAcc += float64(s.PostMix[t*S+st]) * yVal
					hOut[(t*S+st)*D+d] = float32(hAcc)
				}
			}
		}
		s.Residuals = hOut
		s.LayerOutput = nil
		s.PostMix = nil
		s.ResMix = nil
		s.Pending = false
	}
	out := make([]float32, len(s.Residuals))
	copy(out, s.Residuals)
	return out, nil
}

// FusedStepIntoNext computes the next sub-layer's input xInNext [NumTokens, HiddenDim]
// from preMixNext [NumTokens, NumStreams]. When s.Pending is true, it fuses the
// deferred hc_post update of s.Residuals with the hc_pre contraction via
// [ExecuteMHCPrePostCPU]. When s.Pending is false (such as Layer 0 hc_shifted),
// it contracts s.Residuals with preMixNext directly.
func (s *HCLazyState) FusedStepIntoNext(preMixNext []float32) (xInNext []float32, err error) {
	if s == nil {
		return nil, fmt.Errorf("cudafuse: nil HCLazyState")
	}
	T, S, D := s.NumTokens, s.NumStreams, s.HiddenDim
	if len(preMixNext) != T*S {
		return nil, fmt.Errorf("cudafuse: preMixNext length %d, want %d", len(preMixNext), T*S)
	}
	if err := mhcValidateValues(preMixNext); err != nil {
		return nil, err
	}

	if s.Pending {
		xInNext, hOut, err := ExecuteMHCPrePostCPU(s.LayerOutput, s.Residuals, s.PostMix, s.ResMix, preMixNext, T, S, D)
		if err != nil {
			return nil, err
		}
		s.Residuals = hOut
		s.LayerOutput = nil
		s.PostMix = nil
		s.ResMix = nil
		s.Pending = false
		return xInNext, nil
	}

	xInNext = make([]float32, T*D)
	for t := 0; t < T; t++ {
		for d := 0; d < D; d++ {
			var acc float64
			for st := 0; st < S; st++ {
				acc += float64(preMixNext[t*S+st]) * float64(s.Residuals[(t*S+st)*D+d])
			}
			xInNext[t*D+d] = float32(acc)
		}
	}
	return xInNext, nil
}

// CompactActiveTokens gathers the token rows specified by keepIndices from
// s.Residuals (and from pending PostMix, ResMix, and LayerOutput when s.Pending
// is true), returning a new compacted HCLazyState for YoCo / ZOS token compaction.
func (s *HCLazyState) CompactActiveTokens(keepIndices []int) (*HCLazyState, error) {
	if s == nil {
		return nil, fmt.Errorf("cudafuse: nil HCLazyState")
	}
	if len(keepIndices) == 0 {
		return nil, fmt.Errorf("cudafuse: keepIndices must not be empty")
	}
	for i, idx := range keepIndices {
		if idx < 0 || idx >= s.NumTokens {
			return nil, fmt.Errorf("cudafuse: keepIndices[%d] = %d out of range [0, %d)", i, idx, s.NumTokens)
		}
	}

	newT := len(keepIndices)
	S, D := s.NumStreams, s.HiddenDim
	resStride := S * D
	newRes := make([]float32, newT*resStride)
	for i, srcTok := range keepIndices {
		copy(newRes[i*resStride:(i+1)*resStride], s.Residuals[srcTok*resStride:(srcTok+1)*resStride])
	}

	compacted := &HCLazyState{
		Residuals:  newRes,
		NumTokens:  newT,
		NumStreams: S,
		HiddenDim:  D,
		Pending:    s.Pending,
	}

	if s.Pending {
		compacted.LayerOutput = make([]float32, newT*D)
		compacted.PostMix = make([]float32, newT*S)
		compacted.ResMix = make([]float32, newT*S*S)
		for i, srcTok := range keepIndices {
			copy(compacted.LayerOutput[i*D:(i+1)*D], s.LayerOutput[srcTok*D:(srcTok+1)*D])
			copy(compacted.PostMix[i*S:(i+1)*S], s.PostMix[srcTok*S:(srcTok+1)*S])
			copy(compacted.ResMix[i*S*S:(i+1)*S*S], s.ResMix[srcTok*S*S:(srcTok+1)*S*S])
		}
	}
	return compacted, nil
}
