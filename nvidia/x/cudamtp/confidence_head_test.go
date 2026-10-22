// SPDX-License-Identifier: Apache-2.0

package cudamtp

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestConfidenceReferenceCPU(t *testing.T) {
	spec := ConfidenceHeadSpec{
		HiddenDim:        4,
		MarkovVocabSize:  8,
		StepTemperatures: []float32{0.5, 1.0, 2.0},
		Eps:              1e-6,
	}

	// 2 sequences, HiddenDim = 4.
	hidden := []float32{
		1.0, 2.0, -1.0, 2.0,
		0.5, -0.5, 0.5, -0.5,
	}
	rmsW := []float32{1.0, 0.5, 1.5, 1.0}
	// 3 steps * 4 dims.
	wProj := []float32{
		0.4, 0.2, -0.2, 0.1, // step 0
		0.1, 0.3, 0.1, -0.1, // step 1
		-0.2, 0.1, 0.4, 0.2, // step 2
	}
	markov := make([]float32, 8*8)
	// Seq 0 tokens: [10, 3, 5, 12] -> mod 8: (2->3), (3->5), (5->4)
	markov[2*8+3] = 0.25
	markov[3*8+5] = -0.15
	markov[5*8+4] = 0.40
	// Seq 1 tokens: [1, 2, 3, 4] -> mod 8: (1->2), (2->3), (3->4)
	markov[1*8+2] = 0.10
	markov[2*8+3] = 0.25
	markov[3*8+4] = -0.30

	tokenIDs := []int32{
		10, 3, 5, 12,
		1, 2, 3, 4,
	}

	stepProbs, cumProbs, err := ConfidenceReferenceCPU(hidden, rmsW, wProj, markov, tokenIDs, spec)
	if err != nil {
		t.Fatalf("ConfidenceReferenceCPU: %v", err)
	}
	if len(stepProbs) != 2 || len(cumProbs) != 2 {
		t.Fatalf("got batch size (%d, %d), want (2, 2)", len(stepProbs), len(cumProbs))
	}

	// Independent float64 oracle verification.
	for i := 0; i < 2; i++ {
		h := hidden[i*4 : (i+1)*4]
		var sumSq float64
		for d := 0; d < 4; d++ {
			sumSq += float64(h[d]) * float64(h[d])
		}
		invRMS := 1.0 / math.Sqrt(sumSq/4.0+float64(spec.Eps))
		var norm [4]float64
		for d := 0; d < 4; d++ {
			norm[d] = float64(h[d]) * invRMS * float64(rmsW[d])
		}
		wantCum := 1.0
		for s := 0; s < 3; s++ {
			var logit float64
			for d := 0; d < 4; d++ {
				logit += norm[d] * float64(wProj[s*4+d])
			}
			prevIdx := int(tokenIDs[i*4+s]) % 8
			curIdx := int(tokenIDs[i*4+s+1]) % 8
			logit += float64(markov[prevIdx*8+curIdx])
			scaled := logit / float64(spec.StepTemperatures[s])
			wantP := 1.0 / (1.0 + math.Exp(-scaled))
			wantCum *= wantP

			if math.Abs(float64(stepProbs[i][s])-wantP) > 1e-6 {
				t.Fatalf("seq %d step %d: stepProb=%v, want %v", i, s, stepProbs[i][s], wantP)
			}
			if math.Abs(float64(cumProbs[i][s])-wantCum) > 1e-6 {
				t.Fatalf("seq %d step %d: cumProb=%v, want %v", i, s, cumProbs[i][s], wantCum)
			}
		}
	}

	// Validation error checks.
	if _, _, err := ConfidenceReferenceCPU(hidden, rmsW, wProj, markov, tokenIDs, ConfidenceHeadSpec{HiddenDim: 0}); !errors.Is(err, ErrInvalidConfidenceSpec) {
		t.Fatalf("expected ErrInvalidConfidenceSpec for HiddenDim=0, got %v", err)
	}
}

func TestGenerateConfidenceCudaSource(t *testing.T) {
	spec := ConfidenceHeadSpec{
		HiddenDim:        128,
		MarkovVocabSize:  512,
		StepTemperatures: []float32{0.85, 0.90, 1.0},
	}
	src, err := GenerateConfidenceCudaSource(spec)
	if err != nil {
		t.Fatalf("GenerateConfidenceCudaSource: %v", err)
	}
	for _, needle := range []string{
		`extern "C" __global__ void mtp_confidence_head_kernel`,
		`rsqrtf`,
		`markov_matrix`,
		`step_probs`,
		`cum_probs`,
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("generated source missing %q", needle)
		}
	}
}
