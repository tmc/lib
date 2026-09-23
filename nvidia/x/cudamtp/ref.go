// SPDX-License-Identifier: Apache-2.0

package cudamtp

import (
	"errors"
	"fmt"
)

// ErrMTPConfig indicates an invalid configuration for MTP operations.
var ErrMTPConfig = errors.New("cudamtp: invalid MTP configuration")

// Validate verifies the parameters of MTPConfig.
func (cfg MTPConfig) Validate() error {
	if cfg.NumDraftHeads <= 0 || cfg.Dim <= 0 || cfg.VocabSize <= 0 {
		return fmt.Errorf("%w: heads=%d, dim=%d, vocab=%d", ErrMTPConfig, cfg.NumDraftHeads, cfg.Dim, cfg.VocabSize)
	}
	return nil
}

// MTPReference performs host-based CPU reference forward prediction for multi-draft heads.
// Given hidden trunk representation [dim], draft projections [numDraftHeads, dim, dim],
// and LM heads [numDraftHeads, dim, vocabSize], it computes draft logits [numDraftHeads, vocabSize].
func MTPReference(hTrunk []float32, wDraftProj []float32, wLMHeads []float32, cfg MTPConfig) ([]float32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(hTrunk) != cfg.Dim {
		return nil, fmt.Errorf("%w: trunk hidden dimension %d != %d", ErrMTPConfig, len(hTrunk), cfg.Dim)
	}
	projElems := cfg.NumDraftHeads * cfg.Dim * cfg.Dim
	if len(wDraftProj) != projElems {
		return nil, fmt.Errorf("%w: draft proj length %d != %d", ErrMTPConfig, len(wDraftProj), projElems)
	}
	lmElems := cfg.NumDraftHeads * cfg.Dim * cfg.VocabSize
	if len(wLMHeads) != lmElems {
		return nil, fmt.Errorf("%w: lm heads length %d != %d", ErrMTPConfig, len(wLMHeads), lmElems)
	}

	outLogits := make([]float32, cfg.NumDraftHeads*cfg.VocabSize)
	hidden := make([]float32, cfg.Dim)

	for h := 0; h < cfg.NumDraftHeads; h++ {
		// 1. Parallel draft head projection: s_hidden = h_trunk * W_draft_proj[h]
		wProjHead := wDraftProj[h*cfg.Dim*cfg.Dim : (h+1)*cfg.Dim*cfg.Dim]
		for row := 0; row < cfg.Dim; row++ {
			var acc float32
			wRow := wProjHead[row*cfg.Dim : (row+1)*cfg.Dim]
			for k := 0; k < cfg.Dim; k++ {
				acc += hTrunk[k] * wRow[k]
			}
			hidden[row] = acc
		}

		// 2. Draft LM Head logits: headLogits = s_hidden * W_lm_heads[h]
		wLMHead := wLMHeads[h*cfg.Dim*cfg.VocabSize : (h+1)*cfg.Dim*cfg.VocabSize]
		headLogits := outLogits[h*cfg.VocabSize : (h+1)*cfg.VocabSize]
		for v := 0; v < cfg.VocabSize; v++ {
			var acc float32
			for k := 0; k < cfg.Dim; k++ {
				acc += hidden[k] * wLMHead[k*cfg.VocabSize+v]
			}
			headLogits[v] = acc
		}
	}

	return outLogits, nil
}
