package main

import (
	"context"
	"fmt"
	"math"

	"github.com/tmc/lib/x/xla"
	"github.com/tmc/lib/x/xla/llama"
)

func (p *pjrtCheckpoint) newModel(ctx context.Context, engine *xla.Engine) (*pjrtModel, error) {
	if p.retained == nil {
		return newPJRTModel(ctx, engine, &p.config, p.weights, p.theta, p.element)
	}
	c, err := pjrtDecoderConfig(&p.config, p.theta, p.element)
	if err != nil {
		return nil, err
	}
	r := p.retained
	if p.config.NumHiddenLayers != len(r.Layers) {
		return nil, fmt.Errorf("pjrt: layer count does not match weights")
	}
	var firstErr error
	weightBytes := 0
	encode := func(t *RetainedTensor) llama.EncodedTensor {
		if t == nil {
			return llama.EncodedTensor{}
		}
		var element xla.ElementType
		switch t.DType {
		case "F32", "float32":
			element = xla.F32
		case "F16", "float16":
			element = xla.F16
		case "BF16", "bfloat16":
			element = xla.BF16
		default:
			if firstErr == nil {
				firstErr = fmt.Errorf("pjrt: unsupported weight dtype %q", t.DType)
			}
		}
		if len(t.Raw) > math.MaxInt-weightBytes {
			if firstErr == nil {
				firstErr = fmt.Errorf("pjrt: parameter byte count overflows")
			}
		} else {
			weightBytes += len(t.Raw)
		}
		return llama.EncodedTensor{ElementType: element, Data: t.Raw}
	}
	w := llama.EncodedWeights{Embedding: encode(r.EmbedTokens), Norm: encode(r.Norm)}
	if r.LMHead != nil && !(p.config.TieWordEmbeddings && r.LMHead == r.EmbedTokens) {
		w.LMHead = encode(r.LMHead)
	} else if !p.config.TieWordEmbeddings {
		return nil, fmt.Errorf("pjrt: missing untied LM head")
	}
	for _, l := range r.Layers {
		w.Layers = append(w.Layers, llama.EncodedLayerWeights{
			AttentionNorm: encode(l.InputNorm), MLPNorm: encode(l.PostNorm),
			Query: encode(l.WQ), Key: encode(l.WK), Value: encode(l.WV), AttentionOutput: encode(l.WO),
			Gate: encode(l.WGate), Up: encode(l.WUp), Down: encode(l.WDown),
			QueryBias: encode(l.QBias), KeyBias: encode(l.KBias), ValueBias: encode(l.VBias),
		})
	}
	if firstErr != nil {
		return nil, firstErr
	}
	model, err := llama.NewEncoded(ctx, engine, c, w)
	if err != nil {
		return nil, err
	}
	return &pjrtModel{model: model, vocab: p.config.VocabSize, capacity: p.config.MaxSeqLen, weightBytes: weightBytes}, nil
}
