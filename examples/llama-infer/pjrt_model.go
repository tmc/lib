package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/x/xla"
	"github.com/tmc/lib/x/xla/llama"
)

// pjrtModel adapts the example's host weight representation to the decoder.
// Calls must be externally synchronized. It borrows the engine and owns model.
// A failed readback closes model: device history may already have advanced.
type pjrtModel struct {
	model        *llama.Model
	vocab        int
	capacity     int
	weightReport WeightReport
	logits       []float32
}

func newPJRTModel(ctx context.Context, engine *xla.Engine, cfg *ModelConfig, weights *ModelWeights, ropeTheta float64, element xla.ElementType) (*pjrtModel, error) {
	if cfg == nil || weights == nil {
		return nil, fmt.Errorf("pjrt: model config and weights are required")
	}
	if cfg.NumHiddenLayers <= 0 || cfg.NumHiddenLayers != len(weights.Layers) {
		return nil, fmt.Errorf("pjrt: layer count does not match weights")
	}
	c, err := pjrtDecoderConfig(cfg, ropeTheta, element)
	if err != nil {
		return nil, err
	}
	w := llama.Weights{Embedding: weights.EmbedTokens, Norm: weights.Norm, LMHead: weights.LMHead}
	// LoadModel resolves a missing tied head to the same host slice. Preserve
	// that identity so the decoder uploads the shared parameter once. An
	// explicitly stored head remains distinct, even if tying was requested.
	if cfg.TieWordEmbeddings && len(w.LMHead) > 0 && len(w.LMHead) == len(w.Embedding) && &w.LMHead[0] == &w.Embedding[0] {
		w.LMHead = nil
	}
	if w.LMHead == nil && !cfg.TieWordEmbeddings {
		return nil, fmt.Errorf("pjrt: missing untied LM head")
	}
	for _, v := range weights.Layers {
		w.Layers = append(w.Layers, llama.LayerWeights{
			AttentionNorm: v.InputNorm, MLPNorm: v.PostNorm,
			Query: v.WQ, Key: v.WK, Value: v.WV, AttentionOutput: v.WO,
			Gate: v.WGate, Up: v.WUp, Down: v.WDown,
			QueryBias: v.QBias, KeyBias: v.KBias, ValueBias: v.VBias,
		})
	}
	weightBytes := 0
	add := func(data ...[]float32) error {
		for _, values := range data {
			if len(values) > (math.MaxInt-weightBytes)/4 {
				return fmt.Errorf("pjrt: parameter byte count overflows")
			}
			weightBytes += 4 * len(values)
		}
		return nil
	}
	if err := add(w.Embedding, w.Norm, w.LMHead); err != nil {
		return nil, err
	}
	for _, l := range w.Layers {
		if err := add(l.AttentionNorm, l.MLPNorm, l.Query, l.Key, l.Value, l.AttentionOutput, l.Gate, l.Up, l.Down, l.QueryBias, l.KeyBias, l.ValueBias); err != nil {
			return nil, err
		}
	}
	model, err := llama.New(ctx, engine, c, w)
	if err != nil {
		return nil, err
	}
	return &pjrtModel{model: model, vocab: cfg.VocabSize, capacity: cfg.MaxSeqLen, weightReport: WeightReport{DType: pjrtDType(element), Storage: "F32", PerElem: 4, Bytes: weightBytes}}, nil
}

// forward appends a chunk at its exact expected position and returns the last
// row of logits. Returned slices do not alias retained logits or device data.
func (p *pjrtModel) forward(ctx context.Context, ids []int32, position int) ([]float32, error) {
	if p == nil || p.model == nil {
		return nil, fmt.Errorf("pjrt: model is closed")
	}
	lengths := p.model.Lengths()
	if len(lengths) != 1 || int64(position) != lengths[0] {
		return nil, fmt.Errorf("pjrt: position does not match cached length")
	}
	out, err := p.model.Forward(ctx, ids, nil)
	if err != nil {
		if p.model.Lengths() == nil {
			return nil, errors.Join(err, p.close())
		}
		return nil, err
	}
	var data bytes.Buffer
	_, readErr := out.WriteTo(&data)
	if err := errors.Join(readErr, out.Close()); err != nil {
		return nil, errors.Join(err, p.close())
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, p.close())
	}
	// Decoder validation has already checked this product before execution.
	if data.Len() != len(ids)*p.vocab*4 {
		return nil, errors.Join(fmt.Errorf("pjrt: unexpected logit byte length"), p.close())
	}
	raw := data.Bytes()[(len(ids)-1)*p.vocab*4:]
	logits := make([]float32, p.vocab)
	for i := range logits {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.Join(fmt.Errorf("pjrt: nonfinite logits"), p.close())
		}
		logits[i] = v
	}
	p.logits = logits
	return slices.Clone(logits), nil
}

func (p *pjrtModel) readLogits() ([]float32, error) {
	if p == nil || p.model == nil {
		return nil, fmt.Errorf("pjrt: model is closed")
	}
	if p.logits == nil {
		return nil, fmt.Errorf("pjrt: no logits available")
	}
	return slices.Clone(p.logits), nil
}

func (p *pjrtModel) reset() error {
	if p == nil || p.model == nil {
		return fmt.Errorf("pjrt: model is closed")
	}
	if err := p.model.Reset(); err != nil {
		return err
	}
	p.logits = nil
	return nil
}

func (p *pjrtModel) close() error {
	if p == nil {
		return nil
	}
	model := p.model
	p.model = nil
	p.logits = nil
	return model.Close()
}

func pjrtDecoderConfig(cfg *ModelConfig, ropeTheta float64, element xla.ElementType) (llama.Config, error) {
	if cfg == nil {
		return llama.Config{}, fmt.Errorf("pjrt: model config is required")
	}
	if cfg.NumAttentionHeads <= 0 || (cfg.ExplicitHeadDim == 0 && cfg.HiddenSize%cfg.NumAttentionHeads != 0) || cfg.ExplicitHeadDim < 0 {
		return llama.Config{}, fmt.Errorf("pjrt: invalid attention head dimensions")
	}
	c := llama.Config{ElementType: element,
		BatchSize: 1, Capacity: int64(cfg.MaxSeqLen), VocabSize: int64(cfg.VocabSize),
		HiddenSize: int64(cfg.HiddenSize), IntermediateSize: int64(cfg.IntermediateSize),
		QueryHeads: int64(cfg.NumAttentionHeads), KVHeads: int64(cfg.NumKeyValueHeads), HeadSize: int64(cfg.HeadDim()),
		NormEpsilon: cfg.RmsNormEps, RopeTheta: ropeTheta,
	}
	return c, nil
}

func pjrtDType(element xla.ElementType) string {
	switch element {
	case xla.F32:
		return "F32"
	case xla.F16:
		return "F16"
	case xla.BF16:
		return "BF16"
	}
	return ""
}
