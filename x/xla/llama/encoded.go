package llama

import (
	"context"

	"github.com/tmc/lib/x/xla"
)

// EncodedTensor is a row-major tensor payload in little-endian representation.
// ElementType must explicitly identify F32, F16 or BF16; equal byte widths do
// not imply equal types. The model config determines the required shape.
// Data must remain unchanged until NewEncoded returns.
type EncodedTensor struct {
	ElementType xla.ElementType
	Data        []byte
}

// EncodedLayerWeights holds the same parameter roles and shapes as LayerWeights.
// Nil Data for query/key/value biases means zero. Source types may differ.
type EncodedLayerWeights struct {
	AttentionNorm, MLPNorm             EncodedTensor
	Query, Key, Value, AttentionOutput EncodedTensor
	Gate, Up, Down                     EncodedTensor
	QueryBias, KeyBias, ValueBias      EncodedTensor
}

// EncodedWeights holds the same parameter roles and shapes as Weights.
// Nil LMHead.Data shares the embedding. Every supplied tensor carries its own
// storage type, independently of Config.ElementType.
type EncodedWeights struct {
	Embedding, Norm, LMHead EncodedTensor
	Layers                  []EncodedLayerWeights
}

// NewEncoded validates and uploads finite parameters in their source types.
// It does not widen or retain the host payloads. It borrows engine, which must
// remain open until the model is closed. Cancellation waits for in-flight native
// calls before cleanup. All shapes, byte lengths and values are checked before
// device allocation. Graph operations convert source types as needed.
func NewEncoded(ctx context.Context, engine *xla.Engine, c Config, w EncodedWeights) (*Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	spec, err := describeEncoded(c, w)
	if err != nil {
		return nil, err
	}
	return newModel(ctx, engine, c, spec)
}

func describeEncoded(c Config, w EncodedWeights) (layout, error) {
	value := func(v EncodedTensor) weightValue { return weightValue{kind: v.ElementType, raw: v.Data} }
	values := []weightValue{value(w.Embedding)}
	for _, l := range w.Layers {
		for _, v := range []EncodedTensor{l.AttentionNorm, l.MLPNorm, l.Query, l.Key, l.Value, l.AttentionOutput, l.Gate, l.Up, l.Down, l.QueryBias, l.KeyBias, l.ValueBias} {
			values = append(values, value(v))
		}
	}
	values = append(values, value(w.Norm), value(w.LMHead))
	return describeValues(c, values, len(w.Layers))
}
