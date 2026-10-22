package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
)

// validateShapeRopeMetadata admits the pinned default graph, not arbitrary
// shape-compatible GGUF overrides. Optional keys have the upstream defaults.
func validateShapeRopeMetadata(g *gguf.File) error {
	if g == nil {
		return fmt.Errorf("cuda shape rope: missing GGUF metadata")
	}
	for _, scalar := range []struct {
		key      string
		want     float32
		optional bool
	}{
		{"rope.freq_base", 10000, false},
		{"attention.compress_rope_freq_base", 160000, false},
		{"rope.scaling.factor", 16, true},
		{"rope.scaling.yarn_beta_fast", 32, true},
		{"rope.scaling.yarn_beta_slow", 1, true},
	} {
		key := "deepseek4." + scalar.key
		value, ok := g.Metadata[key]
		if !ok && scalar.optional {
			continue
		}
		var got float32
		switch v := value.Value.(type) {
		case float32:
			got = v
		case float64:
			got = float32(v)
		default:
			return fmt.Errorf("cuda shape rope: missing or invalid %s", key)
		}
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got)-float64(scalar.want)) > float64(scalar.want)*1e-6 {
			return fmt.Errorf("cuda shape rope: unsupported %s", key)
		}
	}
	if value, ok := g.Metadata["deepseek4.rope.scaling.original_context_length"]; ok {
		var got uint64
		switch v := value.Value.(type) {
		case uint64:
			got = v
		case uint32:
			got = uint64(v)
		case uint16:
			got = uint64(v)
		case uint8:
			got = uint64(v)
		default:
			return fmt.Errorf("cuda shape rope: invalid original context")
		}
		if got != 65536 {
			return fmt.Errorf("cuda shape rope: unsupported original context")
		}
	}
	return nil
}

type shapeRopeParameters struct {
	originalContext                                             uint32
	base, scale, extension, attentionFactor, betaFast, betaSlow float32
}

// defaultShapeRope matches layer_rope_freq_base/scale and rope_tail_layer_inplace
// in ds4.c:11461 at 11e8cad for the unmodified Flash/Pro shape defaults. A model
// overriding these metadata values must supply separately admitted parameters;
// this function does not silently replace such a model's settings.
func defaultShapeRope(c dsv4shape.Config, layer int) (shapeRopeParameters, error) {
	if err := c.Validate(); err != nil {
		return shapeRopeParameters{}, err
	}
	l, err := c.Layer(layer)
	if err != nil {
		return shapeRopeParameters{}, err
	}
	p := shapeRopeParameters{base: 10000, scale: 1, attentionFactor: 1, betaFast: 32, betaSlow: 1}
	if l.CompressRatio != 0 {
		p.originalContext, p.base, p.scale, p.extension = 65536, 160000, 1.0/16, 1
		p.attentionFactor = 1 / (1 + 0.1*float32(math.Log(float64(1/p.scale))))
	}
	return p, nil
}
