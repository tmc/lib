package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/gguf"
)

func streamingTriplets(g *gguf.File, layers int) (map[int]dsv4expertcache.Triplet, error) {
	if g == nil || layers <= 0 {
		return nil, errors.New("stream model geometry unavailable")
	}
	byName := make(map[string]gguf.Tensor, len(g.Tensors))
	for _, t := range g.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, errors.New("duplicate streaming tensor name")
		}
		byName[t.Name] = t
	}
	triplets := make(map[int]dsv4expertcache.Triplet, layers)
	for layer := 0; layer < layers; layer++ {
		var parts [3]dsv4expertcache.Tensor
		var types [3]uint32
		for i, suffix := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
			name := fmt.Sprintf("blk.%d.%s", layer, suffix)
			t, found := byName[name]
			if !found {
				return nil, fmt.Errorf("stream tensor %q missing", name)
			}
			if len(t.Dimensions) != 3 || t.Dimensions[2] != 256 {
				return nil, fmt.Errorf("stream tensor %q: invalid expert axis", name)
			}
			n, err := t.Bytes()
			if err != nil {
				return nil, err
			}
			if n%256 != 0 || t.Offset > math.MaxUint64-g.DataOffset {
				return nil, errors.New("stream expert extent overflow")
			}
			parts[i] = dsv4expertcache.Tensor{Offset: g.DataOffset + t.Offset, Bytes: n, Stride: n / 256, Experts: 256}
			types[i] = t.TypeID
		}
		format, err := ExpertFormatForTypes(types[0], types[1], types[2])
		if err != nil {
			return nil, err
		}
		if format != ExpertsIQ2Q2K && format != ExpertsQ4K {
			return nil, errors.New("stream expert quantization consumer unavailable")
		}
		triplets[layer] = dsv4expertcache.Triplet{Gate: parts[0], Up: parts[1], Down: parts[2]}
	}
	return triplets, nil
}
