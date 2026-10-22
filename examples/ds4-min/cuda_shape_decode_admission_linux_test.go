//go:build linux

package main

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func shapeTestRopeMetadata() *gguf.File {
	return &gguf.File{Metadata: map[string]gguf.Value{
		"deepseek4.rope.freq_base":                    {Value: float32(10000)},
		"deepseek4.attention.compress_rope_freq_base": {Value: float32(160000)},
	}}
}

func TestShapeDecodeEpsilonAdmission(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	for _, bad := range []float32{0, -1, 2e-6, float32(math.NaN()), float32(math.Inf(1))} {
		for _, phase := range []string{"rms", "hc"} {
			s := &cudaShapeDecode{forward: &cudaShapeForward{plan: shapeForwardPlan{config: c, context: 8}}, position: 3, raw: []shapeRawProgress{{next: 3, rows: 3}}, compressed: []shapeCompressorProgress{{next: 3}}, indexer: []shapeCompressorProgress{{next: 3}}}
			s.forward.scratch = map[string]cudamem.Slice[float32]{"cur": cacheTestSlice(t, c.HC*c.Embed, 1)}
			raw, compressed, indexer := slices.Clone(s.raw), slices.Clone(s.compressed), slices.Clone(s.indexer)
			rms, hc := float32(1e-6), float32(1e-6)
			if phase == "rms" {
				rms = bad
			} else {
				hc = bad
			}
			if _, err := s.Step(0, rms, hc); err == nil || !strings.Contains(err.Error(), "unsupported epsilon") {
				t.Fatalf("%s=%g: %v", phase, bad, err)
			}
			if s.failed || s.position != 3 || !slices.Equal(s.raw, raw) || !slices.Equal(s.compressed, compressed) || !slices.Equal(s.indexer, indexer) {
				t.Fatal("epsilon refusal changed state")
			}
		}
	}
	// Valid defaults reach the next actual-body admission check. Deliberately
	// use an invalid token to keep this host gate before device dispatch.
	s := &cudaShapeDecode{forward: &cudaShapeForward{plan: shapeForwardPlan{config: c, context: 8}}}
	if _, err := s.Step(uint32(c.Vocab), 1e-6, 1e-6); err == nil || !strings.Contains(err.Error(), "invalid token") {
		t.Fatalf("default epsilon refused: %v", err)
	}
}

func TestShapeDecodeRequiredEntriesAndRoutes(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	b := forwardTestBinding(t, c)
	specs, _ := c.Weights()
	tensors := make(map[string]cudaModelTensor)
	for i, spec := range specs {
		kind := spec.Types[0]
		if strings.HasSuffix(spec.Name, "ffn_gate_exps.weight") || strings.HasSuffix(spec.Name, "ffn_up_exps.weight") {
			kind = 16
		}
		if strings.HasSuffix(spec.Name, "ffn_down_exps.weight") {
			kind = 10
		}
		x := gguf.Tensor{TypeID: kind, Dimensions: spec.Dimensions}
		n, err := x.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		x.Dimensions = spec.Dimensions[:1]
		row, err := x.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		b.Weights[i].Type, b.Weights[i].Bytes, b.Weights[i].RowBytes = kind, n, row
		if len(spec.Dimensions) == 3 {
			b.Weights[i].ExpertBytes = n / spec.Dimensions[2]
		}
		tensors[spec.Name] = cudaModelTensor{name: spec.Name, typeID: kind, bytes: n, dimensions: slices.Clone(spec.Dimensions), device: driver.DevicePtr(4096 + i*64)}
	}
	w := shapeDecodeWeights{embedding: tensors["token_embd.weight"], attention: make([]shapeAttentionWeights, c.Layers), ffn: make([]shapeFFNWeights, c.Layers), compressor: make([]shapeCompressorWeights, c.Layers), indexerCompressor: make([]shapeCompressorWeights, c.Layers), indexer: make([]shapeIndexerWeights, c.Layers), sinks: make([]cudamem.Slice[float32], c.Layers), output: shapeOutputWeights{tensors["output_hc_fn.weight"], tensors["output_hc_scale.weight"], tensors["output_hc_base.weight"], tensors["output_norm.weight"], tensors["output.weight"]}}
	for layer := 0; layer < c.Layers; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		at := func(name string) cudaModelTensor { return tensors[prefix+name] }
		w.attention[layer] = shapeAttentionWeights{at("attn_q_a.weight"), at("attn_q_b.weight"), at("attn_kv.weight"), at("attn_output_a.weight"), at("attn_output_b.weight"), at("attn_q_a_norm.weight"), at("attn_kv_a_norm.weight")}
		w.ffn[layer] = shapeFFNWeights{at("ffn_gate_inp.weight"), at("exp_probs_b.bias"), at("ffn_gate_tid2eid.weight"), at("ffn_gate_exps.weight"), at("ffn_up_exps.weight"), at("ffn_down_exps.weight"), at("ffn_gate_shexp.weight"), at("ffn_up_shexp.weight"), at("ffn_down_shexp.weight")}
		compressor := func(prefix string) shapeCompressorWeights {
			return shapeCompressorWeights{at(prefix + "kv.weight"), at(prefix + "gate.weight"), at(prefix + "ape.weight"), at(prefix + "norm.weight")}
		}
		w.compressor[layer], w.indexerCompressor[layer] = compressor("attn_compressor_"), compressor("indexer_compressor_")
		w.indexer[layer] = shapeIndexerWeights{at("indexer.attn_q_b.weight"), at("indexer.proj.weight")}
		w.sinks[layer] = cacheTestSlice(t, c.Heads, layer+1)
	}
	r := &cudaFoundation{gguf: shapeTestRopeMetadata()}
	forward := &cudaShapeForward{r: r, plan: shapeForwardPlan{config: c, context: 8}}
	consumers := &cudaShapeConsumers{r: r, config: c, context: 8, entries: make(map[string]driver.Function), modules: make(map[string]driver.Module)}
	entries := []string{"ds4_shape_attention_decode_external", "ds4_shape_router_select", "ds4_shape_rope_tail", "ds4_shape_output_hc_weights", "ds4_shape_raw_commit", "ds4_shape_indexer_select"}
	for _, entry := range entries {
		consumers.entries[entry], consumers.modules[entry] = 1, 1
	}
	hc, mix := &cudaShapeHC{r: r, config: c}, &cudaShapeHCMix{r: r, config: c}
	attach := func(weights shapeDecodeWeights) (*cudaShapeDecode, error) {
		return newCUDAShapeDecode(forward, consumers, hc, mix, b, weights)
	}
	// Descriptor handles are host surrogates; this exercises actual attachment,
	// not owner destruction, arithmetic or a device module lookup.
	if s, err := attach(w); err != nil || s == nil {
		t.Fatalf("complete 61-layer attachment: %v", err)
	}
	for _, key := range []string{"rope.freq_base", "attention.compress_rope_freq_base", "rope.scaling.factor", "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow", "rope.scaling.original_context_length"} {
		t.Run("rope-metadata/"+key, func(t *testing.T) {
			name := "deepseek4." + key
			saved, present := r.gguf.Metadata[name]
			r.gguf.Metadata[name] = gguf.Value{Value: float32(7)}
			defer func() {
				if present {
					r.gguf.Metadata[name] = saved
				} else {
					delete(r.gguf.Metadata, name)
				}
			}()
			if s, err := attach(w); err == nil || s != nil || !strings.Contains(err.Error(), "rope") {
				t.Fatalf("conflicting RoPE metadata admitted: %v", err)
			}
			if s, err := newCUDAShapeState(r, c, b, nil, nil, w, 8, 0); err == nil || s != nil || !strings.Contains(err.Error(), "rope") {
				t.Fatalf("State allocated before RoPE admission: %v", err)
			}
		})
	}
	t.Run("binding-revalidated", func(t *testing.T) {
		i := len(b.Weights) - 1
		saved := b.Weights[i].Bytes
		b.Weights[i].Bytes++
		defer func() { b.Weights[i].Bytes = saved }()
		if s, err := attach(w); err == nil || s != nil {
			t.Fatal("mutated binding admitted after previous successful attachment")
		}
	})
	for _, entry := range entries {
		for _, field := range []string{"entry", "module"} {
			t.Run(entry+"/"+field, func(t *testing.T) {
				if field == "entry" {
					consumers.entries[entry] = 0
				} else {
					consumers.modules[entry] = 0
				}
				if s, err := attach(w); err == nil || s != nil || !strings.Contains(err.Error(), "missing entry") {
					t.Fatalf("missing entry admitted: %v", err)
				}
				consumers.entries[entry], consumers.modules[entry] = 1, 1
			})
		}
	}
	for _, fault := range []string{"attention", "ffn", "compressor", "indexer-compressor", "indexer-query", "sinks"} {
		t.Run("last-layer/"+fault, func(t *testing.T) {
			bad := w
			bad.attention, bad.ffn, bad.compressor, bad.indexerCompressor, bad.indexer, bad.sinks = slices.Clone(w.attention), slices.Clone(w.ffn), slices.Clone(w.compressor), slices.Clone(w.indexerCompressor), slices.Clone(w.indexer), slices.Clone(w.sinks)
			switch fault {
			case "attention":
				bad.attention[60].kv = cudaModelTensor{}
			case "ffn":
				bad.ffn[60].gate = cudaModelTensor{}
			case "compressor":
				bad.compressor[60].norm = cudaModelTensor{}
			case "indexer-compressor":
				bad.indexerCompressor[60].norm = cudaModelTensor{}
			case "indexer-query":
				bad.indexer[60].query = cudaModelTensor{}
			case "sinks":
				bad.sinks[60] = cudamem.Slice[float32]{}
			}
			if s, err := attach(bad); err == nil || s != nil {
				t.Fatalf("missing final layer route admitted: %s", fault)
			}
		})
	}
}

func TestShapeDecodeFailureTerminal(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	s := &cudaShapeDecode{forward: &cudaShapeForward{plan: shapeForwardPlan{config: c, context: 8}}, mix: &cudaShapeHCMix{prepared: true}, position: 3}
	// An actual Step error exercises the shared error defer without dispatch.
	// The phase-call source controls separately verify that all phase errors
	// return through this same named error, including partial phase writes.
	if _, err := s.Step(uint32(c.Vocab), 1e-6, 1e-6); err == nil || !s.failed {
		t.Fatalf("Step error did not poison owner: %v", err)
	}
	s.forward = nil // Reuse must refuse before dereferencing or changing state.
	if _, err := s.Step(0, 1e-6, 1e-6); err == nil || !strings.Contains(err.Error(), "closed or failed") {
		t.Fatalf("poisoned owner reused: %v", err)
	}
	if s.position != 3 || !s.mix.prepared {
		t.Fatal("refused reuse changed progress or phase")
	}
}
