//go:build linux

package main

import (
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type shapeIndexerWeights struct {
	query, projection cudaModelTensor
}

type shapeDecodeWeights struct {
	embedding                     cudaModelTensor
	attention                     []shapeAttentionWeights
	ffn                           []shapeFFNWeights
	compressor, indexerCompressor []shapeCompressorWeights
	indexer                       []shapeIndexerWeights
	sinks                         []cudamem.Slice[float32]
	output                        shapeOutputWeights
}

// cudaShapeDecode owns the four component owners after successful attachment.
// Uploaded weights and the foundation remain borrowed and must outlive Close.
// The serial decode schedule is executable source, not whole-Pro admission.
// In particular its quantized projection arithmetic still needs an oracle gate.
type cudaShapeDecode struct {
	mu                  sync.Mutex
	forward             *cudaShapeForward
	consumers           *cudaShapeConsumers
	hc                  *cudaShapeHC
	mix                 *cudaShapeHCMix
	binding             *dsv4shape.Binding
	weights             shapeDecodeWeights
	raw                 []shapeRawProgress
	compressed, indexer []shapeCompressorProgress
	position            uint32
	closed, failed      bool
}

// Component ownership transfers only on success. On refusal the caller still
// owns every component. This does not create a public backend Supports claim.
func newCUDAShapeDecode(forward *cudaShapeForward, consumers *cudaShapeConsumers, hc *cudaShapeHC, mix *cudaShapeHCMix, binding *dsv4shape.Binding, weights shapeDecodeWeights) (*cudaShapeDecode, error) {
	if forward == nil || consumers == nil || hc == nil || mix == nil || forward.closed || consumers.closed || hc.closed || mix.closed {
		return nil, errors.New("cuda shape decode: missing or closed component")
	}
	c, r := forward.plan.config, forward.r
	if consumers.r != r || hc.r != r || mix.r != r || consumers.config != c || hc.config != c || mix.config != c || consumers.context != forward.plan.context {
		return nil, errors.New("cuda shape decode: component family, context or foundation mismatch")
	}
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if err := validateShapeRopeMetadata(r.gguf); err != nil {
		return nil, err
	}
	if _, err := planShapeForward(c, binding, forward.plan.context); err != nil {
		return nil, err
	}
	if len(weights.attention) != c.Layers || len(weights.ffn) != c.Layers || len(weights.compressor) != c.Layers || len(weights.indexerCompressor) != c.Layers || len(weights.indexer) != c.Layers || len(weights.sinks) != c.Layers {
		return nil, errors.New("cuda shape decode: incomplete layer weights")
	}
	if weights.embedding.name != "token_embd.weight" || weights.embedding.typeID != 1 || weights.embedding.device == 0 || len(weights.embedding.dimensions) != 2 || weights.embedding.dimensions[0] != uint64(c.Embed) || weights.embedding.dimensions[1] != uint64(c.Vocab) {
		return nil, errors.New("cuda shape decode: unsupported embedding layout")
	}
	if err := validateShapeDecodeTensor(c, binding, weights.embedding); err != nil {
		return nil, err
	}
	for _, entry := range []struct {
		name   string
		kind   uint32
		tensor cudaModelTensor
	}{
		{"output_hc_fn.weight", 1, weights.output.fn}, {"output_hc_scale.weight", 0, weights.output.scale}, {"output_hc_base.weight", 0, weights.output.base}, {"output_norm.weight", 0, weights.output.norm}, {"output.weight", 8, weights.output.output},
	} {
		if entry.tensor.name != entry.name || entry.tensor.typeID != entry.kind {
			return nil, fmt.Errorf("cuda shape decode: invalid output weight %s", entry.name)
		}
		if err := validateShapeDecodeTensor(c, binding, entry.tensor); err != nil {
			return nil, err
		}
	}
	for layer := 0; layer < c.Layers; layer++ {
		if err := validateShapeAttentionWeightsAdmitted(c, binding, layer, weights.attention[layer]); err != nil {
			return nil, err
		}
		if err := validateShapeFFNWeightsAdmitted(c, binding, layer, weights.ffn[layer]); err != nil {
			return nil, err
		}
		if weights.sinks[layer].Len() != c.Heads || weights.sinks[layer].Ptr() == 0 {
			return nil, errors.New("cuda shape decode: missing sinks")
		}
		l, _ := c.Layer(layer)
		if l.CompressRatio != 0 {
			if err := validateShapeCompressorWeightsAdmitted(c, binding, layer, false, weights.compressor[layer]); err != nil {
				return nil, err
			}
		}
		if l.Indexer {
			if err := validateShapeCompressorWeightsAdmitted(c, binding, layer, true, weights.indexerCompressor[layer]); err != nil {
				return nil, err
			}
			w := weights.indexer[layer]
			if (w.query.typeID != 1 && w.query.typeID != 8) || w.projection.typeID != 1 || w.query.device == 0 || w.projection.device == 0 || len(w.query.dimensions) != 2 || w.query.dimensions[0] != uint64(c.QLoRA) || w.query.dimensions[1] != uint64(c.IndexerHeads*c.IndexerHeadDim) || len(w.projection.dimensions) != 2 || w.projection.dimensions[0] != uint64(c.Embed) || w.projection.dimensions[1] != uint64(c.IndexerHeads) {
				return nil, errors.New("cuda shape decode: unsupported indexer projection")
			}
			if w.query.name != fmt.Sprintf("blk.%d.indexer.attn_q_b.weight", layer) || w.projection.name != fmt.Sprintf("blk.%d.indexer.proj.weight", layer) {
				return nil, errors.New("cuda shape decode: mismatched indexer projection")
			}
			for _, tensor := range []cudaModelTensor{w.query, w.projection} {
				if err := validateShapeDecodeTensor(c, binding, tensor); err != nil {
					return nil, err
				}
			}
		}
	}
	for _, entry := range []string{"ds4_shape_attention_decode_external", "ds4_shape_router_select", "ds4_shape_rope_tail", "ds4_shape_output_hc_weights", "ds4_shape_raw_commit", "ds4_shape_indexer_select"} {
		if consumers.entries[entry] == 0 || consumers.modules[entry] == 0 {
			return nil, fmt.Errorf("cuda shape decode: missing entry %s", entry)
		}
	}
	return &cudaShapeDecode{forward: forward, consumers: consumers, hc: hc, mix: mix, binding: binding, weights: weights, raw: make([]shapeRawProgress, c.Layers), compressed: make([]shapeCompressorProgress, c.Layers), indexer: make([]shapeCompressorProgress, c.Layers)}, nil
}

func validateShapeDecodeTensor(c dsv4shape.Config, binding *dsv4shape.Binding, tensor cudaModelTensor) error {
	specs, err := c.Weights()
	if err != nil {
		return err
	}
	for _, spec := range specs {
		if spec.Name != tensor.name {
			continue
		}
		for _, view := range binding.Weights {
			if view.Name == tensor.name && tensor.device != 0 && tensor.typeID == view.Type && tensor.bytes == view.Bytes && tensor.fileOffset == view.Offset && slices.Equal(tensor.dimensions, spec.Dimensions) {
				return nil
			}
		}
		break
	}
	return fmt.Errorf("cuda shape decode: invalid uploaded tensor %s", tensor.name)
}

// Step processes one shared-history token. Any error poisons the State because
// some cache writes may already have been enqueued; retrying the position would
// duplicate compressor updates. Logits are a borrowed view valid until Step or
// Close. Finiteness gates currently synchronize and read full block outputs.
func (s *cudaShapeDecode) Step(token uint32, rmsEpsilon, hcEpsilon float32) (logits cudamem.Slice[float32], err error) {
	if s == nil {
		return logits, errors.New("cuda shape decode: nil state")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.failed {
		return logits, errors.New("cuda shape decode: closed or failed state")
	}
	// Reused RMS and compressor consumers implement the metadata-pinned
	// defaults. Refuse a mixed-epsilon graph before any cache or phase writes.
	if rmsEpsilon != 1e-6 || hcEpsilon != 1e-6 {
		return logits, errors.New("cuda shape decode: unsupported epsilon")
	}
	defer func() {
		if err != nil {
			s.failed = true
		}
	}()
	state, c := s.forward, s.forward.plan.config
	if token >= uint32(c.Vocab) || uint64(s.position) >= state.plan.context {
		return logits, errors.New("cuda shape decode: invalid token or exhausted context")
	}
	for lane := 0; lane < c.HC; lane++ {
		if err = state.r.encodeEmbeddingRow(s.weights.embedding, token, state.scratch["cur"].Slice(lane*c.Embed, (lane+1)*c.Embed)); err != nil {
			return logits, err
		}
	}
	quant, err := s.consumers.attentionQuant()
	if err != nil {
		return logits, err
	}
	ffn, err := s.consumers.ffnBuffers()
	if err != nil {
		return logits, err
	}
	scores, err := shapeConsumerView[float32](s.consumers, "attention-scores")
	if err != nil {
		return logits, err
	}
	for layer := 0; layer < c.Layers; layer++ {
		l, _ := c.Layer(layer)
		if err = s.mix.BeforePhase(s.hc, state, layer, "attn", rmsEpsilon, hcEpsilon); err != nil {
			return logits, err
		}
		if err = encodeShapeQKV(state, s.binding, layer, s.weights.attention[layer], quant); err != nil {
			return logits, err
		}
		if err = encodeShapeRoPE(state, s.consumers.entries["ds4_shape_rope_tail"], layer, s.position); err != nil {
			return logits, err
		}
		if err = commitShapeRaw(state, s.consumers.entries["ds4_shape_raw_commit"], layer, s.position, &s.raw[layer]); err != nil {
			return logits, err
		}
		if l.CompressRatio != 0 {
			if err = encodeShapeCompressor(state, s.binding, layer, s.position, false, s.weights.compressor[layer], &s.compressed[layer]); err != nil {
				return logits, err
			}
		}
		mask := state.scratch["attn_score"]
		if l.Indexer {
			if err = encodeShapeCompressor(state, s.binding, layer, s.position, true, s.weights.indexerCompressor[layer], &s.indexer[layer]); err != nil {
				return logits, err
			}
			if s.indexer[layer].rows != s.compressed[layer].rows {
				return logits, errors.New("cuda shape decode: attention/indexer row count mismatch")
			}
			if s.indexer[layer].rows > c.IndexerTopK {
				w := s.weights.indexer[layer]
				if w.query.typeID == 8 {
					err = state.r.encodeQ80MatvecQ81(w.query, quant.qr, state.scratch["index_q"])
				} else {
					err = state.r.encodeF16MatvecDefault(w.query, state.scratch["qr_norm"], state.scratch["index_q"])
				}
				if err != nil {
					return logits, err
				}
				if err = state.r.encodeF16MatvecDefault(w.projection, state.scratch["attn_norm"], state.scratch["index_weights"]); err != nil {
					return logits, err
				}
			}
			query, e := shapeConsumerView[float32](s.consumers, "indexer-query")
			if e != nil {
				return logits, e
			}
			invalid, e := shapeConsumerView[uint32](s.consumers, "indexer-invalid")
			if e != nil {
				return logits, e
			}
			if err = encodeShapeIndexer(state, s.consumers.entries["ds4_shape_indexer_select"], layer, s.position, s.indexer[layer].rows, query, mask, invalid); err != nil {
				return logits, err
			}
			mask = mask.Slice(0, s.indexer[layer].rows)
		}
		req := shapeAttentionRequest{layer: layer, position: s.position, rawRows: s.raw[layer].rows, rawStart: s.raw[layer].start, compressedRows: uint32(s.compressed[layer].rows), useMask: shapeAttentionUseMask(l.Indexer, uint32(s.compressed[layer].rows))}
		if err = encodeShapeAttentionExternal(state, s.consumers.entries["ds4_shape_attention_decode_external"], req, s.weights.sinks[layer], mask, scores); err != nil {
			return logits, err
		}
		if err = encodeShapeAttentionInverseRoPE(state, s.consumers.entries["ds4_shape_rope_tail"], layer, s.position); err != nil {
			return logits, err
		}
		if err = encodeShapeAttentionOutput(state, s.binding, layer, s.weights.attention[layer], quant); err != nil {
			return logits, err
		}
		if err = readShapeFiniteOutput(state, "attn_out"); err != nil {
			return logits, err
		}
		if err = s.mix.AfterPhase(s.hc, state, layer, "attn"); err != nil {
			return logits, err
		}
		if err = s.mix.FFNPhase(s.hc, state, s.binding, layer, token, s.consumers.entries["ds4_shape_router_select"], s.weights.ffn[layer], ffn, rmsEpsilon, hcEpsilon); err != nil {
			return logits, err
		}
	}
	q, err := shapeConsumerView[byte](s.consumers, "output-input")
	if err != nil {
		return logits, err
	}
	logits, err = shapeConsumerView[float32](s.consumers, "logits")
	if err != nil {
		return logits, err
	}
	if err = encodeShapeOutput(state, s.binding, s.mix, s.hc, s.consumers.entries["ds4_shape_output_hc_weights"], s.weights.output, q, logits, rmsEpsilon, hcEpsilon); err != nil {
		return logits, err
	}
	if err = state.r.withContext(func() error { return syncStream(state.r.stream()) }); err != nil {
		return logits, err
	}
	values := make([]float32, c.Vocab)
	if err = logits.CopyToHost(values); err != nil {
		return logits, err
	}
	if err = checkShapeFiniteOutput(values, c.Vocab); err != nil {
		return logits, err
	}
	s.position++
	return logits, nil
}

func (s *cudaShapeDecode) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	// Each owner drains before release and retains failures for a later retry.
	for _, close := range []func() error{s.consumers.Close, s.mix.Close, s.hc.Close, s.forward.Close} {
		if err := close(); err != nil {
			return err
		}
	}
	return nil
}
