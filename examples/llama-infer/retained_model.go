package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

// A RetainedModel is the model's weights in the dtypes the checkpoint
// declared, arranged by the role each tensor plays.
//
// It is the retained counterpart of [ModelWeights], which holds the same
// tensors already widened to float32. The upload path takes a RetainedModel so
// that the decision of what precision reaches the device is made at the upload
// boundary rather than in the loader.
type RetainedModel struct {
	EmbedTokens *RetainedTensor
	Layers      []RetainedLayer
	Norm        *RetainedTensor
	LMHead      *RetainedTensor
	Consumption CheckpointReceipt
}

// A RetainedLayer holds one decoder layer's weights in their source dtypes.
type RetainedLayer struct {
	InputNorm *RetainedTensor
	PostNorm  *RetainedTensor
	WQ        *RetainedTensor
	WK        *RetainedTensor
	WV        *RetainedTensor
	QBias     *RetainedTensor
	KBias     *RetainedTensor
	VBias     *RetainedTensor
	WO        *RetainedTensor
	WGate     *RetainedTensor
	WUp       *RetainedTensor
	WDown     *RetainedTensor
}

// DTypes returns how many tensors the model retains of each dtype.
//
// A model loaded from a half-precision checkpoint reports its tensors as F16
// or BF16 here. Reporting F32 for all of them means something widened on the
// way in.
func (m *RetainedModel) DTypes() map[string]int {
	counts := map[string]int{}
	m.each(func(_ string, t *RetainedTensor) {
		counts[t.DType]++
	})
	return counts
}

// PayloadBytes returns the total retained payload across every tensor.
func (m *RetainedModel) PayloadBytes() int {
	var n int
	m.each(func(_ string, t *RetainedTensor) { n += t.Bytes() })
	return n
}

// PayloadBytesAsFloat32 returns what the same tensors occupy widened to
// float32, which is what the loader produced before retention.
func (m *RetainedModel) PayloadBytesAsFloat32() int {
	var n int
	m.each(func(_ string, t *RetainedTensor) { n += t.Elems() * 4 })
	return n
}

// each calls fn for every tensor the model holds, skipping absent optional
// ones. The LM head is skipped when it is the embedding table, so a tied model
// does not count the same payload twice.
func (m *RetainedModel) each(fn func(name string, t *RetainedTensor)) {
	visit := func(name string, t *RetainedTensor) {
		if t != nil {
			fn(name, t)
		}
	}
	visit("embed_tokens", m.EmbedTokens)
	visit("norm", m.Norm)
	if m.LMHead != nil && m.LMHead != m.EmbedTokens {
		visit("lm_head", m.LMHead)
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		visit(p+"input_norm", l.InputNorm)
		visit(p+"post_norm", l.PostNorm)
		visit(p+"wq", l.WQ)
		visit(p+"wk", l.WK)
		visit(p+"wv", l.WV)
		visit(p+"q_bias", l.QBias)
		visit(p+"k_bias", l.KBias)
		visit(p+"v_bias", l.VBias)
		visit(p+"wo", l.WO)
		visit(p+"wgate", l.WGate)
		visit(p+"wup", l.WUp)
		visit(p+"wdown", l.WDown)
	}
}

// Float32 widens the model to the float32 representation the existing device
// upload path consumes.
//
// This is the explicit conversion at the upload boundary. Until device kernels
// can read half storage, the bytes are widened here rather than in the loader,
// which is the whole point of retaining them: the precision that reaches the
// device becomes a property of this call, not of how the checkpoint was read.
func (m *RetainedModel) Float32() *ModelWeights {
	decode := func(t *RetainedTensor) []float32 {
		if t == nil {
			return nil
		}
		return t.DecodeFloat32()
	}
	w := &ModelWeights{
		EmbedTokens: decode(m.EmbedTokens),
		Norm:        decode(m.Norm),
		Consumption: m.Consumption,
		Layers:      make([]LayerWeights, len(m.Layers)),
	}
	if m.LMHead == m.EmbedTokens {
		w.LMHead = w.EmbedTokens // tied, and the reference loader shares the slice
	} else {
		w.LMHead = decode(m.LMHead)
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		w.Layers[i] = LayerWeights{
			InputNorm: decode(l.InputNorm),
			PostNorm:  decode(l.PostNorm),
			WQ:        decode(l.WQ),
			WK:        decode(l.WK),
			WV:        decode(l.WV),
			QBias:     decode(l.QBias),
			KBias:     decode(l.KBias),
			VBias:     decode(l.VBias),
			WO:        decode(l.WO),
			WGate:     decode(l.WGate),
			WUp:       decode(l.WUp),
			WDown:     decode(l.WDown),
		}
	}
	return w
}

// retainedFromCheckpoint arranges a loaded checkpoint into model roles.
//
// A tensor the configuration requires but the checkpoint does not provide is
// an error, except for an absent LM head, which is tied to the embedding table
// exactly as the float32 loader ties it.
func retainedFromCheckpoint(c *RetainedCheckpoint, cfg *ModelConfig, receipt CheckpointReceipt) (*RetainedModel, error) {
	get := func(name string) (*RetainedTensor, error) {
		t, ok := c.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("checkpoint has no tensor %q", name)
		}
		return t, nil
	}
	optional := func(name string) *RetainedTensor { return c.Tensors[name] }

	embed, err := get("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	norm, err := get("model.norm.weight")
	if err != nil {
		return nil, err
	}
	m := &RetainedModel{EmbedTokens: embed, Norm: norm, Consumption: receipt}
	if head := optional("lm_head.weight"); head != nil {
		m.LMHead = head
	} else {
		m.LMHead = embed
	}

	m.Layers = make([]RetainedLayer, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		for _, f := range []struct {
			dst  **RetainedTensor
			name string
		}{
			{&m.Layers[i].InputNorm, p + "input_layernorm.weight"},
			{&m.Layers[i].PostNorm, p + "post_attention_layernorm.weight"},
			{&m.Layers[i].WQ, p + "self_attn.q_proj.weight"},
			{&m.Layers[i].WK, p + "self_attn.k_proj.weight"},
			{&m.Layers[i].WV, p + "self_attn.v_proj.weight"},
			{&m.Layers[i].WO, p + "self_attn.o_proj.weight"},
			{&m.Layers[i].WGate, p + "mlp.gate_proj.weight"},
			{&m.Layers[i].WUp, p + "mlp.up_proj.weight"},
			{&m.Layers[i].WDown, p + "mlp.down_proj.weight"},
		} {
			t, err := get(f.name)
			if err != nil {
				return nil, err
			}
			*f.dst = t
		}
		m.Layers[i].QBias = optional(p + "self_attn.q_proj.bias")
		m.Layers[i].KBias = optional(p + "self_attn.k_proj.bias")
		m.Layers[i].VBias = optional(p + "self_attn.v_proj.bias")
	}
	return m, nil
}

// LoadRetainedModel reads the checkpoint in dir without widening it to
// float32.
//
// It is the retention counterpart of [LoadModel] and performs the same
// checkpoint audit, so a checkpoint one accepts the other accepts. The
// difference is only the representation it hands back: every tensor keeps the
// dtype and the bytes the checkpoint declared, and the decision of what
// precision reaches a consumer moves to the consumer.
func LoadRetainedModel(dir string) (*RetainedModel, *ModelConfig, error) {
	cfg, err := LoadModelConfig(dir)
	if err != nil {
		return nil, nil, err
	}

	sw, err := cudasafetensors.OpenSharded(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer sw.Close()

	metadata := make(map[string]checkpointTensor, len(sw.Tensors))
	for name, info := range sw.Tensors {
		metadata[name] = checkpointTensor{DType: info.DType, Shape: info.Shape, Shard: info.ShardFile}
	}
	receipt, err := auditCheckpoint(cfg, metadata, "F32")
	if err != nil {
		return nil, nil, err
	}

	// Only the tensors the model uses are retained. Reading the whole file
	// would hold payload the arrangement then discards, which would make the
	// retained memory accounting measure the checkpoint rather than the model.
	var names []string
	for _, want := range expectedCheckpointTensors(cfg) {
		if _, ok := sw.Tensors[want.name]; ok {
			names = append(names, want.name)
		} else if want.required {
			return nil, nil, fmt.Errorf("checkpoint has no tensor %q", want.name)
		}
	}
	c, err := loadRetainedTensors(sw, names)
	if err != nil {
		return nil, nil, err
	}
	m, err := retainedFromCheckpoint(c, cfg, receipt)
	if err != nil {
		return nil, nil, err
	}
	return m, cfg, nil
}
