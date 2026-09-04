//go:build linux

package main

import "fmt"

const (
	cudaLayers  = 43
	cudaWidth   = 4096
	cudaHCWidth = 4 * cudaWidth
	cudaKVWidth = 512
	cudaRawCap  = 128
)

// cudaLayer holds the device views of one transformer block. Layers 0 and 1
// use raw attention and so bind no compressor. Even layers from 2 additionally
// bind the indexer that selects compressed rows.
type cudaLayer struct {
	attnSinks cudaModelTensor

	attnCompressorAPE  cudaModelTensor
	attnCompressorNorm cudaModelTensor
	attnCompressorGate cudaModelTensor
	attnCompressorKV   cudaModelTensor

	indexerCompressorAPE  cudaModelTensor
	indexerCompressorNorm cudaModelTensor
	indexerCompressorGate cudaModelTensor
	indexerCompressorKV   cudaModelTensor
	indexerProj           cudaModelTensor
	indexerQB             cudaModelTensor

	attnKV     cudaModelTensor
	attnKVNorm cudaModelTensor
	attnNorm   cudaModelTensor
	attnOutA   cudaModelTensor
	attnOutB   cudaModelTensor
	attnQA     cudaModelTensor
	attnQANorm cudaModelTensor
	attnQB     cudaModelTensor

	expertBias  cudaModelTensor
	expertMap   cudaModelTensor
	expertGate  cudaModelTensor
	expertUp    cudaModelTensor
	expertDown  cudaModelTensor
	router      cudaModelTensor
	sharedGate  cudaModelTensor
	sharedUp    cudaModelTensor
	sharedDown  cudaModelTensor
	ffnNorm     cudaModelTensor
	hcAttnBase  cudaModelTensor
	hcAttnFn    cudaModelTensor
	hcAttnScale cudaModelTensor
	hcFFNBase   cudaModelTensor
	hcFFNFn     cudaModelTensor
	hcFFNScale  cudaModelTensor
}

// cudaOutput holds the device views used after the last layer.
type cudaOutput struct {
	embedding cudaModelTensor
	hcBase    cudaModelTensor
	hcFn      cudaModelTensor
	hcScale   cudaModelTensor
	norm      cudaModelTensor
	weight    cudaModelTensor
}

// cudaModel is the immutable device view of one checkpoint.
type cudaModel struct {
	layers [cudaLayers]cudaLayer
	output cudaOutput
}

// bindModel resolves every weight the layer loop reads. It requires host page
// tables: the bounded registration fallback can address only one small range,
// which is far less than a whole model.
func (r *cudaFoundation) bindModel() (*cudaModel, error) {
	if !r.hmm {
		return nil, fmt.Errorf("cuda: bind model: device does not address the model through host page tables")
	}
	m := new(cudaModel)
	bind := func(dst *cudaModelTensor, name string) error {
		t, err := r.tensor(name)
		if err != nil {
			return err
		}
		*dst = t
		return nil
	}
	for dst, name := range map[*cudaModelTensor]string{
		&m.output.embedding: "token_embd.weight",
		&m.output.hcBase:    "output_hc_base.weight",
		&m.output.hcFn:      "output_hc_fn.weight",
		&m.output.hcScale:   "output_hc_scale.weight",
		&m.output.norm:      "output_norm.weight",
		&m.output.weight:    "output.weight",
	} {
		if err := bind(dst, name); err != nil {
			return nil, err
		}
	}
	for i := range m.layers {
		if err := r.bindLayer(&m.layers[i], i, bind); err != nil {
			return nil, fmt.Errorf("cuda: bind layer %d: %w", i, err)
		}
	}
	return m, nil
}

// bindLayer resolves one block's weights. Its dispatch matches the contract:
// layers below 3 route through the identity map, layers from 2 add the
// attention compressor, and even layers from 2 add the indexer.
func (r *cudaFoundation) bindLayer(l *cudaLayer, i int, bind func(*cudaModelTensor, string) error) error {
	required := map[*cudaModelTensor]string{
		&l.attnSinks:   "attn_sinks.weight",
		&l.attnKV:      "attn_kv.weight",
		&l.attnKVNorm:  "attn_kv_a_norm.weight",
		&l.attnNorm:    "attn_norm.weight",
		&l.attnOutA:    "attn_output_a.weight",
		&l.attnOutB:    "attn_output_b.weight",
		&l.attnQA:      "attn_q_a.weight",
		&l.attnQANorm:  "attn_q_a_norm.weight",
		&l.attnQB:      "attn_q_b.weight",
		&l.expertGate:  "ffn_gate_exps.weight",
		&l.expertUp:    "ffn_up_exps.weight",
		&l.expertDown:  "ffn_down_exps.weight",
		&l.router:      "ffn_gate_inp.weight",
		&l.sharedGate:  "ffn_gate_shexp.weight",
		&l.sharedUp:    "ffn_up_shexp.weight",
		&l.sharedDown:  "ffn_down_shexp.weight",
		&l.ffnNorm:     "ffn_norm.weight",
		&l.hcAttnBase:  "hc_attn_base.weight",
		&l.hcAttnFn:    "hc_attn_fn.weight",
		&l.hcAttnScale: "hc_attn_scale.weight",
		&l.hcFFNBase:   "hc_ffn_base.weight",
		&l.hcFFNFn:     "hc_ffn_fn.weight",
		&l.hcFFNScale:  "hc_ffn_scale.weight",
	}
	if i < 3 {
		required[&l.expertMap] = "ffn_gate_tid2eid.weight"
	} else {
		required[&l.expertBias] = "exp_probs_b.bias"
	}
	if i >= 2 {
		required[&l.attnCompressorAPE] = "attn_compressor_ape.weight"
		required[&l.attnCompressorNorm] = "attn_compressor_norm.weight"
		required[&l.attnCompressorGate] = "attn_compressor_gate.weight"
		required[&l.attnCompressorKV] = "attn_compressor_kv.weight"
		if i%2 == 0 {
			required[&l.indexerCompressorAPE] = "indexer_compressor_ape.weight"
			required[&l.indexerCompressorNorm] = "indexer_compressor_norm.weight"
			required[&l.indexerCompressorGate] = "indexer_compressor_gate.weight"
			required[&l.indexerCompressorKV] = "indexer_compressor_kv.weight"
			required[&l.indexerProj] = "indexer.proj.weight"
			required[&l.indexerQB] = "indexer.attn_q_b.weight"
		}
	}
	prefix := fmt.Sprintf("blk.%d.", i)
	for dst, suffix := range required {
		if err := bind(dst, prefix+suffix); err != nil {
			return err
		}
	}
	return nil
}
