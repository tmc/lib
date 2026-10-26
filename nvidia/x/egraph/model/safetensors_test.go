package model

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

func TestLoadLayerSafetensors(t *testing.T) {
	t.Run("separate_qkv_and_gate_up_bf16", func(t *testing.T) {
		const (
			H   = 128
			I   = 256
			G   = 2
			KVH = 1
			D   = 64
		)
		cfg := ArchConfig{
			HiddenSize:        H,
			IntermediateSize:  I,
			NumHiddenLayers:   1,
			NumAttentionHeads: G,
			NumKeyValueHeads:  KVH,
			HeadDim:           D,
			RMSNormEps:        1e-5,
			WeightDType:       BF16,
			KVDType:           BF16,
		}

		r := rand.New(rand.NewPCG(101, 202))
		randBytes := func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(r.IntN(256))
			}
			return b
		}
		inNorm := make([]uint16, H)
		postNorm := make([]uint16, H)
		for i := range H {
			inNorm[i] = tilebridge.BF16Bits(float32(0.8 + 0.4*r.Float64()))
			postNorm[i] = tilebridge.BF16Bits(float32(0.8 + 0.4*r.Float64()))
		}

		qBytes := randBytes(G * D * H * 2)
		kBytes := randBytes(KVH * D * H * 2)
		vBytes := randBytes(KVH * D * H * 2)
		oBytes := randBytes(H * G * D * 2)
		gateBytes := randBytes(I * H * 2)
		upBytes := randBytes(I * H * 2)
		downBytes := randBytes(H * I * 2)

		prefix := "model.layers.0"
		tensors := []safetensors.Tensor{
			{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: EncodeU16LE(inNorm)},
			{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.BF16, Shape: []int{G * D, H}, Data: qBytes},
			{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.BF16, Shape: []int{KVH * D, H}, Data: kBytes},
			{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.BF16, Shape: []int{KVH * D, H}, Data: vBytes},
			{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.BF16, Shape: []int{H, G * D}, Data: oBytes},
			{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: EncodeU16LE(postNorm)},
			{Name: prefix + ".mlp.gate_proj.weight", Dtype: safetensors.BF16, Shape: []int{I, H}, Data: gateBytes},
			{Name: prefix + ".mlp.up_proj.weight", Dtype: safetensors.BF16, Shape: []int{I, H}, Data: upBytes},
			{Name: prefix + ".mlp.down_proj.weight", Dtype: safetensors.BF16, Shape: []int{H, I}, Data: downBytes},
		}
		var buf bytes.Buffer
		if err := safetensors.Write(&buf, tensors, nil); err != nil {
			t.Fatalf("safetensors.Write: %v", err)
		}

		w, err := LoadLayerSafetensors(buf.Bytes(), cfg, prefix, false)
		if err != nil {
			t.Fatalf("LoadLayerSafetensors: %v", err)
		}
		wantQKV := slices.Concat(qBytes, kBytes, vBytes)
		wantGateUp := slices.Concat(gateBytes, upBytes)
		if !slices.Equal(w.QKVWeight, wantQKV) || !slices.Equal(w.GateUpWeight, wantGateUp) {
			t.Fatalf("QKV or GateUp concatenation mismatch")
		}

		encoded, err := EncodeLayerSafetensors(w, cfg)
		if err != nil {
			t.Fatalf("EncodeLayerSafetensors: %v", err)
		}
		w2, err := LoadLayerSafetensors(encoded, cfg, prefix, false)
		if err != nil {
			t.Fatalf("LoadLayerSafetensors(encoded): %v", err)
		}
		if !slices.Equal(w2.QKVWeight, wantQKV) || !slices.Equal(w2.GateUpWeight, wantGateUp) {
			t.Errorf("round-trip QKV or GateUp mismatch")
		}
	})

	t.Run("nvfp4_wgmma_repack_roundtrip", func(t *testing.T) {
		const (
			H   = 128
			I   = 128
			G   = 2
			KVH = 1
			D   = 64
		)
		cfg := ArchConfig{
			HiddenSize:        H,
			IntermediateSize:  I,
			NumHiddenLayers:   1,
			NumAttentionHeads: G,
			NumKeyValueHeads:  KVH,
			HeadDim:           D,
			RMSNormEps:        1e-5,
			WeightDType:       NVFP4,
			KVDType:           E4M3,
		}

		r := rand.New(rand.NewPCG(303, 404))
		randBytes := func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(r.IntN(256))
			}
			return b
		}
		randScales := func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = tilebridge.E4M3Bits(float32(0.125 + 0.5*r.Float64()))
			}
			return b
		}
		inNorm := make([]uint16, H)
		postNorm := make([]uint16, H)
		for i := range H {
			inNorm[i] = tilebridge.BF16Bits(1.0)
			postNorm[i] = tilebridge.BF16Bits(1.0)
		}

		qDim := G * D
		kvDim := KVH * D
		qkvDim := qDim + 2*kvDim

		qW := randBytes(qDim * H / 2)
		kW := randBytes(kvDim * H / 2)
		vW := randBytes(kvDim * H / 2)
		qS := randScales(qDim * H / 16)
		kS := randScales(kvDim * H / 16)
		vS := randScales(kvDim * H / 16)
		oW := randBytes(H * qDim / 2)
		oS := randScales(H * qDim / 16)
		gateW := randBytes(I * H / 2)
		upW := randBytes(I * H / 2)
		gateS := randScales(I * H / 16)
		upS := randScales(I * H / 16)
		downW := randBytes(H * I / 2)
		downS := randScales(H * I / 16)

		prefix := "model.layers.0"
		tensors := []safetensors.Tensor{
			{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: EncodeU16LE(inNorm)},
			{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.U8, Shape: []int{qDim, H / 2}, Data: qW},
			{Name: prefix + ".self_attn.q_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{qDim, H / 16}, Data: qS},
			{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: kW},
			{Name: prefix + ".self_attn.k_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: kS},
			{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: vW},
			{Name: prefix + ".self_attn.v_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: vS},
			{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.U8, Shape: []int{H, qDim / 2}, Data: oW},
			{Name: prefix + ".self_attn.o_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{H, qDim / 16}, Data: oS},
			{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: EncodeU16LE(postNorm)},
			{Name: prefix + ".mlp.gate_proj.weight", Dtype: safetensors.U8, Shape: []int{I, H / 2}, Data: gateW},
			{Name: prefix + ".mlp.gate_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{I, H / 16}, Data: gateS},
			{Name: prefix + ".mlp.up_proj.weight", Dtype: safetensors.U8, Shape: []int{I, H / 2}, Data: upW},
			{Name: prefix + ".mlp.up_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{I, H / 16}, Data: upS},
			{Name: prefix + ".mlp.down_proj.weight", Dtype: safetensors.U8, Shape: []int{H, I / 2}, Data: downW},
			{Name: prefix + ".mlp.down_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{H, I / 16}, Data: downS},
		}
		var buf bytes.Buffer
		if err := safetensors.Write(&buf, tensors, nil); err != nil {
			t.Fatalf("safetensors.Write: %v", err)
		}

		w, err := LoadLayerSafetensors(buf.Bytes(), cfg, prefix, true)
		if err != nil {
			t.Fatalf("LoadLayerSafetensors(repack=true): %v", err)
		}
		if !w.RepackedNVFP4 {
			t.Fatalf("RepackedNVFP4 = false, want true")
		}

		wantQKVW := slices.Concat(qW, kW, vW)
		unpackedQKVW, err := UnpackNVFP4Weights(w.QKVWeight, qkvDim, H)
		if err != nil {
			t.Fatalf("UnpackNVFP4Weights(QKV): %v", err)
		}
		if !slices.Equal(unpackedQKVW, wantQKVW) {
			t.Errorf("UnpackNVFP4Weights(QKV) mismatch")
		}
	})
}

func ExampleRepackNVFP4Weights() {
	packed := make([]byte, 16*16/2)
	for i := range packed {
		packed[i] = byte(i + 1)
	}
	repacked, err := RepackNVFP4Weights(packed, 16, 16)
	if err != nil {
		panic(err)
	}
	unpacked, err := UnpackNVFP4Weights(repacked, 16, 16)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(repacked), slices.Equal(packed, unpacked))
	// Output:
	// 128 true
}
