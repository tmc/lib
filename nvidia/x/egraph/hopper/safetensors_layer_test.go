package hopper

import (
	"bytes"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
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
			inNorm[i] = BF16Bits(float32(0.8 + 0.4*r.Float64()))
			postNorm[i] = BF16Bits(float32(0.8 + 0.4*r.Float64()))
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
			{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(inNorm)},
			{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.BF16, Shape: []int{G * D, H}, Data: qBytes},
			{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.BF16, Shape: []int{KVH * D, H}, Data: kBytes},
			{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.BF16, Shape: []int{KVH * D, H}, Data: vBytes},
			{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.BF16, Shape: []int{H, G * D}, Data: oBytes},
			{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(postNorm)},
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
		if w.Prefix != prefix {
			t.Errorf("Prefix = %q, want %q", w.Prefix, prefix)
		}
		if !slices.Equal(w.InputNorm, inNorm) {
			t.Errorf("InputNorm mismatch")
		}
		if !slices.Equal(w.PostAttnNorm, postNorm) {
			t.Errorf("PostAttnNorm mismatch")
		}
		wantQKV := slices.Concat(qBytes, kBytes, vBytes)
		if !slices.Equal(w.QKVWeight, wantQKV) {
			t.Errorf("QKVWeight concatenation mismatch: got %d bytes, want %d", len(w.QKVWeight), len(wantQKV))
		}
		wantGateUp := slices.Concat(gateBytes, upBytes)
		if !slices.Equal(w.GateUpWeight, wantGateUp) {
			t.Errorf("GateUpWeight concatenation mismatch: got %d bytes, want %d", len(w.GateUpWeight), len(wantGateUp))
		}
		if !slices.Equal(w.OWeight, oBytes) {
			t.Errorf("OWeight mismatch")
		}
		if !slices.Equal(w.DownWeight, downBytes) {
			t.Errorf("DownWeight mismatch")
		}

		// Verify EncodeLayerSafetensors round-trip.
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
				b[i] = E4M3Bits(float32(0.125 + 0.5*r.Float64()))
			}
			return b
		}
		inNorm := make([]uint16, H)
		postNorm := make([]uint16, H)
		for i := range H {
			inNorm[i] = BF16Bits(1.0)
			postNorm[i] = BF16Bits(1.0)
		}

		qDim := G * D
		kvDim := KVH * D
		qkvDim := qDim + 2*kvDim // 256, multiple of 128

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
			{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(inNorm)},
			{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.U8, Shape: []int{qDim, H / 2}, Data: qW},
			{Name: prefix + ".self_attn.q_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{qDim, H / 16}, Data: qS},
			{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: kW},
			{Name: prefix + ".self_attn.k_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: kS},
			{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: vW},
			{Name: prefix + ".self_attn.v_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: vS},
			{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.U8, Shape: []int{H, qDim / 2}, Data: oW},
			{Name: prefix + ".self_attn.o_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{H, qDim / 16}, Data: oS},
			{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(postNorm)},
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
			t.Fatalf("LoadLayerSafetensors(repackForWGMMA=true): %v", err)
		}
		if !w.RepackedNVFP4 {
			t.Fatalf("RepackedNVFP4 = false, want true")
		}

		wantQKVW := slices.Concat(qW, kW, vW)
		wantQKVS := slices.Concat(qS, kS, vS)
		wantGateUpW := slices.Concat(gateW, upW)
		wantGateUpS := slices.Concat(gateS, upS)

		// Verify that UnpackNVFP4Weights and UnpackNVFP4Scales recover original bytes bit-for-bit.
		unpackedQKVW, err := UnpackNVFP4Weights(w.QKVWeight, qkvDim, H)
		if err != nil {
			t.Fatalf("UnpackNVFP4Weights(QKV): %v", err)
		}
		if !slices.Equal(unpackedQKVW, wantQKVW) {
			t.Errorf("UnpackNVFP4Weights(QKV) did not recover original concatenated [Q; K; V] bytes")
		}
		unpackedQKVS, err := UnpackNVFP4Scales(w.QKVScale, qkvDim, H)
		if err != nil {
			t.Fatalf("UnpackNVFP4Scales(QKV): %v", err)
		}
		if !slices.Equal(unpackedQKVS, wantQKVS) {
			t.Errorf("UnpackNVFP4Scales(QKV) did not recover original concatenated [Q; K; V] scales")
		}

		unpackedOW, err := UnpackNVFP4Weights(w.OWeight, H, qDim)
		if err != nil {
			t.Fatalf("UnpackNVFP4Weights(O): %v", err)
		}
		if !slices.Equal(unpackedOW, oW) {
			t.Errorf("UnpackNVFP4Weights(O) mismatch")
		}
		unpackedOS, err := UnpackNVFP4Scales(w.OScale, H, qDim)
		if err != nil {
			t.Fatalf("UnpackNVFP4Scales(O): %v", err)
		}
		if !slices.Equal(unpackedOS, oS) {
			t.Errorf("UnpackNVFP4Scales(O) mismatch")
		}

		unpackedGUW, err := UnpackNVFP4Weights(w.GateUpWeight, 2*I, H)
		if err != nil {
			t.Fatalf("UnpackNVFP4Weights(GateUp): %v", err)
		}
		if !slices.Equal(unpackedGUW, wantGateUpW) {
			t.Errorf("UnpackNVFP4Weights(GateUp) did not recover original [Gate; Up] bytes")
		}
		unpackedGUS, err := UnpackNVFP4Scales(w.GateUpScale, 2*I, H)
		if err != nil {
			t.Fatalf("UnpackNVFP4Scales(GateUp): %v", err)
		}
		if !slices.Equal(unpackedGUS, wantGateUpS) {
			t.Errorf("UnpackNVFP4Scales(GateUp) did not recover original [Gate; Up] scales")
		}

		unpackedDW, err := UnpackNVFP4Weights(w.DownWeight, H, I)
		if err != nil {
			t.Fatalf("UnpackNVFP4Weights(Down): %v", err)
		}
		if !slices.Equal(unpackedDW, downW) {
			t.Errorf("UnpackNVFP4Weights(Down) mismatch")
		}
		unpackedDS, err := UnpackNVFP4Scales(w.DownScale, H, I)
		if err != nil {
			t.Fatalf("UnpackNVFP4Scales(Down): %v", err)
		}
		if !slices.Equal(unpackedDS, downS) {
			t.Errorf("UnpackNVFP4Scales(Down) mismatch")
		}

		// Verify EncodeLayerSafetensors preserves repacked weights and can unpack back when repackForWGMMA=false.
		encRepacked, err := EncodeLayerSafetensors(w, cfg)
		if err != nil {
			t.Fatalf("EncodeLayerSafetensors(repacked): %v", err)
		}
		wUnpacked, err := LoadLayerSafetensors(encRepacked, cfg, prefix, false)
		if err != nil {
			t.Fatalf("LoadLayerSafetensors(encRepacked, false): %v", err)
		}
		if wUnpacked.RepackedNVFP4 {
			t.Errorf("wUnpacked.RepackedNVFP4 = true, want false")
		}
		if !slices.Equal(wUnpacked.QKVWeight, wantQKVW) || !slices.Equal(wUnpacked.GateUpWeight, wantGateUpW) {
			t.Errorf("LoadLayerSafetensors unpack-from-cached-safetensors mismatch")
		}
	})
}

func TestRunCompiledDenseLayer(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}

	const (
		batchM = 2
		H      = 64
		I      = 128
		G      = 4
		KVH    = 2
		D      = 16
		seqLen = 32
		eps    = 1e-5
	)
	cfg := ArchConfig{
		HiddenSize:        H,
		IntermediateSize:  I,
		NumHiddenLayers:   1,
		NumAttentionHeads: G,
		NumKeyValueHeads:  KVH,
		HeadDim:           D,
		RMSNormEps:        eps,
		WeightDType:       BF16,
		KVDType:           BF16,
	}

	r := rand.New(rand.NewPCG(701, 702))
	randBF16Slice := func(n int, scale float32) ([]uint16, []float64) {
		u := make([]uint16, n)
		f := make([]float64, n)
		for i := range n {
			v := BF16.Round(float32(r.NormFloat64()) * scale)
			u[i] = BF16Bits(v)
			f[i] = float64(v)
		}
		return u, f
	}

	qDim := G * HeadDim
	kvDim := KVH * HeadDim

	inNormU16, inNormF64 := randBF16Slice(H, 0.2)
	postNormU16, postNormF64 := randBF16Slice(H, 0.2)
	wqU16, wqF64 := randBF16Slice(qDim*H, 0.2)
	wkU16, _ := randBF16Slice(kvDim*H, 0.2)
	wvU16, _ := randBF16Slice(kvDim*H, 0.2)
	woU16, woF64 := randBF16Slice(H*qDim, 0.2)
	wGateU16, wGateF64 := randBF16Slice(I*H, 0.2)
	wUpU16, wUpF64 := randBF16Slice(I*H, 0.2)
	wDownU16, wDownF64 := randBF16Slice(H*I, 0.2)

	prefix := "model.layers.0"
	tensors := []safetensors.Tensor{
		{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(inNormU16)},
		{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.BF16, Shape: []int{qDim, H}, Data: encodeU16LE(wqU16)},
		{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.BF16, Shape: []int{kvDim, H}, Data: encodeU16LE(wkU16)},
		{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.BF16, Shape: []int{kvDim, H}, Data: encodeU16LE(wvU16)},
		{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.BF16, Shape: []int{H, qDim}, Data: encodeU16LE(woU16)},
		{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(postNormU16)},
		{Name: prefix + ".mlp.gate_proj.weight", Dtype: safetensors.BF16, Shape: []int{I, H}, Data: encodeU16LE(wGateU16)},
		{Name: prefix + ".mlp.up_proj.weight", Dtype: safetensors.BF16, Shape: []int{I, H}, Data: encodeU16LE(wUpU16)},
		{Name: prefix + ".mlp.down_proj.weight", Dtype: safetensors.BF16, Shape: []int{H, I}, Data: encodeU16LE(wDownU16)},
	}
	var stBuf bytes.Buffer
	if err := safetensors.Write(&stBuf, tensors, nil); err != nil {
		t.Fatalf("safetensors.Write: %v", err)
	}

	weights, err := LoadLayerSafetensors(stBuf.Bytes(), cfg, prefix, false)
	if err != nil {
		t.Fatalf("LoadLayerSafetensors: %v", err)
	}

	layer, err := CompileLayer(cfg, batchM, seqLen, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileLayer: %v", err)
	}

	hiddenIn := make([]float32, batchM*H)
	hiddenF64 := make([]float64, batchM*H)
	for i := range hiddenIn {
		v := BF16.Round(float32(r.NormFloat64() * 0.2))
		hiddenIn[i] = v
		hiddenF64[i] = float64(v)
	}

	const pageSize = 16
	maxPages := max((seqLen+pageSize-1)/pageSize, 1)
	cacheElems := batchM * maxPages * pageSize * KVH * HeadDim
	kCache := make([]float32, cacheElems)
	vCache := make([]float32, cacheElems)
	kCacheF64 := make([]float64, cacheElems)
	vCacheF64 := make([]float64, cacheElems)
	for i := range cacheElems {
		kv := E4M3.Round(float32(r.NormFloat64() * 0.25))
		vv := E4M3.Round(float32(r.NormFloat64() * 0.25))
		kCache[i] = kv
		vCache[i] = vv
		kCacheF64[i] = float64(E4M3Bits(kv))
		vCacheF64[i] = float64(E4M3Bits(vv))
	}

	pageTable := make([]int32, batchM*maxPages)
	pageTableF64 := make([]float64, batchM*maxPages)
	for i := range pageTable {
		pageTable[i] = int32(i)
		pageTableF64[i] = float64(i)
	}
	seqLens := []int32{24, 19}
	seqLensF64 := []float64{24, 19}

	got, err := RunCompiledDenseLayer(t.Context(), layer, weights, hiddenIn, kCache, vCache, pageTable, seqLens)
	if err != nil {
		t.Fatalf("RunCompiledDenseLayer: %v", err)
	}
	if len(got) != batchM*H {
		t.Fatalf("len(got) = %d, want %d", len(got), batchM*H)
	}

	env := map[string]Value{
		"x":            {Shape: []int{batchM, H}, Data: hiddenF64},
		"attn_norm_w":  {Shape: []int{H}, Data: inNormF64},
		"wq":           {Shape: []int{qDim, H}, Data: wqF64},
		"k_cache":      {Shape: []int{batchM * maxPages * pageSize, KVH, HeadDim}, Data: kCacheF64},
		"v_cache":      {Shape: []int{batchM * maxPages * pageSize, KVH, HeadDim}, Data: vCacheF64},
		"page_table":   {Shape: []int{batchM, maxPages}, Data: pageTableF64},
		"seqlens":      {Shape: []int{batchM}, Data: seqLensF64},
		"k_scale":      {Shape: []int{1}, Data: []float64{1.0}},
		"v_scale":      {Shape: []int{1}, Data: []float64{1.0}},
		"wo":           {Shape: []int{H, qDim}, Data: woF64},
		"ffn_norm_w":   {Shape: []int{H}, Data: postNormF64},
		"w_gate_up":    {Shape: []int{2 * I, H}, Data: slices.Concat(wGateF64, wUpF64)},
		"w_down":       {Shape: []int{H, I}, Data: wDownF64},
		"post_norm_w":  {Shape: []int{H}, Data: postNormF64},
	}
	wantRef := denseLayerGoReference(t, cfg, batchM, env)

	gotF64 := make([]float64, len(got))
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("got[%d] = %v is not finite", i, v)
		}
		gotF64[i] = float64(v)
	}
	if e := relErr(gotF64, wantRef); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("RunCompiledDenseLayer vs Go float64 reference relative L2 error = %.2e > 5e-3\ngot:  %v\nwant: %v",
			e, gotF64[:8], wantRef[:8])
	}

	t.Run("nvfp4_repacked_end_to_end", func(t *testing.T) {
		nvCfg := cfg
		nvCfg.WeightDType = NVFP4
		nvCfg.KVDType = E4M3

		randNVFP4Tensor := func(rows, cols int) (packed, scales []byte) {
			packed = make([]byte, rows*cols/2)
			scales = make([]byte, rows*cols/16)
			for i := range packed {
				packed[i] = byte(r.IntN(256))
			}
			for i := range scales {
				scales[i] = E4M3Bits(float32(0.0625 + 0.125*r.Float64()))
			}
			return packed, scales
		}
		qP, qS := randNVFP4Tensor(qDim, H)
		kP, kS := randNVFP4Tensor(kvDim, H)
		vP, vS := randNVFP4Tensor(kvDim, H)
		oP, oS := randNVFP4Tensor(H, qDim)
		gP, gS := randNVFP4Tensor(I, H)
		uP, uS := randNVFP4Tensor(I, H)
		dP, dS := randNVFP4Tensor(H, I)

		nvTensors := []safetensors.Tensor{
			{Name: prefix + ".input_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(inNormU16)},
			{Name: prefix + ".self_attn.q_proj.weight", Dtype: safetensors.U8, Shape: []int{qDim, H / 2}, Data: qP},
			{Name: prefix + ".self_attn.q_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{qDim, H / 16}, Data: qS},
			{Name: prefix + ".self_attn.k_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: kP},
			{Name: prefix + ".self_attn.k_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: kS},
			{Name: prefix + ".self_attn.v_proj.weight", Dtype: safetensors.U8, Shape: []int{kvDim, H / 2}, Data: vP},
			{Name: prefix + ".self_attn.v_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{kvDim, H / 16}, Data: vS},
			{Name: prefix + ".self_attn.o_proj.weight", Dtype: safetensors.U8, Shape: []int{H, qDim / 2}, Data: oP},
			{Name: prefix + ".self_attn.o_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{H, qDim / 16}, Data: oS},
			{Name: prefix + ".post_attention_layernorm.weight", Dtype: safetensors.BF16, Shape: []int{H}, Data: encodeU16LE(postNormU16)},
			{Name: prefix + ".mlp.gate_proj.weight", Dtype: safetensors.U8, Shape: []int{I, H / 2}, Data: gP},
			{Name: prefix + ".mlp.gate_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{I, H / 16}, Data: gS},
			{Name: prefix + ".mlp.up_proj.weight", Dtype: safetensors.U8, Shape: []int{I, H / 2}, Data: uP},
			{Name: prefix + ".mlp.up_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{I, H / 16}, Data: uS},
			{Name: prefix + ".mlp.down_proj.weight", Dtype: safetensors.U8, Shape: []int{H, I / 2}, Data: dP},
			{Name: prefix + ".mlp.down_proj.weight_scale", Dtype: safetensors.F8E4M3, Shape: []int{H, I / 16}, Data: dS},
		}
		var nvBuf bytes.Buffer
		if err := safetensors.Write(&nvBuf, nvTensors, nil); err != nil {
			t.Fatalf("safetensors.Write(nvfp4): %v", err)
		}

		nvWeights, err := LoadLayerSafetensors(nvBuf.Bytes(), nvCfg, prefix, true)
		if err != nil {
			t.Fatalf("LoadLayerSafetensors(nvfp4, repack=true): %v", err)
		}
		nvLayer, err := CompileLayer(nvCfg, batchM, seqLen, "NVIDIA H200")
		if err != nil {
			t.Fatalf("CompileLayer(nvfp4): %v", err)
		}
		gotNV, err := RunCompiledDenseLayer(t.Context(), nvLayer, nvWeights, hiddenIn, kCache, vCache, pageTable, seqLens)
		if err != nil {
			t.Fatalf("RunCompiledDenseLayer(nvfp4): %v", err)
		}

		dequantMat := func(packed, scales []byte, rows, cols int) []float64 {
			out := make([]float64, rows*cols)
			rowDst := make([]float32, cols)
			for row := range rows {
				pRow := packed[row*(cols/2) : (row+1)*(cols/2)]
				sRow := scales[row*(cols/16) : (row+1)*(cols/16)]
				if err := DequantNVFP4(rowDst, pRow, sRow, 1.0); err != nil {
					t.Fatal(err)
				}
				for col, v := range rowDst {
					out[row*cols+col] = float64(BF16.Round(v))
				}
			}
			return out
		}
		nvEnv := map[string]Value{
			"x":            {Shape: []int{batchM, H}, Data: hiddenF64},
			"attn_norm_w":  {Shape: []int{H}, Data: inNormF64},
			"wq":           {Shape: []int{qDim, H}, Data: dequantMat(qP, qS, qDim, H)},
			"k_cache":      {Shape: []int{batchM * maxPages * pageSize, KVH, HeadDim}, Data: kCacheF64},
			"v_cache":      {Shape: []int{batchM * maxPages * pageSize, KVH, HeadDim}, Data: vCacheF64},
			"page_table":   {Shape: []int{batchM, maxPages}, Data: pageTableF64},
			"seqlens":      {Shape: []int{batchM}, Data: seqLensF64},
			"k_scale":      {Shape: []int{1}, Data: []float64{1.0}},
			"v_scale":      {Shape: []int{1}, Data: []float64{1.0}},
			"wo":           {Shape: []int{H, qDim}, Data: dequantMat(oP, oS, H, qDim)},
			"ffn_norm_w":   {Shape: []int{H}, Data: postNormF64},
			"w_gate_up":    {Shape: []int{2 * I, H}, Data: dequantMat(slices.Concat(gP, uP), slices.Concat(gS, uS), 2*I, H)},
			"w_down":       {Shape: []int{H, I}, Data: dequantMat(dP, dS, H, I)},
			"post_norm_w":  {Shape: []int{H}, Data: postNormF64},
		}
		wantNVRef := denseLayerGoReference(t, cfg, batchM, nvEnv)
		gotNVF64 := make([]float64, len(gotNV))
		for i, v := range gotNV {
			gotNVF64[i] = float64(v)
		}
		if e := relErr(gotNVF64, wantNVRef); e > 5e-3 || math.IsNaN(e) {
			t.Fatalf("RunCompiledDenseLayer(nvfp4) vs Go float64 reference relative L2 error = %.2e > 5e-3", e)
		}
	})
}
