package main

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestForwardPass(t *testing.T) {
	cfg := DefaultModelConfig()
	weights := GenerateDeterministicWeights(&cfg)
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	// Step 0 forward pass
	logits0 := Forward(weights, &cfg, kvCache, 42, 0)
	if len(logits0) != cfg.VocabSize {
		t.Fatalf("expected %d logits, got %d", cfg.VocabSize, len(logits0))
	}

	tok0 := sampleArgmax(logits0)
	if tok0 < 0 || int(tok0) >= cfg.VocabSize {
		t.Errorf("sampled token out of bounds: %d", tok0)
	}

	// Step 1 forward pass
	logits1 := Forward(weights, &cfg, kvCache, tok0, 1)
	if len(logits1) != cfg.VocabSize {
		t.Fatalf("expected %d logits, got %d", cfg.VocabSize, len(logits1))
	}

	tok1 := sampleArgmax(logits1)
	if tok1 < 0 || int(tok1) >= cfg.VocabSize {
		t.Errorf("sampled token out of bounds: %d", tok1)
	}
}

func TestByteTokenizer(t *testing.T) {
	tok := NewByteTokenizer()
	prompt := "Hello LLaMA 3.2"

	tokens, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, err := tok.Decode(tokens)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if decoded != prompt {
		t.Errorf("expected %q, got %q", prompt, decoded)
	}
}

func TestHFTokenizerParsing(t *testing.T) {
	tmpDir := t.TempDir()
	tokFile := filepath.Join(tmpDir, "tokenizer.json")

	jsonContent := `{
		"model": {
			"vocab": {
				"Hello": 100,
				" world": 101,
				"!": 102
			},
			"merges": ["H ello"]
		},
		"added_tokens": [
			{"id": 0, "content": "<unk>"},
			{"id": 1, "content": "<s>"}
		]
	}`
	if err := os.WriteFile(tokFile, []byte(jsonContent), 0644); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}

	tok, err := LoadHFTokenizer(tokFile)
	if err != nil {
		t.Fatalf("LoadHFTokenizer failed: %v", err)
	}

	if tok.VocabSize() < 5 {
		t.Errorf("expected at least 5 tokens in vocab, got %d", tok.VocabSize())
	}

	decoded, err := tok.Decode([]int32{100, 101, 102})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if decoded != "Hello world!" {
		t.Errorf("expected 'Hello world!', got %q", decoded)
	}
}

func TestRoPEScalingConfigAndFreq(t *testing.T) {
	const (
		theta   = float32(500000.0)
		headDim = 64
	)
	scaling := &RoPEScalingConfig{
		RoPEType:                      "llama3",
		Factor:                        32.0,
		LowFreqFactor:                 1.0,
		HighFreqFactor:                4.0,
		OriginalMaxPositionEmbeddings: 8192,
	}
	if err := scaling.Validate(); err != nil {
		t.Fatalf("Validate valid Llama-3 RoPEScalingConfig: %v", err)
	}

	// Band 1: high-frequency band (d=0), wavelen = 2*pi / 1.0 = 6.2831855 < 8192/4 = 2048 -> unscaled.
	gotHigh := ropeInvFreq(0, headDim, theta, scaling)
	if gotHigh != 1.0 {
		t.Errorf("high-freq band (d=0): got %g, want 1.0", gotHigh)
	}

	// Band 2: intermediate smooth-interpolation band (d=16, 2*d/headDim = 0.5).
	// Hand-computed reference:
	//   unscaled = 1 / 500000^(32/64) = 1 / sqrt(500000)
	//   wavelen  = 2*pi / unscaled (~4442.88, between 2048 and 8192)
	//   smooth   = (8192/wavelen - 1) / (4 - 1)
	//   want     = (1 - smooth)*(unscaled / 32) + smooth*unscaled
	unscaledMid := float32(1.0 / math.Sqrt(float64(theta)))
	wavelenMid := float32(2.0*math.Pi) / unscaledMid
	if !(wavelenMid >= 2048.0 && wavelenMid <= 8192.0) {
		t.Fatalf("d=16 wavelen = %g, expected in [2048, 8192]", wavelenMid)
	}
	smooth := (float32(8192)/wavelenMid - 1.0) / (4.0 - 1.0)
	wantMid := (1.0-smooth)*(unscaledMid/32.0) + smooth*unscaledMid
	gotMid := ropeInvFreq(16, headDim, theta, scaling)
	if diff := math.Abs(float64(gotMid - wantMid)); diff > 1e-7 {
		t.Errorf("intermediate band (d=16): got %g, want %g (diff %g)", gotMid, wantMid, diff)
	}
	// Must-fail control: inverted smooth factor (smooth*(unscaled/factor) + (1-smooth)*unscaled)
	// or unscaled frequency must differ from gotMid by > 1e-4.
	invertedMid := smooth*(unscaledMid/32.0) + (1.0-smooth)*unscaledMid
	if math.Abs(float64(gotMid-invertedMid)) <= 1e-4 {
		t.Errorf("must-fail control: inverted smooth interpolation too close to reference (%g vs %g)", gotMid, invertedMid)
	}
	if math.Abs(float64(gotMid-unscaledMid)) <= 1e-4 {
		t.Errorf("must-fail control: unscaled mid frequency too close to scaled (%g vs %g)", gotMid, unscaledMid)
	}

	// Band 3: low-frequency band (d=30), wavelen > 8192 -> unscaled / factor.
	unscaledLow := float32(1.0 / math.Pow(float64(theta), float64(60)/float64(headDim)))
	wavelenLow := float32(2.0*math.Pi) / unscaledLow
	if !(wavelenLow > 8192.0) {
		t.Fatalf("d=30 wavelen = %g, expected > 8192", wavelenLow)
	}
	wantLow := unscaledLow / 32.0
	gotLow := ropeInvFreq(30, headDim, theta, scaling)
	if diff := math.Abs(float64(gotLow - wantLow)); diff > 1e-7 {
		t.Errorf("low-freq band (d=30): got %g, want %g (diff %g)", gotLow, wantLow, diff)
	}

	// Verify applyRoPEScaled vs applyRoPE at pos=32 on the TestApplyRoPEScaledParity fixture.
	qInit := make([]float32, 2*headDim)
	for i := range qInit {
		qInit[i] = 0.2 + 0.03*float32(i)
	}
	qInit[30+headDim/2] = 15.0
	wantQ, unscaledQ := slices.Clone(qInit), slices.Clone(qInit)
	applyRoPEScaled(wantQ, 32, 2, headDim, theta, scaling)
	applyRoPE(unscaledQ, 32, 2, headDim, theta)
	for _, bandDim := range []int{16, 30} {
		if diff := math.Abs(float64(wantQ[bandDim] - unscaledQ[bandDim])); diff <= 1e-3 {
			t.Fatalf("must-fail control at d=%d: |wantQ[%d] - unscaledQ[%d]| = %g, want > 1e-3", bandDim, bandDim, bandDim, diff)
		}
	}

	// Verify Forward and FusedDecodeLayerPlan.WithRoPEScaling agree on CPU and differ from unscaled at pos > 0.
	fwdScaling := &RoPEScalingConfig{
		RoPEType:                      "llama3",
		Factor:                        8.0,
		LowFreqFactor:                 1.0,
		HighFreqFactor:                4.0,
		OriginalMaxPositionEmbeddings: 16,
	}
	baseCfg := ModelConfig{
		VocabSize:         64,
		HiddenSize:        64,
		IntermediateSize:  128,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RmsNormEps:        1e-5,
		RopeTheta:         100.0,
		MaxSeqLen:         32,
	}
	scaledCfg := baseCfg
	scaledCfg.RoPEScaling = fwdScaling
	w := distinctWeights(&scaledCfg)
	for _, m := range [][]float32{w.LMHead, w.Layers[0].WQ, w.Layers[0].WK, w.Layers[0].WV, w.Layers[0].WO} {
		for i := range m {
			m[i] *= 10
		}
	}

	kvScaled := NewKVCacheState(1, scaledCfg.MaxSeqLen, 64)
	kvUnscaled := NewKVCacheState(1, baseCfg.MaxSeqLen, 64)
	_ = Forward(w, &scaledCfg, kvScaled, 3, 0)
	_ = Forward(w, &baseCfg, kvUnscaled, 3, 0)
	logitsScaled := Forward(w, &scaledCfg, kvScaled, 5, 1)
	logitsUnscaled := Forward(w, &baseCfg, kvUnscaled, 5, 1)
	maxDiff := 0.0
	for i := range logitsScaled {
		if d := math.Abs(float64(logitsScaled[i] - logitsUnscaled[i])); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff <= 1e-4 {
		t.Errorf("must-fail control: scaled vs unscaled Forward max logit diff = %g, want > 1e-4", maxDiff)
	}

	plan, err := NewFusedDecodeLayerPlan(scaledCfg.HiddenSize, scaledCfg.IntermediateSize, scaledCfg.NumAttentionHeads, scaledCfg.NumKeyValueHeads, scaledCfg.HeadDim(), scaledCfg.MaxSeqLen, scaledCfg.RmsNormEps, scaledCfg.RopeTheta)
	if err != nil {
		t.Fatal(err)
	}
	plan, err = plan.WithRoPEScaling(fwdScaling)
	if err != nil {
		t.Fatal(err)
	}
	cache := attnCache{
		k: make([]float32, scaledCfg.MaxSeqLen*64),
		v: make([]float32, scaledCfg.MaxSeqLen*64),
	}
	lw := &w.Layers[0]
	bufs := func(tok int) fusedDecodeLayerBuffers {
		return fusedDecodeLayerBuffers{
			x:        slices.Clone(w.EmbedTokens[tok*64 : (tok+1)*64]),
			attnNorm: lw.InputNorm,
			wq:       lw.WQ,
			wk:       lw.WK,
			wv:       lw.WV,
			qBias:    lw.QBias,
			kBias:    lw.KBias,
			vBias:    lw.VBias,
			wo:       lw.WO,
			ffnNorm:  lw.PostNorm,
			wGate:    lw.WGate,
			wUp:      lw.WUp,
			wDown:    lw.WDown,
		}
	}
	if _, err := executeFusedDecodeLayerCPU(plan, bufs(3), cache, 0); err != nil {
		t.Fatal(err)
	}
	out1, err := executeFusedDecodeLayerCPU(plan, bufs(5), cache, 1)
	if err != nil {
		t.Fatal(err)
	}
	normOut := rmsNorm(out1.Out, w.Norm, scaledCfg.RmsNormEps)
	planLogits := matmulVec(normOut, w.LMHead, scaledCfg.HiddenSize, scaledCfg.VocabSize)
	for i := range planLogits {
		if d := math.Abs(float64(planLogits[i] - logitsScaled[i])); d > 1e-5 {
			t.Fatalf("FusedDecodeLayerPlan.WithRoPEScaling logit[%d] = %g, Forward = %g (diff %g)", i, planLogits[i], logitsScaled[i], d)
		}
	}

	// Test LoadModelConfig parsing of Llama-3.2-1B-style config.json with rope_scaling.
	dir := t.TempDir()
	cfgJSON := `{
		"vocab_size": 128256,
		"hidden_size": 2048,
		"intermediate_size": 8192,
		"num_hidden_layers": 16,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"rms_norm_eps": 1e-5,
		"rope_theta": 500000.0,
		"max_position_embeddings": 131072,
		"rope_scaling": {
			"factor": 32.0,
			"high_freq_factor": 4.0,
			"low_freq_factor": 1.0,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgJSON), 0644); err != nil {
		t.Fatal(err)
	}
	loadedCfg, err := LoadModelConfig(dir)
	if err != nil {
		t.Fatalf("LoadModelConfig: %v", err)
	}
	if loadedCfg.RoPEScaling == nil || *loadedCfg.RoPEScaling != *scaling {
		t.Errorf("LoadModelConfig RoPEScaling = %+v, want %+v", loadedCfg.RoPEScaling, scaling)
	}

	// Invalid rope_scaling configs must be rejected.
	for _, bad := range []RoPEScalingConfig{
		{RoPEType: "linear", Factor: 32, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 8192},
		{RoPEType: "llama3", Factor: 0, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 8192},
		{RoPEType: "llama3", Factor: 32, LowFreqFactor: 4, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 8192},
		{RoPEType: "llama3", Factor: 32, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 0},
	} {
		b := bad
		if err := b.Validate(); err == nil {
			t.Errorf("Validate(%+v) succeeded, want error", b)
		}
	}
}

func TestParseTokenIDs(t *testing.T) {
	got, err := parseTokenIDs("128000,791,6864,315,9822,374", 128256)
	if err != nil {
		t.Fatalf("parseTokenIDs valid: %v", err)
	}
	want := []int32{128000, 791, 6864, 315, 9822, 374}
	if !slices.Equal(got, want) {
		t.Errorf("parseTokenIDs = %v, want %v", got, want)
	}

	spaced, err := parseTokenIDs(" 1 , 42 , 255 ", 256)
	if err != nil || !slices.Equal(spaced, []int32{1, 42, 255}) {
		t.Errorf("parseTokenIDs spaced = %v, %v; want [1 42 255], nil", spaced, err)
	}

	for _, bad := range []string{
		"",
		"   ",
		"1,,2",
		",1,2",
		"1,2,",
		"1,abc,2",
		"-1,2",
		"1,256",
		"999999999999",
	} {
		if ids, err := parseTokenIDs(bad, 256); err == nil {
			t.Errorf("parseTokenIDs(%q, 256) = %v, nil; want error", bad, ids)
		}
	}
}

func TestRunWithTokenIDs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	stdout := os.Stdout
	os.Stdout = devNull
	t.Cleanup(func() {
		os.Stdout = stdout
		if err := devNull.Close(); err != nil {
			t.Error(err)
		}
	})

	saveModel, saveTokIDs, saveSteps := *modelPath, *tokenIDsFlag, *stepsFlag
	t.Cleanup(func() {
		*modelPath, *tokenIDsFlag, *stepsFlag = saveModel, saveTokIDs, saveSteps
	})
	*modelPath = ""
	*stepsFlag = 2
	*tokenIDsFlag = "1,2,3"
	if err := run(); err != nil {
		t.Fatalf("run with valid -token-ids: %v", err)
	}

	*tokenIDsFlag = "1,99999"
	if err := run(); err == nil {
		t.Fatal("run with out-of-range -token-ids succeeded, want error")
	}
}

func BenchmarkByteTokenizerEncode(b *testing.B) {
	tok := NewByteTokenizer()
	prompt := "The quick brown fox jumps over the lazy dog. Antigravity inference engine in pure Go."
	b.SetBytes(int64(len(prompt)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(prompt)
	}
}

func BenchmarkByteTokenizerDecode(b *testing.B) {
	tok := NewByteTokenizer()
	tokens := []int32{84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110}
	b.SetBytes(int64(len(tokens)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Decode(tokens)
	}
}

func BenchmarkModelForwardStep(b *testing.B) {
	cfg := DefaultModelConfig()
	weights := GenerateDeterministicWeights(&cfg)
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Forward(weights, &cfg, kvCache, 42, i%cfg.MaxSeqLen)
	}
}
