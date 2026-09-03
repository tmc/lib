//go:build darwin

package main

import (
	"math"
	"os"
	"testing"
)

func TestMultiBufferProjectionMatchesUnfused(t *testing.T) {
	configs := []struct {
		name string
		cfg  Config
	}{
		{name: "synthetic-f32", cfg: DefaultTinyLlamaConfig()},
	}
	if model := os.Getenv("LLAMA_INFER_TEST_MODEL"); model != "" {
		cfg := Llama3_2_1BConfig(model)
		cfg.Precision = "f16"
		cfg.MaxSeqLen = 128
		configs = append(configs, struct {
			name string
			cfg  Config
		}{name: "real-f16", cfg: cfg})
	}
	for _, test := range configs {
		t.Run(test.name, func(t *testing.T) {
			eng, err := GetOrCompileEngine(test.cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, b := range []int{1, 2, 4} {
				input := make([]float32, b*test.cfg.Dim)
				for i := range input {
					input[i] = float32(math.Sin(float64(i+1) * 0.017))
				}
				fused := runProjectionStages(t, eng, test.cfg, b, input, false)
				unfused := runProjectionStages(t, eng, test.cfg, b, input, true)
				checkClose(t, "q", fused.q, unfused.q, 0)
				checkClose(t, "k", fused.k, unfused.k, 0)
				checkClose(t, "v", fused.v, unfused.v, 0)
				checkClose(t, "gate", fused.gate, unfused.gate, 0)
				checkClose(t, "up", fused.up, unfused.up, 0)
			}
		})
	}
}

type projectionOutputs struct{ q, k, v, gate, up []float32 }

func runProjectionStages(t *testing.T, eng *ModelEngine, cfg Config, b int, input []float32, disable bool) projectionOutputs {
	t.Helper()
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if err := eng.ensurePrefillArena(b, cfg); err != nil {
		t.Fatal(err)
	}
	copy(eng.prefillNormXMat.HostSlice()[:len(input)], input)
	eng.disableProjectionFusion = disable
	defer func() { eng.disableProjectionFusion = false }()
	var l *GPULayerWeights
	var lf *GPULayerWeightsF16
	if eng.isF16 {
		lf = &eng.layersF16[0]
	} else {
		l = &eng.layers[0]
	}
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	encodeQKVProjections(ce, eng, l, lf, cfg, b)
	encodeGateUpProjections(ce, eng, l, lf, cfg, b)
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	q := b * cfg.NumHeads * cfg.HeadDim
	kv := b * cfg.NumKVHeads * cfg.HeadDim
	h := b * cfg.HiddenDim
	return projectionOutputs{
		q:    append([]float32(nil), eng.prefillQMat.HostSlice()[:q]...),
		k:    append([]float32(nil), eng.prefillKMat.HostSlice()[:kv]...),
		v:    append([]float32(nil), eng.prefillVMat.HostSlice()[:kv]...),
		gate: append([]float32(nil), eng.prefillGateMat.HostSlice()[:h]...),
		up:   append([]float32(nil), eng.prefillUpMat.HostSlice()[:h]...),
	}
}
