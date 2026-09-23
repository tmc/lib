//go:build darwin

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
)

func TestPrecisionTelemetry(t *testing.T) {
	eng := &ModelEngine{weights: &metalsafetensors.Weights{Tensors: map[string]metalsafetensors.TensorInfo{
		"layer.0.weight": {DType: "F16", OffsetStart: 0, OffsetEnd: 32},
		"layer.1.weight": {DType: "F16", OffsetStart: 32, OffsetEnd: 96},
		"layer.0.scale":  {DType: "F32", OffsetStart: 96, OffsetEnd: 112},
	}}}

	got := eng.precisionTelemetry()
	want := []TensorPrecision{
		{DType: "F16", TensorCount: 2, Bytes: 96},
		{DType: "F32", TensorCount: 1, Bytes: 16},
	}
	if len(got) != len(want) {
		t.Fatalf("precisionTelemetry() = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("precisionTelemetry()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	response := NewTelemetryServer(eng, Config{MaxSeqLen: 128}).memoryTelemetry()
	if response.Precision.State != "observed" || len(response.Precision.DTypes) != len(want) {
		t.Errorf("precision telemetry = %+v, want observed dtype aggregate", response.Precision)
	}
}

func writeEngineCheckpoint(t *testing.T, dtype string) (string, Config) {
	t.Helper()
	dir := t.TempDir()
	model := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        4,
		IntermediateSize:  8,
		NumHiddenLayers:   1,
		NumAttentionHeads: 2,
		NumKeyValueHeads:  1,
		ExplicitHeadDim:   2,
		VocabSize:         16,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         8,
		TieWordEmbeddings: true,
	}
	config, err := json.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0o600); err != nil {
		t.Fatal(err)
	}
	type tensorHeader struct {
		DType      string    `json:"dtype"`
		Shape      []int     `json:"shape"`
		DataOffset [2]uint64 `json:"data_offsets"`
	}
	header := make(map[string]tensorHeader)
	var data []byte
	tensors := expectedCheckpointTensors(&model)
	slices.SortFunc(tensors, func(a, b expectedTensor) int {
		if a.name < b.name {
			return -1
		}
		if a.name > b.name {
			return 1
		}
		return 0
	})
	bytesPerElement := 4
	if dtype == "F16" {
		bytesPerElement = 2
	}
	for _, tensor := range tensors {
		if tensor.name == "lm_head.weight" {
			continue
		}
		elements := 1
		for _, n := range tensor.shape {
			elements *= n
		}
		start := len(data)
		data = append(data, make([]byte, elements*bytesPerElement)...)
		// Norm weights must be non-zero for a useful buffer, although this
		// ownership test deliberately submits no inference work.
		if len(tensor.shape) == 1 {
			for i := 0; i < elements; i++ {
				if dtype == "F16" {
					binary.LittleEndian.PutUint16(data[start+i*2:], 0x3c00)
				} else {
					binary.LittleEndian.PutUint32(data[start+i*4:], 0x3f800000)
				}
			}
		}
		header[tensor.name] = tensorHeader{DType: dtype, Shape: tensor.shape, DataOffset: [2]uint64{uint64(start), uint64(len(data))}}
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := make([]byte, 8, 8+len(headerJSON)+len(data))
	binary.LittleEndian.PutUint64(file, uint64(len(headerJSON)))
	file = append(file, headerJSON...)
	file = append(file, data...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), file, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Name: "ownership-" + dtype, ModelPath: dir, Precision: map[string]string{"F16": "f16", "F32": "f32"}[dtype],
		VocabSize: 16, Dim: 4, HiddenDim: 8, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 2,
		MaxSeqLen: 8, Eps: 1e-5, RopeTheta: 10000,
	}
	return dir, cfg
}

func TestRealCheckpointWeightsJoinEngineLifetime(t *testing.T) {
	for _, dtype := range []string{"F16", "F32"} {
		t.Run(dtype, func(t *testing.T) {
			_, cfg := writeEngineCheckpoint(t, dtype)
			eng, err := GetOrCompileDarwinEngine(cfg)
			if err != nil {
				t.Fatalf("load checkpoint: %v", err)
			}
			var embed, tied metalmem.Slice[float32]
			var embedF16, tiedF16 metalmem.Slice[uint16]
			if eng.isF16 {
				embedF16, tiedF16 = eng.embedTableF16, eng.lmHeadWF16
				if embedF16.Buffer() == nil || tiedF16.Buffer() == nil || embedF16.Buffer().GetID() != tiedF16.Buffer().GetID() {
					t.Fatal("tied F16 weights do not share one live buffer")
				}
			} else {
				embed, tied = eng.embedTable, eng.lmHeadW
				if embed.Buffer() == nil || tied.Buffer() == nil || embed.Buffer().GetID() != tied.Buffer().GetID() {
					t.Fatal("tied F32 weights do not share one live buffer")
				}
			}
			eng.Free()
			eng.Free()
			if eng.isF16 {
				if embedF16.Buffer() != nil || tiedF16.Buffer() != nil {
					t.Error("F16 checkpoint weights remain live after Free")
				}
			} else if embed.Buffer() != nil || tied.Buffer() != nil {
				t.Error("F32 checkpoint weights remain live after Free")
			}
		})
	}
}

func TestMemoryTelemetryUsesRuntimeMetrics(t *testing.T) {
	metrics := NewRuntimeMetrics(4)
	metrics.SetKVAllocation(8192, 32)
	metrics.RecordForward(ForwardMeasurement{Position: 7})
	prefixes := newPrefixCache(1024, 2)
	prefixes.store([]int32{1, 2}, &prefixSnapshot{tokens: 2, bytes: 16})
	prefixes.load([]int32{1, 2, 3})

	srv := NewTelemetryServer(&ModelEngine{metrics: metrics, prefixes: prefixes}, Config{MaxSeqLen: 32})
	response := srv.memoryTelemetry()
	if response.KVCache.State != "observed" || response.KVCache.AllocatedBytes != 8192 || response.KVCache.ActiveTokens != 8 {
		t.Errorf("KV cache telemetry = %+v, want observed runtime allocation", response.KVCache)
	}
	if response.PrefixCache.State != "observed" || response.PrefixCache.Entries != 1 || response.PrefixCache.Hits != 1 || response.PrefixCache.SavedTokens != 2 {
		t.Errorf("prefix cache telemetry = %+v, want observed prefix cache metrics", response.PrefixCache)
	}
}

func TestGenerateRestoresPrefixKVSnapshot(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		t.Skipf("Metal engine unavailable: %v", err)
	}
	before, _, _, _, _ := eng.metrics.Snapshot()
	if err := eng.Generate(context.Background(), cfg, "repeat prefix", 1, func(int32, string) bool { return true }); err != nil {
		t.Fatalf("first generate: %v", err)
	}
	afterFirst, _, _, _, _ := eng.metrics.Snapshot()
	if err := eng.Generate(context.Background(), cfg, "repeat prefix", 1, func(int32, string) bool { return true }); err != nil {
		t.Fatalf("second generate: %v", err)
	}
	afterSecond, _, _, _, _ := eng.metrics.Snapshot()
	firstForwards := len(afterFirst) - len(before)
	secondForwards := len(afterSecond) - len(afterFirst)
	if firstForwards <= 1 {
		t.Fatalf("first generate ran %d forwards, want prefill plus decode", firstForwards)
	}
	if secondForwards != 1 {
		t.Errorf("second generate ran %d forwards, want only decode after prefix restore", secondForwards)
	}
	metrics := eng.prefixCacheMetrics()
	if metrics.Hits < 1 || metrics.SavedTokens < 1 {
		t.Errorf("prefix metrics = %+v, want observed restored prefix", metrics)
	}
}

func TestModelEngineFreeReleasesOwnedBuffers(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = t.Name()
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		t.Skipf("Metal engine unavailable: %v", err)
	}
	embed := eng.embedTable
	layer := eng.layers[0].WQ
	scratch := eng.xBuf
	if eng.isF16 {
		if eng.embedTableF16.Buffer() == nil || eng.layersF16[0].WQ.Buffer() == nil {
			t.Fatal("engine weight buffer is nil before Free")
		}
	} else if embed.Buffer() == nil || layer.Buffer() == nil {
		t.Fatal("engine weight buffer is nil before Free")
	}
	if scratch.Buffer() == nil {
		t.Fatal("engine scratch buffer is nil before Free")
	}

	eng.Free()
	eng.Free()
	if eng.isF16 {
		if eng.embedTableF16.Buffer() != nil || eng.layersF16[0].WQ.Buffer() != nil {
			t.Error("engine weight buffer remains live after Free")
		}
	} else if embed.Buffer() != nil || layer.Buffer() != nil {
		t.Error("engine weight buffer remains live after Free")
	}
	if scratch.Buffer() != nil {
		t.Error("engine scratch buffer remains live after Free")
	}
	darwinEngineMu.Lock()
	_, cached := darwinEngineCache[eng.cacheKey]
	darwinEngineMu.Unlock()
	if cached {
		t.Error("freed engine remains cached")
	}
}
