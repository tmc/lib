package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func pjrtConfigJSON(t *testing.T) map[string]any {
	t.Helper()
	c, _ := pjrtFixture()
	c.ModelType = "llama"
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestPJRTConfig(t *testing.T) {
	for _, tt := range []struct {
		name  string
		key   string
		value any
		want  string
	}{
		{"model", "model_type", "mistral", "model_type"},
		{"activation", "hidden_act", "gelu", "hidden_act"},
		{"empty activation", "hidden_act", "", "hidden_act"},
		{"empty layers", "layer_types", []string{}, "layer_types"},
		{"zero parallel", "pretraining_tp", 0, "pretraining_tp"},
		{"scaled", "rope_scaling", map[string]any{"rope_type": "linear", "factor": 2}, "rope_scaling"},
		{"quantized", "quantization_config", map[string]any{}, "unknown field"},
		{"partial rotary", "partial_rotary_factor", 0.5, "unknown field"},
		{"sliding", "use_sliding_window", true, "sliding attention"},
		{"layer pattern", "layer_types", []string{"full_attention", "sliding_attention"}, "layer type"},
		{"attention bias", "attention_bias", true, "bias"},
		{"MLP bias", "mlp_bias", true, "bias"},
		{"nested", "text_config", map[string]any{"hidden_size": 8}, "nested"},
		{"epsilon", "rms_norm_eps", 0, "positive"},
		{"theta", "rope_theta", 0, "positive"},
		{"heads", "num_key_value_heads", 0, "positive"},
		{"negative head", "head_dim", -1, "heads"},
		{"zero dimension", "hidden_size", 0, "positive"},
		{"parallel", "pretraining_tp", 2, "pretraining_tp"},
		{"EOS", "eos_token_id", -1, "EOS"},
		{"BOS", "bos_token_id", 999, "BOS"},
		{"overflow", "hidden_size", math.MaxInt64, "overflows"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := pjrtConfigJSON(t)
			v[tt.key] = tt.value
			if tt.name == "overflow" {
				v["head_dim"] = 4
			}
			b, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := readPJRTConfig(b); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %s", err, tt.want)
			}
		})
	}
	v := pjrtConfigJSON(t)
	v["rope_theta"] = 10000.1
	v["max_position_embeddings"] = 16384
	v["eos_token_id"] = []int{0, 2}
	v["bos_token_id"] = 0
	b, _ := json.Marshal(v)
	p, err := readPJRTConfig(b)
	if err != nil {
		t.Fatal(err)
	}
	if p.theta != 10000.1 || p.config.MaxSeqLen != 16384 || !slices.Equal(p.eos, []int32{0, 2}) || p.bos == nil || *p.bos != 0 {
		t.Fatalf("config lost precision or metadata: %+v", p)
	}
	if _, err := readPJRTConfig(append(b, []byte(" {}")...)); err == nil {
		t.Fatal("trailing JSON accepted")
	}
	v["model_type"] = "qwen2"
	v["sliding_window"] = 4096
	v["use_sliding_window"] = false
	b, _ = json.Marshal(v)
	if _, err := readPJRTConfig(b); err != nil {
		t.Fatal(err)
	}
}

func TestPJRTCheckpointLoad(t *testing.T) {
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, dtype)
			p, err := loadPJRTCheckpoint(dir, 3)
			if err != nil {
				t.Fatal(err)
			}
			if p.config.MaxSeqLen != 3 {
				t.Fatal("capacity not applied")
			}
			want := float32(0.125)
			if dtype == "F16" {
				want = 0.25
			}
			if p.weights.EmbedTokens[0] != want {
				t.Fatalf("first value %g, want %g", p.weights.EmbedTokens[0], want)
			}
			if &p.weights.LMHead[0] != &p.weights.EmbedTokens[0] {
				t.Fatal("lost tied storage")
			}
			if p.weights.Consumption.Verdict != "accepted" {
				t.Fatal("missing receipt")
			}
			f := filepath.Join(dir, "model.safetensors")
			info, err := os.Stat(f)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Truncate(f, info.Size()-1); err != nil {
				t.Fatal(err)
			}
			if _, err := loadPJRTCheckpoint(dir, 3); err == nil {
				t.Fatal("truncated tensor payload accepted")
			}
		})
	}
	dir, _ := writeRetainedTestCheckpoint(t, "F32")
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	json.Unmarshal(data, &v)
	v["rope_scaling"] = map[string]any{"factor": 2}
	data, _ = json.Marshal(v)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := loadPJRTCheckpoint(dir, 3); err == nil || !strings.Contains(err.Error(), "rope_scaling") {
		t.Fatalf("configuration not validated before weights: %v", err)
	}
	if _, _, err := preparePJRTModel(Config{}); err == nil {
		t.Fatal("implicit synthetic model accepted")
	}
	if _, _, err := preparePJRTModel(Config{Precision: "bf16"}); err == nil {
		t.Fatal("half execution silently accepted")
	}
}

func TestNativePJRTCheckpointGeneration(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, dtype)
			p, err := loadPJRTCheckpoint(dir, 6)
			if err != nil {
				t.Fatal(err)
			}
			model, err := newPJRTModel(context.Background(), engine, &p.config, p.weights, p.theta)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := model.close(); err != nil {
					t.Error(err)
				}
			}()
			c := &p.config
			kv := NewKVCacheState(c.NumHiddenLayers, c.MaxSeqLen, c.NumKeyValueHeads*c.HeadDim())
			prompt := []int32{1, 3, 2}
			var logits []float32
			for pos, id := range prompt {
				logits = Forward(p.weights, c, kv, id, pos)
			}
			var want []int32
			for step := 0; step < 4; step++ {
				id := sampleArgmax(logits)
				want = append(want, id)
				if step < 3 {
					logits = Forward(p.weights, c, kv, id, len(prompt)+step)
				}
			}
			var got []int32
			err = model.generate(context.Background(), prompt, 100, nil, func(id int32) bool { got = append(got, id); return true })
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("generation %v, %v; want %v", got, err, want)
			}
			if !slices.Equal(model.model.Lengths(), []int64{6}) {
				t.Fatal("incorrect capacity boundary")
			}
			assertOne := func(eos []int32, keepGoing bool) {
				t.Helper()
				var got []int32
				err := model.generate(context.Background(), prompt, 4, eos, func(id int32) bool { got = append(got, id); return keepGoing })
				if err != nil || !slices.Equal(got, want[:1]) || !slices.Equal(model.model.Lengths(), []int64{3}) {
					t.Fatalf("stop: %v %v lengths %v", got, err, model.model.Lengths())
				}
			}
			assertOne([]int32{want[0]}, true)
			assertOne(nil, false)
			before := model.model.Lengths()
			calls := 0
			emit := func(int32) bool { calls++; return true }
			if err := model.generate(context.Background(), prompt, 0, nil, emit); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if err := model.generate(ctx, prompt, 4, nil, emit); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if err := model.generate(context.Background(), []int32{999}, 4, nil, emit); err == nil {
				t.Fatal("invalid prompt accepted")
			}
			if calls != 0 || !slices.Equal(before, model.model.Lengths()) {
				t.Fatal("validation or zero limit changed state")
			}
			// A full-capacity prompt may emit its one prediction without appending it.
			calls = 0
			if err := model.generate(context.Background(), []int32{1, 2, 3, 4, 5, 6}, 5, nil, emit); err != nil || calls != 1 {
				t.Fatalf("full prompt: %d %v", calls, err)
			}
		})
	}
}

func TestModelConfigPreservesContextLimit(t *testing.T) {
	dir, _ := writeRetainedTestCheckpoint(t, "F32")
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	raw["max_position_embeddings"] = 16384
	data, err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	c, err := LoadModelConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if c.MaxSeqLen != 16384 {
		t.Fatalf("context limit %d, want 16384", c.MaxSeqLen)
	}
}

func TestPJRTGenerationMetadata(t *testing.T) {
	dir, _ := writeRetainedTestCheckpoint(t, "F32")
	file := filepath.Join(dir, "generation_config.json")
	if err := os.WriteFile(file, []byte(`{"eos_token_id":[0,2],"bos_token_id":0,"temperature":0.5}`), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := loadPJRTCheckpoint(dir, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(p.eos, []int32{0, 2}) || p.bos == nil || *p.bos != 0 {
		t.Fatalf("metadata: %+v", p)
	}
	if err := os.WriteFile(file, []byte(`{"eos_token_id":[99]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadPJRTCheckpoint(dir, 3); err == nil || !strings.Contains(err.Error(), "EOS") {
		t.Fatalf("invalid metadata accepted: %v", err)
	}
}
