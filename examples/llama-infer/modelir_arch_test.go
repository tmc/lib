package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/llama-infer/internal/llamamodel"
	"github.com/tmc/modelir"
)

const modelirFixture = "internal/llamamodel/testdata/llama-3.2-1b"

func TestModelirSwitch(t *testing.T) {
	t.Setenv("LLAMA_INFER_MODELIR", "")
	if modelirEnabled() {
		t.Fatal("enabled with the variable unset")
	}
	t.Setenv("LLAMA_INFER_MODELIR", "1")
	if !modelirEnabled() {
		t.Fatal("not enabled with LLAMA_INFER_MODELIR=1")
	}
}

// The resolved architecture projected onto ModelConfig says what the JSON
// loader read, and the switch does not change what LoadModelConfig returns.
func TestModelirLoadMatchesLegacy(t *testing.T) {
	t.Setenv("LLAMA_INFER_MODELIR", "")
	legacy, err := LoadModelConfig(modelirFixture)
	if err != nil {
		t.Fatal(err)
	}
	arch, _, err := llamamodel.Load(modelirFixture)
	if err != nil {
		t.Fatal(err)
	}
	got, err := modelConfigFromArch(arch)
	if err != nil {
		t.Fatal(err)
	}
	if diff := modelConfigDiff(got, legacy); len(diff) > 0 {
		t.Fatalf("modelir and the JSON loader disagree on %v:\n modelir %+v\n json    %+v", diff, got, legacy)
	}
	t.Setenv("LLAMA_INFER_MODELIR", "1")
	checked, err := LoadModelConfig(modelirFixture)
	if err != nil {
		t.Fatalf("LoadModelConfig with the switch: %v", err)
	}
	if !reflect.DeepEqual(checked, legacy) {
		t.Fatalf("the switch changed what LoadModelConfig returns:\n with    %+v\n without %+v", checked, legacy)
	}
}

// A configuration without rms_norm_eps: the JSON loader fills in 1e-5,
// modelir's config frontend 1e-6. The switch surfaces the disagreement
// instead of picking a side.
func TestModelirRefusesDisagreement(t *testing.T) {
	dir := t.TempDir()
	const cfg = `{"model_type":"llama","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
		"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":256,"rope_theta":10000.0,"hidden_act":"silu"}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLAMA_INFER_MODELIR", "")
	if _, err := LoadModelConfig(dir); err != nil {
		t.Fatalf("JSON loader: %v", err)
	}
	t.Setenv("LLAMA_INFER_MODELIR", "1")
	_, err := LoadModelConfig(dir)
	if err == nil || !strings.Contains(err.Error(), "rms_norm_eps") {
		t.Fatalf("LoadModelConfig with the switch: err = %v, want the rms_norm_eps disagreement", err)
	}
}

func TestModelConfigFromArchRefuses(t *testing.T) {
	base := func() *llamamodel.Arch {
		arch, _, err := llamamodel.Load(modelirFixture)
		if err != nil {
			t.Fatal(err)
		}
		return arch
	}
	tests := []struct {
		name  string
		edit  func(a *llamamodel.Arch)
		wants string
	}{
		{"layernorm", func(a *llamamodel.Arch) { a.Norm = modelir.NormLayerNorm }, "RMSNorm only"},
		{"gelu gate", func(a *llamamodel.Arch) { a.MLP = modelir.MLPGELUGate }, "SwiGLU with SiLU only"},
		{"mixed layers", func(a *llamamodel.Arch) { a.Layers[3].KVHeads = 4 }, "layer 3 differs"},
		{"sliding", func(a *llamamodel.Arch) {
			for i := range a.Layers {
				a.Layers[i].Type = modelir.LayerTypeSliding
			}
		}, "dense attention only"},
		{"no layers", func(a *llamamodel.Arch) { a.Layers = nil }, "no layers"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := base()
			tt.edit(a)
			_, err := modelConfigFromArch(a)
			if err == nil || !strings.Contains(err.Error(), tt.wants) {
				t.Fatalf("err = %v, want %q", err, tt.wants)
			}
		})
	}
}

func TestModelConfigDiff(t *testing.T) {
	a := DefaultModelConfig()
	b := a
	if diff := modelConfigDiff(&a, &b); len(diff) != 0 {
		t.Fatalf("identical configurations differ on %v", diff)
	}
	b.ExplicitHeadDim = a.HeadDim() // explicit and derived head_dim agree
	if diff := modelConfigDiff(&a, &b); len(diff) != 0 {
		t.Fatalf("explicit head_dim %d reads as a difference: %v", b.ExplicitHeadDim, diff)
	}
	b.VocabSize++
	b.EOSTokenIDs = []int32{2, 3}
	if diff := modelConfigDiff(&a, &b); !reflect.DeepEqual(diff, []string{"vocab_size", "eos_token_id"}) {
		t.Fatalf("diff = %v, want vocab_size and eos_token_id", diff)
	}
}
