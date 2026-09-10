package llamamodel

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/modelir"
	"github.com/tmc/modelir/encoding/sexpr"
)

const fixture = "testdata/llama-3.2-1b"

func TestLoadDirectory(t *testing.T) {
	arch, ms, err := Load(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if ms == nil || ms.Arch == nil {
		t.Fatal("Load returned no module set")
	}
	want := &Arch{
		Family:       "llama",
		Vocab:        128256,
		Hidden:       2048,
		Intermediate: 8192,
		Norm:         modelir.NormRMSNorm,
		NormEps:      1e-5,
		MLP:          modelir.MLPSwiGLU,
		Activation:   modelir.MLPActivationSiLU,
		RoPE:         RoPE{Theta: 500000},
		TieEmbed:     true,
		MaxPositions: 131072,
		BOS:          128000,
		EOS:          []int32{128001, 128008, 128009},
	}
	for i := 0; i < 16; i++ {
		want.Layers = append(want.Layers, Layer{Index: i, Type: modelir.LayerTypeDense, Heads: 32, KVHeads: 8, HeadDim: 64})
	}
	if !reflect.DeepEqual(arch, want) {
		t.Fatalf("Load(%s):\n got %+v\nwant %+v", fixture, arch, want)
	}
}

// The module set Load resolves, written as .sexpr and loaded again, resolves
// to the same Arch and encodes to the same bytes.
func TestLoadSexprMatchesDirectory(t *testing.T) {
	fromDir, ms, err := Load(fixture)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := sexpr.Marshal(ms)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	path := filepath.Join(t.TempDir(), "llama.sexpr")
	if err := os.WriteFile(path, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	fromSexpr, ms2, err := Load(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	if !reflect.DeepEqual(fromSexpr, fromDir) {
		t.Fatalf("sexpr and directory resolve differently:\n sexpr %+v\n dir   %+v", fromSexpr, fromDir)
	}
	again, err := sexpr.Marshal(ms2)
	if err != nil {
		t.Fatalf("marshal again: %v", err)
	}
	if !bytes.Equal(again, encoded) {
		t.Fatalf("encoding is not stable across a round trip:\n%s\n---\n%s", encoded, again)
	}
}

func TestResolveRefuses(t *testing.T) {
	tests := []struct {
		name  string
		edit  func(ms *modelir.ModuleSet)
		code  string // modelir diagnostic code, or ""
		wants string // substring of the error
	}{
		{
			name: "unknown role",
			edit: func(ms *modelir.ModuleSet) { ms.Modules[0].Roles = append(ms.Modules[0].Roles, "oracle") },
			code: "module.role.unknown",
		},
		{
			name: "missing field",
			edit: func(ms *modelir.ModuleSet) { ms.Arch.Rope.BaseField = "rope_theta_v2" },
			code: "arch.field.missing",
		},
		{
			name: "zero dimension",
			edit: func(ms *modelir.ModuleSet) {
				fields := ms.Configs[0].Fields[:0]
				for _, f := range ms.Configs[0].Fields {
					if f.Name != "vocab_size" {
						fields = append(fields, f)
					}
				}
				ms.Configs[0].Fields = fields
			},
			wants: "vocab_size unresolved",
		},
		{
			// modelir's validator refuses the value before Resolve's own
			// norm check would.
			name: "unknown norm",
			edit: func(ms *modelir.ModuleSet) { ms.Arch.Model.NormKind = "batchnorm" },
			code: "arch.value.unknown",
		},
		{
			name:  "token id out of range",
			edit:  func(ms *modelir.ModuleSet) { ms.Arch.Generation.EOSIDs = []int64{1 << 40} },
			wants: "eos_id",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ms, err := Load(fixture)
			if err != nil {
				t.Fatal(err)
			}
			tt.edit(ms)
			_, err = Resolve(ms)
			if err == nil {
				t.Fatal("Resolve accepted the module set")
			}
			if tt.code != "" {
				var ds modelir.Diagnostics
				if !errors.As(err, &ds) {
					t.Fatalf("error %v does not carry modelir diagnostics", err)
				}
				found := false
				for _, d := range ds {
					found = found || d.Code == tt.code
				}
				if !found {
					t.Fatalf("diagnostics %v lack code %s", ds, tt.code)
				}
			}
			if tt.wants != "" && !strings.Contains(err.Error(), tt.wants) {
				t.Fatalf("error %q does not mention %q", err, tt.wants)
			}
		})
	}
}

func TestLoadMissing(t *testing.T) {
	if _, _, err := Load(filepath.Join(t.TempDir(), "nowhere")); err == nil {
		t.Fatal("Load of a missing path succeeded")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"llama"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("Load of a configuration without dimensions succeeded")
	}
}
