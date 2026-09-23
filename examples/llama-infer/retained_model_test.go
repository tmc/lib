package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// writeRetainedTestCheckpoint writes a small but complete checkpoint in dtype
// and returns its directory.
//
// The payload is deliberately not uniform: every element gets a distinct bit
// pattern derived from its position, so a tensor read at the wrong offset, or
// interpreted as the wrong dtype, produces different values rather than the
// same constant everywhere.
func writeRetainedTestCheckpoint(t *testing.T, dtype string) (string, *ModelConfig) {
	t.Helper()
	dir := t.TempDir()
	model := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        4,
		IntermediateSize:  8,
		NumHiddenLayers:   2,
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
		switch {
		case a.name < b.name:
			return -1
		case a.name > b.name:
			return 1
		}
		return 0
	})

	size := 4
	if dtype != "F32" {
		size = 2
	}
	for _, tensor := range tensors {
		if tensor.name == "lm_head.weight" {
			continue // tied to the embedding table
		}
		elements := 1
		for _, n := range tensor.shape {
			elements *= n
		}
		start := len(data)
		data = append(data, make([]byte, elements*size)...)
		for i := 0; i < elements; i++ {
			switch dtype {
			case "F32":
				// A small positive float whose mantissa varies per element.
				binary.LittleEndian.PutUint32(data[start+i*4:], math.Float32bits(float32(i%13+1)/8))
			case "F16":
				binary.LittleEndian.PutUint16(data[start+i*2:], uint16(0x3400+i%97))
			case "BF16":
				binary.LittleEndian.PutUint16(data[start+i*2:], uint16(0x3E00+i%97))
			default:
				t.Fatalf("unsupported test dtype %q", dtype)
			}
		}
		header[tensor.name] = tensorHeader{
			DType:      dtype,
			Shape:      tensor.shape,
			DataOffset: [2]uint64{uint64(start), uint64(len(data))},
		}
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
	return dir, &model
}

// TestLoadRetainedModelMatchesReferenceLoader checks that the retained load
// path, widened at the upload boundary, produces exactly what the float32
// loader produces.
//
// This is what makes the retained path usable as the production input: the
// upload sees the same numbers it saw before, so device behavior is unchanged
// while the representation upstream of the upload is no longer float32.
func TestLoadRetainedModelMatchesReferenceLoader(t *testing.T) {
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, dtype)

			want, wantCfg, err := LoadModel(dir)
			if err != nil {
				t.Fatalf("LoadModel: %v", err)
			}
			retained, gotCfg, err := LoadRetainedModel(dir)
			if err != nil {
				t.Fatalf("LoadRetainedModel: %v", err)
			}
			if gotCfg.NumHiddenLayers != wantCfg.NumHiddenLayers || gotCfg.HiddenSize != wantCfg.HiddenSize {
				t.Fatalf("config = %+v, want %+v", gotCfg, wantCfg)
			}

			got := retained.Float32()
			eq := func(name string, got, want []float32) {
				t.Helper()
				if len(got) != len(want) {
					t.Fatalf("%s has %d elements, want %d", name, len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
					}
				}
			}
			eq("embed_tokens", got.EmbedTokens, want.EmbedTokens)
			eq("norm", got.Norm, want.Norm)
			eq("lm_head", got.LMHead, want.LMHead)
			if len(got.Layers) != len(want.Layers) {
				t.Fatalf("%d layers, want %d", len(got.Layers), len(want.Layers))
			}
			for i := range want.Layers {
				g, w := got.Layers[i], want.Layers[i]
				eq("input_norm", g.InputNorm, w.InputNorm)
				eq("post_norm", g.PostNorm, w.PostNorm)
				eq("wq", g.WQ, w.WQ)
				eq("wk", g.WK, w.WK)
				eq("wv", g.WV, w.WV)
				eq("wo", g.WO, w.WO)
				eq("wgate", g.WGate, w.WGate)
				eq("wup", g.WUp, w.WUp)
				eq("wdown", g.WDown, w.WDown)
			}
		})
	}
}

// TestLoadRetainedModelKeepsSourceDType checks that the model reaching the
// upload boundary is still in the checkpoint's dtype.
//
// Without this, the equivalence test above would also pass for a loader that
// widened everything immediately, which is exactly the behavior retention
// replaces.
func TestLoadRetainedModelKeepsSourceDType(t *testing.T) {
	for _, dtype := range []string{"F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, dtype)
			m, _, err := LoadRetainedModel(dir)
			if err != nil {
				t.Fatalf("LoadRetainedModel: %v", err)
			}
			counts := m.DTypes()
			if counts[dtype] == 0 || len(counts) != 1 {
				t.Fatalf("dtypes = %v, want every tensor tagged %s", counts, dtype)
			}
			if !m.EmbedTokens.IsHalf() {
				t.Error("embedding table is not retained at two bytes per element")
			}
			if got, want := m.PayloadBytes()*2, m.PayloadBytesAsFloat32(); got != want {
				t.Errorf("retained %d bytes, which is not half of the %d float32 bytes", m.PayloadBytes(), want)
			}
		})
	}
}

// TestLoadRetainedModelTiesLMHead checks that a checkpoint without an LM head
// ties it to the embedding table, as the float32 loader does, and that the
// tied payload is only counted once.
func TestLoadRetainedModelTiesLMHead(t *testing.T) {
	dir, _ := writeRetainedTestCheckpoint(t, "F16")
	m, _, err := LoadRetainedModel(dir)
	if err != nil {
		t.Fatalf("LoadRetainedModel: %v", err)
	}
	if m.LMHead != m.EmbedTokens {
		t.Fatal("LM head is not tied to the embedding table")
	}
	w := m.Float32()
	if &w.LMHead[0] != &w.EmbedTokens[0] {
		t.Error("widening a tied model produced two separate slices")
	}
	var embedTwice int
	m.each(func(name string, _ *RetainedTensor) {
		if name == "embed_tokens" || name == "lm_head" {
			embedTwice++
		}
	})
	if embedTwice != 1 {
		t.Errorf("tied embedding table counted %d times, want 1", embedTwice)
	}
}

// TestRetainedFromCheckpointRejectsMissingTensor checks that a checkpoint
// missing a required tensor fails the arrangement rather than yielding a model
// with a nil weight.
func TestRetainedFromCheckpointRejectsMissingTensor(t *testing.T) {
	full, err := NewRetainedTensor(make([]byte, 8), "F16", []int{4})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &ModelConfig{NumHiddenLayers: 1}
	c := &RetainedCheckpoint{Tensors: map[string]*RetainedTensor{
		"model.embed_tokens.weight": full,
		"model.norm.weight":         full,
	}}
	if _, err := retainedFromCheckpoint(c, cfg, CheckpointReceipt{}); err == nil {
		t.Fatal("accepted a checkpoint with no layer weights")
	}
	if _, err := retainedFromCheckpoint(&RetainedCheckpoint{Tensors: map[string]*RetainedTensor{}}, cfg, CheckpointReceipt{}); err == nil {
		t.Fatal("accepted an empty checkpoint")
	}
}
