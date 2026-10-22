package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestInspectModelStableSummary(t *testing.T) {
	g := &gguf.File{
		Size: 4096, DataOffset: 1024,
		Metadata: map[string]gguf.Value{
			"deepseek4.block_count":   {TypeID: gguf.TypeUint32, Value: uint32(43)},
			"general.architecture":    {TypeID: gguf.TypeString, Value: "deepseek4"},
			"general.context_length":  {TypeID: gguf.TypeUint32, Value: uint32(65536)},
			"general.vocabulary_size": {TypeID: gguf.TypeUint32, Value: uint32(129280)},
		},
		Tensors: []gguf.Tensor{
			{Name: "b", Dimensions: []uint64{32}, TypeID: 0},
			{Name: "a", Dimensions: []uint64{64}, TypeID: 0},
			{Name: "q", Dimensions: []uint64{32}, TypeID: 1},
		},
	}
	var out bytes.Buffer
	if err := inspectModel(&out, "model.gguf", g); err != nil {
		t.Fatal(err)
	}
	var got modelSummary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "ds4-min-model-summary-v1" || got.Model != "model.gguf" || got.TensorCount != 3 {
		t.Fatalf("summary header = %+v", got)
	}
	if got.Architecture != "deepseek4" || got.ContextLength != 65536 || got.VocabularySize != 129280 || got.BlockCount != 43 {
		t.Fatalf("summary metadata = %+v", got)
	}
	if len(got.Quantization) != 2 || got.Quantization[0].TypeID != 0 || got.Quantization[0].Tensors != 2 || got.Quantization[0].Bytes != 384 {
		t.Fatalf("quantization = %+v", got.Quantization)
	}
	if got.Quantization[1].TypeID != 1 || got.Quantization[1].Bytes != 64 {
		t.Fatalf("quantization = %+v", got.Quantization)
	}
	if !strings.HasSuffix(out.String(), "\n") {
		t.Fatal("summary lacks trailing newline")
	}
}

func TestInspectModelRejectsUnsupportedTensor(t *testing.T) {
	g := &gguf.File{Tensors: []gguf.Tensor{{Name: "bad", Dimensions: []uint64{32}, TypeID: 999}}}
	if err := inspectModel(&bytes.Buffer{}, "bad.gguf", g); err == nil || !strings.Contains(err.Error(), `tensor "bad"`) {
		t.Fatalf("inspect error = %v", err)
	}
}

func TestInspectParseDoesNotSelectBackend(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "arbitrary.gguf", "-backend", "unknown", "-inspect"}, strings.NewReader("ignored"), &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.inspect {
		t.Fatal("inspect flag not set")
	}
}
