package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestExtractManifest(t *testing.T) {
	modelPath := os.Getenv("DS4_GGUF_MANIFEST_MODEL")
	outPath := os.Getenv("DS4_GGUF_MANIFEST_OUT")
	if modelPath == "" || outPath == "" {
		t.Skip("manifest paths not set")
	}
	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	manifest := struct {
		Schema     string          `json:"schema"`
		FileBytes  int64           `json:"file_bytes"`
		DataOffset uint64          `json:"data_offset"`
		Metadata   []metadataEntry `json:"metadata"`
		Tensors    []tensorEntry   `json:"tensors"`
	}{Schema: "ds4-min-gguf-manifest-v1", FileBytes: info.Size(), DataOffset: g.DataOffset}
	keys := make([]string, 0, len(g.Metadata))
	for key := range g.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		v := g.Metadata[key]
		manifest.Metadata = append(manifest.Metadata, metadataEntry{key, v.TypeID, v.Value})
	}
	for _, tensor := range g.Tensors {
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		manifest.Tensors = append(manifest.Tensors, tensorEntry{
			tensor.Name, tensor.Dimensions, tensor.TypeID, tensor.Offset, n,
		})
	}
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	t.Logf("manifest bytes=%d sha256=%s", len(b), hex.EncodeToString(sum[:]))
	if err := os.WriteFile(outPath, append(b, '\n'), 0o644); err != nil {
		t.Fatal(fmt.Errorf("write manifest: %w", err))
	}
}

func TestPinnedProfile(t *testing.T) {
	modelPath := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if modelPath == "" {
		t.Skip("profile model path not set")
	}
	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := validatePinnedModel(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	digests, err := profileHashes(g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("metadata=%s tensors=%s quantization=%s tokenizer=%s", digests.metadata, digests.tensors, digests.quant, digests.tokenizer)
	t.Run("metadata corruption", func(t *testing.T) {
		v := g.Metadata["general.architecture"]
		g.Metadata["general.architecture"] = gguf.Value{TypeID: v.TypeID, Value: "deepseek3"}
		defer func() { g.Metadata["general.architecture"] = v }()
		if err := validatePinnedProfile(g); err == nil {
			t.Fatal("validation succeeded")
		}
	})
	t.Run("tensor corruption", func(t *testing.T) {
		d := g.Tensors[0].Dimensions[0]
		g.Tensors[0].Dimensions[0] = d + 1
		defer func() { g.Tensors[0].Dimensions[0] = d }()
		if err := validatePinnedProfile(g); err == nil {
			t.Fatal("validation succeeded")
		}
	})
}
