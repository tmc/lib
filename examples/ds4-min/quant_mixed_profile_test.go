package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestRegisteredModelHeader(t *testing.T) {
	path := os.Getenv("DS4_GGUF_MANIFEST_MODEL")
	if path == "" {
		t.Skip("manifest model not set")
	}
	f, err := os.Open(path)
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
	d, err := profileHashes(g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("metadata=%s tensors=%s quantization=%s tokenizer=%s", d.metadata, d.tensors, d.quant, d.tokenizer)
	p, err := MatchProfile(g)
	if err != nil {
		t.Fatalf("registered profile = %v, %v", p, err)
	}
	layout, err := LayoutOf(g)
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range layout {
		want := ExpertsIQ2Q2K
		if p.Name == "ds4f-q2-q4" && i >= 37 {
			want = ExpertsQ4K
		}
		if f != want {
			t.Fatalf("layer %d=%v want %v", i, f, want)
		}
	}
	first, count := -1, 0
	for i, tensor := range g.Tensors {
		if !strings.HasSuffix(tensor.Name, ".ffn_gate_tid2eid.weight") {
			continue
		}
		if tensor.TypeID != 26 {
			t.Fatalf("hash-router tensor %s type=%d, want I32=26", tensor.Name, tensor.TypeID)
		}
		if first < 0 {
			first = i
		}
		count++
	}
	if count != 3 {
		t.Fatalf("hash-router tensor count=%d, want 3", count)
	}
	g.Tensors[first].TypeID = 5
	if _, err := LayoutOf(g); err == nil {
		t.Fatal("accepted obsolete I32 type 5")
	}
}

func TestProfileFileChecksumAndIsolation(t *testing.T) {
	payload := []byte("registered model payload")
	g := &gguf.File{Size: uint64(len(payload))}
	d, err := profileHashes(g)
	if err != nil {
		t.Fatal(err)
	}
	profile := Profile{Name: "test", Bytes: g.Size, SHA256: fmt.Sprintf("%x", sha256.Sum256(payload)), Metadata: d.metadata, Tensors: d.tensors, Quant: d.quant, Tokenizer: d.tokenizer}
	saved := supportedProfiles
	supportedProfiles = append(append([]Profile(nil), saved...), profile)
	defer func() { supportedProfiles = saved }()
	if err := validateProfileFile(g, bytes.NewReader(payload), int64(len(payload))); err != nil {
		t.Fatal(err)
	}
	for _, size := range []int64{-1, int64(len(payload)) - 1, int64(len(payload)) + 1} {
		if err := validateProfileFile(g, bytes.NewReader(payload), size); err == nil {
			t.Fatalf("accepted inconsistent reader size %d", size)
		}
	}
	for _, digest := range []string{"", "abcd", strings.Repeat("z", 64)} {
		supportedProfiles[len(supportedProfiles)-1].SHA256 = digest
		if err := validateProfileFile(g, bytes.NewReader(payload), int64(len(payload))); err == nil {
			t.Fatalf("accepted invalid registered checksum %q", digest)
		}
	}
	supportedProfiles[len(supportedProfiles)-1].SHA256 = profile.SHA256
	mutated := append([]byte(nil), payload...)
	mutated[0] ^= 1
	if err := validateProfileFile(g, bytes.NewReader(mutated), int64(len(mutated))); err == nil {
		t.Fatal("accepted changed payload with matching header")
	}
	g.Size++
	if _, err := MatchProfile(g); err == nil {
		t.Fatal("accepted wrong model size")
	}
	g.Size--
	g.Metadata = map[string]gguf.Value{"general.architecture": {TypeID: 8, Value: "mutation"}}
	if _, err := MatchProfile(g); err == nil {
		t.Fatal("accepted wrong metadata")
	}
	if saved[0].SHA256 == saved[1].SHA256 || saved[0].Quant == saved[1].Quant {
		t.Fatal("mixed profile aliases Q2 identity")
	}
}
