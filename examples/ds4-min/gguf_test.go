package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"testing"
)

func TestReadGGUF(t *testing.T) {
	b := fixtureGGUF()
	g, err := readGGUF(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	if g.dataOffset != 192 {
		t.Fatalf("data offset = %d, want 192", g.dataOffset)
	}
	if got := g.metadata["general.architecture"].value; got != "deepseek4" {
		t.Fatalf("architecture = %q", got)
	}
	if len(g.tensors) != 1 || g.tensors[0].name != "token_embd.weight" {
		t.Fatalf("tensors = %#v", g.tensors)
	}
}

func TestReadGGUFTruncated(t *testing.T) {
	b := fixtureGGUF()
	for n := range len(b) {
		if _, err := readGGUF(bytes.NewReader(b[:n]), int64(n)); err == nil {
			t.Fatalf("length %d succeeded", n)
		}
	}
}

func TestReadGGUFMalformed(t *testing.T) {
	tests := []struct {
		name string
		edit func([]byte)
	}{
		{"magic", func(b []byte) { b[0] = 'X' }},
		{"version", func(b []byte) { put32(b[4:], 2) }},
		{"tensor count", func(b []byte) { put64(b[8:], maxTensors+1) }},
		{"metadata count", func(b []byte) { put64(b[16:], maxMetadata+1) }},
		{"zero alignment", func(b []byte) {
			i := bytes.Index(b, []byte("general.alignment")) + len("general.alignment") + 4
			put32(b[i:], 0)
		}},
		{"bad dimensions", func(b []byte) {
			i := bytes.Index(b, []byte("token_embd.weight")) + len("token_embd.weight")
			put32(b[i:], 5)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := fixtureGGUF()
			tt.edit(b)
			if _, err := readGGUF(bytes.NewReader(b), int64(len(b))); err == nil {
				t.Fatal("read succeeded")
			}
		})
	}
}

func TestReadGGUFStringBound(t *testing.T) {
	b := fixtureGGUF()
	put64(b[24:], maxString+1)
	_, err := readGGUF(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, errInvalidGGUF) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadGGUFArrayAllocationBound(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 0)
	write64(&b, 1)
	writeString(&b, "test.values")
	write32(&b, 9)
	write32(&b, 0)
	write64(&b, maxArray)
	_, err := readGGUF(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if !errors.Is(err, errInvalidGGUF) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadGGUFRejectsInvalidUTF8(t *testing.T) {
	b := fixtureGGUF()
	i := bytes.Index(b, []byte("general.architecture"))
	b[i] = 0xff
	_, err := readGGUF(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, errInvalidGGUF) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestCheckSHA256RejectsPayloadCorruption(t *testing.T) {
	b := fixtureGGUF()
	sum := sha256.Sum256(b)
	want := hex.EncodeToString(sum[:])
	if err := checkSHA256(bytes.NewReader(b), int64(len(b)), want); err != nil {
		t.Fatal(err)
	}
	b[len(b)-1] ^= 1
	if err := checkSHA256(bytes.NewReader(b), int64(len(b)), want); err == nil {
		t.Fatal("corrupt payload passed validation")
	}
}

func TestAlign(t *testing.T) {
	if got, err := align(97, 32); err != nil || got != 128 {
		t.Fatalf("align = %d, %v", got, err)
	}
	if _, err := align(^uint64(0)-1, 8); err == nil {
		t.Fatal("overflow succeeded")
	}
}

func TestTensorBytesChecksRows(t *testing.T) {
	_, err := tensorBytes(ggufTensor{name: "x", typeID: 16, dimensions: []uint64{128, 2}})
	if !errors.Is(err, errInvalidGGUF) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
	n, err := tensorBytes(ggufTensor{name: "x", typeID: 16, dimensions: []uint64{256, 2}})
	if err != nil || n != 132 {
		t.Fatalf("bytes = %d, %v, want 132", n, err)
	}
}

func TestReadGGUFNestedArray(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 0)
	write64(&b, 1)
	writeString(&b, "test.values")
	write32(&b, 9)
	write32(&b, 9)
	write64(&b, 1)
	write32(&b, 4)
	write64(&b, 2)
	write32(&b, 7)
	write32(&b, 11)
	if _, err := readGGUF(bytes.NewReader(b.Bytes()), int64(b.Len())); err != nil {
		t.Fatal(err)
	}
}

func TestReadGGUFUnalignedTensor(t *testing.T) {
	b := fixtureGGUF()
	i := bytes.Index(b, []byte("token_embd.weight")) + len("token_embd.weight") + 4 + 16 + 4
	put64(b[i:], 1)
	_, err := readGGUF(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, errInvalidGGUF) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadGGUFAlignmentMultipleOfEight(t *testing.T) {
	b := fixtureGGUF()
	i := bytes.Index(b, []byte("general.alignment")) + len("general.alignment") + 4
	put32(b[i:], 24)
	if _, err := readGGUF(bytes.NewReader(b), int64(len(b))); err != nil {
		t.Fatal(err)
	}
}

func FuzzReadGGUF(f *testing.F) {
	f.Add(fixtureGGUF())
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, b []byte) {
		_, _ = readGGUF(bytes.NewReader(b), int64(len(b)))
	})
}

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
	g, err := readGGUF(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	manifest := struct {
		Schema     string          `json:"schema"`
		FileBytes  int64           `json:"file_bytes"`
		DataOffset uint64          `json:"data_offset"`
		Metadata   []metadataEntry `json:"metadata"`
		Tensors    []tensorEntry   `json:"tensors"`
	}{Schema: "ds4-min-gguf-manifest-v1", FileBytes: info.Size(), DataOffset: g.dataOffset}
	keys := make([]string, 0, len(g.metadata))
	for key := range g.metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		v := g.metadata[key]
		manifest.Metadata = append(manifest.Metadata, metadataEntry{key, v.typeID, v.value})
	}
	for _, tensor := range g.tensors {
		n, err := tensorBytes(tensor)
		if err != nil {
			t.Fatal(err)
		}
		manifest.Tensors = append(manifest.Tensors, tensorEntry{
			tensor.name, tensor.dimensions, tensor.typeID, tensor.offset, n,
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
		v := g.metadata["general.architecture"]
		g.metadata["general.architecture"] = ggufValue{v.typeID, "deepseek3"}
		defer func() { g.metadata["general.architecture"] = v }()
		if err := validatePinnedProfile(g); err == nil {
			t.Fatal("validation succeeded")
		}
	})
	t.Run("tensor corruption", func(t *testing.T) {
		d := g.tensors[0].dimensions[0]
		g.tensors[0].dimensions[0] = d + 1
		defer func() { g.tensors[0].dimensions[0] = d }()
		if err := validatePinnedProfile(g); err == nil {
			t.Fatal("validation succeeded")
		}
	})
}

func fixtureGGUF() []byte {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 1)
	write64(&b, 2)
	writeString(&b, "general.architecture")
	write32(&b, 8)
	writeString(&b, "deepseek4")
	writeString(&b, "general.alignment")
	write32(&b, 4)
	write32(&b, 32)
	writeString(&b, "token_embd.weight")
	write32(&b, 2)
	write64(&b, 4)
	write64(&b, 8)
	write32(&b, 0)
	write64(&b, 0)
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, 128))
	return b.Bytes()
}

func writeString(b *bytes.Buffer, s string) { write64(b, uint64(len(s))); b.WriteString(s) }
func write32(b *bytes.Buffer, n uint32)     { var p [4]byte; put32(p[:], n); b.Write(p[:]) }
func write64(b *bytes.Buffer, n uint64)     { var p [8]byte; put64(p[:], n); b.Write(p[:]) }
func put32(b []byte, n uint32)              { binary.LittleEndian.PutUint32(b, n) }
func put64(b []byte, n uint64)              { binary.LittleEndian.PutUint64(b, n) }
