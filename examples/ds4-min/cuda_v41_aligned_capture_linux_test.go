//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"math"
	"os"
	"path/filepath"
	"testing"
)

var v41AlignedOutput = flag.String("v41-aligned-output", "", "fresh absolute directory for aligned Q8 raw qualification vectors")

type v41AlignedVector struct {
	Path     string `json:"path"`
	Encoding string `json:"encoding"`
	Bytes    int    `json:"bytes"`
	SHA256   string `json:"sha256"`
}

type v41AlignedCapture struct {
	root  string
	files []v41AlignedVector
}

func newV41AlignedCapture(t *testing.T) *v41AlignedCapture {
	t.Helper()
	if !filepath.IsAbs(*v41AlignedOutput) || filepath.Clean(*v41AlignedOutput) != *v41AlignedOutput {
		t.Fatal("-v41-aligned-output must name a clean absolute fresh directory")
	}
	if err := os.Mkdir(*v41AlignedOutput, 0700); err != nil {
		t.Fatal(err)
	}
	c := &v41AlignedCapture{root: *v41AlignedOutput}
	c.manifest(t)
	return c
}

func (c *v41AlignedCapture) manifest(t *testing.T) {
	t.Helper()
	b, err := json.MarshalIndent(c.files, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(c.root, "manifest.json"), append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}

func (c *v41AlignedCapture) save(t *testing.T, name string, value any) {
	t.Helper()
	var b []byte
	encoding := "bytes"
	switch v := value.(type) {
	case []byte:
		b = v
	case []float32:
		encoding = "float32-le"
		b = make([]byte, len(v)*4)
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
		}
	case []int32:
		encoding = "int32-le"
		b = make([]byte, len(v)*4)
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[i*4:], uint32(x))
		}
	default:
		t.Fatalf("unsupported vector type %T", value)
	}
	path := filepath.Join(t.Name(), name+".bin")
	abs := filepath.Join(c.root, path)
	if err := os.MkdirAll(filepath.Dir(abs), 0700); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(abs, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	n, writeErr := f.Write(b)
	closeErr := f.Close()
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if n != len(b) {
		t.Fatal("short vector write")
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	stored, err := os.ReadFile(abs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(stored, b) {
		t.Fatal("vector readback mismatch")
	}
	h := sha256.Sum256(b)
	c.files = append(c.files, v41AlignedVector{filepath.ToSlash(path), encoding, len(b), hex.EncodeToString(h[:])})
	c.manifest(t)
}
