package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestModelStateIdentityCache(t *testing.T) {
	g := &gguf.File{Tensors: []gguf.Tensor{{Name: "weight", TypeID: 1, Dimensions: []uint64{4}}}}
	data := []byte("weights")
	c := new(modelStateIdentityCache)
	m, q, err := c.get(bytes.NewReader(data), int64(len(data)), g)
	if err != nil || m != sha256.Sum256(data) || q == ([32]byte{}) {
		t.Fatalf("identity=%x/%x err=%v", m, q, err)
	}
	data[0] = 'W'
	m2, q2, err := c.get(bytes.NewReader(data), int64(len(data)), g)
	if err != nil || m2 != m || q2 != q {
		t.Fatal("shared cache recomputed immutable model")
	}
	m3, _, err := new(modelStateIdentityCache).get(bytes.NewReader(data), int64(len(data)), g)
	if err != nil || m3 == m {
		t.Fatal("different model accepted as same identity")
	}
	g.Tensors[0].TypeID++
	_, q3, err := new(modelStateIdentityCache).get(bytes.NewReader(data), int64(len(data)), g)
	if err != nil || q3 == q {
		t.Fatal("different quant profile accepted as same identity")
	}
	bad := new(modelStateIdentityCache)
	_, _, err = bad.get(bytes.NewReader(data), int64(len(data)+1), g)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated model err=%v", err)
	}
	_, _, err = bad.get(bytes.NewReader(data), int64(len(data)), g)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal("failed cached hash was silently retried")
	}
}
