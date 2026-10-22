//go:build linux

package main

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestCUDAQuantMMQBlocksByteEqual checks the eight-slice launch used by the
// attention-output A path against the existing one-slice launch. The slices
// are interleaved exactly as cudaBatchDense.attention stores them, and the
// output is compared byte-for-byte rather than by widened values.
func TestCUDAQuantMMQBlocksByteEqual(t *testing.T) {
	r := cudaSynthFoundation(t)
	const (
		slices = 8
		tokens = 5
		cols   = 4096
	)
	stride := slices * cols
	sliceStride := cols
	rnd := rand.New(rand.NewSource(1))
	x := make([]float32, tokens*stride)
	for i := range x {
		x[i] = float32(rnd.Intn(2001)-1000) / 1000
	}
	in := cudaSynthUpload(t, r, x)
	blockBytes := (cols / 128) * tokens * 144
	want := cudaExpertAllocBytes(t, r, slices*blockBytes)
	got := cudaExpertAllocBytes(t, r, slices*blockBytes)
	for g := 0; g < slices; g++ {
		if err := r.encodeQuantMMQ(in.Slice(g*sliceStride, in.Len()), want.Slice(g*blockBytes, (g+1)*blockBytes), stride, cols, tokens); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.encodeQuantMMQBlocks(in, got, stride, cols, tokens, sliceStride, slices); err != nil {
		t.Fatal(err)
	}
	wantBytes := make([]byte, want.Len())
	gotBytes := make([]byte, got.Len())
	if err := want.CopyToHost(wantBytes); err != nil {
		t.Fatal(err)
	}
	if err := got.CopyToHost(gotBytes); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wantBytes, gotBytes) {
		t.Fatalf("batched quantization differs from eight launches")
	}
}
