//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// TestCUDADenseQ8D2RBlockDiagByteEqual checks that a single unified
// ds4_dense_q8_d2r_blockdiag launch produces identical outputs to 8 separate
// ds4_dense_q8_d2r launches on identical Q8_1 activations and Q8_0 weights.
//
// As requested by the test harness specification:
// 1. The input does NOT have 8 identical blocks, ensuring stride or offset bugs fail.
// 2. A negative control proves that the test fails when a block offset is corrupted.
func TestCUDADenseQ8D2RBlockDiagByteEqual(t *testing.T) {
	r := cudaSynthFoundation(t)

	const (
		m         = 1024
		n         = 5
		k         = 4096
		numBlocks = 8
		outStride = m * numBlocks
	)

	nblk := uint64(m) * uint64(k/32)
	alignedBlockBytes := denseQ8AlignedBytes(m, k)
	totalWBytes := alignedBlockBytes * numBlocks

	blockQBlocks := uint64(k/128) * uint64(n)
	reqQ8Blocks := blockQBlocks*uint64(numBlocks) + 128
	totalQBytes := int(reqQ8Blocks * 144)

	rnd := rand.New(rand.NewSource(42))

	wBuf := cudaExpertAllocBytes(t, r, totalWBytes)

	// Repack 8 non-identical blocks of Q8_0 weights
	for g := 0; g < numBlocks; g++ {
		rawW := cudaSynthQ80(rnd, m, k, -8)
		rawWDev := cudaExpertAllocBytes(t, r, len(rawW))
		if err := rawWDev.CopyFromHost(rawW); err != nil {
			t.Fatalf("copy rawW: %v", err)
		}
		dstSlice := wBuf.Slice(g*alignedBlockBytes, (g+1)*alignedBlockBytes)
		if err := r.repackQ80(dstSlice, rawWDev.Ptr(), nblk); err != nil {
			t.Fatalf("repack block %d: %v", g, err)
		}
	}

	// Generate non-zero float activations and quantize them into MMQ layout
	// across the 8 blocks.
	actStride := numBlocks * k
	x := make([]float32, n*actStride)
	for i := range x {
		x[i] = float32((i*37+i/17)%257-128) / 128.0
	}
	inDev := cudaSynthUpload(t, r, x)

	qBuf := cudaExpertAllocBytes(t, r, totalQBytes)
	if err := r.encodeQuantMMQBlocks(inDev, qBuf, actStride, k, n, k, numBlocks); err != nil {
		t.Fatalf("encodeQuantMMQBlocks: %v", err)
	}

	totalOutFloats := outStride * n
	wantOut := cudaTestAlloc(t, r, totalOutFloats)
	gotOut := cudaTestAlloc(t, r, totalOutFloats)

	// Baseline: 8 separate ds4_dense_q8_d2r launches
	for g := 0; g < numBlocks; g++ {
		wG := wBuf.Slice(g*alignedBlockBytes, (g+1)*alignedBlockBytes)
		qG := qBuf.Slice(g*int(blockQBlocks*144), qBuf.Len())
		outG := wantOut.Slice(g*m, wantOut.Len())
		if err := r.encodeDenseQ8D2R(wG, qG, outG, m, n, k, outStride); err != nil {
			t.Fatalf("baseline block %d launch failed: %v", g, err)
		}
	}

	// Candidate: 1 unified ds4_dense_q8_d2r_blockdiag launch
	if err := r.encodeDenseQ8D2RBlockDiag(wBuf, qBuf, gotOut, m, n, k, outStride, numBlocks); err != nil {
		t.Fatalf("blockdiag launch failed: %v", err)
	}

	wantFloats := make([]float32, totalOutFloats)
	gotFloats := make([]float32, totalOutFloats)
	if err := wantOut.CopyToHost(wantFloats); err != nil {
		t.Fatalf("copy wantOut: %v", err)
	}
	if err := gotOut.CopyToHost(gotFloats); err != nil {
		t.Fatalf("copy gotOut: %v", err)
	}

	nonZeroCount := 0
	for i := range wantFloats {
		if wantFloats[i] != 0.0 {
			nonZeroCount++
		}
		if wantFloats[i] != gotFloats[i] {
			t.Fatalf("blockdiag float mismatch at index %d: want %v, got %v", i, wantFloats[i], gotFloats[i])
		}
	}
	if nonZeroCount == 0 {
		t.Fatalf("test produced all zeros across %d floats", totalOutFloats)
	}

	// Byte-for-byte check of the memory buffers
	wantBytes := make([]byte, totalOutFloats*4)
	gotBytes := make([]byte, totalOutFloats*4)
	for i, v := range wantFloats {
		binary.LittleEndian.PutUint32(wantBytes[i*4:], math.Float32bits(v))
	}
	for i, v := range gotFloats {
		binary.LittleEndian.PutUint32(gotBytes[i*4:], math.Float32bits(v))
	}
	if !bytes.Equal(wantBytes, gotBytes) {
		t.Fatalf("byte mismatch between baseline and blockdiag output")
	}

	// Negative control: corrupt block 3 weights by replacing with another block's weights
	corruptWBuf := cudaExpertAllocBytes(t, r, totalWBytes)
	wBytes := make([]byte, totalWBytes)
	if err := wBuf.CopyToHost(wBytes); err != nil {
		t.Fatalf("copy wBuf: %v", err)
	}
	// Copy block 0 weights over block 3
	copy(wBytes[3*alignedBlockBytes:4*alignedBlockBytes], wBytes[0:alignedBlockBytes])
	if err := corruptWBuf.CopyFromHost(wBytes); err != nil {
		t.Fatalf("copy corruptWBuf: %v", err)
	}

	corruptOut := cudaTestAlloc(t, r, totalOutFloats)
	if err := r.encodeDenseQ8D2RBlockDiag(corruptWBuf, qBuf, corruptOut, m, n, k, outStride, numBlocks); err != nil {
		t.Fatalf("corrupt blockdiag launch failed: %v", err)
	}
	corruptFloats := make([]float32, totalOutFloats)
	if err := corruptOut.CopyToHost(corruptFloats); err != nil {
		t.Fatalf("copy corruptOut: %v", err)
	}

	matched := true
	for i := range wantFloats {
		if wantFloats[i] != corruptFloats[i] {
			matched = false
			break
		}
	}
	if matched {
		t.Fatalf("NEGATIVE CONTROL FAILED: corrupting block 3 weights still produced identical output")
	}
}
