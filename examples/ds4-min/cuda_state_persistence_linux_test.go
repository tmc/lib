//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestValidateCUDAStateLayerCounts(t *testing.T) {
	var s cudaSessionState
	s.layers[0].attn.cap = 3
	s.layers[0].indexer.cap = 1
	attn, indexer := make([]int, len(s.layers)), make([]int, len(s.layers))
	attn[0], indexer[0] = 3, 1
	attentionOverflow := append([]int(nil), attn...)
	attentionOverflow[0] = 4
	indexerOverflow := append([]int(nil), indexer...)
	indexerOverflow[0] = 2
	tests := []struct {
		name          string
		attn, indexer []int
		wantErr       bool
	}{
		{name: "within capacity", attn: attn, indexer: indexer},
		{name: "attention overflow", attn: attentionOverflow, indexer: indexer, wantErr: true},
		{name: "indexer overflow", attn: attn, indexer: indexerOverflow, wantErr: true},
		{name: "missing layer", attn: nil, indexer: nil, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCUDAStateLayerCounts(s, tt.attn, tt.indexer)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateCUDAStateLayerCounts error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCUDAStatePersistenceHostField(t *testing.T) {
	f := cudaStateField{name: "logits", kind: cudaStateFieldFloat, hostFloats: []float32{1, -2.5, float32(math.SmallestNonzeroFloat32)}}
	b, err := cudaStateFieldBytes(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 12 || binary.LittleEndian.Uint32(b[0:]) != math.Float32bits(1) || binary.LittleEndian.Uint32(b[4:]) != math.Float32bits(-2.5) || binary.LittleEndian.Uint32(b[8:]) != math.Float32bits(math.SmallestNonzeroFloat32) {
		t.Fatalf("encoded host field = %x", b)
	}
	if err := applyCUDAField(f, b); err != nil {
		t.Fatal(err)
	}
	if f.hostFloats[1] != -2.5 {
		t.Fatalf("decoded host field = %v", f.hostFloats)
	}
}

func TestCUDAStatePersistenceRejectsMalformedFields(t *testing.T) {
	if _, _, err := readCUDAStateFields([]byte{1, 2, 3}, 1); err == nil {
		t.Fatal("short field header accepted")
	}
	var h [20]byte
	binary.LittleEndian.PutUint16(h[0:2], 1)
	binary.LittleEndian.PutUint64(h[12:20], 1)
	if _, _, err := readCUDAStateFields(h[:], 1); err == nil {
		t.Fatal("truncated field payload accepted")
	}

	f := cudaStateField{name: "bytes", kind: cudaStateFieldByte, hostFloats: nil}
	if err := applyCUDAField(f, bytes.Repeat([]byte{0}, 4)); err == nil {
		t.Fatal("byte field with wrong length accepted")
	}
}
