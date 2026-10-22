//go:build linux

package main

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

func TestCUDAUpstreamValues(t *testing.T) {
	m := dsv4kvstore.UpstreamPayloadMetadata{Tokens: 129, Layers: cudaLayers, Vocab: 256}
	layout := dsv4kvstore.UpstreamPayloadLayout{Metadata: m}
	state := cudaSessionState{}
	counts := 52 + int(m.Tokens)*4 + int(m.Vocab)*4
	payload := make([]byte, counts+cudaLayers*8)
	for i := range state.layers {
		if i < 2 {
			continue
		}
		ratio := 4
		if i%2 != 0 {
			ratio = 128
		}
		state.layers[i].attn.ratio, state.layers[i].attn.cap = ratio, 64
		binary.LittleEndian.PutUint32(payload[counts+i*4:], m.Tokens/uint32(ratio))
		if ratio == 4 {
			state.layers[i].indexer.ratio, state.layers[i].indexer.cap = 4, 64
			binary.LittleEndian.PutUint32(payload[counts+cudaLayers*4+i*4:], m.Tokens/4)
		}
	}
	for _, tc := range []struct {
		name      string
		mutate    func([]byte)
		wantError bool
	}{
		{"valid", func([]byte) {}, false},
		{"invalid token", func(b []byte) { binary.LittleEndian.PutUint32(b[52:], m.Vocab) }, true},
		{"nonfinite logits", func(b []byte) { binary.LittleEndian.PutUint32(b[52+int(m.Tokens)*4:], 0x7fc00000) }, true},
		{"missing count", func(b []byte) { binary.LittleEndian.PutUint32(b[counts+2*4:], 0) }, true},
		{"overflow count", func(b []byte) { binary.LittleEndian.PutUint32(b[counts+2*4:], 1000) }, true},
		{"raw layer count", func(b []byte) { binary.LittleEndian.PutUint32(b[counts:], 1) }, true},
		{"indexer count", func(b []byte) { binary.LittleEndian.PutUint32(b[counts+cudaLayers*4+2*4:], 31) }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := append([]byte(nil), payload...)
			tc.mutate(b)
			if err := validateCUDAUpstreamValues(b, layout, &state); (err != nil) != tc.wantError {
				t.Fatalf("error = %v, want error %v", err, tc.wantError)
			}
		})
	}
	if err := validateCUDAUpstreamValues(payload[:len(payload)-1], layout, &state); err == nil {
		t.Fatal("truncated counts accepted")
	}
}

func TestCUDAUpstreamRollback(t *testing.T) {
	copyErr, rollbackErr := errors.New("copy failed"), errors.New("restore failed")
	for _, tc := range []struct {
		name                 string
		copyErr, rollbackErr error
		wantCalls            int
		poison               bool
	}{
		{"commit", nil, nil, 0, false},
		{"rollback", copyErr, nil, 1, false},
		{"failed rollback", copyErr, rollbackErr, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			err := commitCUDAUpstreamState(func() error { return tc.copyErr }, func() error { calls++; return tc.rollbackErr })
			if calls != tc.wantCalls || errors.Is(err, dsv4.ErrDesynchronized) != tc.poison {
				t.Fatalf("calls=%d error=%v", calls, err)
			}
			if tc.copyErr != nil && !errors.Is(err, copyErr) {
				t.Fatal("lost copy failure")
			}
			if tc.rollbackErr != nil && !errors.Is(err, rollbackErr) {
				t.Fatal("lost rollback failure")
			}
		})
	}
}

func TestUpstreamFloatValues(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0x3f800000, 0xff800000}
	b := make([]byte, len(bits)*4)
	for i, v := range bits {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	v, err := upstreamFloatValues(b)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range bits {
		if math.Float32bits(v[i]) != want {
			t.Fatalf("index %d bits differ", i)
		}
	}
	if _, err := upstreamFloatValues(b[:len(b)-1]); err == nil {
		t.Fatal("unaligned floats accepted")
	}
}
