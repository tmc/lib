//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

func TestCUDAUpstreamReadbackBits(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0xff800000, 0x7fc00123, 0x3f800000}
	want := make([]byte, 4*len(bits))
	got := make([]float32, len(bits))
	for i, b := range bits {
		binary.LittleEndian.PutUint32(want[4*i:], b)
		got[i] = math.Float32frombits(b)
	}
	if err := compareCUDAUpstreamSection(2, "attention_state_score", want, got); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		index int
		bits  uint32
	}{
		{"signed zero", 1, 0}, {"negative infinity", 2, 0x7f800000},
		{"NaN payload", 3, 0x7fc00124}, {"one ULP", 4, 0x3f800001},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutant := append([]float32(nil), got...)
			mutant[tc.index] = math.Float32frombits(tc.bits)
			err := compareCUDAUpstreamSection(2, "attention_state_score", want, mutant)
			if err == nil || !strings.Contains(err.Error(), "layer 2 attention_state_score element") {
				t.Fatalf("missing first difference: %v", err)
			}
		})
	}
	if err := compareCUDAUpstreamSection(2, "raw", want[:len(want)-1], got); err == nil {
		t.Fatal("partial float accepted")
	}
	if err := compareCUDAUpstreamSection(2, "raw", want, got[:len(got)-1]); err == nil {
		t.Fatal("short readback accepted")
	}
}

func TestCUDAUpstreamReadbackCounts(t *testing.T) {
	m := dsv4kvstore.UpstreamPayloadMetadata{Tokens: 1, Layers: cudaLayers, Vocab: 1}
	layout := dsv4kvstore.UpstreamPayloadLayout{Metadata: m}
	counts := 52 + int(m.Tokens)*4 + int(m.Vocab)*4
	for _, tc := range []struct {
		name   string
		offset int
	}{
		{"compressed", counts + 4*7},
		{"indexer", counts + 4*cudaLayers + 4*7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := make([]byte, counts+cudaLayers*8)
			binary.LittleEndian.PutUint32(payload[tc.offset:], 1)
			err := verifyCUDAUpstreamImport(payload, layout, &cudaSessionState{})
			if err == nil || !strings.Contains(err.Error(), "layer 7 "+tc.name+" count") {
				t.Fatalf("wrong first failing count: %v", err)
			}
		})
	}
}

func TestCUDAUpstreamReadbackCensus(t *testing.T) {
	sections := []dsv4kvstore.UpstreamPayloadSection{{Layer: 0, Name: "raw"}}
	if err := validateCUDAUpstreamSectionCensus(sections, []int{0}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		sections []dsv4kvstore.UpstreamPayloadSection
	}{
		{"missing", nil},
		{"duplicate", append(append([]dsv4kvstore.UpstreamPayloadSection(nil), sections...), sections[0])},
		{"unexpected", append(append([]dsv4kvstore.UpstreamPayloadSection(nil), sections...), dsv4kvstore.UpstreamPayloadSection{Layer: 0, Name: "indexer"})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateCUDAUpstreamSectionCensus(tc.sections, []int{0}); err == nil {
				t.Fatal("malformed section census accepted")
			}
		})
	}
}
