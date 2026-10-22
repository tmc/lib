package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

type dsparkOutputReader struct {
	dsparkSparseReader
	outputScale float32
	failOutput  bool
}

func (r dsparkOutputReader) ReadAt(p []byte, off int64) (int, error) {
	for _, tensor := range r.tensors {
		if strings.HasSuffix(tensor.Name, "attn_output_b.weight") && uint64(off) == tensor.Offset && r.failOutput {
			return 0, fmt.Errorf("planted output read failure")
		}
	}
	n, err := r.dsparkSparseReader.ReadAt(p, off)
	if err != nil {
		return n, err
	}
	for _, tensor := range r.tensors {
		if uint64(off) != tensor.Offset {
			continue
		}
		if strings.HasSuffix(tensor.Name, "attn_output_a.weight") {
			binary.LittleEndian.PutUint32(p, math.Float32bits(1))
		}
		if strings.HasSuffix(tensor.Name, "attn_output_b.weight") {
			binary.LittleEndian.PutUint32(p, math.Float32bits(r.outputScale))
		}
	}
	return n, nil
}

func TestDSparkAttentionOutputAndRollback(t *testing.T) {
	g := supportFixture()
	offset := uint64(0)
	for i := range g.Tensors {
		g.Tensors[i].Offset = offset
		bytes, err := g.Tensors[i].Bytes()
		if err != nil {
			t.Fatal(err)
		}
		offset += bytes
	}
	m, err := admitDSparkModel(g)
	if err != nil {
		t.Fatal(err)
	}
	m.reader = dsparkOutputReader{dsparkSparseReader{g.Tensors}, 1, false}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	s.beginCapture(0)
	input := make([]float32, (m.block+1)*Flash.Embed*Flash.HC)
	for i := range input {
		input[i] = 1
	}
	positions := make([]int, m.block+1)
	for row := 1; row < len(positions); row++ {
		positions[row] = row - 1
	}
	baseline, err := s.stageAttention(0, 0, 0, m.block+1, input, positions)
	if err != nil {
		t.Fatal(err)
	}
	// Identical nonrotary keys, a strongly aligned first query, and a zero
	// sink give V=22. The actual sparse A and B rows retain coordinate 0.
	if len(baseline) != m.block || math.Abs(float64(baseline[0][0]-23)) > .001 || math.Abs(float64(baseline[0][1]-1)) > .001 {
		t.Fatalf("projection/residual: %v", baseline[0][:2])
	}
	m.reader = dsparkOutputReader{dsparkSparseReader{g.Tensors}, 2, false}
	mutant, err := s.stageAttention(0, 0, 0, m.block+1, input, positions)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(mutant[0][0]-45)) > .001 || mutant[0][1] != baseline[0][1] {
		t.Fatal("actual B tensor mutation did not change only selected output")
	}
	for i := range s.raw[0] {
		s.raw[0][i] = 7
	}
	m.reader = dsparkOutputReader{dsparkSparseReader{g.Tensors}, 1, true}
	if _, err := s.stageAttention(0, 0, 0, m.block+1, input, positions); err == nil {
		t.Fatal("output failure accepted")
	}
	for _, v := range s.raw[0] {
		if v != 7 {
			t.Fatal("failed stage mutated support cache")
		}
	}
	if _, err := s.stageAttention(0, 1, 0, m.block+1, input, positions); err == nil {
		t.Fatal("append outside visible window accepted")
	}
}
