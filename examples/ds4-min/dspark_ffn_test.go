package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestDSparkExpertBinding(t *testing.T) {
	data := make([]byte, 3*2*34)
	for expert := 0; expert < 3; expert++ {
		for row := 0; row < 2; row++ {
			b := data[(expert*2+row)*34:]
			binary.LittleEndian.PutUint16(b, 0x3c00)
			for i := 0; i < 32; i++ {
				b[2+i] = byte(expert*2 + row + 1)
			}
		}
	}
	m := &dsparkModel{reader: bytes.NewReader(data), tensors: map[string]gguf.Tensor{"expert": {Name: "expert", TypeID: 8, Dimensions: []uint64{32, 2, 3}}}}
	row, err := m.expertRow("expert", 2, 1)
	if err != nil || row[0] != 6 || row[31] != 6 {
		t.Fatalf("expert stride: %v %v", row, err)
	}
	wrong, err := m.expertRow("expert", 1, 1)
	if err != nil || wrong[0] != 4 || wrong[0] == row[0] {
		t.Fatal("expert-vs-row mutation not observable")
	}
	if _, err := m.expertRow("expert", 3, 0); err == nil {
		t.Fatal("out of range expert accepted")
	}
	tensor, _ := m.tensor("expert")
	tensor.TypeID, tensor.Dimensions = 10, []uint64{256, 1, 1}
	m.tensors["expert"] = tensor
	q2 := make([]byte, 84)
	for i := 0; i < 16; i++ {
		q2[i] = 0x10
	}
	binary.LittleEndian.PutUint16(q2[82:], 0x3c00)
	m.reader = bytes.NewReader(q2)
	row, err = m.expertRow("expert", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range row {
		if v != -1 {
			t.Fatal("Q2 minimum term")
		}
	}
}

func TestDSparkRouterAndClampReference(t *testing.T) {
	ids, weights, err := dsparkSelect([]float32{0, 1, 2}, []float32{10, 0, -10}, 2)
	if err != nil || ids[0] != 0 || ids[1] != 1 {
		t.Fatalf("biased selection %v %v", ids, err)
	}
	a, b := math.Sqrt(math.Log(2)), math.Sqrt(math.Log1p(math.Exp(1)))
	if math.Abs(float64(weights[0])-a/(a+b)*float64(Flash.ExpertWeightScale)) > 1e-7 {
		t.Fatal("bias leaked into normalized routing weight")
	}
	ids, _, err = dsparkSelect([]float32{0, 0, 0}, []float32{0, 0, 0}, 2)
	if err != nil || ids[0] != 0 || ids[1] != 1 {
		t.Fatal("router ties are not lowest-id first")
	}
	x, err := dsparkSwiGLU([]float32{-30, 30}, []float32{20, -20}, .25)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{-30 / (1 + math.Exp(30)) * 10 * .25, 10 / (1 + math.Exp(-10)) * -10 * .25}
	for i := range x {
		if math.Abs(float64(x[i])-want[i]) > 1e-6 {
			t.Fatalf("asymmetric clamp %v want %v", x, want)
		}
	}
}

type dsparkFFNReader struct{ dsparkSparseReader }

func (r dsparkFFNReader) ReadAt(p []byte, off int64) (int, error) {
	n, err := r.dsparkSparseReader.ReadAt(p, off)
	if err != nil {
		return n, err
	}
	for _, tensor := range r.tensors {
		if !strings.Contains(tensor.Name, "_exps.weight") {
			continue
		}
		rowBytes := tensor.Dimensions[0] / 32 * 34
		expertBytes := rowBytes * tensor.Dimensions[1]
		if uint64(off) >= tensor.Offset && (uint64(off)-tensor.Offset)%expertBytes == 0 && uint64(off) < tensor.Offset+expertBytes*tensor.Dimensions[2] {
			binary.LittleEndian.PutUint16(p, 0x3c00)
			p[2] = 1
		}
	}
	return n, nil
}

func TestDSparkFFNAdmittedStage(t *testing.T) {
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
	m.reader = dsparkFFNReader{dsparkSparseReader{g.Tensors}}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	input := make([][]float32, m.block)
	for row := range input {
		input[row] = make([]float32, Flash.Embed*Flash.HC)
		for i := range input[row] {
			input[row][i] = 1
		}
	}
	out, err := s.stageFFN(0, input)
	if err != nil {
		t.Fatal(err)
	}
	// Six equal router weights sum to 1.5. Each admitted expert contains
	// one unit coefficient in gate/up/down; the shared expert is zero.
	want := 1 + float64(Flash.ExpertWeightScale)/(1+math.Exp(-1))
	for _, row := range out {
		if math.Abs(float64(row[0])-want) > 2e-5 || math.Abs(float64(row[1])-1) > 2e-5 {
			t.Fatalf("routed/shared HC %v want %g", row[:2], want)
		}
	}
}
