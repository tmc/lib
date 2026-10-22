package main

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

type dsparkZeroReader struct {
	dsparkSparseReader
	failName string
}

func (r dsparkZeroReader) ReadAt(p []byte, off int64) (int, error) {
	for _, tensor := range r.tensors {
		if tensor.Name == r.failName && uint64(off) == tensor.Offset {
			return 0, fmt.Errorf("planted stage read failure")
		}
	}
	n, err := r.dsparkSparseReader.ReadAt(p, off)
	if err != nil {
		return n, err
	}
	for _, tensor := range r.tensors {
		bytes, err := tensor.Bytes()
		if err != nil {
			return 0, err
		}
		if uint64(off) >= tensor.Offset && uint64(off) < tensor.Offset+bytes && !strings.Contains(tensor.Name, "norm.weight") {
			clear(p)
			break
		}
	}
	return n, nil
}

func TestDSparkFullCPUStageChain(t *testing.T) {
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
	m.reader = dsparkZeroReader{dsparkSparseReader{g.Tensors}, ""}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.beginCapture(127); err != nil {
		t.Fatal(err)
	}
	hc := make([]float32, Flash.Embed*Flash.HC)
	for i := range hc {
		hc[i] = 1
	}
	for _, layer := range m.targets {
		if err := s.capture(layer, 127, hc); err != nil {
			t.Fatal(err)
		}
	}
	embeddings := make([][]float32, m.block)
	for row := range embeddings {
		embeddings[row] = make([]float32, Flash.Embed)
		for i := range embeddings[row] {
			embeddings[row][i] = 1
		}
	}
	hidden, err := s.forwardCPU(embeddings, 0, 0, m.block+1)
	if err != nil {
		t.Fatal(err)
	}
	// Actual full-shape zero projection weights leave the HC residual at 1
	// in every stage. Four head weights (.5+eps) give one constant embedding;
	// the final unit norm makes every coordinate approximately one.
	for _, row := range hidden {
		if len(row) != Flash.Embed {
			t.Fatal("hidden width")
		}
		for _, v := range row {
			if math.Abs(float64(v)-1) > 2e-5 {
				t.Fatalf("full-stage reference %g want1", v)
			}
		}
	}
	for stage := range s.raw {
		for i := range s.raw[stage] {
			s.raw[stage][i] = float32(stage + 7)
		}
	}
	m.reader = dsparkZeroReader{dsparkSparseReader{g.Tensors}, "mtp.1.attn_norm.weight"}
	if _, err := s.forwardCPU(embeddings, 0, 0, m.block+1); err == nil {
		t.Fatal("intermediate stage failure accepted")
	}
	for stage := range s.raw {
		for _, v := range s.raw[stage] {
			if v != float32(stage+7) {
				t.Fatal("stage chain failure left cache changes")
			}
		}
	}
}
