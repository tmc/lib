package main

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

type dsparkProposalReader struct{ dsparkZeroReader }

func (r dsparkProposalReader) ReadAt(p []byte, off int64) (int, error) {
	n, err := r.dsparkZeroReader.ReadAt(p, off)
	if err != nil {
		return n, err
	}
	for _, tensor := range r.tensors {
		if tensor.Name == "mtp.2.markov_head.markov_w1.weight" && uint64(off) == tensor.Offset+2*tensor.Dimensions[0]*4 {
			binary.LittleEndian.PutUint32(p, math.Float32bits(1))
		}
		if tensor.Name == "mtp.2.markov_head.markov_w2.weight" && uint64(off) == tensor.Offset+7*tensor.Dimensions[0]*4 {
			binary.LittleEndian.PutUint32(p, math.Float32bits(4))
		}
	}
	return n, nil
}

type dsparkBaseRowsReader struct {
	outputOffset uint64
	embeddingIDs []int
}

func (r *dsparkBaseRowsReader) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	if uint64(off) < r.outputOffset {
		r.embeddingIDs = append(r.embeddingIDs, int(off)/(Flash.Embed*4))
	}
	return len(p), nil
}

func TestDSparkCPUProposalProducer(t *testing.T) {
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
	m.reader = dsparkProposalReader{dsparkZeroReader{dsparkSparseReader{g.Tensors}, ""}}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.beginCapture(128); err != nil {
		t.Fatal(err)
	}
	for _, layer := range m.targets {
		if err := s.capture(layer, 128, make([]float32, Flash.Embed*Flash.HC)); err != nil {
			t.Fatal(err)
		}
	}
	reader := &dsparkBaseRowsReader{outputOffset: uint64(Flash.Embed) * uint64(Flash.Vocab) * 4}
	base := &dsparkModel{reader: reader, tensors: map[string]gguf.Tensor{
		"token_embd.weight": {Name: "token_embd.weight", TypeID: 0, Dimensions: []uint64{uint64(Flash.Embed), uint64(Flash.Vocab)}},
		"output.weight":     {Name: "output.weight", TypeID: 0, Offset: reader.outputOffset, Dimensions: []uint64{uint64(Flash.Embed), uint64(Flash.Vocab)}},
	}}
	ids, probabilities, err := s.proposalCPU(base, 2, 128, 2, 0, 0, 0, m.block+1)
	if err != nil || len(ids) != 2 || ids[0] != 7 || ids[1] != 0 || len(probabilities) != 2 || probabilities[0] != .5 {
		t.Fatalf("actual reference proposal %v %v %v", ids, probabilities, err)
	}
	if len(reader.embeddingIDs) != m.block || reader.embeddingIDs[0] != 2 {
		t.Fatal("seed embedding not read")
	}
	for _, id := range reader.embeddingIDs[1:] {
		if id != m.noise {
			t.Fatal("draft noise embedding not read")
		}
	}
}
