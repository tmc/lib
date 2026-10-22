package main

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

type dsparkIdentityReader struct {
	normOffset int64
	columns    int
}

func (r dsparkIdentityReader) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	if off == r.normOffset {
		for i := 0; i < len(p); i += 4 {
			binary.LittleEndian.PutUint32(p[i:], math.Float32bits(1))
		}
	} else {
		row := int(off) / (r.columns * 4)
		binary.LittleEndian.PutUint32(p[row*4:], math.Float32bits(1))
	}
	return len(p), nil
}

func TestDSparkStageInputActualProjection(t *testing.T) {
	m, err := admitDSparkModel(supportFixture())
	if err != nil {
		t.Fatal(err)
	}
	columns := len(m.targets) * Flash.Embed
	normOffset := int64(columns * Flash.Embed * 4)
	m.reader = dsparkIdentityReader{normOffset, columns}
	m.tensors["mtp.0.main_proj.weight"] = gguf.Tensor{Name: "mtp.0.main_proj.weight", TypeID: 0, Dimensions: []uint64{uint64(columns), uint64(Flash.Embed)}}
	m.tensors["mtp.0.main_norm.weight"] = gguf.Tensor{Name: "mtp.0.main_norm.weight", TypeID: 0, Offset: uint64(normOffset), Dimensions: []uint64{uint64(Flash.Embed)}}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.beginCapture(127); err != nil {
		t.Fatal(err)
	}
	for i := range s.features {
		s.features[i] = 2
	}
	for i := range s.captured {
		s.captured[i] = true
	}
	embeddings := make([][]float32, m.block)
	for row := range embeddings {
		embeddings[row] = make([]float32, Flash.Embed)
		for i := range embeddings[row] {
			embeddings[row][i] = float32(row + 3)
		}
	}
	input, positions, err := s.stageInput(embeddings)
	if err != nil {
		t.Fatal(err)
	}
	if positions[0] != 127 || positions[1] != 127 || positions[2] != 128 {
		t.Fatalf("positions%v", positions)
	}
	width := Flash.Embed * Flash.HC
	for lane := 0; lane < Flash.HC; lane++ {
		if math.Abs(float64(input[lane*Flash.Embed])-1) > 1e-6 {
			t.Fatal("feature projection")
		}
		for row := range embeddings {
			if input[(row+1)*width+lane*Flash.Embed] != float32(row+3) {
				t.Fatal("embedding HC layout")
			}
		}
	}
}

func TestDSparkPrivateStateCapture(t *testing.T) {
	m, err := admitDSparkModel(supportFixture())
	if err != nil {
		t.Fatal(err)
	}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.beginCapture(127); err != nil {
		t.Fatal(err)
	}
	hc := make([]float32, Flash.Embed*Flash.HC)
	for lane := 0; lane < Flash.HC; lane++ {
		for i := 0; i < Flash.Embed; i++ {
			hc[lane*Flash.Embed+i] = float32(lane + 1)
		}
	}
	if err := s.capture(1, 127, hc); err != nil {
		t.Fatal(err)
	}
	if s.features[0] != 2.5 || !s.captured[0] || s.captured[1] {
		t.Fatal("feature capture layout")
	}
	if err := s.capture(20, 128, hc); err == nil {
		t.Fatal("mixed positions accepted")
	}
	c := s.clone()
	c.features[0] = 99
	c.raw[0][0] = 99
	c.captured[0] = false
	if s.features[0] != 2.5 || s.raw[0][0] != 0 || !s.captured[0] {
		t.Fatal("request snapshot alias")
	}
}

func TestDSparkPrivateRawRing(t *testing.T) {
	m, err := admitDSparkModel(supportFixture())
	if err != nil {
		t.Fatal(err)
	}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]float32, Flash.HeadDim*2)
	for i := 0; i < Flash.HeadDim; i++ {
		rows[i] = 1
		rows[Flash.HeadDim+i] = 2
	}
	if err := s.writeRaw(1, 7, rows); err != nil {
		t.Fatal(err)
	}
	if s.raw[1][7*Flash.HeadDim] != 1 || s.raw[1][0] != 2 || s.raw[0][0] != 0 {
		t.Fatal("stage/ring coordinates")
	}
	rows[0] = float32(math.NaN())
	if err := s.writeRaw(1, 7, rows); err == nil {
		t.Fatal("NaN ring accepted")
	}
	if s.raw[1][7*Flash.HeadDim] != 1 {
		t.Fatal("failed write mutated ring")
	}
}
