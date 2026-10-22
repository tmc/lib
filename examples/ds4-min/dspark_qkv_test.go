package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

// A sparse, byte-level weight fixture avoids allocating the 128MB Q-B
// matrix merely to exercise a single nonzero coefficient. It is not a model
// corpus or an upstream execution oracle.
type dsparkSparseReader struct{ tensors []gguf.Tensor }

func (r dsparkSparseReader) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	for _, tensor := range r.tensors {
		bytes, err := tensor.Bytes()
		if err != nil {
			return 0, err
		}
		if uint64(off) < tensor.Offset || uint64(off) >= tensor.Offset+bytes {
			continue
		}
		if strings.Contains(tensor.Name, "norm.weight") {
			for i := 0; i < len(p); i += 4 {
				binary.LittleEndian.PutUint32(p[i:], math.Float32bits(1))
			}
		}
		if (strings.HasSuffix(tensor.Name, "attn_kv.weight") || strings.HasSuffix(tensor.Name, "attn_q_a.weight") || strings.HasSuffix(tensor.Name, "attn_q_b.weight")) && uint64(off) == tensor.Offset {
			binary.LittleEndian.PutUint32(p, math.Float32bits(1))
		}
		return len(p), nil
	}
	return 0, fmt.Errorf("unknown sparse offset %d", off)
}

func TestDSparkQKVActualWeightPath(t *testing.T) {
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
	m.reader = dsparkSparseReader{g.Tensors}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.beginCapture(127); err != nil {
		t.Fatal(err)
	}
	input := make([]float32, (m.block+1)*Flash.Embed*Flash.HC)
	for i := range input {
		input[i] = 1
	}
	positions := make([]int, m.block+1)
	positions[0] = 127
	for row := 1; row < len(positions); row++ {
		positions[row] = 126 + row
	}
	queries, kv, post, combine, err := s.stageQKV(0, input, positions)
	if err != nil {
		t.Fatal(err)
	}
	if len(queries) != m.block || len(kv) != m.block+1 || len(post) != m.block || len(combine) != m.block {
		t.Fatal("feature/draft row counts")
	}
	for _, q := range queries {
		if math.Abs(float64(q[0])-math.Sqrt(512)) > 1e-4 {
			t.Fatalf("query%g", q[0])
		}
		if q[1] != 0 {
			t.Fatal("unexpected query coordinate")
		}
	}
	for _, k := range kv {
		if k[0] != 22 || k[1] != 0 {
			t.Fatalf("KV%g,%g", k[0], k[1])
		}
	}
	positions[1]++
	if _, _, _, _, err := s.stageQKV(0, input, positions); err == nil {
		t.Fatal("wrong positional pairing admitted")
	}
}
