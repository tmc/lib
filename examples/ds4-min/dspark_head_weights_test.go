package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

type dsparkZeroOutputReader struct{}

func (dsparkZeroOutputReader) ReadAt(p []byte, _ int64) (int, error) {
	clear(p)
	return len(p), nil
}

func TestDSparkAdmittedHeadWeights(t *testing.T) {
	m := &dsparkModel{stages: 3, block: 2, rank: 1, tensors: make(map[string]gguf.Tensor)}
	var data []byte
	add := func(name string, dims []uint64, values []float32) {
		m.tensors[name] = gguf.Tensor{Name: name, TypeID: 0, Offset: uint64(len(data)), Dimensions: dims}
		for _, v := range values {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
	}
	w1, w2 := make([]float32, Flash.Vocab), make([]float32, Flash.Vocab)
	w1[2], w2[7] = 1, 4
	add("mtp.2.markov_head.markov_w1.weight", []uint64{1, uint64(Flash.Vocab)}, w1)
	add("mtp.2.markov_head.markov_w2.weight", []uint64{1, uint64(Flash.Vocab)}, w2)
	confidence := make([]float32, Flash.Embed+1)
	confidence[Flash.Embed] = 1
	add("mtp.2.confidence_head.proj.weight", []uint64{uint64(len(confidence)), 1}, confidence)
	m.reader = bytes.NewReader(data)
	base := &dsparkModel{reader: dsparkZeroOutputReader{}, tensors: map[string]gguf.Tensor{"output.weight": {Name: "output.weight", TypeID: 0, Dimensions: []uint64{uint64(Flash.Embed), uint64(Flash.Vocab)}}}}
	hidden := [][]float32{make([]float32, Flash.Embed), make([]float32, Flash.Embed)}
	ids, probabilities, err := m.headCPU(base, hidden, 2, 2, 0)
	if err != nil || len(ids) != 2 || ids[0] != 7 || ids[1] != 0 || math.Abs(float64(probabilities[0])-1/(1+math.Exp(-1))) > 1e-7 || probabilities[1] != .5 {
		t.Fatalf("sequential actual head %v %v %v", ids, probabilities, err)
	}
	ids, _, err = m.headCPU(base, hidden, 2, 2, .6)
	if err != nil || len(ids) != 1 || ids[0] != 7 {
		t.Fatal("confidence prefix gate")
	}
	// An actual stored W2 coefficient changes the first proposed token. The
	// second still reads the newly selected token's W1 row, not frozen seed 2.
	tensor, _ := m.tensor("mtp.2.markov_head.markov_w2.weight")
	binary.LittleEndian.PutUint32(data[tensor.Offset+9*4:], math.Float32bits(5))
	ids, _, err = m.headCPU(base, hidden, 2, 2, 0)
	if err != nil || len(ids) != 2 || ids[0] != 9 || ids[1] != 0 {
		t.Fatal("stored Markov mutation or sequential state inactive")
	}
}
