//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"math"
)

// A requestTensorRecipe describes deterministic synthetic fixture bytes, not
// an admitted GGUF model. Zero spans are hashed without allocating the tensor.
type requestTensorRecipe struct {
	name       string
	typeID     uint32
	dimensions []uint64
	size       int
	edits      []requestTensorEdit
}

type requestTensorEdit struct {
	offset int
	data   []byte
}

func requestHeadRecipes(poison bool) []requestTensorRecipe {
	const rank = 32
	const q8Block = 34
	output := requestTensorRecipe{
		name: "target.output.weight", typeID: 8,
		dimensions: []uint64{4096, uint64(Flash.Vocab)},
		size:       Flash.Vocab * (4096 / 32) * q8Block,
	}
	if poison {
		output.edits = []requestTensorEdit{{0, []byte{0x00, 0x7e}}}
	}
	w1 := requestTensorRecipe{
		name: "mtp.2.markov_head.markov_w1.weight", typeID: 0,
		dimensions: []uint64{rank, uint64(Flash.Vocab)},
		size:       Flash.Vocab * rank * 4,
	}
	latent := make([]byte, 8)
	binary.LittleEndian.PutUint32(latent, math.Float32bits(1))
	binary.LittleEndian.PutUint32(latent[4:], math.Float32bits(.001))
	for row := 0; row < Flash.Vocab; row++ {
		w1.edits = append(w1.edits, requestTensorEdit{row * rank * 4, latent})
	}
	// Half scale 1, weight q[1] = 1. Activation quantization drops .001
	// beside the maximum 1; an unquantized dot does not.
	selected := make([]byte, q8Block)
	binary.LittleEndian.PutUint16(selected, 0x3c00)
	selected[3] = 1
	w2 := requestTensorRecipe{
		name: "mtp.2.markov_head.markov_w2.weight", typeID: 8,
		dimensions: []uint64{rank, uint64(Flash.Vocab)},
		size:       Flash.Vocab * q8Block,
		edits:      []requestTensorEdit{{q8Block, selected}},
	}
	confidence := requestTensorRecipe{
		name: "mtp.2.confidence_head.proj.weight", typeID: 8,
		dimensions: []uint64{4096 + rank, 1},
		size:       ((4096 + rank) / 32) * q8Block,
		edits:      []requestTensorEdit{{(4096 / 32) * q8Block, selected}},
	}
	return []requestTensorRecipe{output, w1, w2, confidence}
}

func (r requestTensorRecipe) validate() error {
	if r.name == "" || r.size < 1 {
		return fmt.Errorf("request fixture: invalid tensor recipe")
	}
	end := 0
	for _, edit := range r.edits {
		if edit.offset < end || len(edit.data) == 0 || edit.offset > r.size-len(edit.data) {
			return fmt.Errorf("request fixture %s: invalid or overlapping edit", r.name)
		}
		end = edit.offset + len(edit.data)
	}
	return nil
}

func requestHashZeros(h hash.Hash, n int) {
	var zeros [64 << 10]byte
	for n > 0 {
		m := min(n, len(zeros))
		h.Write(zeros[:m])
		n -= m
	}
}

func (r requestTensorRecipe) digest() ([32]byte, error) {
	if err := r.validate(); err != nil {
		return [32]byte{}, err
	}
	h := sha256.New()
	end := 0
	for _, edit := range r.edits {
		requestHashZeros(h, edit.offset-end)
		h.Write(edit.data)
		end = edit.offset + len(edit.data)
	}
	requestHashZeros(h, r.size-end)
	var sum [32]byte
	copy(sum[:], h.Sum(nil))
	return sum, nil
}

func (r requestTensorRecipe) fill(dst []byte) error {
	if err := r.validate(); err != nil {
		return err
	}
	if len(dst) != r.size {
		return fmt.Errorf("request fixture %s: tensor extent %d, want %d", r.name, len(dst), r.size)
	}
	clear(dst)
	for _, edit := range r.edits {
		copy(dst[edit.offset:], edit.data)
	}
	return nil
}
