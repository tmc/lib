package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"sync"

	"github.com/tmc/lib/gguf"
)

// modelStateIdentityCache belongs to an immutable model owner. Request states
// share its pointer, so native persistence hashes the model only once and
// evaluation does not pay any hash cost.
type modelStateIdentityCache struct {
	once         sync.Once
	model, quant [sha256.Size]byte
	err          error
}

func (c *modelStateIdentityCache) get(r io.ReaderAt, size int64, g *gguf.File) ([32]byte, [32]byte, error) {
	if c == nil || r == nil || size <= 0 || g == nil {
		return [32]byte{}, [32]byte{}, errors.New("model state: identity unavailable")
	}
	c.once.Do(func() {
		h := sha256.New()
		n, err := io.Copy(h, io.NewSectionReader(r, 0, size))
		if err != nil {
			c.err = err
			return
		}
		if n != size {
			c.err = io.ErrUnexpectedEOF
			return
		}
		copy(c.model[:], h.Sum(nil))
		h.Reset()
		var word [8]byte
		for _, t := range g.Tensors {
			binary.LittleEndian.PutUint64(word[:], uint64(len(t.Name)))
			h.Write(word[:])
			h.Write([]byte(t.Name))
			binary.LittleEndian.PutUint64(word[:], uint64(t.TypeID))
			h.Write(word[:])
			binary.LittleEndian.PutUint64(word[:], uint64(len(t.Dimensions)))
			h.Write(word[:])
			for _, d := range t.Dimensions {
				binary.LittleEndian.PutUint64(word[:], d)
				h.Write(word[:])
			}
		}
		copy(c.quant[:], h.Sum(nil))
	})
	return c.model, c.quant, c.err
}
