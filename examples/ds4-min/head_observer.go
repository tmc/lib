package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
)

// outputHeadHistory tracks only diagnostic evaluations. Pending history is
// visible to the head producer, but never survives a failed evaluation.
type outputHeadHistory struct {
	committed, pending []int
}

func (h *outputHeadHistory) invalidate() { *h = outputHeadHistory{} }

func (h *outputHeadHistory) begin(position int, tokens []int) (func(bool), error) {
	if h.pending != nil || position != len(h.committed) || len(tokens) == 0 {
		return nil, fmt.Errorf("head observer: missing or noncontiguous evaluated history")
	}
	for _, token := range tokens {
		if token < 0 || int64(token) > math.MaxInt32 {
			return nil, fmt.Errorf("head observer: invalid history token")
		}
	}
	h.pending = append(append([]int(nil), h.committed...), tokens...)
	return func(success bool) {
		if success {
			h.committed = h.pending
			h.pending = nil
		} else {
			h.invalidate()
		}
	}, nil
}

func (h *outputHeadHistory) hash(position int) (string, error) {
	if position < 0 || position >= len(h.pending) {
		return "", fmt.Errorf("head observer: missing pending history")
	}
	data := make([]byte, (position+1)*4)
	for i, token := range h.pending[:position+1] {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(token))
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func writeHeadSnapshot(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("head observer: create snapshot: %w", err)
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return fmt.Errorf("head observer: write snapshot: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("head observer: close snapshot: %w", closeErr)
	}
	return nil
}
