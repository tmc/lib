package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// sampledDecision records the actual sampler result, before any evaluation
// of that token. A stop decision consumes RNG but does not advance KV state.
type sampledDecision struct {
	Step          int    `json:"step"`
	Token         int    `json:"token"`
	RNGBefore     uint64 `json:"rng_before"`
	RNGAfter      uint64 `json:"rng_after"`
	Stop          bool   `json:"stop"`
	HistorySHA256 string `json:"history_sha256"`
	LogitsSHA256  string `json:"logits_sha256"`
}

func writeSampledDecision(dir string, step, token int, before, after uint64, stop bool, history []int, logits []float32) error {
	if step < 0 || token < 0 || token >= len(logits) || len(history) == 0 {
		return fmt.Errorf("sampled trace: invalid decision geometry")
	}
	if e := nonFiniteExtent(logits); e.bad() {
		return fmt.Errorf("sampled trace: non-finite %v", e)
	}
	data := make([]byte, 4*len(logits))
	for i, v := range logits {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	h := sha256.New()
	var id [4]byte
	for _, v := range history {
		if v < 0 || int64(v) > math.MaxInt32 {
			return fmt.Errorf("sampled trace: invalid history token")
		}
		binary.LittleEndian.PutUint32(id[:], uint32(v))
		h.Write(id[:])
	}
	sum := sha256.Sum256(data)
	record := sampledDecision{step, token, before, after, stop, hex.EncodeToString(h.Sum(nil)), hex.EncodeToString(sum[:])}
	metadata, err := json.Marshal(record)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	base := filepath.Join(dir, fmt.Sprintf("step-%04d", step))
	// Exclusive creation refuses stale evidence rather than mixing attempts.
	for _, file := range []struct {
		path string
		data []byte
	}{{base + ".f32", data}, {base + ".json", append(metadata, '\n')}} {
		f, err := os.OpenFile(file.path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return fmt.Errorf("sampled trace: create: %w", err)
		}
		_, writeErr := f.Write(file.data)
		closeErr := f.Close()
		if writeErr != nil {
			return writeErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}
