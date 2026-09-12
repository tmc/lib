package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
)

func normalizeLogitsDigest(value string) (string, error) {
	digest, err := hex.DecodeString(value)
	if err != nil || len(digest) != sha256.Size {
		return "", fmt.Errorf("expected %d hexadecimal digits", 2*sha256.Size)
	}
	return hex.EncodeToString(digest), nil
}

// checkLogits hashes the exact float32 bits, in the oracle's little-endian
// encoding. An unset expectation leaves the ordinary generation path alone.
func checkLogits(phase string, logits []float32, vocab int, want string, log io.Writer) error {
	if want == "" {
		return nil
	}
	if len(logits) != vocab || vocab == 0 {
		return fmt.Errorf("%s logits: got %d values, want %d", phase, len(logits), vocab)
	}
	h := sha256.New()
	var buf [4]byte
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%s logits: non-finite value at %d", phase, i)
		}
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		h.Write(buf[:])
	}
	got := fmt.Sprintf("%x", h.Sum(nil))
	fmt.Fprintf(log, "ds4-min: %s logits sha256=%s\n", phase, got)
	if got != want {
		return fmt.Errorf("%s logits sha256=%s, want %s", phase, got, want)
	}
	return nil
}

func (cfg config) checkingLogits() bool {
	return cfg.expectPrefillLogits != "" || cfg.expectFinalLogits != ""
}

func (cfg config) logLogitsWorkload(prompt []int, log io.Writer) {
	if !cfg.checkingLogits() {
		return
	}
	h := sha256.New()
	var buf [4]byte
	for _, id := range prompt {
		binary.LittleEndian.PutUint32(buf[:], uint32(id))
		h.Write(buf[:])
	}
	fmt.Fprintf(log, "ds4-min: logits check backend=%s model=%q context=%d prefill_tokens=%d requested_tokens=%d token_ids_sha256=%x\n", cfg.backend, cfg.model, cfg.context, len(prompt), cfg.tokens, h.Sum(nil))
}

func (cfg config) checkFinalLogits(logits []float32, generated, vocab int, log io.Writer) error {
	if err := checkLogits("final", logits, vocab, cfg.expectFinalLogits, log); err != nil {
		return err
	}
	if cfg.expectFinalLogits != "" && generated != cfg.tokens {
		return fmt.Errorf("final logits: generated %d tokens, want %d", generated, cfg.tokens)
	}
	return nil
}
