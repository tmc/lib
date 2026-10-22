package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	dsv4bench "github.com/tmc/lib/examples/ds4-min/bench/dsv4bench"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

// dumpFrontierLogits writes the validated JSON frontier artifact used by
// ds4-bench comparisons. The directory form keeps the prefill and final
// vectors separate without changing the existing raw dump contract.
func dumpFrontierLogits(dir, phase string, logits []float32, tokenIDs, generated []int, cfg config) error {
	if dir == "" || (phase != "prefill" && phase != "final") || len(logits) == 0 {
		return fmt.Errorf("invalid frontier logits dump request")
	}
	if err := checkFiniteLogits(phase, logits); err != nil {
		return err
	}
	argmax := sampler.Argmax(logits)
	promptTokens := len(tokenIDs) - len(generated)
	if promptTokens <= 0 {
		return fmt.Errorf("%s frontier logits: missing prompt tokens", phase)
	}
	frontier := dsv4bench.FrontierLogits{
		Source:         "ds4-min",
		Model:          cfg.model,
		Backend:        cfg.backend,
		Quality:        "default",
		QuantBits:      2,
		PromptTokens:   promptTokens,
		FrontierTokens: len(tokenIDs),
		Context:        cfg.context,
		Vocab:          len(logits),
		ArgmaxID:       argmax,
		ArgmaxLogit:    logits[argmax],
		Logits:         append([]float32(nil), logits...),
	}
	var data bytes.Buffer
	if err := dsv4bench.WriteFrontierJSON(&data, frontier); err != nil {
		return fmt.Errorf("encode %s frontier logits: %w", phase, err)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create frontier logits directory: %w", err)
	}
	name := filepath.Join(dir, "frontier_"+phase+".logits.json")
	tmp, err := os.CreateTemp(dir, ".frontier-*.json")
	if err != nil {
		return fmt.Errorf("create %s frontier logits: %w", phase, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data.Bytes()); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write %s frontier logits: %w", phase, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close %s frontier logits: %w", phase, err)
	}
	if err := os.Chmod(tmpName, 0o600); err != nil {
		return fmt.Errorf("protect %s frontier logits: %w", phase, err)
	}
	if err := os.Rename(tmpName, name); err != nil {
		return fmt.Errorf("install %s frontier logits: %w", phase, err)
	}
	return nil
}

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
	// Finiteness is decided over the whole slice before any hashing, so a
	// poisoned draw still produces no digest -- the property the corruption
	// tabulation depends on to keep its two failure classes apart -- while the
	// error reports how far the poisoning extends rather than only where it
	// starts.
	if err := checkFiniteLogits(phase, logits); err != nil {
		return err
	}
	h := sha256.New()
	var buf [4]byte
	for _, v := range logits {
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
	fmt.Fprintf(log, "ds4-min: logits check backend=%s model=%q context=%d prefill_tokens=%d requested_tokens=%d token_ids_sha256=%s\n", cfg.backend, cfg.model, cfg.context, len(prompt), cfg.tokens, tokenIDsHash(prompt))
}

type logitsDumpMetadata struct {
	Schema         int    `json:"schema"`
	Phase          string `json:"phase"`
	Position       int    `json:"position"`
	Vocab          int    `json:"vocab"`
	Bytes          int    `json:"bytes"`
	LogitsSHA256   string `json:"logits_sha256"`
	TokenIDsSHA256 string `json:"token_ids_sha256"`
	PromptTokens   int    `json:"prompt_tokens"`
	GeneratedIDs   []int  `json:"generated_ids,omitempty"`
	Model          string `json:"model"`
	Backend        string `json:"backend"`
}

func tokenIDsHash(prompt []int) string {
	h := sha256.New()
	var buf [4]byte
	for _, id := range prompt {
		binary.LittleEndian.PutUint32(buf[:], uint32(id))
		h.Write(buf[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// dumpLogits writes a raw vector and a sidecar. The prefix is caller-owned;
// phase suffixes keep prefill and final artifacts separate.
func dumpLogits(prefix, phase string, logits []float32, tokenIDs []int, position int, generated []int, cfg config) error {
	for _, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s logits: non-finite value", phase)
		}
	}
	return dumpLogitsRaw(prefix, phase, logits, tokenIDs, position, generated, cfg)
}

// dumpLogitsRaw preserves exact float32 bits, including non-finite values, for
// a failed expectation. The ordinary dump path rejects non-finite artifacts.
func dumpLogitsRaw(prefix, phase string, logits []float32, tokenIDs []int, position int, generated []int, cfg config) error {
	if prefix == "" || phase == "" || len(logits) == 0 {
		return fmt.Errorf("invalid logits dump request")
	}
	data := make([]byte, 4*len(logits))
	h := sha256.New()
	var buf [4]byte
	for i, value := range logits {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		copy(data[4*i:], buf[:])
		h.Write(buf[:])
	}
	base := prefix + "." + phase
	if err := os.WriteFile(base+".f32le", data, 0o600); err != nil {
		return fmt.Errorf("write %s logits: %w", phase, err)
	}
	meta := logitsDumpMetadata{
		Schema: 1, Phase: phase, Position: position, Vocab: len(logits),
		Bytes: len(data), LogitsSHA256: fmt.Sprintf("%x", h.Sum(nil)),
		TokenIDsSHA256: tokenIDsHash(tokenIDs), PromptTokens: len(tokenIDs) - len(generated),
		GeneratedIDs: append([]int(nil), generated...),
		Model:        cfg.model, Backend: cfg.backend,
	}
	encoded, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("encode %s logits metadata: %w", phase, err)
	}
	if err := os.WriteFile(base+".json", append(encoded, '\n'), 0o600); err != nil {
		return fmt.Errorf("write %s logits metadata: %w", phase, err)
	}
	return nil
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

// logitExtent is how far the non-finite values in a logits vector reach.
// The extent is the diagnostic, not the position. A single bad element at the
// front is a layout or offset fault -- an offset, a mapping boundary, a first
// tile -- where a wholly poisoned vector is the memory-pressure corruption
// cudaDenseArtifactLayers describes, and the two want different fixes. An
// instrument that records only the first index reports the same thing for
// both and can distinguish neither, which is why every non-finite check in
// this package goes through nonFiniteExtent rather than stopping at the first
// value it finds. There were five such checks, written independently, and all
// five had the defect.
type logitExtent struct {
	first, last, n, len int
}

// nonFiniteExtent reports the non-finite values in logits. It walks the whole
// slice; the zero count means the vector is clean.
func nonFiniteExtent(logits []float32) logitExtent {
	e := logitExtent{first: -1, last: -1, len: len(logits)}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			if e.first < 0 {
				e.first = i
			}
			e.last, e.n = i, e.n+1
		}
	}
	return e
}

// bad reports whether logits held any non-finite value.
func (e logitExtent) bad() bool { return e.n > 0 }

// String renders the extent as first-last/count/length, so a total poisoning
// and a single bad element at the front do not print the same.
func (e logitExtent) String() string {
	if e.n == 0 {
		return "none"
	}
	return fmt.Sprintf("%d-%d/%d/%d", e.first, e.last, e.n, e.len)
}

// checkFiniteLogits reports the non-finite values in logits, naming the phase
// that produced them.
func checkFiniteLogits(phase string, logits []float32) error {
	if e := nonFiniteExtent(logits); e.bad() {
		return fmt.Errorf("%s logits: non-finite %v", phase, e)
	}
	return nil
}
