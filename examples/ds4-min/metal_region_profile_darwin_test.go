//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// metalRegionTokens is the batch this instrument profiles, and
// metalRegionContext the context it opens. Both match the CUDA phase
// profile so the two rankings cover the same input at the same
// positions: a ranking taken over a different corpus or a different
// context is not comparable to one taken over this one, and the whole
// point of the instrument is the comparison.
//
// Like the CUDA profile, it says nothing about the positions a
// 2048-token run reaches, because attention cost grows with position and
// the FFN's does not.
const (
	metalRegionTokens  = 64
	metalRegionContext = 1024
)

// metalRegionPrefixSHA256 is the sha256 of the first metalRegionTokens IDs
// as little-endian uint32, registered before any run. It is derived from
// the already-registered 2048-token oracle prefix
// 5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380,
// not fitted to anything observed, and a run whose tokens do not hash to
// it is refused rather than recorded.
//
// This is a gate because the alternative failed: a CUDA phase profile and
// this one were both described as consuming "the retained tokens" and
// hashing the first 64 IDs the same way on both hosts showed they differ
// from the first token. A file hash binds a file; only this binds the
// sequence, and only on both sides does it bind the comparison.
const metalRegionPrefixSHA256 = "8e2073735828d7714e51be5fdc31d2b5b40ed2ce091d226371dba62517354310"

// metalRegionProfileLogitsSHA256 is the SHA-256 digest of the metalRegionVocab
// float32 logits produced by evaluating the first metalRegionTokens tokens of
// the canonical oracle prefix at metalRegionContext.
//
// Unlike the two oracle workload digests, this one is measured: four qualified
// runs on 2026-09-07 produced it on both the instrumented and the reference
// arm, recorded under evidence/worker-DF6906F6/metal-region-*-n64-*/result.json
// with tokens=64, context=1024 and the token prefix this file already pins.
//
// It is the cheap anchor. A restructuring of evalLayer that changes the output
// changes this digest, and reaching it costs one model load rather than the
// 2048-token prefill and 128 decodes the oracle workload needs.
const metalRegionProfileLogitsSHA256 = "d06e9c92a278e6079030cc3a204aff3ace54032beb5695ea25231a683a746d80"

// metalRegionVocab is the logit count both arms must produce. A run that
// returns a different count has not evaluated this model, and comparing
// two such runs to each other proves nothing about it.
const metalRegionVocab = 129280

// expectedMetalRegions returns the spans a run of tokens positions must
// produce, in completion order.
//
// This inventory covers the batched regime, before zero-based position 2051.
// Each position submits an embedding lookup and one buffer per layer. The last
// position also submits the four output-logit operations. Standalone operations
// currently record zero layer and position coordinates.
func expectedMetalRegions(tokens int) []metalRegion {
	if tokens < 1 || tokens > 2051 {
		return nil
	}
	regions := make([]metalRegion, 0, tokens*(ds4Layers+1)+4)
	for position := range tokens {
		regions = append(regions, metalRegion{Region: "get-row-f16"})
		for layer := range ds4Layers {
			regions = append(regions, metalRegion{Region: "layer", Layer: layer, Position: position})
		}
	}
	for _, name := range []string{"rms", "output-head", "rms", "matvec-q80"} {
		regions = append(regions, metalRegion{Region: name})
	}
	return regions
}

// validateMetalRegions refuses a log that does not match the expected
// inventory exactly, coordinate by coordinate and in order.
//
// The check is over coordinates, not durations: an instrument that
// stopped recording, recorded a buffer twice, or attributed a span to the
// wrong layer produces a wrong ranking that reads as a plausible one, and
// only the inventory catches that. Durations are checked only for the
// properties a GPU clock cannot violate.
func validateMetalRegions(tokens int, regions []metalRegion) error {
	want := expectedMetalRegions(tokens)
	if len(want) == 0 {
		return fmt.Errorf("metal region profile: token count %d is outside the batched inventory (1..2051)", tokens)
	}
	if len(regions) != len(want) {
		return fmt.Errorf("metal region profile: got %d spans, want %d", len(regions), len(want))
	}
	for i, got := range regions {
		if got.Region != want[i].Region || got.Layer != want[i].Layer || got.Position != want[i].Position {
			return fmt.Errorf("metal region profile: span %d is (%q,%d,%d), want (%q,%d,%d)",
				i, got.Region, got.Layer, got.Position, want[i].Region, want[i].Layer, want[i].Position)
		}
		if err := got.valid(); err != nil {
			return fmt.Errorf("metal region profile: span %d: %w", i, err)
		}
	}
	// A region whose spans sum to zero carries no timing, which is what a
	// clock that never advanced looks like: every bound passes the
	// per-span checks and the ranking is between two zeros. Reject it
	// here rather than publish a ranking of nothing.
	for name, total := range metalRegionTotals(regions) {
		if !(total > 0) {
			return fmt.Errorf("metal region profile: %s spans total %g s, want a positive total", name, total)
		}
	}
	return nil
}

// valid reports whether a span could have come from a working GPU clock.
// Zero bounds are refused because they are what an unavailable clock
// reports, and an infinite bound is refused before it can subtract into
// a NaN duration that every ordering check then silently passes.
func (r metalRegion) valid() error {
	for name, v := range map[string]float64{"start": r.Start, "end": r.End} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%s bound %g is not finite", name, v)
		}
		if v <= 0 {
			return fmt.Errorf("%s bound %g is not a positive clock reading", name, v)
		}
	}
	if r.End < r.Start {
		return fmt.Errorf("ends %g s before it starts", r.Start-r.End)
	}
	return nil
}

// metalRegionTotals sums the spans by region name.
func metalRegionTotals(regions []metalRegion) map[string]float64 {
	totals := make(map[string]float64)
	for _, r := range regions {
		totals[r.Region] += r.Seconds()
	}
	return totals
}

func TestMetalRegionInventory(t *testing.T) {
	want := expectedMetalRegions(metalRegionTokens)
	if n := len(want); n != metalRegionTokens*(ds4Layers+1)+4 {
		t.Fatalf("inventory has %d spans, want %d", n, metalRegionTokens*(ds4Layers+1)+4)
	}
	fill := func(regions []metalRegion) []metalRegion {
		out := append([]metalRegion(nil), regions...)
		for i := range out {
			out[i].Start = float64(i+1) * 1e-3
			out[i].End = out[i].Start + 1e-4
		}
		return out
	}
	if err := validateMetalRegions(metalRegionTokens, fill(want)); err != nil {
		t.Fatalf("exact inventory rejected: %v", err)
	}
	if err := validateMetalRegions(2052, fill(want)); err == nil {
		t.Fatal("accepted a fallback run as a batched inventory")
	}
	for _, tt := range []struct {
		name    string
		regions []metalRegion
	}{
		{"short", fill(want[:len(want)-1])},
		{"long", fill(append(append([]metalRegion(nil), want...), want[0]))},
		{"old split layer", func() []metalRegion {
			out := fill(want)
			out[1].Region = "opening"
			closing := out[1]
			closing.Region = "closing-ffn"
			split := append([]metalRegion(nil), out[:2]...)
			split = append(split, closing)
			return append(split, out[2:]...)
		}()},
		{"swapped order", func() []metalRegion {
			out := fill(want)
			out[0], out[1] = out[1], out[0]
			return out
		}()},
		{"duplicated span", func() []metalRegion {
			out := fill(want)
			out[3] = out[1]
			return out
		}()},
		{"wrong layer", func() []metalRegion {
			out := fill(want)
			out[2].Layer++
			return out
		}()},
		{"unnamed buffer", func() []metalRegion {
			out := fill(want)
			out[4].Region = ""
			return out
		}()},
		{"span ends before it starts", func() []metalRegion {
			out := fill(want)
			out[5].Start, out[5].End = out[5].End, out[5].Start
			return out
		}()},
		{"NaN bound", func() []metalRegion {
			out := fill(want)
			out[6].End = math.NaN()
			return out
		}()},
		{"zero clock", func() []metalRegion {
			out := fill(want)
			for i := range out {
				out[i].Start, out[i].End = 0, 0
			}
			return out
		}()},
		{"one zero bound", func() []metalRegion {
			out := fill(want)
			out[7].Start = 0
			return out
		}()},
		{"infinite bound", func() []metalRegion {
			out := fill(want)
			out[8].End = math.Inf(1)
			return out
		}()},
		{"infinite difference", func() []metalRegion {
			out := fill(want)
			out[9].Start, out[9].End = math.Inf(1), math.Inf(1)
			return out
		}()},
		{"region with no duration", func() []metalRegion {
			out := fill(want)
			for i := range out {
				if out[i].Region == "layer" {
					out[i].End = out[i].Start
				}
			}
			return out
		}()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateMetalRegions(metalRegionTokens, tt.regions); err == nil {
				t.Fatal("accepted a log the inventory should refuse")
			}
		})
	}
	if err := validateMetalRegions(0, nil); err == nil {
		t.Fatal("accepted a zero token count")
	}
}

// TestMetalRegionProfile ranks the attention opening against the closing
// and FFN, and refuses to report the ranking unless the instrumented run
// reproduces the uninstrumented run's logits bit for bit.
//
// Identity is the gate because the instrument is meant to be inert: it
// reads two properties of a command buffer that has already completed. A
// bit that moves means it is not, and no ranking from it is worth having.
// TestMetalRegionPrefixRegistration checks the registered prefix hash
// against the retained token file, so a wrong constant is caught here
// rather than by a refusal after a model slot has been taken.
func TestMetalRegionPrefixRegistration(t *testing.T) {
	// The fixture is tracked, so a read error is a broken checkout, not an
	// absent optional input. Skipping on it would turn the one control that
	// binds the constant to real bytes into a silent pass.
	input, err := os.ReadFile(metalRegionTokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < metalRegionTokens {
		t.Fatalf("token file holds %d tokens, want at least %d", len(tokens), metalRegionTokens)
	}
	if _, err := checkMetalRegionPrefix(tokens[:metalRegionTokens]); err != nil {
		t.Fatal(err)
	}
}

// TestMetalRegionPrefixRefusal exercises the predicate the production run
// depends on. The registration test above proves only that the constant
// agrees with the fixture; deleting the refusal outright leaves it green.
// This is the control with power over the refusal itself, and it needs no
// model, so it runs on every build rather than behind a gate.
func TestMetalRegionPrefixRefusal(t *testing.T) {
	input, err := os.ReadFile(metalRegionTokensPath)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	canonical = canonical[:metalRegionTokens]

	changed := append([]int(nil), canonical...)
	changed[metalRegionTokens-1]++
	first := append([]int(nil), canonical...)
	first[0]++
	// The CUDA lane's superseded capture, whose first ID is 0 where the
	// canonical corpus begins 36662. This is the corpus confusion that
	// motivated the gate, so it is a case rather than a hypothetical.
	other := append([]int(nil), canonical...)
	other[0] = 0

	for _, c := range []struct {
		name   string
		tokens []int
		want   bool
	}{
		{"canonical", canonical, true},
		{"last token changed", changed, false},
		{"first token changed", first, false},
		{"superseded CUDA first ID", other, false},
		{"one token short", canonical[:metalRegionTokens-1], false},
		{"one token long", append(append([]int(nil), canonical...), 1), false},
		{"empty", nil, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := checkMetalRegionPrefix(c.tokens)
			if c.want {
				if err != nil {
					t.Fatalf("accepted corpus refused: %v", err)
				}
				if got != metalRegionPrefixSHA256 {
					t.Fatalf("hash %s, want %s", got, metalRegionPrefixSHA256)
				}
				return
			}
			if err == nil {
				t.Fatalf("refusal did not fire; hashed to %s", got)
			}
		})
	}
}

// metalRegionTokensPath is the retained canonical corpus.
const metalRegionTokensPath = "../../scripts/testdata/oracle-workload-tokens2048.json"

// checkMetalRegionPrefix returns the hash of the IDs that will be consumed,
// refusing any sequence that is not the registered corpus.
//
// There is no separate length check. An earlier version had one, and a
// mutation check showed it was dead: the encoding is four bytes per ID with
// no terminator, so a sequence of the wrong length hashes differently and the
// comparison below already refuses it. A predicate that cannot fail is worse
// than no predicate, because it reads as protection.
func checkMetalRegionPrefix(tokens []int) (string, error) {
	got := fmt.Sprintf("%x", sha256.Sum256(oracleWorkloadTokenBytes(tokens)))
	if got != metalRegionPrefixSHA256 {
		return got, fmt.Errorf("token prefix is %s, want the registered %s: this is not the corpus this profile is scoped to", got, metalRegionPrefixSHA256)
	}
	return got, nil
}

func TestMetalRegionProfile(t *testing.T) {
	if os.Getenv("DS4_METAL_REGION_PROFILE") == "" {
		t.Skip("DS4_METAL_REGION_PROFILE is not set")
	}
	if os.Getenv("DS4_METAL_REGION_PROFILE") != "1" {
		t.Fatal("DS4_METAL_REGION_PROFILE must be 1")
	}
	required := func(name string) string {
		t.Helper()
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is required", name)
		}
		return value
	}
	model := required("DS4_MODEL")
	tokensPath := required("DS4_PREFILL_TOKENS")
	resultPath := required("DS4_METAL_REGION_PROFILE_OUT")

	// Create the receipt before spending model time, and refuse to
	// overwrite: a profile written over an earlier one leaves no way to
	// tell which run produced the file that survived.
	out, err := os.OpenFile(resultPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	input, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < metalRegionTokens {
		t.Fatalf("token file holds %d tokens, want at least %d", len(tokens), metalRegionTokens)
	}
	tokens = tokens[:metalRegionTokens]

	// Bind the IDs this process actually consumed, in the same encoding
	// the registered oracle prefix uses. Naming the token file binds the
	// file; this binds the sequence that reached the runtime, which is
	// the thing a later comparison against the CUDA lane depends on.
	tokenPrefixHash, err := checkMetalRegionPrefix(tokens)
	if err != nil {
		t.Fatal(err)
	}

	r, err := newMetalRuntime(model, metalRegionContext)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Error(err)
			}
		}
	}()

	// eval returns the run's logits as bits, after refusing anything that
	// is not a full set of finite values: a short or non-finite arm makes
	// the identity comparison below vacuous rather than false.
	eval := func(arm string) ([]uint32, string) {
		t.Helper()
		if err := r.Reset(); err != nil {
			t.Fatalf("%s arm: reset: %v", arm, err)
		}
		if err := r.EvalBatch(tokens, 0); err != nil {
			t.Fatalf("%s arm: eval: %v", arm, err)
		}
		logits := r.Logits()
		if len(logits) != metalRegionVocab {
			t.Fatalf("%s arm produced %d logits, want %d", arm, len(logits), metalRegionVocab)
		}
		bits := make([]uint32, len(logits))
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s arm logit %d is not finite: %v", arm, i, v)
			}
			bits[i] = math.Float32bits(v)
		}
		hash, err := oracleWorkloadHash(logits)
		if err != nil {
			t.Fatalf("%s arm: %v", arm, err)
		}
		return bits, hash
	}

	reference, referenceHash := eval("reference")

	log := &metalRegionLog{}
	r.ops.regions = log
	instrumented, instrumentedHash := eval("instrumented")
	r.ops.regions = nil

	for i := range reference {
		if instrumented[i] != reference[i] {
			t.Fatalf("logit %d differs: instrumented %#08x, reference %#08x", i, instrumented[i], reference[i])
		}
	}
	if instrumentedHash != referenceHash {
		t.Fatalf("logit hashes differ: instrumented %s, reference %s", instrumentedHash, referenceHash)
	}
	if instrumentedHash != metalRegionProfileLogitsSHA256 {
		t.Fatalf("metal region profile logits hash %s, want registered %s", instrumentedHash, metalRegionProfileLogitsSHA256)
	}

	regions := log.all()
	if err := validateMetalRegions(metalRegionTokens, regions); err != nil {
		t.Fatal(err)
	}

	totals := metalRegionTotals(regions)
	receipt := struct {
		Tokens          int                `json:"tokens"`
		Context         int                `json:"context"`
		Model           string             `json:"model"`
		TokensPath      string             `json:"tokens_path"`
		TokenPrefix     string             `json:"token_prefix_sha256"`
		Executable      string             `json:"executable"`
		LogitsSHA256    string             `json:"logits_sha256"`
		ReferenceSHA256 string             `json:"reference_logits_sha256"`
		Totals          map[string]float64 `json:"region_seconds"`
		Regions         []metalRegion      `json:"regions"`
	}{
		Tokens:          metalRegionTokens,
		Context:         metalRegionContext,
		Model:           model,
		TokensPath:      tokensPath,
		TokenPrefix:     tokenPrefixHash,
		LogitsSHA256:    instrumentedHash,
		ReferenceSHA256: referenceHash,
		Totals:          totals,
		Regions:         regions,
	}
	if receipt.Executable, err = os.Executable(); err != nil {
		t.Fatal(err)
	}

	// Close the runtime before publishing: a profile is only a receipt of
	// a completed run if the run completed.
	err = r.Close()
	closed = true
	if err != nil {
		t.Fatal(err)
	}

	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := out.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"get-row-f16", "layer", "rms", "output-head", "matvec-q80"} {
		count := 0
		for _, region := range regions {
			if region.Region == name {
				count++
			}
		}
		t.Logf("%s: %d spans, %.6f s GPU time", name, count, totals[name])
	}
}

// TestMetalRegionProfileLogitsAnchor computes the cheap anchor logits over 64 tokens
// with context 1024 and asserts exact bit-level reproduction of the registered digest.
func TestMetalRegionProfileLogitsAnchor(t *testing.T) {
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Skip("DS4_MODEL is not set")
	}
	input, err := os.ReadFile(metalRegionTokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < metalRegionTokens {
		t.Fatalf("token file holds %d tokens, want at least %d", len(tokens), metalRegionTokens)
	}
	tokens = tokens[:metalRegionTokens]
	if _, err := checkMetalRegionPrefix(tokens); err != nil {
		t.Fatal(err)
	}

	r, err := newMetalRuntime(model, metalRegionContext)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if err := r.EvalBatch(tokens, 0); err != nil {
		t.Fatalf("eval: %v", err)
	}
	logits := r.Logits()
	if len(logits) != metalRegionVocab {
		t.Fatalf("produced %d logits, want %d", len(logits), metalRegionVocab)
	}
	hash, err := oracleWorkloadHash(logits)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("metal region profile logits sha256=%s", hash)
	if hash != metalRegionProfileLogitsSHA256 {
		t.Fatalf("logits hash = %s, want registered %s", hash, metalRegionProfileLogitsSHA256)
	}
}
