package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/gguf"
)

const oracleWorkloadPrefill = 2048
const oracleWorkloadDecode = 128
const oracleWorkloadContext = 65665

// SHA-256 of the first 2048 pinned-oracle IDs as little-endian uint32.
const oracleWorkloadTokenHash = "5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380"

// oracleWorkloadPrefillLogitsSHA256 and oracleWorkloadFinalLogitsSHA256 are the
// digests this project has quoted for the 2048-token prefill and for the state
// after 128 further non-EOS greedy decodes.
//
// Neither has a measurement behind it that anyone can point to. Both were
// searched for across the tracked tree, the whole of git history, and every
// result.json under evidence/, and appear in none of them: they exist only in
// prose. They are registered here so that a restructuring which moves the
// output moves a tracked file, and so the first gated run writes the measured
// pair into its result.json and finally gives them a provenance. Until that run
// happens, a mismatch is as likely to mean the constant was never right as to
// mean the code regressed -- which is what the failure message says. Do not
// edit a constant to match a run; record the run and say which one is wrong.
//
// Contrast metalRegionProfileLogitsSHA256, which four qualified runs on
// 2026-09-07 reproduced on both arms.
const oracleWorkloadPrefillLogitsSHA256 = "1877b6d33bcc9f790b6c5b9247bfdf06c861168f507ab8f07cfa5bffc490bc78"
const oracleWorkloadFinalLogitsSHA256 = "ab3364ea856d3a0ffe215809afb17ace90a9720e600d18aec33562edc466f372"

type oracleWorkloadRuntime interface {
	EvalBatch([]int, int) error
	Eval(int, int) error
	Logits() []float32
}

type oracleWorkloadResult struct {
	PrefillTokens     int     `json:"prefill_tokens"`
	Generated         []int   `json:"generated_tokens"`
	PrefillSeconds    float64 `json:"prefill_seconds"`
	GenerationSeconds float64 `json:"generation_seconds"`
	FirstSeconds      float64 `json:"first_seconds"`
	SteadySeconds     float64 `json:"steady_seconds"`
	PrefillHash       string  `json:"prefill_logits_sha256"`
	FinalHash         string  `json:"final_logits_sha256"`
}

func oracleWorkloadHash(logits []float32) (string, error) {
	if len(logits) != 129280 {
		return "", fmt.Errorf("workload vocabulary %d, want 129280", len(logits))
	}
	data := make([]byte, 4*len(logits))
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return "", fmt.Errorf("workload non-finite logit %d", i)
		}
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), nil
}

func oracleWorkloadToken(logits []float32, eos int) int {
	if len(logits) != 129280 {
		return -1
	}
	best := -1
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return -1
		}
		if i != eos && (best < 0 || v > logits[best]) {
			best = i
		}
	}
	return best
}

func measureOracleWorkload(r oracleWorkloadRuntime, tokens []int, eos int, now func() time.Time, checkpoint func(oracleWorkloadResult)) (result oracleWorkloadResult, err error) {
	if len(tokens) != oracleWorkloadPrefill || eos < 0 || eos >= 129280 {
		return result, fmt.Errorf("invalid oracle workload input")
	}
	for _, token := range tokens {
		if token < 0 || token >= 129280 {
			return result, fmt.Errorf("invalid oracle workload token %d", token)
		}
	}
	start := now()
	if err = r.EvalBatch(tokens, 0); err != nil {
		return result, fmt.Errorf("workload prefill: %w", err)
	}
	result.PrefillSeconds = now().Sub(start).Seconds()
	result.PrefillTokens = len(tokens)
	if result.PrefillHash, err = oracleWorkloadHash(r.Logits()); err != nil {
		return result, err
	}
	// The prefill hash is final here, but the decode loop below runs for
	// minutes and dies by panic on a test timeout. That panic is raised in
	// testing's own goroutine and kills the process without unwinding this
	// function, so a result held only in memory is lost even though half of it
	// is already computed. Hand it to the caller now.
	if checkpoint != nil {
		checkpoint(result)
	}
	result.Generated = make([]int, 0, oracleWorkloadDecode)
	start = now()
	for i := 0; i < oracleWorkloadDecode; i++ {
		token := oracleWorkloadToken(r.Logits(), eos)
		if token < 0 {
			return result, fmt.Errorf("workload no non-EOS token")
		}
		tokenStart := now()
		if err = r.Eval(token, len(tokens)+i); err != nil {
			return result, fmt.Errorf("workload decode %d: %w", i, err)
		}
		elapsed := now().Sub(tokenStart).Seconds()
		if i == 0 {
			result.FirstSeconds = elapsed
		} else {
			result.SteadySeconds += elapsed
		}
		result.Generated = append(result.Generated, token)
	}
	result.GenerationSeconds = now().Sub(start).Seconds()
	if result.FinalHash, err = oracleWorkloadHash(r.Logits()); err != nil {
		return result, err
	}
	for _, seconds := range []float64{result.PrefillSeconds, result.GenerationSeconds, result.FirstSeconds, result.SteadySeconds} {
		if seconds <= 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
			return result, fmt.Errorf("invalid workload duration")
		}
	}
	return result, nil
}

// TestOracleWorkload measures the first frontier, not a warmed correctness gate.
// Host qualification, source binding and registered identity gates are external.
func TestOracleWorkload(t *testing.T) {
	if os.Getenv("DS4_ORACLE_WORKLOAD_GATE") == "" {
		t.Skip("DS4_ORACLE_WORKLOAD_GATE is not set")
	}
	if os.Getenv("DS4_ORACLE_WORKLOAD_GATE") != "1" {
		t.Fatal("DS4_ORACLE_WORKLOAD_GATE must be 1")
	}
	required := func(key string) string {
		t.Helper()
		v := os.Getenv(key)
		if v == "" {
			t.Fatalf("%s is required", key)
		}
		return v
	}
	model := required("DS4_MODEL")
	backend := required("DS4_ORACLE_WORKLOAD_BACKEND")
	if backend != "cuda" && backend != "metal" {
		t.Fatal("workload backend must be cuda or metal")
	}
	promptPath := required("DS4_ORACLE_WORKLOAD_PROMPT")
	tokensPath := required("DS4_ORACLE_WORKLOAD_TOKENS")
	dir := required("DS4_ORACLE_WORKLOAD_DIR")
	discrepancyPath := required("DS4_ORACLE_WORKLOAD_TOKENIZER_RECEIPT")
	discrepancy, err := os.ReadFile(discrepancyPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(discrepancy) == 0 {
		t.Fatal("empty tokenizer discrepancy receipt")
	}
	input, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens = tokens[:oracleWorkloadPrefill]
	tokenBytes := oracleWorkloadTokenBytes(tokens)
	tokenHash := fmt.Sprintf("%x", sha256.Sum256(tokenBytes))
	if tokenHash != oracleWorkloadTokenHash {
		t.Fatal("token prefix does not match registered oracle workload")
	}
	prompt, err := os.ReadFile(promptPath)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	g, readErr := gguf.Read(f, info.Size())
	closeErr := f.Close()
	if readErr != nil {
		t.Fatal(readErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	tok, err := newTokenizer(g)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	write := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	hash := func(path string) string {
		t.Helper()
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		e := f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if e != nil {
			t.Fatal(e)
		}
		return fmt.Sprintf("%x", h.Sum(nil))
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	identity := map[string]any{
		"backend":                       backend,
		"host":                          host,
		"goos":                          runtime.GOOS,
		"goarch":                        runtime.GOARCH,
		"context":                       oracleWorkloadContext,
		"model_sha256":                  hash(model),
		"executable_sha256":             hash(executable),
		"prompt_sha256":                 fmt.Sprintf("%x", sha256.Sum256(prompt)),
		"tokens_sha256":                 fmt.Sprintf("%x", sha256.Sum256(input)),
		"token_ids_sha256":              tokenHash,
		"token_ids_encoding":            "little-endian uint32",
		"token_ids_count":               len(tokens),
		"input_contract":                "port evaluated on captured oracle tokenization, which its own tokenizer does not produce; no front-end equivalence claim",
		"eos":                           tok.eos,
		"eos_provenance":                "vocabulary lookup of <｜end▁of▁sentence｜> in tokenizer.ggml.tokens; port validates ID 1",
		"tokenizer_discrepancy_receipt": discrepancyPath,
		"tokenizer_discrepancy_sha256":  fmt.Sprintf("%x", sha256.Sum256(discrepancy)),
		"lifecycle":                     "fresh runtime; first prefill; 128 greedy non-EOS decodes; close before COMPLETE",
		"scope":                         "measurement only; external oracle-token provenance, qualification and registered correctness gates required",
	}
	data, err := json.MarshalIndent(identity, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	write("identity.json", data)
	write("tokens.json", input)
	write("token-prefix.u32le", tokenBytes)
	write("tokenizer-discrepancy.json", discrepancy)
	r, closer, err := openSessionBackend(config{model: model, backend: backend, context: oracleWorkloadContext, log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := closer.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	batch, ok := r.(oracleWorkloadRuntime)
	if !ok {
		t.Fatal("backend lacks EvalBatch")
	}
	result, err := measureOracleWorkload(batch, tokens, tok.eos, time.Now, func(partial oracleWorkloadResult) {
		partialData, err := json.MarshalIndent(partial, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		write("prefill.json", partialData)
	})
	if err != nil {
		t.Fatal(err)
	}
	err = closer.Close()
	closed = true
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	write("result.json", data)
	write("COMPLETE", []byte("runtime closed; 2048 prefill and 128 non-EOS decodes completed\n"))
	t.Logf("prefill %.2f tokens/s, generation %.2f tokens/s", float64(result.PrefillTokens)/result.PrefillSeconds, float64(len(result.Generated))/result.GenerationSeconds)
	t.Logf("measured prefill logits sha256=%s", result.PrefillHash)
	t.Logf("measured final logits sha256=%s", result.FinalHash)
	if result.PrefillHash != oracleWorkloadPrefillLogitsSHA256 {
		t.Errorf("prefill logits hash %s, registered %s; the registered digest has never been reproduced, so this may be the constant rather than the code", result.PrefillHash, oracleWorkloadPrefillLogitsSHA256)
	}
	if result.FinalHash != oracleWorkloadFinalLogitsSHA256 {
		t.Errorf("final logits hash %s, registered %s; the registered digest has never been reproduced, so this may be the constant rather than the code", result.FinalHash, oracleWorkloadFinalLogitsSHA256)
	}
}

type oracleWorkloadFake struct {
	logits              []float32
	batch               []int
	positions, selected []int
	failure             string
}

func (r *oracleWorkloadFake) EvalBatch(tokens []int, position int) error {
	if r.failure == "prefill" {
		return fmt.Errorf("injected prefill error")
	}
	r.batch = append([]int(nil), tokens...)
	if position != 0 {
		return fmt.Errorf("wrong position")
	}
	return nil
}
func (r *oracleWorkloadFake) Eval(token, position int) error {
	if r.failure == "decode" {
		return fmt.Errorf("injected decode error")
	}
	r.positions = append(r.positions, position)
	r.selected = append(r.selected, token)
	return nil
}
func (r *oracleWorkloadFake) Logits() []float32 { return r.logits }
func newOracleWorkloadFake() *oracleWorkloadFake {
	r := &oracleWorkloadFake{logits: make([]float32, 129280)}
	r.logits[1] = 100
	r.logits[2] = 10
	r.logits[3] = 10
	return r
}
func oracleWorkloadClock() func() time.Time {
	next := time.Unix(0, 0)
	return func() time.Time { next = next.Add(time.Millisecond); return next }
}

// TestOracleWorkloadPrefillCheckpoint pins the reason the checkpoint exists: the
// prefill hash must reach the caller BEFORE the decode loop starts, because a
// test-timeout panic during decode kills the process without unwinding
// measureOracleWorkload. Asserting only that the hash is eventually correct
// would pass even if the checkpoint fired at the very end, which is the bug.
func TestOracleWorkloadPrefillCheckpoint(t *testing.T) {
	r := newOracleWorkloadFake()
	tokens := make([]int, 2048)
	for i := range tokens {
		tokens[i] = i
	}
	var calls int
	var atCall oracleWorkloadResult
	var decodedAtCall int
	got, err := measureOracleWorkload(r, tokens, 1, oracleWorkloadClock(), func(partial oracleWorkloadResult) {
		calls++
		atCall = partial
		decodedAtCall = len(r.positions)
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("checkpoint called %d times, want exactly 1", calls)
	}
	if decodedAtCall != 0 {
		t.Fatalf("checkpoint fired after %d decode positions, want 0", decodedAtCall)
	}
	if atCall.PrefillHash == "" || atCall.PrefillHash != got.PrefillHash {
		t.Fatalf("checkpoint hash %q, final %q", atCall.PrefillHash, got.PrefillHash)
	}
	if atCall.PrefillTokens != len(tokens) {
		t.Fatalf("checkpoint prefill tokens %d, want %d", atCall.PrefillTokens, len(tokens))
	}
	if len(atCall.Generated) != 0 {
		t.Fatalf("checkpoint carried %d generated tokens, want 0", len(atCall.Generated))
	}
	if atCall.FinalHash != "" {
		t.Fatal("checkpoint carried a final hash before decode ran")
	}
}

func TestOracleWorkloadContract(t *testing.T) {
	r := newOracleWorkloadFake()
	tokens := make([]int, 2048)
	for i := range tokens {
		tokens[i] = i
	}
	got, err := measureOracleWorkload(r, tokens, 1, oracleWorkloadClock(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r.batch, tokens) || len(r.positions) != 128 || len(got.Generated) != 128 {
		t.Fatal("wrong prefix or counts")
	}
	for i, pos := range r.positions {
		if pos != 2048+i || r.selected[i] != 2 || got.Generated[i] != 2 {
			t.Fatalf("step %d: position%d token%d", i, pos, r.selected[i])
		}
	}
	if got.PrefillSeconds != .001 || got.GenerationSeconds != .257 || got.FirstSeconds != .001 || math.Abs(got.SteadySeconds-.127) > 1e-9 {
		t.Fatalf("wrong timing windows: %+v", got)
	}
}
func TestOracleWorkloadFailures(t *testing.T) {
	for _, name := range []string{"prefill", "decode", "short vocabulary", "nan", "infinity", "short prefix", "invalid token", "invalid eos"} {
		t.Run(name, func(t *testing.T) {
			r := newOracleWorkloadFake()
			r.failure = name
			tokens := make([]int, 2048)
			eos := 1
			switch name {
			case "short vocabulary":
				r.logits = r.logits[:1]
			case "nan":
				r.logits[5] = float32(math.NaN())
			case "infinity":
				r.logits[5] = float32(math.Inf(1))
			case "short prefix":
				tokens = tokens[:2047]
			case "invalid token":
				tokens[9] = -1
			case "invalid eos":
				eos = -1
			}
			if _, err := measureOracleWorkload(r, tokens, eos, oracleWorkloadClock(), nil); err == nil {
				t.Fatal("invalid workload accepted")
			}
		})
	}
}

func oracleWorkloadTokenBytes(tokens []int) []byte {
	data := make([]byte, 4*len(tokens))
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(data[4*i:], uint32(token))
	}
	return data
}
func TestOracleWorkloadTokenEncoding(t *testing.T) {
	tokens := []int{0, 1, 129279}
	data := oracleWorkloadTokenBytes(tokens)
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "e6975f35fedae15219b5c539423d289ee45a9902268066d6d04622201a7553dc" {
		t.Fatalf("wrong token encoding digest %s", got)
	}
	if !reflect.DeepEqual(data, []byte{0, 0, 0, 0, 1, 0, 0, 0, 255, 248, 1, 0}) {
		t.Fatalf("wrong token bytes %v", data)
	}
}
