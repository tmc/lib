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

func measureOracleWorkload(r oracleWorkloadRuntime, tokens []int, eos int, now func() time.Time) (result oracleWorkloadResult, err error) {
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
	input, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens = tokens[:oracleWorkloadPrefill]
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
	actual, err := tok.Encode(string(prompt))
	if err != nil {
		t.Fatal(err)
	}
	if len(actual) < len(tokens) || !reflect.DeepEqual(actual[:len(tokens)], tokens) {
		t.Fatal("raw prompt does not match retained 2048-token prefix")
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
	identity := map[string]any{"backend": backend, "host": host, "goos": runtime.GOOS, "goarch": runtime.GOARCH, "context": oracleWorkloadContext, "model_sha256": hash(model), "executable_sha256": hash(executable), "prompt_sha256": fmt.Sprintf("%x", sha256.Sum256(prompt)), "tokens_sha256": fmt.Sprintf("%x", sha256.Sum256(input)), "eos": tok.eos, "lifecycle": "fresh runtime; first prefill; 128 greedy non-EOS decodes; close before COMPLETE", "scope": "measurement only; external oracle-token provenance, qualification and registered correctness gates required"}
	data, err := json.MarshalIndent(identity, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	write("identity.json", data)
	write("tokens.json", input)
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
	result, err := measureOracleWorkload(batch, tokens, tok.eos, time.Now)
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
func TestOracleWorkloadContract(t *testing.T) {
	r := newOracleWorkloadFake()
	tokens := make([]int, 2048)
	for i := range tokens {
		tokens[i] = i
	}
	got, err := measureOracleWorkload(r, tokens, 1, oracleWorkloadClock())
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
	if got.PrefillSeconds != .001 || got.FirstSeconds != .001 || math.Abs(got.SteadySeconds-.127) > 1e-9 {
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
			if _, err := measureOracleWorkload(r, tokens, eos, oracleWorkloadClock()); err == nil {
				t.Fatal("invalid workload accepted")
			}
		})
	}
}
