package main

import (
	"bytes"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func TestParseConfig(t *testing.T) {
	tests := []struct {
		name string
		args []string
		in   string
		want string
	}{
		{"stdin", []string{"-model", "m", "-backend", "cpu"}, "hello", "hello"},
		{"prompt", []string{"-model", "m", "-backend", "cpu", "-prompt", "hello"}, "ignored", "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig(tt.args, strings.NewReader(tt.in), new(bytes.Buffer))
			if err != nil {
				t.Fatal(err)
			}
			if got := string(cfg.prompt); got != tt.want {
				t.Fatalf("prompt = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseRawAndDumpTokens(t *testing.T) {
	tests := []struct {
		name string
		args []string
		raw  bool
		dump bool
	}{
		{name: "raw", args: []string{"-model", "m", "-backend", "cpu", "-raw", "-prompt", "hello"}, raw: true},
		{name: "dump", args: []string{"-model", "m", "-backend", "cpu", "-dump-tokens", "-prompt", "hello"}, dump: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig(tt.args, strings.NewReader("ignored"), new(bytes.Buffer))
			if err != nil {
				t.Fatal(err)
			}
			if cfg.raw != tt.raw || cfg.dumpTokens != tt.dump {
				t.Fatalf("raw=%v dump=%v, want raw=%v dump=%v", cfg.raw, cfg.dumpTokens, tt.raw, tt.dump)
			}
		})
	}
}

func TestParseFrontierLogitsDump(t *testing.T) {
	for _, flag := range []string{"-dump-frontier-logits", "-dump-frontier-logits-dir"} {
		cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", flag, "out", "-prompt", "hello"}, strings.NewReader(""), new(bytes.Buffer))
		if err != nil {
			t.Fatal(err)
		}
		if cfg.dumpFrontierLogitsDir != "out" {
			t.Fatalf("%s: directory = %q", flag, cfg.dumpFrontierLogitsDir)
		}
	}
	if _, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-dump-frontier-logits", "a", "-dump-frontier-logits-dir", "b", "-prompt", "hello"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("conflicting frontier flags error = %v", err)
	}
}

func TestParseSystemAndThinkModes(t *testing.T) {
	tests := []struct {
		name string
		args []string
		sys  string
		mode ThinkMode
	}{
		{"defaults", []string{"-model", "m", "-backend", "cpu"}, "You are a helpful assistant", ThinkHigh},
		{"aliases", []string{"-model", "m", "-backend", "cpu", "-sys", "Be terse.", "-think"}, "Be terse.", ThinkHigh},
		{"think then nothink", []string{"-model", "m", "-backend", "cpu", "-think", "-nothink"}, "You are a helpful assistant", ThinkNone},
		{"nothink then max", []string{"-model", "m", "-backend", "cpu", "-nothink", "-think-max"}, "You are a helpful assistant", ThinkMax},
		{"last wins", []string{"-model", "m", "-backend", "cpu", "-think-max", "-nothink", "-think"}, "You are a helpful assistant", ThinkHigh},
	}
	if cfg, err := parseConfig([]string{"-model", "m", "-system", ""}, strings.NewReader(""), new(bytes.Buffer)); err != nil || cfg.system != "" || !cfg.systemSet {
		t.Fatalf("explicit empty system = %+v, %v", cfg, err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig(tt.args, strings.NewReader("ignored"), new(bytes.Buffer))
			if err != nil {
				t.Fatal(err)
			}
			if cfg.system != tt.sys || cfg.thinkMode != tt.mode {
				t.Fatalf("system=%q mode=%v, want %q %v", cfg.system, cfg.thinkMode, tt.sys, tt.mode)
			}
		})
	}
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.sampling != sampler.Default {
		t.Fatalf("default sampling = %+v, want %+v", cfg.sampling, sampler.Default)
	}
	if _, err := parseConfig([]string{"-model", "m", "-think-level", "high"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("think-level error = %v", err)
	}
}

func TestWritePerplexityUsesFullTokenCount(t *testing.T) {
	var out bytes.Buffer
	p := dsv4eval.Perplexity{Tokens: 8, Scored: 8, NLL: 1.25, AvgNLL: 0.15625, PPL: 1.1692}
	if err := writePerplexity(&out, 40, p); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); !strings.HasPrefix(got, "tokens=40 scored=8 ") {
		t.Fatalf("perplexity output = %q", got)
	}
}

func TestDecodeConsistencyArgumentsAndOutput(t *testing.T) {
	var out bytes.Buffer
	if err := writeConsistency(&out, dsv4eval.Consistency{Positions: 3, Top1Matches: 3, MaxAbs: 0.25, RMS: 0.125}); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "positions=3 max_abs=0.25 rms=0.125 top1=3/3\n" {
		t.Fatalf("consistency output = %q", got)
	}
	if _, err := parseConfig([]string{"-model", "m", "-decode-consistency", "-1"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "non-negative") {
		t.Fatalf("negative decode-consistency error = %v", err)
	}
}

func TestRawPromptBypassesChatTemplate(t *testing.T) {
	tok := &tokenizer{tokens: []string{"h", "i", "<bos>", "<user>", "<assistant>", "</think>"}, tokenID: map[string]int{"h": 0, "i": 1}}
	for i := range 256 {
		tok.byteRunes[i] = gpt2ByteRune(byte(i))
	}
	cfg := config{raw: true, prompt: []byte("hi"), context: 8, tokens: 1}
	got, err := preparePrompt(cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("raw prompt = %v, want %v", got, want)
	}
}

func TestDumpTokens(t *testing.T) {
	tok := &tokenizer{tokens: []string{"zero", "one", "two"}}
	var out bytes.Buffer
	if err := dumpTokens(&out, []int{0, 2}, tok); err != nil {
		t.Fatal(err)
	}
	if got, want := out.String(), "[0, 2]\n     0  zero\n     2  two\n"; got != want {
		t.Fatalf("dump = %q, want %q", got, want)
	}
}

func TestDumpTokensRejectsOutOfRangeID(t *testing.T) {
	for _, id := range []int{-1, 2} {
		t.Run(strconv.Itoa(id), func(t *testing.T) {
			if err := dumpTokens(new(bytes.Buffer), []int{id}, &tokenizer{tokens: []string{"zero", "one"}}); err == nil {
				t.Fatal("accepted out-of-range token ID")
			}
		})
	}
}

func TestParseConfigErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"model", nil, "model is required"},
		{"arguments", []string{"-model", "m", "extra"}, "unexpected arguments"},
		{"backend", []string{"-model", "m", "-backend", "bogus"}, `unknown backend "bogus"`},
		{"prompts", []string{"-model", "m", "-prompt", "a", "-prompt-file", "b"}, "prompt and prompt-file conflict"},
		{"context", []string{"-model", "m", "-ctx", "0"}, "context must be positive"},
		{"tokens", []string{"-model", "m", "-tokens", "0"}, "tokens must be positive"},
		{"prefill chunk too small", []string{"-model", "m", "-prefill-chunk", "128"}, "prefill-chunk must be a multiple"},
		{"prefill chunk misaligned", []string{"-model", "m", "-prefill-chunk", "300"}, "prefill-chunk must be a multiple"},
		{"prefill chunk too large", []string{"-model", "m", "-prefill-chunk", "4096"}, "prefill-chunk must be a multiple"},
		{"ballast zero", []string{"-model", "m", "-ballast-gib", "0"}, "ballast-gib must be positive"},
		{"ballast negative", []string{"-model", "m", "-ballast-gib", "-1"}, "ballast-gib must be positive"},
		{"profiler phase", []string{"-model", "m", "-cuda-profiler-phase", "bogus"}, "cuda-profiler-phase: want prefill or generation"},
		{"prompt file", []string{"-model", "m", "-prompt-file", "not-present"}, "read prompt:"},
		{"dump source", []string{"-model", "m", "-dump-tokens"}, "dump-tokens requires prompt or prompt-file"},
		{"dump token IDs", []string{"-model", "m", "-dump-tokens", "-prompt-tokens", "x"}, "dump-tokens requires prompt or prompt-file"},
		{"raw prefix", []string{"-model", "m", "-raw", "-prefix-file", "x"}, "raw conflicts"},
		{"perplexity prompt", []string{"-model", "m", "-perplexity-file", "scores.txt", "-prompt", "x"}, "perplexity-file conflicts"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseConfig(tt.args, strings.NewReader(""), new(bytes.Buffer))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestParseBallastGib(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-ballast-gib", "1.5"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.ballastGibSet || cfg.ballastGib != 1.5 {
		t.Fatalf("ballast = %v/%v, want 1.5/true", cfg.ballastGib, cfg.ballastGibSet)
	}
}

func TestParseWarmWeights(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-warm-weights"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.warmWeights || !cfg.warmWeightsSet {
		t.Fatalf("warm weights = %v/%v, want true/true", cfg.warmWeights, cfg.warmWeightsSet)
	}
	cfg, err = parseConfig([]string{"-model", "m", "-backend", "cpu"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.warmWeights || cfg.warmWeightsSet {
		t.Fatalf("default warm weights = %v/%v, want false/false", cfg.warmWeights, cfg.warmWeightsSet)
	}
}

func TestParsePrefillChunk(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		want int
	}{
		{name: "default", args: []string{"-model", "m"}, want: 2048},
		{name: "small", args: []string{"-model", "m", "-prefill-chunk", "256"}, want: 256},
		{name: "middle", args: []string{"-model", "m", "-prefill-chunk", "1024"}, want: 1024},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig(tt.args, strings.NewReader(""), new(bytes.Buffer))
			if err != nil {
				t.Fatal(err)
			}
			if cfg.prefillChunk != tt.want {
				t.Fatalf("prefill chunk = %d, want %d", cfg.prefillChunk, tt.want)
			}
		})
	}
}

func TestParseProfilerPhase(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		want       profilerPhase
		prefill    bool
		generation bool
	}{
		{name: "disabled", args: []string{"-model", "m", "-backend", "cpu"}},
		{name: "legacy range", args: []string{"-model", "m", "-backend", "cpu", "-cuda-profiler-range"}, want: profilerPhaseBoth, prefill: true, generation: true},
		{name: "prefill", args: []string{"-model", "m", "-backend", "cpu", "-cuda-profiler-phase", "prefill"}, want: profilerPhasePrefill, prefill: true},
		{name: "generation", args: []string{"-model", "m", "-backend", "cpu", "-cuda-profiler-phase", "generation"}, want: profilerPhaseGeneration, generation: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig(tt.args, strings.NewReader(""), new(bytes.Buffer))
			if err != nil {
				t.Fatal(err)
			}
			if cfg.profilerPhase != tt.want {
				t.Fatalf("profiler phase = %q, want %q", cfg.profilerPhase, tt.want)
			}
			if got := cfg.phaseProfileEnabled(profilerPhasePrefill); got != tt.prefill {
				t.Fatalf("prefill enabled = %v, want %v", got, tt.prefill)
			}
			if got := cfg.phaseProfileEnabled(profilerPhaseGeneration); got != tt.generation {
				t.Fatalf("generation enabled = %v, want %v", got, tt.generation)
			}
		})
	}
}

func TestParseTraceLoad(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-trace", "trace.out", "-trace-load"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.traceLoad {
		t.Fatal("trace-load = false, want true")
	}
	if cfg.traceProfile != "trace.out" {
		t.Fatalf("trace = %q, want trace.out", cfg.traceProfile)
	}
}

func TestParseSamplingFlags(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-temp", "0.6", "-top-k", "40", "-top-p", "0.95", "-min-p", "0.05", "-seed", "12345", "-dump-logprobs", "scores.json", "-logprobs-top-k", "8", "-dump-decode-logits-dir", "steps"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.sampling.Temperature != 0.6 || cfg.sampling.TopK != 40 || cfg.sampling.TopP != 0.95 || cfg.sampling.MinP != 0.05 || cfg.seed != 12345 || !cfg.seedSet {
		t.Fatalf("sampling config = %+v, seed=%d/%v", cfg.sampling, cfg.seed, cfg.seedSet)
	}
	if cfg.dumpLogprobs != "scores.json" || cfg.logprobsTopK != 8 || cfg.decodeLogitsDir != "steps" {
		t.Fatalf("dump config = logprobs=%q topk=%d logits=%q", cfg.dumpLogprobs, cfg.logprobsTopK, cfg.decodeLogitsDir)
	}
}

func TestParseSamplingSeedZero(t *testing.T) {
	_, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-seed", "0"}, strings.NewReader(""), new(bytes.Buffer))
	if err == nil || !strings.Contains(err.Error(), "seed must be positive") {
		t.Fatalf("error = %v, want explicit zero-seed rejection", err)
	}
}

func TestParseKVFlags(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-kv-save", "state.dsv4"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil || cfg.kvSave != "state.dsv4" || cfg.kvLoad != "" {
		t.Fatalf("save config = %+v, err=%v", cfg, err)
	}
	cfg, err = parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-kv-load", "state.dsv4"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil || cfg.kvLoad != "state.dsv4" || cfg.kvSave != "" {
		t.Fatalf("load config = %+v, err=%v", cfg, err)
	}
	if _, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-kv-save", "same", "-kv-load", "same"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "different files") {
		t.Fatalf("same KV path error = %v", err)
	}
}

func TestParseSyncPrefix(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-sync-prefix", "1"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil || cfg.syncPrefix != 1 {
		t.Fatalf("sync-prefix config = %+v, err=%v", cfg, err)
	}
	if _, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-sync-prefix", "-1"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "non-negative") {
		t.Fatalf("negative sync-prefix error = %v", err)
	}
	if _, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-sync-prefix", "1", "-kv-load", "state.dsv4"}, strings.NewReader(""), new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("kv-load conflict error = %v", err)
	}
}

func TestHelp(t *testing.T) {
	var stderr bytes.Buffer
	err := run([]string{"-help"}, strings.NewReader(""), new(bytes.Buffer), &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if got := stderr.String(); got != usage {
		t.Fatalf("help:\n%s\nwant:\n%s", got, usage)
	}
}

func TestReadPromptError(t *testing.T) {
	want := errors.New("broken")
	_, err := parseConfig([]string{"-model", "m", "-backend", "cpu"}, errorReader{want}, new(bytes.Buffer))
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want wrapping %v", err, want)
	}
}

type errorReader struct{ err error }

func (r errorReader) Read([]byte) (int, error) { return 0, r.err }
