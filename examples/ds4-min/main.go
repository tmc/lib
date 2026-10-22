package main

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
	"github.com/tmc/lib/examples/ds4-min/sampler"
	"github.com/tmc/lib/gguf"
)

const usage = `usage: ds4-min -model file [-backend auto|metal|cuda|cpu] [-prompt text | -prompt-file file | -prompt-tokens file] [-tokens n] [-ctx n]

Load a DeepSeek model and generate a response. V4.1 uses the CPU reference
with explicit -backend cpu; GPU execution is not implemented for that family.

  -backend string
        execution backend: auto, metal, cuda, or cpu (default "auto")
  -family string
        model family: auto, flash, or v41 (default "auto"); overrides must match GGUF
  -model-sha256 string
        independent V4.1 complete-file identity required for portable state
  -ctx int
        session context length (default 4096)
  -power int
        GPU duty cycle percentage, 1..100 (default 100)
  -model string
        path to the pinned GGUF model (required)
  -inspect
        print a backend-free GGUF summary and exit
  -interactive
        read one chat message per line; /reset clears history, /quit exits
  -prompt string
        user prompt; conflicts with -prompt-file and -prompt-tokens
  -prompt-file string
        read the user prompt from file; conflicts with -prompt and -prompt-tokens
  -prompt-tokens string
        read a JSON token-id array verbatim; conflicts with prose prompts
  -system string
        system prompt (default "You are a helpful assistant"); -sys is an alias
  -think, -nothink, -think-max
        select the chat thinking mode; the last flag wins (default -think)
  -dir-steering-file file
        load raw little-endian per-layer directions; file alone selects FFN scale 1
  -dir-steering-attn float
        attention steering scale in [-100,100] (default 0)
  -dir-steering-ffn float
        FFN steering scale in [-100,100]; an explicit scale overrides the file default
  -raw
        tokenize the prompt as plain text without the chat template
  -dump-tokens
        print the prompt token IDs and raw vocabulary strings, then exit
  -tokens int
        maximum generated tokens (default 16)
  -temp float
        sampling temperature; zero selects greedy (default 1)
  -top-k int
        retain at most this many candidates; zero means full vocabulary (default 0)
  -top-p float
        cumulative probability cutoff (default 1)
  -min-p float
        minimum probability relative to the best candidate (default 0.05)
  -seed uint
        sampling seed; zero is rejected when explicitly supplied
  -dump-logprobs file
        write per-step top-logprob JSON
  -logprobs-top-k int
        number of per-step logprob entries (default 20)
  -dump-decode-logits-dir dir
        write raw float32 logits before each generated token
  -expect-prefill-logits string
        require this SHA-256 of little-endian float32 prefill logits
  -expect-final-logits string
        require this final logits SHA-256 and exactly -tokens accepted tokens
  -dump-logits string
        write prefill/final little-endian float32 logits and provenance sidecars using this prefix
  -dump-frontier-logits-dir string
        write validated prefill/final frontier logits JSON into this directory
  -dump-frontier-logits string
        alias for -dump-frontier-logits-dir
  -kv-save file
        save the backend session after prompt prefill
  -kv-load file
        load a backend session before prompt evaluation
  -kv-load-contract file
        qualify an upstream KVC load with a model/token/payload receipt
  -sync-prefix n
        evaluate the first n prompt tokens, then evaluate the full prompt
  -prefill-chunk int
        maximum tokens evaluated by one prefill batch (default 2048)
  -ballast-gib float
        allocate and touch this many GiB of CUDA ballast before evaluation;
        useful for reproducing unified-memory pressure (default 0)
  -warm-weights
        fault the CUDA model mapping into memory before evaluation

Profiling. These start after the model is loaded, so a profile describes
generation rather than the mmap residency wait.

  -cuda-profiler-phase string
        profile one phase with cuProfilerStart/Stop: prefill or generation
  -cuda-profiler-range
        bracket prefill and generation with cuProfilerStart/Stop
  -cuda-profiler-stop
        call cuProfilerStop after the last kernel so the profiler flushes its buffers

  -cpuprofile file
        write a Go CPU profile
  -memprofile file
        write a Go heap profile at exit
  -trace file
        write a Go execution trace
  -trace-load
        start requested Go profiling before model loading (use with -trace to include CUDA load-time work)
  -spans file
        write a gputrace.host-event/v1 sidecar for "gputrace host-receipt"
  -spans-stream file
        write that sidecar a record at a time, with no cap and an
        accounting receipt beside it; use this for -span-level=op
  -span-level string
        span depth: off, phase, token, layer, or op (default "token")
  -run-id string
        run identity shared with a gputrace timing sidecar; "gputrace
        host-receipt" binds the two only when they name the same run

Spans are also appended to $GPUTRACE_APP_EVENTS when that is set, which
is what "gputrace capture" does, so a captured run carries them with no
flag.

-spans holds every interval in memory and stops recording at 8 MiB or
100,000 records, whichever binds first — near 40,000 spans in practice,
which one 2048-token prefill exceeds at -span-level=layer. Those are the
recorder's own caps. The converter's reader imposes none: readHost in
examples/cmd/profile2trace/spans.go decodes a file of any length, though
it does materialize every record.

-spans-stream writes each record as its span closes and holds only the
spans currently open, so it is bounded by nesting depth rather than by run
length. It writes <file>.accounting.json beside the sidecar, which reports
how many spans began, ended and encoded, and refuses to call itself
complete if any of those disagree or a write failed. The two flags are
alternatives; -span-level=op needs the streaming one.
`

const defaultSystemPrompt = "You are a helpful assistant"

type config struct {
	power             int
	streaming         streamingOptions
	collectors        collectorOptions
	evalSuite         string
	evalTrace         string
	evalCorpus        string
	evalArtifactsDir  string
	evalOutput        *os.File
	model             string
	inspect           bool
	interactive       bool
	backend           string
	family            string
	modelSHA256       string
	system            string
	systemSet         bool
	thinkMode         ThinkMode
	perplexityFile    string
	decodeConsistency int
	prompt            []byte
	promptTokens      []int
	prefixFile        string
	raw               bool
	dumpTokens        bool
	context           int
	tokens            int
	log               io.Writer

	expectPrefillLogits   string
	expectFinalLogits     string
	dumpLogits            string
	dumpFrontierLogitsDir string
	expectFailureDump     string

	cpuProfile        string
	memProfile        string
	traceProfile      string
	traceLoad         bool
	spanPath          string
	spanStream        string
	calibEvery        int
	spanLevel         spanLevel
	runID             string
	calibPath         string
	profilerStop      bool
	profilerRange     bool
	warmPrefill       bool
	ballastGib        float64
	ballastGibSet     bool
	warmWeights       bool
	warmWeightsSet    bool
	sampling          sampler.Params
	seed              uint64
	seedSet           bool
	dirSteeringFile   string
	dirSteeringAttn   float64
	dirSteeringFFN    float64
	dumpLogprobs      string
	logprobsTopK      int
	decodeLogitsDir   string
	sampledTraceDir   string
	sampledTranscript string
	kvSave            string
	kvLoad            string
	kvLoadContract    string
	syncPrefix        int
	prefillChunk      int
	profilerPhase     profilerPhase
}

type profilerPhase string

const (
	profilerPhaseBoth       profilerPhase = "both"
	profilerPhasePrefill    profilerPhase = "prefill"
	profilerPhaseGeneration profilerPhase = "generation"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-min:", err)
		exitProcess(1)
	}
	exitProcess(0)
}

// greedyToken remains as a compatibility helper for backend tests; generation
// itself goes through sampler.Sample so temperature-zero uses the same rules.
func greedyToken(logits []float32) int { return sampler.Argmax(logits) }

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) (err error) {
	if len(args) > 0 && args[0] == "-frontier-bench" {
		return runFrontierBenchmark(context.Background(), args[1:], stdout, stderr)
	}
	if len(args) > 0 && args[0] == "-server" {
		return runServer(args[1:], stdout, stderr)
	}
	cfg, err := parseConfig(args, stdin, stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if err := validateCollectorOptions(cfg); err != nil {
		return err
	}
	cfg.evalOutput, err = admitEvalOutputs(cfg)
	if err != nil {
		return err
	}
	if cfg.evalOutput != nil {
		defer func() { err = errors.Join(err, cfg.evalOutput.Close()) }()
	}
	f, err := os.Open(cfg.model)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}
	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("stat model: %w", errors.Join(err, f.Close()))
	}
	g, err := gguf.Read(f, info.Size())
	err = errors.Join(err, f.Close())
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	if cfg.inspect {
		return inspectModel(stdout, cfg.model, g)
	}
	if err := validateFamilyProfile(&cfg, g); err != nil {
		return fmt.Errorf("model profile: %w", err)
	}
	cfg.log = stderr
	tok, err := newTokenizer(g)
	if err != nil {
		return err
	}
	if cfg.evalSuite != "" {
		return runEvalSmoke(cfg, tok)
	}
	if cfg.collectors.dataset != "" || cfg.collectors.imatrix != "" {
		return runImatrix(cfg, tok, stderr)
	}
	if cfg.interactive {
		return runInteractive(cfg, tok, stdin, stdout, stderr)
	}
	prompt, err := preparePrompt(cfg, tok)
	if err != nil {
		return err
	}
	if cfg.syncPrefix > 0 && cfg.syncPrefix >= len(prompt) {
		return fmt.Errorf("sync-prefix %d must be less than prompt length %d", cfg.syncPrefix, len(prompt))
	}
	if cfg.dumpTokens {
		return dumpTokens(stdout, prompt, tok)
	}
	cfg.logLogitsWorkload(prompt, stderr)
	var prof *profiler
	if cfg.traceLoad {
		prof, err = startProfiling(cfg)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, prof.Stop()) }()
	}
	transcript, err := newSampleTranscript(cfg.sampledTranscript)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, transcript.Close()) }()
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return err
	}
	var expertProfile *profileCollectors
	defer func() {
		err = finishProfileOutput(err, closer, cfg, expertProfile)
	}()
	if err := configureSteering(b, cfg); err != nil {
		return err
	}
	s, err := newSession(b, cfg.context, tok.stops)
	if err != nil {
		return err
	}
	if err := configurePrefillChunk(s.backend, cfg.prefillChunk); err != nil {
		return err
	}
	expertProfile, err = attachExpertProfile(cfg, b, len(prompt))
	if err != nil {
		return err
	}
	if cfg.kvLoadContract != "" {
		if err := loadUpstreamSessionPaths(s, cfg.kvLoad, cfg.kvLoadContract); err != nil {
			return fmt.Errorf("load upstream kv state: %w", err)
		}
	} else if cfg.kvLoad != "" {
		f, err := os.Open(cfg.kvLoad)
		if err != nil {
			return fmt.Errorf("open kv load: %w", err)
		}
		err = s.Load(f)
		closeErr := f.Close()
		if err != nil {
			return fmt.Errorf("load kv state: %w", err)
		}
		if closeErr != nil {
			return fmt.Errorf("close kv load: %w", closeErr)
		}
	}
	if cfg.perplexityFile != "" {
		return runPerplexity(s, prompt, cfg, stdout)
	}
	if cfg.decodeConsistency != 0 {
		return runDecodeConsistency(s, prompt, cfg, stdout)
	}
	if cfg.warmPrefill {
		if err := warmPrefill(s, prompt, len(tok.tokens), cfg.expectPrefillLogits, stderr); err != nil {
			return err
		}
	}
	if prof == nil {
		prof, err = startProfiling(cfg)
		if err != nil {
			return err
		}
	}
	defer func() { err = errors.Join(err, prof.Stop(), writeSpans(cfg, stderr)) }()
	runID := cfg.runID
	if runID == "" {
		runID = defaultRunID()
	}
	if cfg.spanLevel != levelOff {
		if cfg.spanStream != "" {
			r, err := newStreamRecorder(cfg.spanLevel, runID, cfg.spanStream)
			if err != nil {
				return err
			}
			streamSpans = r
		} else {
			spans = newSpanRecorder(cfg.spanLevel, runID)
		}
	}
	// Bracket the calibration ranges around the whole workload rather than
	// beside it: two readings far apart are what test whether one constant
	// offset relates the two clocks at all.
	var calibrations []calibration
	if cfg.calibPath != "" {
		calibrations = append(calibrations, calibrate(runID, "start"))
		interiorCalib = &interiorCalibrator{runID: runID, every: cfg.calibEvery}
	}
	fmt.Fprintf(stderr, "ds4-min: processing %d input tokens\n", len(prompt))
	prefillStart := time.Now()
	if err := cfg.beginProfilerRange(profilerPhasePrefill); err != nil {
		return err
	}
	if err := tracePhase(levelPhase, "prefill", func() error {
		if cfg.syncPrefix > 0 {
			if err := s.Sync(prompt[:cfg.syncPrefix]); err != nil {
				return fmt.Errorf("sync prefix: %w", err)
			}
		}
		return s.Sync(prompt)
	}); err != nil {
		return err
	}
	if err := cfg.endProfilerRange(profilerPhasePrefill); err != nil {
		return err
	}
	prefill := time.Since(prefillStart)
	if err := checkLogits("prefill", s.logits, len(tok.tokens), cfg.expectPrefillLogits, stderr); err != nil {
		if cfg.expectFailureDump != "" {
			if dumpErr := dumpLogitsRaw(cfg.expectFailureDump, "prefill", s.logits, prompt, len(prompt)-1, nil, cfg); dumpErr != nil {
				return errors.Join(err, dumpErr)
			}
		}
		return err
	}
	if cfg.kvSave != "" {
		f, err := os.Create(cfg.kvSave)
		if err != nil {
			return fmt.Errorf("create kv save: %w", err)
		}
		err = s.Save(f)
		closeErr := f.Close()
		if err != nil {
			return fmt.Errorf("save kv state: %w", err)
		}
		if closeErr != nil {
			return fmt.Errorf("close kv save: %w", closeErr)
		}
	}
	if cfg.dumpLogits != "" {
		if err := dumpLogits(cfg.dumpLogits, "prefill", s.logits, prompt, len(prompt)-1, nil, cfg); err != nil {
			return err
		}
	}
	if cfg.dumpFrontierLogitsDir != "" {
		if err := dumpFrontierLogits(cfg.dumpFrontierLogitsDir, "prefill", s.logits, prompt, nil, cfg); err != nil {
			return err
		}
	}
	var text textAssembler
	generated := 0
	generatedIDs := make([]int, 0, cfg.tokens)
	smp := sampler.New(len(s.logits))
	rng := sampler.RNG{State: cfg.seed}
	if !cfg.seedSet && cfg.sampling.Temperature > 0 {
		var seedBytes [8]byte
		if _, err := rand.Read(seedBytes[:]); err != nil {
			return fmt.Errorf("generate sampling seed: %w", err)
		}
		rng.State = binary.LittleEndian.Uint64(seedBytes[:])
	}
	if cfg.sampling.Temperature > 0 {
		fmt.Fprintf(stderr, "ds4-min: seed %d\n", rng.State)
	}
	var logprobSteps []logprobStep
	decodeStart := time.Now()
	if err := cfg.beginProfilerRange(profilerPhaseGeneration); err != nil {
		return err
	}
	if err := tracePhase(levelPhase, "generation", func() error {
		for step := range cfg.tokens {
			if cfg.decodeLogitsDir != "" {
				if err := dumpDecodeLogits(cfg.decodeLogitsDir, step, s.logits); err != nil {
					return err
				}
			}
			before := rng.State
			id := 0
			if cfg.dumpLogprobs != "" {
				// Upstream --dump-logprobs is an observation mode: it uses
				// argmax and does not consume the sampling RNG, regardless of
				// the generation flags supplied to the command.
				id = sampler.Argmax(s.logits)
			} else {
				id = smp.Sample(s.logits, cfg.sampling, &rng)
			}
			if transcript != nil {
				if err := transcript.Decision(step, id, before, rng.State, tok.IsStopForMode(id, cfg.thinkMode)); err != nil {
					return err
				}
			}
			if cfg.dumpLogprobs != "" {
				logprobSteps = append(logprobSteps, logprobStep{Step: step, Selected: id, Scores: sampler.TopLogprobs(s.logits, cfg.logprobsTopK)})
			}
			if cfg.sampledTraceDir != "" {
				history := append(append([]int(nil), prompt...), generatedIDs...)
				if err := writeSampledDecision(cfg.sampledTraceDir, step, id, before, rng.State, tok.IsStopForMode(id, cfg.thinkMode), history, s.logits); err != nil {
					return err
				}
			}
			if tok.IsStopForMode(id, cfg.thinkMode) {
				break
			}
			fragment, err := tok.Decode([]int{id})
			if err != nil {
				return err
			}
			out, err := text.Add(fragment)
			if err != nil {
				return err
			}
			if _, err := io.WriteString(stdout, out); err != nil {
				return fmt.Errorf("write output: %w", err)
			}
			if _, err := s.Accept(id); err != nil {
				return err
			}
			generated++
			generatedIDs = append(generatedIDs, id)
			if cfg.checkingLogits() {
				fmt.Fprintf(stderr, "ds4-min: accepted token %d=%d\n", generated, id)
			}
		}
		return nil
	}); err != nil {
		return err
	}
	if err := cfg.endProfilerRange(profilerPhaseGeneration); err != nil {
		return err
	}
	// Same definition as ds4_cli.c:679-686: prompt tokens over the prefill
	// wall time, generated tokens over the decode wall time.
	decode := time.Since(decodeStart)
	if err := cfg.checkFinalLogits(s.logits, generated, len(tok.tokens), stderr); err != nil {
		if cfg.expectFailureDump != "" {
			history := append(append([]int(nil), prompt...), generatedIDs...)
			if dumpErr := dumpLogitsRaw(cfg.expectFailureDump, "final", s.logits, history, len(history)-1, generatedIDs, cfg); dumpErr != nil {
				return errors.Join(err, dumpErr)
			}
		}
		return err
	}
	if cfg.dumpLogits != "" {
		history := append(append([]int(nil), prompt...), generatedIDs...)
		if err := dumpLogits(cfg.dumpLogits, "final", s.logits, history, len(history)-1, generatedIDs, cfg); err != nil {
			return err
		}
	}
	if cfg.dumpFrontierLogitsDir != "" {
		history := append(append([]int(nil), prompt...), generatedIDs...)
		if err := dumpFrontierLogits(cfg.dumpFrontierLogitsDir, "final", s.logits, history, generatedIDs, cfg); err != nil {
			return err
		}
	}
	if cfg.dumpLogprobs != "" {
		if err := writeLogprobs(cfg.dumpLogprobs, len(prompt), cfg.context, cfg.logprobsTopK, logprobSteps, tok); err != nil {
			return err
		}
	}
	if cfg.calibPath != "" {
		// Interior markers go between the two boundary brackets in the
		// order they were taken, so the receipt reads in run order.
		calibrations = append(calibrations, interiorCalib.calibrations()...)
		calibrations = append(calibrations, calibrate(runID, "end"))
		f, err := os.Create(cfg.calibPath)
		if err != nil {
			return err
		}
		if err := errors.Join(writeCalibrations(f, calibrations), f.Close()); err != nil {
			return err
		}
		for _, c := range calibrations {
			fmt.Fprintf(stderr, "ds4-min: calibration %s host %d..%d\n", c.Name, c.Before, c.After)
		}
	}
	rate := func(n int, d time.Duration) float64 {
		if d <= 0 {
			return 0
		}
		return float64(n) / d.Seconds()
	}
	if err := text.Finish(); err != nil {
		return err
	}
	if generated > 0 {
		if _, err := io.WriteString(stdout, "\n"); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
	}
	fmt.Fprintf(stderr, "ds4-min: prefill: %.2f t/s, generation: %.2f t/s\n",
		rate(len(prompt), prefill), rate(generated, decode))
	// Last thing before the process unwinds: the context is still alive, all
	// device work is done, and every NVTX range is closed, so a flush here
	// costs nothing and loses nothing.
	if cfg.profilerStop {
		if err := stopCUDAProfiler(); err != nil {
			return err
		}
		fmt.Fprintf(stderr, "ds4-min: cuProfilerStop ok\n")
	}
	return nil
}

func parseConfig(args []string, stdin io.Reader, stderr io.Writer) (config, error) {
	var cfg config
	var prompt, promptFile, promptTokens string
	fs := flag.NewFlagSet("ds4-min", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	fs.StringVar(&cfg.backend, "backend", "auto", "execution backend")
	fs.StringVar(&cfg.family, "family", "auto", "model family: auto, flash or v41 (CPU reference)")
	fs.StringVar(&cfg.modelSHA256, "model-sha256", "", "independent V4.1 complete-file SHA-256 for portable state")
	addStreamingFlags(fs, &cfg.streaming)
	fs.IntVar(&cfg.context, "ctx", 4096, "session context length")
	fs.IntVar(&cfg.power, "power", 100, "GPU duty cycle percentage, 1..100")
	fs.StringVar(&cfg.model, "model", "", "path to model")
	fs.BoolVar(&cfg.inspect, "inspect", false, "print a backend-free GGUF summary and exit")
	fs.BoolVar(&cfg.interactive, "interactive", false, "read one chat message per line")
	fs.StringVar(&cfg.perplexityFile, "perplexity-file", "", "score raw text token perplexity and exit")
	fs.IntVar(&cfg.decodeConsistency, "decode-consistency", 0, "compare N greedy decode logits against a fresh re-prefill")
	fs.StringVar(&cfg.system, "system", "You are a helpful assistant", "system prompt; empty disables it")
	fs.StringVar(&cfg.evalSuite, "eval-suite", "", "execute the hard-smoke evaluation suite")
	fs.StringVar(&cfg.evalTrace, "eval-trace", "", "write the runtime evaluation trace")
	fs.StringVar(&cfg.evalCorpus, "eval-corpus", "", "external source-generated hard-smoke corpus directory")
	fs.StringVar(&cfg.evalArtifactsDir, "eval-artifacts-dir", "", "write diagnostic eval tokens, decision hashes and terminal logits to a new directory")
	fs.StringVar(&cfg.system, "sys", "You are a helpful assistant", "alias for -system")
	var think, nothink, thinkMax bool
	fs.BoolVar(&think, "think", false, "enable high thinking mode")
	fs.BoolVar(&nothink, "nothink", false, "disable thinking mode")
	fs.BoolVar(&thinkMax, "think-max", false, "enable maximum thinking mode")
	thinkLevel := fs.String("think-level", "", "unsupported compatibility flag")
	fs.StringVar(&prompt, "prompt", "", "user prompt")
	fs.StringVar(&promptFile, "prompt-file", "", "read user prompt from file")
	fs.StringVar(&promptTokens, "prompt-tokens", "", "read JSON token IDs from file")
	fs.StringVar(&cfg.prefixFile, "prefix-file", "", "read alternating USER:/ASSISTANT: chat turns from a file")
	fs.BoolVar(&cfg.raw, "raw", false, "tokenize the prompt as plain text without the chat template")
	fs.BoolVar(&cfg.dumpTokens, "dump-tokens", false, "print prompt token IDs and raw vocabulary strings, then exit")
	fs.IntVar(&cfg.tokens, "tokens", 16, "maximum generated tokens")
	defaults := sampler.Default
	var temperature = float64(defaults.Temperature)
	var topP = float64(defaults.TopP)
	var minP = float64(defaults.MinP)
	fs.Float64Var(&temperature, "temp", temperature, "sampling temperature; zero selects greedy")
	fs.IntVar(&cfg.sampling.TopK, "top-k", 0, "sampling top-k; zero means full vocabulary")
	fs.Float64Var(&topP, "top-p", topP, "sampling top-p")
	fs.Float64Var(&minP, "min-p", minP, "sampling minimum probability")
	fs.Uint64Var(&cfg.seed, "seed", 0, "sampling seed")
	steeringFlags(fs, &cfg)
	fs.StringVar(&cfg.dumpLogprobs, "dump-logprobs", "", "write per-step top-logprob JSON")
	fs.IntVar(&cfg.logprobsTopK, "logprobs-top-k", 20, "number of per-step logprob entries")
	fs.StringVar(&cfg.decodeLogitsDir, "dump-decode-logits-dir", "", "write raw decode logits to a directory")
	fs.StringVar(&cfg.sampledTraceDir, "dump-sampled-trace-dir", "", "write actual sampled tokens, RNG states, and preselection logits")
	fs.StringVar(&cfg.sampledTranscript, "dump-sampled-transcript", "", "write actual sampled decisions without copying logits")
	fs.StringVar(&cfg.kvSave, "kv-save", "", "save the backend session after prompt prefill")
	fs.StringVar(&cfg.kvLoad, "kv-load", "", "load a backend session before prompt evaluation")
	fs.StringVar(&cfg.kvLoadContract, "kv-load-contract", "", "qualified upstream model, token, payload and geometry receipt for kv-load")
	fs.IntVar(&cfg.syncPrefix, "sync-prefix", 0, "evaluate this prompt prefix before evaluating the full prompt")
	fs.IntVar(&cfg.prefillChunk, "prefill-chunk", 2048, "maximum tokens evaluated by one prefill batch")
	fs.Float64Var(&cfg.ballastGib, "ballast-gib", 0, "allocate and touch this many GiB of CUDA ballast before evaluation")
	fs.BoolVar(&cfg.warmWeights, "warm-weights", false, "fault the CUDA model mapping into memory before evaluation")
	fs.StringVar(&cfg.expectPrefillLogits, "expect-prefill-logits", "", "expected prefill logits SHA-256")
	fs.StringVar(&cfg.expectFinalLogits, "expect-final-logits", "", "expected final logits SHA-256")
	fs.StringVar(&cfg.dumpLogits, "dump-logits", "", "write prefill/final float32 logits and provenance sidecars using this prefix")
	var frontierLogitsDir, frontierLogitsAlias string
	fs.StringVar(&frontierLogitsDir, "dump-frontier-logits-dir", "", "write validated prefill/final frontier logits JSON into this directory")
	fs.StringVar(&frontierLogitsAlias, "dump-frontier-logits", "", "alias for -dump-frontier-logits-dir")
	fs.StringVar(&cfg.expectFailureDump, "expect-failure-dump", "", "write mismatching prefill/final logits and provenance sidecars before failing an expectation")
	addCollectorFlags(fs, &cfg.collectors)
	fs.StringVar(&cfg.cpuProfile, "cpuprofile", "", "write a Go CPU profile")
	fs.StringVar(&cfg.memProfile, "memprofile", "", "write a Go heap profile")
	fs.StringVar(&cfg.traceProfile, "trace", "", "write a Go execution trace")
	fs.BoolVar(&cfg.traceLoad, "trace-load", false, "start requested Go profiling before model loading")
	fs.StringVar(&cfg.spanPath, "spans", "", "write a gputrace host-event sidecar")
	fs.StringVar(&cfg.spanStream, "spans-stream", "", "stream that sidecar with no record cap")
	var spanLevel string
	fs.StringVar(&spanLevel, "span-level", "token", "span depth: off, phase, token, layer, or op")
	fs.StringVar(&cfg.runID, "run-id", "", "run identity shared with a gputrace timing sidecar")
	fs.StringVar(&cfg.calibPath, "calibration", "", "write NVTX clock-calibration brackets to this file")
	fs.IntVar(&cfg.calibEvery, "calibration-every", 64, "also bracket every Nth prefill batch; 0 for boundaries only")
	fs.BoolVar(&cfg.profilerStop, "cuda-profiler-stop", false, "call cuProfilerStop after the last kernel so the profiler flushes its buffers")
	fs.BoolVar(&cfg.profilerRange, "cuda-profiler-range", false, "bracket prefill and generation with cuProfilerStart/Stop so a profiler run with -c cudaProfilerApi bounds each phase the way ds4_bench.c does")
	fs.BoolVar(&cfg.warmPrefill, "warm-prefill", false, "evaluate the prompt once and reset the session before the measured prefill, as the C range runner does")
	var profilerPhaseValue string
	fs.StringVar(&profilerPhaseValue, "cuda-profiler-phase", "", "profile one phase with cuProfilerStart/Stop: prefill or generation")
	if err := fs.Parse(args); err != nil {
		return config{}, err
	}
	// Upstream defaults to HIGH thinking with sampling enabled. Explicit
	// -nothink and -temp 0 retain the deterministic compatibility gates.
	cfg.thinkMode = ThinkHigh
	if *thinkLevel != "" {
		return config{}, errors.New("think-level is not supported; use think, nothink, or think-max")
	}
	for _, arg := range args {
		name, value, hasValue := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		if hasValue && value == "false" {
			continue
		}
		switch name {
		case "think":
			cfg.thinkMode = ThinkHigh
		case "nothink":
			cfg.thinkMode = ThinkNone
		case "think-max":
			cfg.thinkMode = ThinkMax
		}
	}
	_ = think
	_ = nothink
	_ = thinkMax
	for _, name := range []string{"system", "sys"} {
		if f := fs.Lookup(name); f != nil {
			// Visit reports explicitly supplied aliases, while both aliases
			// intentionally share the same destination.
			fs.Visit(func(v *flag.Flag) {
				if v.Name == name {
					cfg.systemSet = true
				}
			})
		}
	}
	if fs.NArg() != 0 {
		return config{}, errors.New("unexpected arguments")
	}
	if err := validateSteeringConfig(&cfg, fs); err != nil {
		return config{}, err
	}
	cfg.sampling.Temperature = float32(temperature)
	cfg.sampling.TopP = float32(topP)
	cfg.sampling.MinP = float32(minP)
	if cfg.model == "" {
		return config{}, errors.New("model is required")
	}
	if cfg.context <= 0 {
		return config{}, errors.New("context must be positive")
	}
	if cfg.tokens <= 0 {
		return config{}, errors.New("tokens must be positive")
	}
	if cfg.prefillChunk < 256 || cfg.prefillChunk > 2048 || cfg.prefillChunk%256 != 0 {
		return config{}, errors.New("prefill-chunk must be a multiple of 256 in [256,2048]")
	}
	if cfg.kvSave != "" && cfg.kvLoad != "" && cfg.kvSave == cfg.kvLoad {
		return config{}, errors.New("kv-save and kv-load must name different files")
	}
	if cfg.kvLoadContract != "" && cfg.kvLoad == "" {
		return config{}, errors.New("kv-load-contract requires kv-load")
	}
	if cfg.syncPrefix < 0 {
		return config{}, errors.New("sync-prefix must be non-negative")
	}
	if cfg.syncPrefix > 0 && cfg.kvLoad != "" {
		return config{}, errors.New("sync-prefix conflicts with kv-load")
	}
	if err := validatePower(cfg.power, cfg.family); err != nil {
		return config{}, err
	}
	if cfg.inspect {
		return cfg, nil
	}
	if cfg.decodeConsistency < 0 {
		return config{}, errors.New("decode-consistency must be non-negative")
	}
	if profilerPhaseValue != "" {
		switch profilerPhase(profilerPhaseValue) {
		case profilerPhasePrefill, profilerPhaseGeneration:
			cfg.profilerPhase = profilerPhase(profilerPhaseValue)
		default:
			return config{}, fmt.Errorf("cuda-profiler-phase: want prefill or generation, got %q", profilerPhaseValue)
		}
		cfg.profilerRange = true
	} else if cfg.profilerRange {
		cfg.profilerPhase = profilerPhaseBoth
	}
	set := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	cfg.ballastGibSet = set["ballast-gib"]
	cfg.warmWeightsSet = set["warm-weights"]
	if cfg.ballastGibSet && cfg.ballastGib <= 0 {
		return config{}, errors.New("ballast-gib must be positive")
	}
	cfg.seedSet = set["seed"]
	if cfg.seedSet && cfg.seed == 0 {
		return config{}, errors.New("seed must be positive")
	}
	if cfg.sampling.Temperature < 0 || cfg.sampling.Temperature > 100 {
		return config{}, errors.New("temp must be in [0,100]")
	}
	if cfg.sampling.TopP < 0 || cfg.sampling.TopP > 1 || cfg.sampling.MinP < 0 || cfg.sampling.MinP > 1 {
		return config{}, errors.New("top-p and min-p must be in [0,1]")
	}
	if cfg.sampling.TopK < 0 || cfg.logprobsTopK <= 0 {
		return config{}, errors.New("top-k must be non-negative and logprobs-top-k must be positive")
	}
	if cfg.logprobsTopK > 128 {
		cfg.logprobsTopK = 128
	}
	for _, check := range []struct {
		name  string
		value *string
	}{
		{"expect-prefill-logits", &cfg.expectPrefillLogits},
		{"expect-final-logits", &cfg.expectFinalLogits},
	} {
		if set[check.name] {
			value, err := normalizeLogitsDigest(*check.value)
			if err != nil {
				return config{}, fmt.Errorf("%s: %w", check.name, err)
			}
			*check.value = value
		}
	}
	if set["prompt-tokens"] && (set["prompt"] || set["prompt-file"]) {
		return config{}, errors.New("prompt-tokens conflicts with prompt and prompt-file")
	}
	if (cfg.sampledTraceDir != "" || cfg.sampledTranscript != "") && cfg.dumpLogprobs != "" {
		return config{}, errors.New("dump-sampled-trace-dir conflicts with argmax dump-logprobs mode")
	}
	if prompt != "" && promptFile != "" {
		return config{}, errors.New("prompt and prompt-file conflict")
	}
	if cfg.prefixFile != "" && (prompt != "" || promptFile != "" || promptTokens != "") {
		return config{}, errors.New("prefix-file conflicts with prompt, prompt-file, and prompt-tokens")
	}
	if frontierLogitsDir != "" && frontierLogitsAlias != "" {
		return config{}, errors.New("dump-frontier-logits and dump-frontier-logits-dir conflict")
	}
	cfg.dumpFrontierLogitsDir = frontierLogitsDir
	if cfg.dumpFrontierLogitsDir == "" {
		cfg.dumpFrontierLogitsDir = frontierLogitsAlias
	}
	if cfg.raw && (cfg.prefixFile != "" || set["prompt-tokens"]) {
		return config{}, errors.New("raw conflicts with prefix-file and prompt-tokens")
	}
	if cfg.interactive {
		if set["dump-sampled-transcript"] {
			return config{}, errors.New("interactive conflicts with dump-sampled-transcript")
		}
		for _, name := range []string{"prompt", "prompt-file", "prompt-tokens", "prefix-file", "raw", "dump-tokens", "inspect", "eval-suite", "eval-trace", "eval-corpus", "perplexity-file", "decode-consistency", "dump-logprobs", "dump-decode-logits-dir", "dump-logits", "dump-frontier-logits", "dump-frontier-logits-dir", "dump-sampled-trace-dir", "expect-prefill-logits", "expect-final-logits", "kv-load", "kv-load-contract", "kv-save", "sync-prefix", "cpuprofile", "memprofile", "trace", "trace-load", "spans", "spans-stream", "cuda-profiler-phase", "cuda-profiler-range", "cuda-profiler-stop"} {
			if set[name] {
				return config{}, fmt.Errorf("interactive conflicts with %s", name)
			}
		}
	}
	if err := validateEvalFlags(cfg, set); err != nil {
		return config{}, err
	}
	if cfg.dumpTokens {
		if !set["prompt"] && !set["prompt-file"] {
			return config{}, errors.New("dump-tokens requires prompt or prompt-file")
		}
		if cfg.prefixFile != "" || set["prompt-tokens"] {
			return config{}, errors.New("dump-tokens requires a prose prompt")
		}
	}
	if cfg.perplexityFile != "" && (set["prompt"] || set["prompt-file"] || set["prompt-tokens"] || cfg.prefixFile != "" || cfg.raw || cfg.dumpTokens) {
		return config{}, errors.New("perplexity-file conflicts with prompt, raw, prefix-file, and dump-tokens")
	}
	if cfg.decodeConsistency != 0 && cfg.perplexityFile != "" {
		return config{}, errors.New("decode-consistency conflicts with perplexity-file")
	}
	if cfg.spanPath != "" && cfg.spanStream != "" {
		return config{}, errors.New("-spans and -spans-stream both write the host-event sidecar; choose one")
	}
	level, err := parseSpanLevel(spanLevel)
	if err != nil {
		return config{}, err
	}
	cfg.spanLevel = level
	if cfg.calibPath != "" {
		if err := requireNVTXProvider(); err != nil {
			return config{}, err
		}
	}
	backend, err := selectBackend(cfg.backend)
	if err != nil {
		return config{}, err
	}
	cfg.backend = backend
	if err := validateFamilySelection(cfg); err != nil {
		return config{}, err
	}
	if err := cfg.streaming.validate(backend); err != nil {
		return config{}, err
	}
	switch {
	case cfg.evalSuite != "":
		// The pinned evaluator owns its twelve inputs; never read ordinary stdin.
	case cfg.collectors.dataset != "" || cfg.collectors.imatrix != "":
		if err := validateCalibrationFlags(set, cfg.backend); err != nil {
			return config{}, err
		}
		if set["prompt"] || set["prompt-file"] || set["prompt-tokens"] || cfg.prefixFile != "" {
			return config{}, errors.New("imatrix dataset conflicts with prompt inputs")
		}
		// The dataset runner owns its input; never block on ordinary stdin.
	case cfg.interactive:
		// Run reads lines directly; parsing must leave stdin untouched.
	case cfg.perplexityFile != "":
		data, readErr := os.ReadFile(cfg.perplexityFile)
		if readErr != nil {
			return config{}, fmt.Errorf("read perplexity file: %w", readErr)
		}
		cfg.prompt = data
	case set["prompt-tokens"]:
		data, readErr := os.ReadFile(promptTokens)
		if readErr != nil {
			return config{}, fmt.Errorf("read prompt tokens: %w", readErr)
		}
		cfg.promptTokens, err = parsePromptTokens(data)
		if err != nil {
			return config{}, err
		}
		if err := cfg.checkPromptCapacity(len(cfg.promptTokens)); err != nil {
			return config{}, err
		}
	case set["prompt-file"]:
		cfg.prompt, err = os.ReadFile(promptFile)
		if err != nil {
			return config{}, fmt.Errorf("read prompt: %w", err)
		}
	case set["prompt"]:
		cfg.prompt = []byte(prompt)
	case cfg.prefixFile != "":
		data, readErr := os.ReadFile(cfg.prefixFile)
		if readErr != nil {
			return config{}, fmt.Errorf("read prefix file: %w", readErr)
		}
		cfg.prompt = data
	default:
		cfg.prompt, err = io.ReadAll(stdin)
		if err != nil {
			return config{}, fmt.Errorf("read prompt: %w", err)
		}
	}
	return cfg, nil
}

// defaultRunID names one run in both gputrace sidecars. The host-event reader
// refuses a file that mixes IDs, and the capture path merges every
// process of a target tree into one file, so the ID has to distinguish
// processes as well as runs.
func defaultRunID() string {
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return fmt.Sprintf("ds4-min-%s-%d-%d", host, os.Getpid(), time.Now().UnixNano())
}

// writeSpans writes whichever span sidecars this run was asked for. The
// GPUTRACE_APP_EVENTS path needs no flag: "gputrace capture" sets it.
func writeSpans(cfg config, log io.Writer) error {
	if sr := streamSpans; sr != nil {
		streamSpans = nil
		err := sr.Close()
		acc := accountingPath(cfg.spanStream)
		if err != nil {
			return fmt.Errorf("streaming span sidecar: %w (see %s)", err, acc)
		}
		fmt.Fprintf(log, "ds4-min: streamed %d spans to %s; accounting in %s\n", sr.written, cfg.spanStream, acc)
		return nil
	}
	r := spans
	if r == nil {
		return nil
	}
	spans = nil
	var errs []error
	if cfg.spanPath != "" {
		errs = append(errs, r.WriteHostEvents(cfg.spanPath))
	}
	if path := os.Getenv("GPUTRACE_APP_EVENTS"); path != "" {
		errs = append(errs, r.AppendGPUEvents(path))
	}
	if dropped := r.Dropped(); dropped > 0 {
		fmt.Fprintf(log, "ds4-min: recorded %d spans, dropped %d at the record limit; use a shallower -span-level\n", r.Len(), dropped)
	}
	return errors.Join(errs...)
}

// parsePromptTokens reads IDs without adding a chat template or special tokens.
func parsePromptTokens(data []byte) ([]int, error) {
	var values []*int
	if err := json.Unmarshal(data, &values); err != nil {
		return nil, fmt.Errorf("prompt tokens: %w", err)
	}
	if len(values) == 0 {
		return nil, errors.New("prompt tokens: empty array")
	}
	tokens := make([]int, len(values))
	for i, value := range values {
		if value == nil || *value < 0 {
			return nil, fmt.Errorf("prompt token %d: expected a nonnegative integer", i)
		}
		tokens[i] = *value
	}
	return tokens, nil
}

func runPerplexity(s *session, tokens []int, cfg config, stdout io.Writer) error {
	const prefix = 32
	if len(tokens) <= prefix {
		return fmt.Errorf("perplexity needs more than %d tokens, got %d", prefix, len(tokens))
	}
	scored := len(tokens) - prefix
	if cfg.tokens > 0 && scored > cfg.tokens {
		scored = cfg.tokens
	}
	if scored > s.context-prefix {
		scored = s.context - prefix
	}
	if scored <= 0 {
		return fmt.Errorf("context too small for perplexity scoring")
	}
	if err := s.Sync(tokens[:prefix]); err != nil {
		return fmt.Errorf("perplexity prefix: %w", err)
	}
	logits := make([][]float32, 0, scored)
	targets := make([]int, 0, scored)
	for i, target := range tokens[prefix : prefix+scored] {
		logits = append(logits, append([]float32(nil), s.logits...))
		targets = append(targets, target)
		if i == scored-1 {
			break
		}
		if len(s.tokens) >= s.context {
			return fmt.Errorf("perplexity context exhausted at token %d", prefix+i)
		}
		if err := s.backend.Eval(target, len(s.tokens)); err != nil {
			return fmt.Errorf("perplexity token %d: %w", prefix+i, err)
		}
		s.tokens = append(s.tokens, target)
		s.logits = append(s.logits[:0], s.backend.Logits()...)
	}
	p, err := dsv4eval.ScorePerplexity(logits, targets)
	if err != nil {
		return err
	}
	return writePerplexity(stdout, len(tokens), p)
}

func writePerplexity(w io.Writer, tokens int, p dsv4eval.Perplexity) error {
	if tokens < p.Scored {
		return fmt.Errorf("perplexity: %d tokens for %d scored", tokens, p.Scored)
	}
	_, err := fmt.Fprintf(w, "tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n", tokens, p.Scored, p.NLL, p.AvgNLL, p.PPL)
	return err
}

func runDecodeConsistency(live *session, prompt []int, cfg config, stdout io.Writer) error {
	return checkFinalDecodeConsistency(live, prompt, cfg, stdout)
}

func advanceSession(s *session, token int) error {
	if len(s.tokens) >= s.context {
		return errContextOverflow
	}
	if err := s.backend.Eval(token, len(s.tokens)); err != nil {
		return err
	}
	s.tokens = append(s.tokens, token)
	s.logits = append(s.logits[:0], s.backend.Logits()...)
	return nil
}

func writeConsistency(w io.Writer, c dsv4eval.Consistency) error {
	_, err := fmt.Fprintf(w, "positions=%d max_abs=%.9g rms=%.9g top1=%d/%d\n", c.Positions, c.MaxAbs, c.RMS, c.Top1Matches, c.Positions)
	return err
}

// preparePrompt checks the loaded vocabulary and reserves room for every
// requested decode step before the backend is opened.
func preparePrompt(cfg config, tok *tokenizer) ([]int, error) {
	prompt := cfg.promptTokens
	if prompt == nil {
		var err error
		if cfg.perplexityFile != "" {
			prompt, err = tok.Encode(string(cfg.prompt))
		} else if cfg.raw {
			prompt, err = tok.Encode(string(cfg.prompt))
		} else if cfg.prefixFile != "" {
			turns, parseErr := parsePrefixFile(cfg.prompt)
			if parseErr != nil {
				return nil, fmt.Errorf("parse prefix file: %w", parseErr)
			}
			system := cfg.system
			if system == "" && !cfg.systemSet {
				system = defaultSystemPrompt
			}
			prompt, err = tok.RenderPrefix(system, turns, cfg.thinkMode)
		} else {
			system := cfg.system
			if system == "" && !cfg.systemSet {
				system = defaultSystemPrompt
			}
			prompt, err = tok.RenderChat(system, string(cfg.prompt), cfg.thinkMode, cfg.context)
		}
		if err != nil {
			return nil, fmt.Errorf("render prompt: %w", err)
		}
	}
	for i, id := range prompt {
		if id < 0 || id >= len(tok.tokens) {
			return nil, fmt.Errorf("prompt token %d: ID %d outside model vocabulary of %d tokens", i, id, len(tok.tokens))
		}
	}
	if cfg.dumpTokens {
		if len(prompt) > cfg.context {
			return nil, fmt.Errorf("prompt length %d exceeds context %d", len(prompt), cfg.context)
		}
		return prompt, nil
	}
	if cfg.perplexityFile != "" {
		if len(prompt) <= 32 {
			return nil, fmt.Errorf("perplexity file needs more than 32 tokens, got %d", len(prompt))
		}
		if cfg.context <= 32 {
			return nil, fmt.Errorf("context too small for perplexity scoring")
		}
		return prompt, nil
	}
	if err := cfg.checkPromptCapacity(len(prompt)); err != nil {
		return nil, err
	}
	return prompt, nil
}

// dumpTokens writes the token vector followed by the raw vocabulary spelling
// for each ID. The format matches ds4's dump_tokens_fp output: the first line
// is a JSON-shaped array, while the remaining lines deliberately preserve the
// GGUF token spelling instead of decoding byte-runes to text.
func dumpTokens(w io.Writer, ids []int, tok *tokenizer) error {
	parts := make([]string, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(tok.tokens) {
			return fmt.Errorf("dump token %d: ID %d outside model vocabulary of %d tokens", i, id, len(tok.tokens))
		}
		parts[i] = strconv.Itoa(id)
	}
	if _, err := fmt.Fprintf(w, "[%s]\n", strings.Join(parts, ", ")); err != nil {
		return fmt.Errorf("write token IDs: %w", err)
	}
	for i, id := range ids {
		if _, err := fmt.Fprintf(w, "%6d  %s\n", id, tok.tokens[id]); err != nil {
			return fmt.Errorf("write token %d: %w", i, err)
		}
	}
	return nil
}

// warmPrefill evaluates prompt once and resets s, so the measured prefill
// that follows starts with the batch resources allocated and the kernels
// loaded. matched-c-range-runner.c does the same before its prefill range,
// and a range around the first prefill instead also holds that setup. The
// warm digest is reported but a mismatch does not stop the run, so a run
// can say whether the first evaluation, the second, or both went wrong.
func warmPrefill(s *session, prompt []int, vocab int, want string, log io.Writer) error {
	if err := s.Sync(prompt); err != nil {
		return fmt.Errorf("warm prefill: %w", err)
	}
	if err := checkLogits("warmup", s.logits, vocab, want, log); err != nil {
		fmt.Fprintf(log, "ds4-min: warmup: %v\n", err)
	}
	return s.Reset()
}

// beginProfilerRange and endProfilerRange bracket one phase with the
// driver's profiler controls. ds4_bench.c:861-879 brackets the C oracle the
// same way, so with -c cudaProfilerApi both implementations hand the
// profiler the same phase boundary instead of one being an NVTX window and
// the other a capture range.
func (cfg config) phaseProfileEnabled(phase profilerPhase) bool {
	if !cfg.profilerRange {
		return false
	}
	// Treat the zero value as the legacy both-phases mode. This keeps config
	// literals in tests and callers equivalent to -cuda-profiler-range.
	return cfg.profilerPhase == "" || cfg.profilerPhase == profilerPhaseBoth || cfg.profilerPhase == phase
}

func (cfg config) beginProfilerRange(phase profilerPhase) error {
	if !cfg.phaseProfileEnabled(phase) {
		return nil
	}
	return startCUDAProfiler()
}

func (cfg config) endProfilerRange(phase profilerPhase) error {
	if !cfg.phaseProfileEnabled(phase) {
		return nil
	}
	return stopCUDAProfiler()
}

func (cfg config) checkPromptCapacity(n int) error {
	// Subtraction avoids overflow for a very large -tokens value.
	if n > cfg.context || cfg.tokens > cfg.context-n {
		return fmt.Errorf("prompt length %d and %d generated tokens exceed context %d", n, cfg.tokens, cfg.context)
	}
	return nil
}
