package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/tmc/lib/gguf"
)

const usage = `usage: ds4-min -model file [-backend auto|metal|cuda|cpu] [-prompt text | -prompt-file file | -prompt-tokens file] [-tokens n] [-ctx n]

Load one DeepSeek-V4-Flash Q2 model and generate a non-thinking response.

  -backend string
        execution backend: auto, metal, cuda, or cpu (default "auto")
  -ctx int
        session context length (default 4096)
  -model string
        path to the pinned GGUF model (required)
  -prompt string
        user prompt; conflicts with -prompt-file and -prompt-tokens
  -prompt-file string
        read the user prompt from file; conflicts with -prompt and -prompt-tokens
  -prompt-tokens string
        read a JSON token-id array verbatim; conflicts with prose prompts
  -tokens int
        maximum generated tokens (default 16)
  -expect-prefill-logits string
        require this SHA-256 of little-endian float32 prefill logits
  -expect-final-logits string
        require this final logits SHA-256 and exactly -tokens accepted tokens

Profiling. These start after the model is loaded, so a profile describes
generation rather than the mmap residency wait.

  -cpuprofile file
        write a Go CPU profile
  -memprofile file
        write a Go heap profile at exit
  -trace file
        write a Go execution trace
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

type config struct {
	model        string
	backend      string
	prompt       []byte
	promptTokens []int
	context      int
	tokens       int
	log          io.Writer

	expectPrefillLogits string
	expectFinalLogits   string

	cpuProfile   string
	memProfile   string
	traceProfile string
	spanPath     string
	spanStream   string
	calibEvery   int
	spanLevel    spanLevel
	runID        string
	calibPath    string
	profilerStop bool
}

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-min:", err)
		exitProcess(1)
	}
	exitProcess(0)
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) (err error) {
	cfg, err := parseConfig(args, stdin, stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
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
	if err := validatePinnedProfile(g); err != nil {
		return fmt.Errorf("model profile: %w", err)
	}
	cfg.log = stderr
	tok, err := newTokenizer(g)
	if err != nil {
		return err
	}
	prompt, err := preparePrompt(cfg, tok)
	if err != nil {
		return err
	}
	cfg.logLogitsWorkload(prompt, stderr)
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := closer.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close backend: %w", closeErr))
		}
	}()
	s, err := newSession(b, cfg.context, tok.stops)
	if err != nil {
		return err
	}
	prof, err := startProfiling(cfg)
	if err != nil {
		return err
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
	if err := tracePhase(levelPhase, "prefill", func() error { return s.Sync(prompt) }); err != nil {
		return err
	}
	prefill := time.Since(prefillStart)
	if err := checkLogits("prefill", s.logits, len(tok.tokens), cfg.expectPrefillLogits, stderr); err != nil {
		return err
	}
	var text textAssembler
	generated := 0
	decodeStart := time.Now()
	if err := tracePhase(levelPhase, "generation", func() error {
		for range cfg.tokens {
			id := greedyToken(s.logits)
			if tok.IsStop(id) {
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
			if cfg.checkingLogits() {
				fmt.Fprintf(stderr, "ds4-min: accepted token %d=%d\n", generated, id)
			}
		}
		return nil
	}); err != nil {
		return err
	}
	// Same definition as ds4_cli.c:679-686: prompt tokens over the prefill
	// wall time, generated tokens over the decode wall time.
	decode := time.Since(decodeStart)
	if err := cfg.checkFinalLogits(s.logits, generated, len(tok.tokens), stderr); err != nil {
		return err
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

func greedyToken(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

func parseConfig(args []string, stdin io.Reader, stderr io.Writer) (config, error) {
	var cfg config
	var prompt, promptFile, promptTokens string
	fs := flag.NewFlagSet("ds4-min", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	fs.StringVar(&cfg.backend, "backend", "auto", "execution backend")
	fs.IntVar(&cfg.context, "ctx", 4096, "session context length")
	fs.StringVar(&cfg.model, "model", "", "path to model")
	fs.StringVar(&prompt, "prompt", "", "user prompt")
	fs.StringVar(&promptFile, "prompt-file", "", "read user prompt from file")
	fs.StringVar(&promptTokens, "prompt-tokens", "", "read JSON token IDs from file")
	fs.IntVar(&cfg.tokens, "tokens", 16, "maximum generated tokens")
	fs.StringVar(&cfg.expectPrefillLogits, "expect-prefill-logits", "", "expected prefill logits SHA-256")
	fs.StringVar(&cfg.expectFinalLogits, "expect-final-logits", "", "expected final logits SHA-256")
	fs.StringVar(&cfg.cpuProfile, "cpuprofile", "", "write a Go CPU profile")
	fs.StringVar(&cfg.memProfile, "memprofile", "", "write a Go heap profile")
	fs.StringVar(&cfg.traceProfile, "trace", "", "write a Go execution trace")
	fs.StringVar(&cfg.spanPath, "spans", "", "write a gputrace host-event sidecar")
	fs.StringVar(&cfg.spanStream, "spans-stream", "", "stream that sidecar with no record cap")
	var spanLevel string
	fs.StringVar(&spanLevel, "span-level", "token", "span depth: off, phase, token, layer, or op")
	fs.StringVar(&cfg.runID, "run-id", "", "run identity shared with a gputrace timing sidecar")
	fs.StringVar(&cfg.calibPath, "calibration", "", "write NVTX clock-calibration brackets to this file")
	fs.IntVar(&cfg.calibEvery, "calibration-every", 64, "also bracket every Nth prefill batch; 0 for boundaries only")
	fs.BoolVar(&cfg.profilerStop, "cuda-profiler-stop", false, "call cuProfilerStop after the last kernel so the profiler flushes its buffers")
	if err := fs.Parse(args); err != nil {
		return config{}, err
	}
	if fs.NArg() != 0 {
		return config{}, errors.New("unexpected arguments")
	}
	if cfg.model == "" {
		return config{}, errors.New("model is required")
	}
	if cfg.context <= 0 {
		return config{}, errors.New("context must be positive")
	}
	if cfg.tokens <= 0 {
		return config{}, errors.New("tokens must be positive")
	}
	set := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
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
	if prompt != "" && promptFile != "" {
		return config{}, errors.New("prompt and prompt-file conflict")
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
	switch {
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
	case promptFile != "":
		cfg.prompt, err = os.ReadFile(promptFile)
		if err != nil {
			return config{}, fmt.Errorf("read prompt: %w", err)
		}
	case prompt != "":
		cfg.prompt = []byte(prompt)
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

// preparePrompt checks the loaded vocabulary and reserves room for every
// requested decode step before the backend is opened.
func preparePrompt(cfg config, tok *tokenizer) ([]int, error) {
	prompt := cfg.promptTokens
	if prompt == nil {
		var err error
		prompt, err = tok.RenderNonThinking("You are a helpful assistant", string(cfg.prompt))
		if err != nil {
			return nil, fmt.Errorf("render prompt: %w", err)
		}
	}
	for i, id := range prompt {
		if id < 0 || id >= len(tok.tokens) {
			return nil, fmt.Errorf("prompt token %d: ID %d outside model vocabulary of %d tokens", i, id, len(tok.tokens))
		}
	}
	if err := cfg.checkPromptCapacity(len(prompt)); err != nil {
		return nil, err
	}
	return prompt, nil
}

func (cfg config) checkPromptCapacity(n int) error {
	// Subtraction avoids overflow for a very large -tokens value.
	if n > cfg.context || cfg.tokens > cfg.context-n {
		return fmt.Errorf("prompt length %d and %d generated tokens exceed context %d", n, cfg.tokens, cfg.context)
	}
	return nil
}
