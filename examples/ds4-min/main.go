package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/tmc/lib/gguf"
)

const usage = `usage: ds4-min -model file [-backend auto|metal|cuda|cpu] [-prompt text | -prompt-file file] [-tokens n] [-ctx n]

Load one DeepSeek-V4-Flash Q2 model and generate a non-thinking response.

  -backend string
        execution backend: auto, metal, cuda, or cpu (default "auto")
  -ctx int
        session context length (default 4096)
  -model string
        path to the pinned GGUF model (required)
  -prompt string
        user prompt; conflicts with -prompt-file
  -prompt-file string
        read the user prompt from file; conflicts with -prompt
  -tokens int
        maximum generated tokens (default 16)

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
  -span-level string
        span depth: off, phase, token, layer, or op (default "token")

Spans are also appended to $GPUTRACE_APP_EVENTS when that is set, which
is what "gputrace capture" does, so a captured run carries them with no
flag. Both sidecars are bounded: recording stops at 100,000 spans, which
one 2048-token prefill exceeds at -span-level=op.
`

type config struct {
	model   string
	backend string
	prompt  []byte
	context int
	tokens  int
	log     io.Writer

	cpuProfile   string
	memProfile   string
	traceProfile string
	spanPath     string
	spanLevel    spanLevel
}

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-min:", err)
		os.Exit(1)
	}
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
	prompt, err := tok.RenderNonThinking("You are a helpful assistant", string(cfg.prompt))
	if err != nil {
		return fmt.Errorf("render prompt: %w", err)
	}
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
	if cfg.spanLevel != levelOff {
		spans = newSpanRecorder(cfg.spanLevel, runID())
	}
	fmt.Fprintf(stderr, "ds4-min: processing %d input tokens\n", len(prompt))
	prefillStart := time.Now()
	prefillSpan := begin(levelPhase, "prefill", nil)
	if err := s.Sync(prompt); err != nil {
		return err
	}
	prefillSpan.end()
	prefill := time.Since(prefillStart)
	var text textAssembler
	generated := 0
	decodeStart := time.Now()
	decodeSpan := begin(levelPhase, "generation", nil)
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
	}
	decodeSpan.end()
	// Same definition as ds4_cli.c:679-686: prompt tokens over the prefill
	// wall time, generated tokens over the decode wall time.
	decode := time.Since(decodeStart)
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
	var prompt, promptFile string
	fs := flag.NewFlagSet("ds4-min", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	fs.StringVar(&cfg.backend, "backend", "auto", "execution backend")
	fs.IntVar(&cfg.context, "ctx", 4096, "session context length")
	fs.StringVar(&cfg.model, "model", "", "path to model")
	fs.StringVar(&prompt, "prompt", "", "user prompt")
	fs.StringVar(&promptFile, "prompt-file", "", "read user prompt from file")
	fs.IntVar(&cfg.tokens, "tokens", 16, "maximum generated tokens")
	fs.StringVar(&cfg.cpuProfile, "cpuprofile", "", "write a Go CPU profile")
	fs.StringVar(&cfg.memProfile, "memprofile", "", "write a Go heap profile")
	fs.StringVar(&cfg.traceProfile, "trace", "", "write a Go execution trace")
	fs.StringVar(&cfg.spanPath, "spans", "", "write a gputrace host-event sidecar")
	var spanLevel string
	fs.StringVar(&spanLevel, "span-level", "token", "span depth: off, phase, token, layer, or op")
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
	if prompt != "" && promptFile != "" {
		return config{}, errors.New("prompt and prompt-file conflict")
	}
	level, err := parseSpanLevel(spanLevel)
	if err != nil {
		return config{}, err
	}
	cfg.spanLevel = level
	backend, err := selectBackend(cfg.backend)
	if err != nil {
		return config{}, err
	}
	cfg.backend = backend
	switch {
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

// runID names one run in both gputrace sidecars. The host-event reader
// refuses a file that mixes IDs, and the capture path merges every
// process of a target tree into one file, so the ID has to distinguish
// processes as well as runs.
func runID() string {
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return fmt.Sprintf("ds4-min-%s-%d-%d", host, os.Getpid(), time.Now().UnixNano())
}

// writeSpans writes whichever span sidecars this run was asked for. The
// GPUTRACE_APP_EVENTS path needs no flag: "gputrace capture" sets it.
func writeSpans(cfg config, log io.Writer) error {
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
