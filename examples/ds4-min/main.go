package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
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
`

type config struct {
	model   string
	backend string
	prompt  []byte
	context int
	tokens  int
}

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-min:", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	cfg, err := parseConfig(args, stdin, stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	_ = cfg
	_ = stdout
	return errors.New("model execution not implemented")
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
