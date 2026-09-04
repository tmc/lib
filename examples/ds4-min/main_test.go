package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
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
		{"prompt file", []string{"-model", "m", "-prompt-file", "not-present"}, "read prompt:"},
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
