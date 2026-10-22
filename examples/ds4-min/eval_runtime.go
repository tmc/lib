package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime/debug"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
)

// evalModel borrows the existing session's State; it owns no second cache.
type evalModel struct {
	session   *session
	tokenizer *tokenizer
}

func (m evalModel) Render(system, question string) ([]int, error) {
	return m.tokenizer.RenderChat(system, question, ThinkHigh, m.session.context)
}
func (m evalModel) Sync(ids []int) error { return m.session.Sync(ids) }
func (m evalModel) Eval(id int) error    { _, err := m.session.Accept(id); return err }
func (m evalModel) Logits() []float32    { return m.session.logits }
func (m evalModel) Text(id int) (string, error) {
	b, err := m.tokenizer.Decode([]int{id})
	return string(b), err
}
func (m evalModel) IsStop(id int) bool { return m.tokenizer.stops.Contains(id) }
func (m evalModel) ThinkClose() []int  { return []int{m.tokenizer.thinkingEnd} }

func runEvalSmoke(cfg config, tok *tokenizer) (err error) {
	if cfg.evalSuite != "hard-smoke" || cfg.evalTrace == "" {
		return fmt.Errorf("eval: require -eval-suite hard-smoke and -eval-trace")
	}
	cases, err := dsv4eval.LoadSmokeCorpus(cfg.evalCorpus)
	if err != nil {
		return fmt.Errorf("eval: corpus admission: %w", err)
	}
	if cfg.sampling.Temperature != 0 {
		return fmt.Errorf("eval: temperature must be zero")
	}
	build := ""
	modified := false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" {
				build = s.Value
			}
			if s.Key == "vcs.modified" && s.Value == "true" {
				modified = true
			}
		}
	}
	if build == "" || modified {
		return fmt.Errorf("eval: require a clean revision-stamped binary")
	}
	f, err := os.Open(cfg.model)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, readErr := io.Copy(hash, f)
	if err = errors.Join(readErr, f.Close()); err != nil {
		return fmt.Errorf("eval: hash model: %w", err)
	}
	return runEvalBackend(cfg, openSessionBackend, func(backend sessionBackend) error {
		s, err := newSession(backend, cfg.context, tok.stops)
		if err != nil {
			return err
		}
		if err := configurePrefillChunk(backend, cfg.prefillChunk); err != nil {
			return err
		}
		if cfg.evalOutput == nil {
			return fmt.Errorf("eval: trace output was not admitted")
		}
		return runEvalCases(cfg.evalOutput, evalModel{s, tok}, cases, dsv4eval.RunOptions{Build: build, ModelSHA256: fmt.Sprintf("%x", hash.Sum(nil)), Context: cfg.context, Soft: 1024, Hard: 512, Rank: 3}, cfg)
	})
}

// runEvalBackend retains failed request cleanup before releasing its owner.
// A trace emitted by body is diagnostic until this function returns nil.
func runEvalBackend(cfg config, open func(config) (sessionBackend, io.Closer, error), body func(sessionBackend) error) (err error) {
	backend, closer, err := open(cfg)
	if err != nil {
		return closeEvaluationBackend(err, closer)
	}
	defer func() { err = closeEvaluationBackend(err, closer) }()
	return body(backend)
}

func admitEvalOutputs(cfg config) (*os.File, error) {
	if cfg.evalSuite == "" {
		return nil, nil
	}
	if err := admitEvalArtifacts(cfg); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(cfg.evalTrace, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, fmt.Errorf("eval: admit new trace: %w", err)
	}
	return f, nil
}

func runEvalCases(out io.Writer, model dsv4eval.Runtime, cases []dsv4eval.Case, options dsv4eval.RunOptions, cfg config) error {
	if cfg.evalArtifactsDir != "" {
		options.Capture = (dsv4eval.ArtifactRecorder{Dir: cfg.evalArtifactsDir}).Record
	}
	return dsv4eval.RunSmoke(out, model, cases, options)
}

// admitEvalArtifacts runs before even the model metadata file is opened.
// A partial directory remains diagnostic evidence, never an admitted rerun.
func admitEvalArtifacts(cfg config) error {
	if cfg.evalArtifactsDir == "" {
		return nil
	}
	if err := os.Mkdir(cfg.evalArtifactsDir, 0700); err != nil {
		return fmt.Errorf("eval: admit new artifacts directory: %w", err)
	}
	return nil
}

func validateEvalFlags(cfg config, set map[string]bool) error {
	active := cfg.evalSuite != "" || set["eval-trace"] || set["eval-corpus"] || set["eval-artifacts-dir"]
	if !active {
		return nil
	}
	if cfg.evalSuite != "hard-smoke" || cfg.evalTrace == "" || cfg.evalCorpus == "" || cfg.sampling.Temperature != 0 || cfg.thinkMode != ThinkHigh {
		return fmt.Errorf("eval requires hard-smoke suite, trace, corpus and temp 0")
	}
	for _, name := range []string{"prompt", "prompt-file", "prompt-tokens", "prefix-file", "raw", "dump-tokens", "inspect", "interactive", "system", "perplexity-file", "decode-consistency", "dataset", "imatrix", "kv-load", "kv-save", "warm-prefill", "dump-logits", "dump-decode-logits-dir", "dump-logprobs", "dump-sampled-trace-dir"} {
		if set[name] {
			return fmt.Errorf("eval conflicts with %s", name)
		}
	}
	return nil
}
