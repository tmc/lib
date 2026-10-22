package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/tmc/lib/examples/ds4-min/bench/dsv4bench"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/gguf"
)

// -frontier-bench -evaluate is a provenance-bearing measurement producer.
// It does not replace the existing upstream-compatible scalar CLI outputs.
func runFrontierEvaluation(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("frontier-bench -evaluate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "pinned GGUF model")
	backend := fs.String("backend", "auto", "execution backend")
	prompt := fs.String("prompt-tokens", "", "JSON prompt IDs")
	forced := fs.String("forced-tokens", "", "JSON teacher-forced continuation IDs")
	revision := fs.String("revision", "", "clean executable VCS revision")
	output := fs.String("receipt", "", "new output JSON receipt path")
	allocated := fs.Int("ctx", 4096, "allocated request context")
	maxAbs := fs.Float64("max-abs", -1, "pre-registered consistency bound")
	maxRMS := fs.Float64("max-rms", -1, "pre-registered consistency bound")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *model == "" || *prompt == "" || *forced == "" || *revision == "" || *output == "" || *allocated <= 0 || *maxAbs < 0 || *maxRMS < 0 || math.IsNaN(*maxAbs) || math.IsNaN(*maxRMS) || math.IsInf(*maxAbs, 0) || math.IsInf(*maxRMS, 0) {
		return fmt.Errorf("evaluation: model, explicit IDs, clean revision, receipt and finite consistency bounds required")
	}
	if err := frontierBuildRevision(*revision); err != nil {
		return err
	}
	cfg, err := parseConfig([]string{"-model", *model, "-backend", *backend, "-prompt-tokens", *prompt, "-ctx", strconv.Itoa(*allocated), "-tokens", "0"}, nil, stderr)
	if err != nil {
		return err
	}
	cfg.backend, err = selectBackend(cfg.backend)
	if err != nil {
		return err
	}
	cfg.log = stderr
	f, err := os.Open(cfg.model)
	if err != nil {
		return err
	}
	info, err := f.Stat()
	if err != nil {
		return errors.Join(err, f.Close())
	}
	g, err := gguf.Read(f, info.Size())
	err = errors.Join(err, f.Close())
	if err != nil {
		return err
	}
	if err := validatePinnedProfile(g); err != nil {
		return err
	}
	tok, err := newTokenizer(g)
	if err != nil {
		return err
	}
	ids, err := preparePrompt(cfg, tok)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(*forced)
	if err != nil {
		return err
	}
	continuation, err := parsePromptTokens(data)
	if err != nil {
		return err
	}
	if len(ids) == 0 || len(continuation) == 0 || len(ids) > cfg.context-len(continuation) {
		return fmt.Errorf("evaluation: token history exceeds context or is empty")
	}
	for _, token := range append(append([]int(nil), ids...), continuation...) {
		if token < 0 || token >= len(tok.tokens) {
			return fmt.Errorf("evaluation: token %d outside vocabulary", token)
		}
	}
	modelSHA, err := frontierFileHash(cfg.model)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	binarySHA, err := frontierFileHash(executable)
	if err != nil {
		return err
	}
	host, err := os.Hostname()
	if err != nil {
		return err
	}
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return closeEvaluationBackend(err, closer)
	}
	var owner *stateSessionBackend
	switch b := b.(type) {
	case *stateSessionBackend:
		owner = b
	case *batchStateSessionBackend:
		owner = b.stateSessionBackend
	default:
		return finishBackendCleanup(fmt.Errorf("evaluation: independent backend unavailable"), closer)
	}
	if err := owner.state.Close(); err != nil {
		return finishBackendCleanup(err, closer)
	}
	owner.state = nil
	return produceEvaluation(ctx, owner.backend, closer, dsv4bench.EvaluationConfig{Prompt: ids, Continuation: continuation, Provenance: dsv4bench.Provenance{Revision: *revision, BinarySHA256: binarySHA, ModelSHA256: modelSHA, Host: host, Backend: cfg.backend, Context: cfg.context}}, *output, stdout, *maxAbs, *maxRMS)
}

type evaluationCLIReceipt struct {
	dsv4bench.EvaluationReceipt
	MaxAbs, MaxRMS     float64
	ConsistencyVerdict string
}

func produceEvaluation(ctx context.Context, backend dsv4.Backend, closer io.Closer, cfg dsv4bench.EvaluationConfig, path string, stdout io.Writer, maxAbs, maxRMS float64) error {
	if closer == nil || stdout == nil || path == "" || maxAbs < 0 || maxRMS < 0 || math.IsNaN(maxAbs) || math.IsNaN(maxRMS) || math.IsInf(maxAbs, 0) || math.IsInf(maxRMS, 0) {
		return closeEvaluationBackend(fmt.Errorf("evaluation: output and finite consistency bounds required"), closer)
	}
	r, err := dsv4bench.Evaluate(ctx, backend, cfg)
	err = finishBackendCleanup(err, closer)
	if err != nil {
		return err
	}
	verdict := "PASS"
	if r.Consistency.Top1Matches != r.Expected || r.Consistency.MaxAbs > maxAbs || r.Consistency.RMS > maxRMS {
		verdict = "FAIL"
	}
	data, err := json.Marshal(evaluationCLIReceipt{EvaluationReceipt: r, MaxAbs: maxAbs, MaxRMS: maxRMS, ConsistencyVerdict: verdict})
	if err != nil {
		return err
	}
	data = append(data, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data)
	err = finishBackendCleanup(writeErr, f)
	if err != nil {
		return errors.Join(err, os.Remove(path))
	}
	n, err := stdout.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	if err != nil {
		return errors.Join(err, os.Remove(path))
	}
	// A failed numerical gate retains its complete diagnostic measurement;
	// it is not an execution/cleanup failure or a promoted parity result.
	if verdict == "FAIL" {
		return fmt.Errorf("evaluation consistency gate failed: max_abs=%g rms=%g top1=%d/%d", r.Consistency.MaxAbs, r.Consistency.RMS, r.Consistency.Top1Matches, r.Expected)
	}
	return nil
}

func closeEvaluationBackend(err error, closer io.Closer) error {
	if closer == nil {
		return err
	}
	return finishBackendCleanup(err, closer)
}
