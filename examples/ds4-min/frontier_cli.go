package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strconv"

	"github.com/tmc/lib/examples/ds4-min/bench/dsv4bench"
	"github.com/tmc/lib/gguf"
)

// runFrontierBenchmark is routed by the -frontier-bench subcommand. Token IDs
// are explicit so teacher-forced histories can be reproduced independently.
func runFrontierBenchmark(ctx context.Context, args []string, stdout, stderr io.Writer) (err error) {
	if len(args) > 0 && args[0] == "-evaluate" {
		return runFrontierEvaluation(ctx, args[1:], stdout, stderr)
	}
	fs := flag.NewFlagSet("frontier-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "GGUF weights")
	backend := fs.String("backend", "auto", "execution backend")
	prompt := fs.String("prompt-tokens", "", "JSON prompt token IDs")
	revision := fs.String("revision", "", "candidate source revision")
	receipt := fs.String("receipt", "", "complete JSON measurement receipt (required)")
	start := fs.Int("ctx-start", 2048, "first frontier")
	maximum := fs.Int("ctx-max", 32768, "last frontier")
	alloc := fs.Int("ctx-alloc", 0, "allocated context (default max+gen+1)")
	step := fs.Int("step-incr", 2048, "additive frontier step")
	mul := fs.Float64("step-mul", 1, "multiplicative frontier step")
	gen := fs.Int("gen-tokens", 128, "decode budget")
	forced := fs.Bool("teacher-forced-decode", false, "evaluate prompt continuation")
	replay := fs.Bool("replay", false, "reset/replay instead of snapshots")
	csv := fs.Bool("csv", false, "emit fully measured upstream CSV (requires memory metadata capability)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *revision == "" || *receipt == "" || *model == "" || *prompt == "" {
		return fmt.Errorf("frontier-bench: model, prompt-tokens, revision and receipt required; positional arguments forbidden")
	}
	if err := frontierBuildRevision(*revision); err != nil {
		return err
	}
	if *alloc == 0 {
		*alloc = *maximum + *gen + 1
	}
	cfg, err := parseConfig([]string{"-model", *model, "-backend", *backend, "-prompt-tokens", *prompt, "-ctx", strconv.Itoa(*alloc), "-tokens", "0"}, nil, stderr)
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
	tokens, err := preparePrompt(cfg, tok)
	if err != nil {
		return err
	}
	modelHash, err := frontierFileHash(cfg.model)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	binaryHash, err := frontierFileHash(executable)
	if err != nil {
		return err
	}
	host, err := os.Hostname()
	if err != nil {
		return err
	}
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return err
	}
	defer func() {
		if closer != nil {
			err = finishBackendCleanup(err, closer)
		}
	}()
	var owner *stateSessionBackend
	switch b := b.(type) {
	case *stateSessionBackend:
		owner = b
	case *batchStateSessionBackend:
		owner = b.stateSessionBackend
	default:
		return fmt.Errorf("frontier-bench: backend does not expose independent states")
	}
	// The factory's unused initial request is closed before Run creates its own.
	initialState := owner.state
	owner.state = nil
	if err := initialState.Close(); err != nil {
		return &backendCleanupOwner{cause: err, request: initialState}
	}
	result, err := dsv4bench.Run(ctx, owner.backend, dsv4bench.RunConfig{Prompt: tokens, Start: *start, Max: *maximum, Step: *step, Generate: *gen, EOS: tok.eos, Multiplier: *mul, TeacherForced: *forced, Replay: *replay, MeasureMemory: *csv, Provenance: dsv4bench.Provenance{Revision: *revision, BinarySHA256: binaryHash, ModelSHA256: modelHash, Host: host, Backend: cfg.backend, Context: cfg.context}})
	if err != nil {
		return err
	}
	// Publish only after the benchmark State and immutable backend release.
	err = finishBackendCleanup(nil, closer)
	closer = nil
	if err != nil {
		return err
	}
	output, err := os.OpenFile(*receipt, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("create benchmark receipt: %w", err)
	}
	capture := frontierCLIReceipt{RunResult: result, MemoryAvailability: frontierMemoryAvailability()}
	if *csv {
		for field := range capture.MemoryAvailability {
			capture.MemoryAvailability[field] = "observed: live request backing storage, serialized snapshot payload and process peak RSS"
		}
	}
	err = json.NewEncoder(output).Encode(capture)
	err = errors.Join(err, output.Close())
	if err != nil {
		return err
	}
	if *csv {
		return dsv4bench.WriteMeasuredCSV(stdout, result)
	}
	return json.NewEncoder(stdout).Encode(capture)
}

type frontierCLIReceipt struct {
	dsv4bench.RunResult
	MemoryAvailability map[string]string
}

func frontierMemoryAvailability() map[string]string {
	result := make(map[string]string)
	for _, field := range []string{"kvcache_bytes", "context_total_bytes", "context_raw_bytes", "context_compressed_bytes", "context_scratch_bytes", "peak_rss_bytes"} {
		result[field] = "UNREAD: backend does not expose measured metadata; numeric placeholders are not measurements"
	}
	return result
}

func frontierFileHash(name string) (string, error) {
	f, err := os.Open(name)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	_, err = io.Copy(h, f)
	err = errors.Join(err, f.Close())
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func frontierBuildRevision(want string) error {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return fmt.Errorf("frontier-bench: missing executable build metadata")
	}
	var revision, modified string
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value
		}
	}
	if revision == "" || modified != "false" || revision != want {
		return fmt.Errorf("frontier-bench: revision must match clean executable VCS metadata")
	}
	return nil
}
