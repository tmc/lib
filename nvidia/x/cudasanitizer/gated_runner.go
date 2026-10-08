package cudasanitizer

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sync"
)

// GatedTrial defines one execution trial to be evaluated by GatedRunner.
type GatedTrial struct {
	// Name identifies the trial.
	Name string

	// Prog and AppArgs define the program executed under compute-sanitizer
	// when gating criteria pass.
	Prog    string
	AppArgs []string

	// FastRun executes the un-instrumented trial and returns its raw edge coverage trace.
	// Either FastRun or Trace must be non-empty/non-nil.
	FastRun func(ctx context.Context) ([]byte, error)

	// Trace is a precomputed raw coverage trace. Used when FastRun is nil.
	Trace []byte

	// SanitizeRun optionally overrides default compute-sanitizer execution (e.g. for testing).
	// If nil, [Run] is invoked with the runner's Config, Prog, and AppArgs.
	SanitizeRun func(ctx context.Context) (*Result, error)
}

// GatedRunResult records the outcome of a gated trial execution.
type GatedRunResult struct {
	// TrialName is the name of the evaluated trial.
	TrialName string

	// Novel reports whether the execution trace was admitted by the gating policy.
	Novel bool

	// SanitizerDispatched reports whether compute-sanitizer was launched.
	SanitizerDispatched bool

	// TraceDigest is the SHA-256 hash of the evaluated raw trace.
	TraceDigest [32]byte

	// Result is the compute-sanitizer result if dispatched, or nil if bypassed.
	Result *Result

	// FastErr is any error returned by FastRun.
	FastErr error
}

// GatedRunner decouples heavy sanitizer checks from the execution loop by
// evaluating un-instrumented trace novelty and only triggering compute-sanitizer
// when gating criteria are satisfied.
type GatedRunner struct {
	mu      sync.Mutex
	config  Config
	novelty *TraceNovelty
}

// NewGatedRunner constructs a GatedRunner with the given config, map size, and max traces limit.
// If config.Gating is unset, it defaults to GatingCoverageIncrease.
func NewGatedRunner(cfg Config, mapSize, maxTraces int) (*GatedRunner, error) {
	if cfg.Gating == 0 {
		cfg.Gating = GatingCoverageIncrease
	}
	novelty, err := NewTraceNovelty(cfg.Gating, mapSize, maxTraces)
	if err != nil {
		return nil, fmt.Errorf("gated runner: %w", err)
	}
	return &GatedRunner{
		config:  cfg,
		novelty: novelty,
	}, nil
}

// RunTrial executes trial on the fast path, evaluates trace novelty, and conditionally
// dispatches compute-sanitizer if novel.
func (r *GatedRunner) RunTrial(ctx context.Context, trial GatedTrial) (*GatedRunResult, error) {
	if r == nil || r.novelty == nil {
		return nil, fmt.Errorf("gated runner: uninitialized runner")
	}

	var raw []byte
	if trial.FastRun != nil {
		trace, err := trial.FastRun(ctx)
		if err != nil {
			return &GatedRunResult{
				TrialName: trial.Name,
				FastErr:   err,
			}, err
		}
		raw = trace
	} else if len(trial.Trace) > 0 {
		raw = trial.Trace
	} else {
		return nil, fmt.Errorf("gated runner: trial has neither FastRun nor Trace")
	}

	novel, err := r.novelty.Evaluate(raw)
	if err != nil {
		return nil, fmt.Errorf("gated runner: trace evaluation failed: %w", err)
	}

	res := &GatedRunResult{
		TrialName:   trial.Name,
		Novel:       novel,
		TraceDigest: sha256.Sum256(raw),
	}

	if !novel {
		// Bypassed!
		return res, nil
	}

	// Novel trace: dispatch sanitizer!
	res.SanitizerDispatched = true
	if trial.SanitizeRun != nil {
		sanRes, err := trial.SanitizeRun(ctx)
		res.Result = sanRes
		return res, err
	}

	sanRes, err := Run(ctx, r.config, trial.Prog, trial.AppArgs...)
	res.Result = sanRes
	return res, err
}

// Config returns the runner's configuration.
func (r *GatedRunner) Config() Config {
	if r == nil {
		return Config{}
	}
	return r.config
}

// Novelty returns the runner's TraceNovelty filter.
func (r *GatedRunner) Novelty() *TraceNovelty {
	if r == nil {
		return nil
	}
	return r.novelty
}
