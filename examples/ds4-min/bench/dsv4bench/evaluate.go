package dsv4bench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
)

// EvaluationConfig pins an explicit prompt and teacher-forced continuation.
// Consistency compares every resulting frontier with a fresh independent State.
type EvaluationConfig struct {
	Prompt, Continuation []int
	Provenance           Provenance
}

// EvaluationStep records the distribution predicting Token, before Eval,
// and the resulting frontier. History hashes use int32 little-endian IDs.
type EvaluationStep struct {
	Position, Token                    int
	HistorySHA256, ResultHistorySHA256 string
	Logits, ResultLogits               []float32
}

// EvaluationReceipt is emitted only after all steps and cleanup complete.
// Complete means operational completeness, not numerical parity with C.
type EvaluationReceipt struct {
	Provenance                       Provenance
	PromptSHA256, ContinuationSHA256 string
	Prompt, Continuation             []int
	Steps                            []EvaluationStep
	Perplexity                       dsv4eval.Perplexity
	Consistency                      dsv4eval.Consistency
	Expected, Ran, Skipped           int
	Complete                         bool
}

// Evaluate produces actual State-based teacher-forced logits, perplexity and
// live-versus-fresh-prefix consistency. It never invokes a model subprocess.
// Backend remains caller-owned; every created State is closed before return.
// A failed or incomplete run returns no receipt. Numeric disagreement remains
// in Consistency for a caller's pre-registered comparison, not hidden by Complete.
func Evaluate(ctx context.Context, backend dsv4.Backend, cfg EvaluationConfig) (receipt EvaluationReceipt, err error) {
	if ctx == nil || backend == nil {
		return receipt, fmt.Errorf("evaluation: missing context or backend")
	}
	c := RunConfig{Prompt: append(append([]int(nil), cfg.Prompt...), cfg.Continuation...), Start: len(cfg.Prompt), Max: len(cfg.Prompt), Step: 1, Generate: len(cfg.Continuation), EOS: 0, Multiplier: 1, TeacherForced: true, Provenance: cfg.Provenance}
	if len(cfg.Continuation) == 0 {
		return receipt, fmt.Errorf("evaluation: empty continuation")
	}
	if err := validateRun(c); err != nil {
		return receipt, err
	}
	live, err := backend.NewState(cfg.Provenance.Context)
	if live != nil {
		defer func() {
			err = closeEvaluationState(err, live)
			if err != nil {
				receipt = EvaluationReceipt{}
			}
		}()
	}
	if err != nil {
		return receipt, fmt.Errorf("evaluation create live state: %w", err)
	}
	if live == nil {
		return receipt, fmt.Errorf("evaluation: nil live state")
	}
	if err = evalInterval(ctx, backend, live, cfg.Prompt, 0); err != nil {
		return receipt, err
	}
	receipt = EvaluationReceipt{Provenance: cfg.Provenance, Prompt: append([]int(nil), cfg.Prompt...), Continuation: append([]int(nil), cfg.Continuation...), PromptSHA256: historyHash(cfg.Prompt), ContinuationSHA256: historyHash(cfg.Continuation), Expected: len(cfg.Continuation)}
	history := append([]int(nil), cfg.Prompt...)
	var scoreRows, liveRows, freshRows [][]float32
	for step, token := range cfg.Continuation {
		if err = ctx.Err(); err != nil {
			return receipt, err
		}
		before := append([]float32(nil), backend.Logits(live)...)
		if _, err = choose(before, -1); err != nil {
			return receipt, err
		}
		if token >= len(before) {
			return receipt, fmt.Errorf("evaluation step %d: token outside vocabulary", step)
		}
		row := EvaluationStep{Position: len(history), Token: token, HistorySHA256: historyHash(history), Logits: before}
		if err = backend.Eval(live, token, len(history)); err != nil {
			return receipt, fmt.Errorf("evaluation step %d: %w", step, err)
		}
		history = append(history, token)
		row.ResultLogits = append([]float32(nil), backend.Logits(live)...)
		if _, err = choose(row.ResultLogits, -1); err != nil {
			return receipt, err
		}
		if len(row.ResultLogits) != len(before) {
			return receipt, fmt.Errorf("evaluation step %d: vocabulary changed", step)
		}
		row.ResultHistorySHA256 = historyHash(history)
		fresh, createErr := backend.NewState(cfg.Provenance.Context)
		if createErr != nil {
			if fresh != nil {
				createErr = closeEvaluationState(createErr, fresh)
			}
			return receipt, createErr
		}
		if fresh == nil {
			return receipt, fmt.Errorf("evaluation: nil fresh state")
		}
		if reflect.TypeOf(fresh) == reflect.TypeOf(live) && reflect.TypeOf(live).Comparable() && fresh == live {
			return receipt, fmt.Errorf("evaluation: backend reused live State")
		}
		freshErr := evalInterval(ctx, backend, fresh, history, 0)
		var values []float32
		if freshErr == nil {
			values = append([]float32(nil), backend.Logits(fresh)...)
		}
		freshErr = closeEvaluationState(freshErr, fresh)
		if freshErr != nil {
			return receipt, freshErr
		}
		if len(values) != len(row.ResultLogits) {
			return receipt, fmt.Errorf("evaluation: fresh vocabulary changed")
		}
		if _, err = choose(values, -1); err != nil {
			return receipt, err
		}
		scoreRows = append(scoreRows, before)
		liveRows = append(liveRows, row.ResultLogits)
		freshRows = append(freshRows, values)
		receipt.Steps = append(receipt.Steps, row)
		receipt.Ran++
	}
	receipt.Perplexity, err = dsv4eval.ScorePerplexity(scoreRows, cfg.Continuation)
	if err != nil {
		return receipt, err
	}
	receipt.Consistency, err = dsv4eval.Compare(liveRows, freshRows)
	if err != nil {
		return receipt, err
	}
	if receipt.Perplexity.Scored != receipt.Expected || receipt.Consistency.Positions != receipt.Expected || math.IsInf(receipt.Perplexity.PPL, 0) {
		return receipt, fmt.Errorf("evaluation: incomplete or nonfinite summary")
	}
	receipt.Complete = true
	return receipt, nil
}

// evaluationCleanupError transfers a failed child cleanup to the caller.
// The immutable backend must stay alive until this owner closes successfully.
type evaluationCleanupError struct {
	cause   error
	state   dsv4.State
	pending []io.Closer
}

func (e *evaluationCleanupError) Error() string { return e.cause.Error() }
func (e *evaluationCleanupError) Unwrap() error { return e.cause }
func (e *evaluationCleanupError) Close() error {
	for len(e.pending) > 0 {
		if err := e.pending[0].Close(); err != nil {
			return err
		}
		e.pending = e.pending[1:]
	}
	if e.state == nil {
		return nil
	}
	if err := e.state.Close(); err != nil {
		return err
	}
	e.state = nil
	return nil
}

func closeEvaluationState(err error, state dsv4.State) error {
	if closeErr := state.Close(); closeErr != nil {
		return &evaluationCleanupError{cause: errors.Join(err, fmt.Errorf("close evaluation state: %w", closeErr)), state: state, pending: evaluationErrorOwners(err)}
	}
	return err
}

func evaluationErrorOwners(err error) []io.Closer {
	if owner, ok := err.(io.Closer); ok {
		return []io.Closer{owner}
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		var owners []io.Closer
		for _, e := range joined.Unwrap() {
			owners = append(owners, evaluationErrorOwners(e)...)
		}
		return owners
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return evaluationErrorOwners(wrapped.Unwrap())
	}
	return nil
}
