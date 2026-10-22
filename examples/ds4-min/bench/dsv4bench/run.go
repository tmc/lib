package dsv4bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// Provenance identifies the executable, weights, and environment of a run.
type Provenance struct {
	Revision, BinarySHA256, ModelSHA256, Host, Backend string
	Context                                            int
}

// RunConfig describes an incremental frontier sweep. Prompt contains already
// tokenized input. Replay resets and reconstructs prefixes instead of taking
// snapshots; reconstruction is outside the measured prefill interval.
type RunConfig struct {
	Prompt                          []int
	Start, Max, Step, Generate, EOS int
	Multiplier                      float64
	TeacherForced, Replay           bool
	// MeasureMemory requires observed owner storage and process RSS for CSV.
	MeasureMemory bool
	Provenance    Provenance
}

// Measurement contains timing and the full evaluated token histories. The
// histories include the frontier prefix and all evaluated decode tokens.
// Row memory fields are observed only when MemoryObserved is true. CSV export
// refuses measurements without observed owner storage and process RSS.
type Measurement struct {
	Row Row
	// Logits is the prefill frontier; FinalLogits follows History.
	Logits         []float32
	FinalLogits    []float32
	History        []int
	HistorySHA256  string
	MemoryObserved bool
}

// RunResult is a complete sweep; failed sweeps return no partial result.
type RunResult struct {
	Provenance   Provenance
	Measurements []Measurement
}

// Run evaluates a sweep on an independent State and closes it before return.
// Backend remains owned by the caller. Device backends must synchronize Eval
// and Logits before returning, so elapsed time includes completed work.
func Run(ctx context.Context, backend dsv4.Backend, cfg RunConfig) (result RunResult, err error) {
	if ctx == nil || backend == nil {
		return result, fmt.Errorf("run benchmark: missing context or backend")
	}
	if err = validateRun(cfg); err != nil {
		return result, err
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	state, err := backend.NewState(cfg.Provenance.Context)
	if err != nil {
		if state != nil {
			if closeErr := state.Close(); closeErr != nil {
				return result, &CleanupError{cause: errors.Join(err, closeErr), state: state}
			}
		}
		return result, fmt.Errorf("create benchmark state: %w", err)
	}
	if state == nil {
		return result, fmt.Errorf("create benchmark state: nil state")
	}
	cleanup := &CleanupError{state: state}
	defer func() {
		if closeErr := state.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close benchmark state: %w", closeErr))
		} else {
			cleanup.state = nil
		}
		if cleanup.state != nil || len(cleanup.snapshots) != 0 {
			cleanup.cause = err
			err = cleanup
		}
		if err != nil {
			result = RunResult{}
		}
	}()
	result.Provenance = cfg.Provenance
	previous := 0
	var boundaries []int
	for frontier := cfg.Start; ; {
		if err = ctx.Err(); err != nil {
			return result, err
		}
		begin := time.Now()
		if err = evalInterval(ctx, backend, state, cfg.Prompt[previous:frontier], previous); err != nil {
			return result, fmt.Errorf("prefill frontier %d: %w", frontier, err)
		}
		prefill := time.Since(begin)
		boundaries = append(boundaries, frontier)
		logits := append([]float32(nil), backend.Logits(state)...)
		if _, err = choose(logits, -1); err != nil {
			return result, err
		}
		var snapshot dsv4.Snapshot
		restore := cfg.Generate > 0 && frontier < cfg.Max
		if restore && !cfg.Replay {
			snapshot, err = state.Snapshot()
			if err != nil {
				if snapshot != nil {
					if closeErr := snapshot.Close(); closeErr != nil {
						cleanup.snapshots = append(cleanup.snapshots, snapshot)
						err = errors.Join(err, closeErr)
					}
				}
				return result, fmt.Errorf("snapshot frontier %d: %w", frontier, err)
			}
			if snapshot == nil {
				return result, fmt.Errorf("snapshot frontier %d: nil snapshot", frontier)
			}
		}
		measurement, decodeErr := decode(ctx, backend, state, cfg, frontier, logits, prefill, previous)
		if decodeErr == nil && restore {
			if snapshot != nil {
				decodeErr = state.Restore(snapshot)
			} else {
				decodeErr = state.Reset()
				if decodeErr == nil {
					position := 0
					for _, end := range boundaries {
						decodeErr = evalInterval(ctx, backend, state, cfg.Prompt[position:end], position)
						if decodeErr != nil {
							break
						}
						position = end
					}
				}
			}
		}
		if decodeErr == nil && cfg.MeasureMemory {
			decodeErr = observeMemory(state, snapshot, &measurement.Row)
			measurement.MemoryObserved = decodeErr == nil
		}
		if snapshot != nil {
			if closeErr := snapshot.Close(); closeErr != nil {
				cleanup.snapshots = append(cleanup.snapshots, snapshot)
				decodeErr = errors.Join(decodeErr, closeErr)
			}
		}
		if decodeErr != nil {
			return result, fmt.Errorf("decode or restore frontier %d: %w", frontier, decodeErr)
		}
		result.Measurements = append(result.Measurements, measurement)
		if frontier == cfg.Max {
			break
		}
		previous = frontier
		next := float64(frontier) + float64(cfg.Step)
		if cfg.Multiplier > 1 {
			next = math.Ceil(float64(frontier) * cfg.Multiplier)
		}
		if next >= float64(cfg.Max) {
			frontier = cfg.Max
		} else {
			frontier = int(next)
		}
	}
	return result, nil
}

func decode(ctx context.Context, backend dsv4.Backend, state dsv4.State, cfg RunConfig, frontier int, logits []float32, prefill time.Duration, previous int) (Measurement, error) {
	m := Measurement{Logits: logits, History: append([]int(nil), cfg.Prompt[:frontier]...)}
	row := Row{CtxTokens: int64(frontier), PrefillTokens: int64(frontier - previous), PrefillTPS: rate(frontier-previous, prefill)}
	begin := time.Now()
	var first, steady time.Duration
	for n := 0; n < cfg.Generate; n++ {
		if err := ctx.Err(); err != nil {
			return m, err
		}
		token, err := choose(backend.Logits(state), cfg.EOS)
		if err != nil {
			return m, err
		}
		if cfg.TeacherForced {
			token, err = cfg.Prompt[frontier+n], nil
		}
		if err != nil {
			return m, err
		}
		start := time.Now()
		if err := backend.Eval(state, token, frontier+n); err != nil {
			return m, err
		}
		elapsed := time.Since(start)
		m.History = append(m.History, token)
		if n == 0 {
			first = elapsed
		} else {
			steady += elapsed
		}
		// Upstream evaluates a forced EOS but excludes it from gen_done.
		if token == cfg.EOS {
			break
		}
		row.GenTokens++
	}
	row.GenTPS = rate(int(row.GenTokens), time.Since(begin))
	row.GenFirstMS = float64(first) / float64(time.Millisecond)
	if row.GenTokens > 0 {
		row.GenSteadyTokens = row.GenTokens - 1
	}
	row.GenSteadyTPS = rate(int(row.GenSteadyTokens), steady)
	m.Row = row
	m.FinalLogits = append([]float32(nil), backend.Logits(state)...)
	if _, err := choose(m.FinalLogits, -1); err != nil {
		return m, err
	}
	m.HistorySHA256 = historyHash(m.History)
	return m, nil
}

func evalInterval(ctx context.Context, b dsv4.Backend, s dsv4.State, tokens []int, pos int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if batch, ok := b.(dsv4.BatchEvaluator); ok {
		if err := batch.EvalBatch(s, tokens, pos); err != nil {
			return err
		}
		return ctx.Err()
	}
	for i, token := range tokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := b.Eval(s, token, pos+i); err != nil {
			return err
		}
	}
	return ctx.Err()
}

func choose(logits []float32, excluded int) (int, error) {
	best := -1
	for i, v := range logits {
		if !finite(float64(v)) {
			return -1, fmt.Errorf("benchmark logits: non-finite value at %d", i)
		}
		if i != excluded && (best < 0 || v > logits[best]) {
			best = i
		}
	}
	if best < 0 {
		return -1, fmt.Errorf("benchmark logits: no eligible token")
	}
	return best, nil
}
func rate(tokens int, d time.Duration) float64 {
	if tokens == 0 || d <= 0 {
		return 0
	}
	return float64(tokens) / d.Seconds()
}
func historyHash(tokens []int) string {
	h := sha256.New()
	var word [4]byte
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(word[:], uint32(token))
		h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
func validateRun(c RunConfig) error {
	p := c.Provenance
	if p.Revision == "" || p.Host == "" || p.Backend == "" {
		return fmt.Errorf("benchmark provenance: revision, host and backend required")
	}
	for _, sum := range []string{p.BinarySHA256, p.ModelSHA256} {
		raw, err := hex.DecodeString(sum)
		if err != nil || len(raw) != sha256.Size {
			return fmt.Errorf("benchmark provenance: invalid sha256")
		}
	}
	if c.Start <= 0 || c.Max < c.Start || c.Step <= 0 || c.Generate < 0 || c.EOS < 0 || !finite(c.Multiplier) || c.Multiplier < 1 {
		return fmt.Errorf("benchmark sweep: invalid bounds")
	}
	if c.Max >= p.Context || c.Generate >= p.Context-c.Max {
		return fmt.Errorf("benchmark sweep: allocated context too small")
	}
	needed := c.Max
	if c.TeacherForced {
		needed += c.Generate
	}
	if len(c.Prompt) < needed {
		return fmt.Errorf("benchmark sweep: prompt has %d tokens, need %d", len(c.Prompt), needed)
	}
	for _, token := range c.Prompt {
		if token < 0 || int64(token) > math.MaxInt32 {
			return fmt.Errorf("benchmark sweep: prompt token outside int32 range")
		}
	}
	return nil
}
