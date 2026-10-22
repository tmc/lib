package main

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type dsparkSupportConfig struct {
	path, sha256 string
	confidence   float64
}

type dsparkSupportFile interface {
	io.ReaderAt
	io.Closer
	Stat() (os.FileInfo, error)
}

func dsparkGenerationLimit(context, history, limit int) (int, error) {
	if history < 0 || history >= context || limit < 1 {
		return 0, errContextOverflow
	}
	return min(limit, context-history), nil
}

func (c dsparkSupportConfig) validate(cfg config) error {
	if c.path == "" {
		if c.sha256 != "" {
			return errors.New("dspark-sha256 requires dspark")
		}
		return nil
	}
	b, err := hex.DecodeString(c.sha256)
	if err != nil || len(b) != 32 || c.sha256 != strings.ToLower(c.sha256) {
		return errors.New("dspark requires a lowercase full support sha256")
	}
	if math.IsNaN(c.confidence) || math.IsInf(c.confidence, 0) || c.confidence < 0 || c.confidence > 1 {
		return errors.New("dspark-confidence must be in [0,1]")
	}
	if (cfg.backend != "metal" && cfg.backend != "cuda") || cfg.sampling.Temperature != 0 {
		return errors.New("dspark currently requires a native backend and temp 0")
	}
	if cfg.streaming.enabled || cfg.streaming.cold || cfg.streaming.autoCache || cfg.streaming.budget != "" || cfg.streaming.preload != 0 {
		return errors.New("dspark does not support ssd streaming or its modifiers")
	}
	if cfg.interactive || cfg.perplexityFile != "" || cfg.decodeConsistency != 0 || cfg.evalSuite != "" || cfg.collectors.dataset != "" || cfg.collectors.imatrix != "" || cfg.kvLoad != "" || cfg.kvSave != "" || cfg.dumpLogprobs != "" || cfg.decodeLogitsDir != "" || cfg.sampledTraceDir != "" {
		return errors.New("dspark does not support the selected persistence, evaluation or per-token observation mode")
	}
	return nil
}

// finishDSparkOpen releases failed acquisitions before the support file.
// A successful open closes the support file once; a failed Close retains
// that exact owner for retry after the constructed resources are released.
func finishDSparkOpen(err error, resources, file io.Closer) error {
	if err != nil {
		if resources != nil {
			err = finishBackendCleanup(err, resources)
		}
		if file != nil {
			err = finishBackendCleanup(err, file)
		}
		return err
	}
	if file == nil {
		return nil
	}
	if closeErr := file.Close(); closeErr != nil {
		err = closeErr
		if resources != nil {
			err = finishBackendCleanup(err, resources)
		}
		var pending io.Closer
		if owners := backendErrorOwners(err); len(owners) > 0 {
			pending = &backendCleanupGroup{owners}
		}
		return &backendCleanupOwner{cause: err, request: pending, backend: file}
	}
	return nil
}

// speculativeSessionBackend returns tokens already evaluated by the target.
// The stop predicate excludes the first stop and every following proposal
// from the published target history.
type speculativeSessionBackend interface {
	sessionBackend
	Generate(history []int, limit int, stop func(int) bool) ([]int, bool, error)
}

func runDSparkAcceptedPrefix(state dsv4.State, run func() ([]int, error), eval func(int, int) error, history []int, limit int, stop func(int) bool) (ids []int, stopped bool, err error) {
	if state == nil || run == nil || eval == nil || stop == nil || limit < 1 {
		return nil, false, errors.New("dspark generation: invalid request")
	}
	frontier, err := state.Snapshot()
	if err != nil {
		return nil, false, err
	}
	if frontier == nil {
		return nil, false, errors.New("dspark generation: missing frontier snapshot")
	}
	defer func() {
		if closeErr := frontier.Close(); closeErr != nil {
			err = &backendCleanupOwner{cause: errors.Join(err, closeErr), request: frontier}
		}
	}()
	ids, err = run()
	if err == nil && (len(ids) == 0 || len(ids) > limit) {
		err = errors.New("dspark generation: invalid accepted count")
	}
	if err == nil {
		for i, id := range ids {
			if id < 0 || id >= Flash.Vocab {
				err = fmt.Errorf("dspark generation: invalid accepted token %d", id)
				break
			}
			if stop(id) {
				ids, stopped = ids[:i], true
				break
			}
		}
	}
	if err != nil || stopped {
		if restoreErr := state.Restore(frontier); restoreErr != nil {
			return nil, false, errors.Join(err, restoreErr, dsv4.ErrDesynchronized)
		}
		if err != nil {
			return nil, false, err
		}
		for i, id := range ids {
			if replayErr := eval(id, len(history)+i); replayErr != nil {
				if restoreErr := state.Restore(frontier); restoreErr != nil {
					return nil, false, errors.Join(replayErr, restoreErr, dsv4.ErrDesynchronized)
				}
				return nil, false, replayErr
			}
		}
	}
	return append([]int(nil), ids...), stopped, nil
}
