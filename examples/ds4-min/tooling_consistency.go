package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

// checkFinalDecodeConsistency mirrors ds4_cli.c at 11e8cad: evaluate greedy
// tokens live (including a selected stop), then sync the complete prefix on
// one fresh State belonging to the same immutable backend.
func checkFinalDecodeConsistency(live *session, prompt []int, cfg config, stdout io.Writer) error {
	if live == nil || stdout == nil || cfg.decodeConsistency < 0 || cfg.context != live.context {
		return fmt.Errorf("decode consistency: invalid active prompt or context")
	}
	var owner *stateSessionBackend
	switch b := live.backend.(type) {
	case *stateSessionBackend:
		owner = b
	case *batchStateSessionBackend:
		owner = b.stateSessionBackend
	default:
		return fmt.Errorf("decode consistency: independent request State unavailable")
	}
	if err := live.Sync(prompt); err != nil {
		return fmt.Errorf("decode consistency live prefix: %w", err)
	}
	prefix := append([]int(nil), prompt...)
	for i := 0; i < cfg.decodeConsistency; i++ {
		if _, err := dsv4eval.LogProb(live.logits, 0); err != nil {
			return err
		}
		id := sampler.Argmax(live.logits)
		if err := advanceSession(live, id); err != nil {
			return fmt.Errorf("decode consistency live step %d: %w", i, err)
		}
		prefix = append(prefix, id)
		if live.stop.Contains(id) {
			break
		}
	}
	liveLogits := append([]float32(nil), live.logits...)
	state, err := owner.backend.NewState(cfg.context)
	if err != nil {
		if state != nil {
			return finishBackendCleanup(err, state)
		}
		return err
	}
	if state == nil {
		return fmt.Errorf("decode consistency: nil fresh State")
	}
	if reflect.TypeOf(state) == reflect.TypeOf(owner.state) && reflect.TypeOf(state).Comparable() && state == owner.state {
		return fmt.Errorf("decode consistency: backend reused live State")
	}
	adapter := &stateSessionBackend{backend: owner.backend, state: state}
	var b sessionBackend = adapter
	if batch, ok := owner.backend.(dsv4.BatchEvaluator); ok {
		b = &batchStateSessionBackend{stateSessionBackend: adapter, batch: batch}
	}
	fresh, err := newSession(b, cfg.context, live.stop)
	if err == nil {
		err = configurePrefillChunk(b, cfg.prefillChunk)
	}
	if err == nil {
		err = fresh.Sync(prefix)
	}
	var freshLogits []float32
	if err == nil {
		freshLogits = append([]float32(nil), fresh.logits...)
	}
	// Only this child is released; the caller still owns the foundation.
	err = finishBackendCleanup(err, state)
	if err != nil {
		return err
	}
	c, err := dsv4eval.Compare([][]float32{liveLogits}, [][]float32{freshLogits})
	if err != nil {
		return err
	}
	var out bytes.Buffer
	if err = writeConsistency(&out, c); err != nil {
		return err
	}
	for _, row := range []struct {
		name   string
		logits []float32
	}{{"live_top", liveLogits}, {"fresh_top", freshLogits}} {
		ids := make([]int, len(row.logits))
		for i := range ids {
			ids[i] = i
		}
		sort.Slice(ids, func(i, j int) bool {
			if row.logits[ids[i]] == row.logits[ids[j]] {
				return ids[i] < ids[j]
			}
			return row.logits[ids[i]] > row.logits[ids[j]]
		})
		if len(ids) > 10 {
			ids = ids[:10]
		}
		for _, id := range ids {
			p, e := dsv4eval.LogProb(row.logits, id)
			if e != nil {
				return e
			}
			fmt.Fprintf(&out, "%s id=%d logit=%.9g logprob=%.9g\n", row.name, id, row.logits[id], p)
		}
	}
	diagnostics := cfg.log
	if diagnostics == nil {
		diagnostics = stdout
	}
	n, err := diagnostics.Write(out.Bytes())
	if err == nil && n != out.Len() {
		err = io.ErrShortWrite
	}
	if err != nil {
		return err
	}
	if c.Top1Matches != 1 {
		return errors.New("decode consistency failed: final top1 differs")
	}
	return nil
}
