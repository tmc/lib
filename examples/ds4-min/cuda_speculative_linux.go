//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// cudaSpeculativeVerifier implements the rollback speculative contract. The
// default evaluation route is scalar; the opt-in grouped route reads all row
// logits from one transformer batch. The seed is already represented by the request's
// current logits; it evaluates drafts in order, retaining one
// device snapshot so CommitPrefix can discard rejected work and replay only
// the accepted prefix. Grouped verification must pass its independent
// state/argmax gate before it can replace the default path.
type cudaSpeculativeVerifier struct {
	adapter *cudaStateAdapter
	state   *cudaRequestState

	frontier      dsv4.Snapshot
	seed          int
	drafts        []int
	rows          [][]float32
	rowGeneration uint64
	rowEpoch      uint64
	pos           int
	batched       bool
	failed        error
	closed        bool
}

var _ dsv4.SpeculativeVerifier = (*cudaSpeculativeVerifier)(nil)

func (v *cudaSpeculativeVerifier) CurrentLogits() []float32 {
	if v == nil || v.closed || v.failed != nil || v.adapter == nil || v.adapter.closed || v.state == nil || v.state.closed {
		return nil
	}
	return append([]float32(nil), v.adapter.Logits(v.state)...)
}

// BoundTo reports the exact target State and backend ownership, not a matching
// position or an equivalent model. Closed owners cannot supply sampled rows.
func (v *cudaSpeculativeVerifier) BoundTo(state dsv4.State) bool {
	return v != nil && !v.closed && v.failed == nil && v.adapter != nil && !v.adapter.closed && v.state != nil && !v.state.closed && v.state.owner == v.adapter && v.state == state
}

// RowLogits returns detached target logits after draft i, while its verification
// transaction remains live. Support logits and argmax IDs are not substitutes.
func (v *cudaSpeculativeVerifier) RowLogits(i int) ([]float32, error) {
	if v == nil || v.closed || v.state == nil || v.state.closed {
		return nil, dsv4.ErrClosed
	}
	if v.failed != nil {
		return nil, v.failed
	}
	if err := v.validateRowFrontier(); err != nil {
		return nil, err
	}
	if i < 0 || i >= len(v.rows) {
		return nil, fmt.Errorf("cuda speculative verifier: unavailable target row")
	}
	return append([]float32(nil), v.rows[i]...), nil
}

// validateRowFrontier rejects foreign state changes before rollback can replace
// them. The monotonic epoch also detects a Restore to the same visible position.
func (v *cudaSpeculativeVerifier) validateRowFrontier() error {
	if !v.BoundTo(v.state) {
		return dsv4.ErrClosed
	}
	epoch, err := v.state.speculativeEpoch()
	if err != nil {
		return err
	}
	if v.frontier == nil || epoch != v.rowEpoch || v.state.runtime == nil || v.state.generation != v.rowGeneration || v.state.runtime.pos != v.pos+len(v.drafts) || len(v.rows) != len(v.drafts) {
		return fmt.Errorf("cuda speculative verifier: unavailable target frontier")
	}
	return nil
}

func (v *cudaSpeculativeVerifier) Snapshot() (dsv4.Snapshot, error) {
	if v == nil || v.state == nil {
		return nil, dsv4.ErrClosed
	}
	if v.failed != nil {
		return nil, v.failed
	}
	if v.closed {
		return nil, dsv4.ErrClosed
	}
	return v.state.Snapshot()
}

func (v *cudaSpeculativeVerifier) Restore(snapshot dsv4.Snapshot) error {
	if v == nil || v.state == nil {
		return dsv4.ErrClosed
	}
	if v.failed != nil {
		return v.failed
	}
	if v.closed {
		return dsv4.ErrClosed
	}
	v.rows = nil
	return v.state.Restore(snapshot)
}

func (v *cudaSpeculativeVerifier) VerifyBatch(seed int, drafts []int, pos int) ([]int, error) {
	if v == nil || v.closed || v.adapter == nil || v.state == nil || v.state.closed {
		return nil, dsv4.ErrClosed
	}
	if v.failed != nil {
		return nil, v.failed
	}
	if v.batched && len(drafts) > 1 && v.state.runtime != nil {
		if err := admitGroupedPower(v.state.runtime.power); err != nil {
			return nil, err
		}
	}
	if err := validateCUDASpeculativeBatch(seed, drafts, pos); err != nil {
		return nil, err
	}
	vocab, err := cudaSpeculativeVocabSize(v.state)
	if err != nil {
		return nil, err
	}
	if err := validateCUDASpeculativeBatchVocab(seed, drafts, vocab); err != nil {
		return nil, err
	}
	if v.frontier != nil {
		if err := v.closeFrontier(nil); err != nil {
			return nil, err
		}
	}
	frontier, err := v.state.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("cuda speculative verifier snapshot: %w", err)
	}
	v.frontier = frontier
	v.seed, v.pos = seed, pos
	v.drafts = append(v.drafts[:0], drafts...)
	if v.batched && len(drafts) > 1 {
		return v.verifyGrouped(drafts, pos)
	}
	tops := make([]int, len(drafts))
	rows := make([][]float32, len(drafts))
	for i, draft := range drafts {
		logits := v.CurrentLogits()
		if len(logits) == 0 {
			return nil, v.abort(fmt.Errorf("cuda speculative verifier: empty logits at draft %d", i))
		}
		tops[i] = cudaSpeculativeArgmax(logits)
		if err := v.adapter.Eval(v.state, draft, pos+i); err != nil {
			return nil, v.abort(err)
		}
		rows[i] = v.CurrentLogits()
		if len(rows[i]) == 0 {
			return nil, v.abort(fmt.Errorf("cuda speculative verifier: empty output row %d", i))
		}
		if err := checkFiniteLogits("cuda speculative row", rows[i]); err != nil {
			return nil, v.abort(err)
		}
	}
	if err := v.retainRows(rows); err != nil {
		return nil, v.abort(err)
	}
	return tops, nil
}

// verifyGrouped consumes drafts once through the transformer batch path. The
// frontier already includes seed, so top[0] comes from its current logits and
// top[i] comes from the output of draft[i-1], not draft[i]. CommitPrefix still
// restores and replays the accepted prefix through the admitted scalar route.
func (v *cudaSpeculativeVerifier) verifyGrouped(drafts []int, pos int) ([]int, error) {
	r := v.state.runtime
	if pos != r.pos || pos < 0 || pos > r.ctx || len(drafts) > r.ctx-pos {
		return nil, v.abort(fmt.Errorf("cuda speculative verifier: batch exceeds timeline"))
	}
	current := v.CurrentLogits()
	if len(current) == 0 {
		return nil, v.abort(fmt.Errorf("cuda speculative verifier: empty frontier logits"))
	}
	if err := checkFiniteLogits("cuda speculative frontier", current); err != nil {
		return nil, v.abort(err)
	}
	first := cudaSpeculativeArgmax(current)
	f := v.state.dspark
	if f != nil {
		if err := f.beginGrouped(drafts, pos); err != nil {
			return nil, v.abort(err)
		}
		r.dspark = f
	}
	fail := func(err error) ([]int, error) {
		r.dspark = nil
		if f != nil {
			f.invalidate()
		}
		return nil, v.abort(err)
	}
	var rows [][]float32
	if err := r.evalBatchReadout(drafts, pos, true, &rows); err != nil {
		return fail(fmt.Errorf("cuda speculative verifier batch: %w", err))
	}
	tops, err := cudaSpeculativeRowTops(first, rows, len(drafts))
	if err != nil {
		return fail(err)
	}
	if f != nil {
		if err := f.sealGrouped(); err != nil {
			return fail(err)
		}
	}
	r.dspark = nil
	if err := v.state.advanceSpeculativeEpoch(); err != nil {
		return fail(err)
	}
	if err := v.retainRows(rows); err != nil {
		return fail(err)
	}
	return tops, nil
}

func (v *cudaSpeculativeVerifier) retainRows(rows [][]float32) error {
	epoch, err := v.state.speculativeEpoch()
	if err != nil {
		return err
	}
	v.rows = make([][]float32, len(rows))
	for i, row := range rows {
		v.rows[i] = append([]float32(nil), row...)
	}
	v.rowGeneration = v.state.generation
	v.rowEpoch = epoch
	return nil
}

func cudaSpeculativeRowTops(first int, rows [][]float32, count int) ([]int, error) {
	if count <= 0 || count > cudaSpeculativeMaxBatch || len(rows) != count {
		return nil, fmt.Errorf("cuda speculative verifier: got %d rows, want %d", len(rows), count)
	}
	tops := make([]int, count)
	tops[0] = first
	for i, row := range rows {
		if len(row) == 0 || len(row) != len(rows[0]) || first < 0 || first >= len(row) {
			return nil, fmt.Errorf("cuda speculative verifier: invalid vocabulary row %d", i)
		}
		if err := checkFiniteLogits("cuda speculative row", row); err != nil {
			return nil, err
		}
		if i+1 < count {
			tops[i+1] = cudaSpeculativeArgmax(row)
		}
	}
	return tops, nil
}

// Grouped verification is opt-in until its ordered-state and argmax gates
// pass on the target model. Enabling it does not qualify numerical parity.
func cudaSpeculativeGroupedEnabled() bool {
	return os.Getenv("DS4_CUDA_SPECULATIVE_GROUPED") == "1"
}

func validateCUDASpeculativeBatch(seed int, drafts []int, pos int) error {
	if seed < 0 {
		return fmt.Errorf("cuda speculative verifier: negative seed %d", seed)
	}
	if pos < 0 {
		return fmt.Errorf("cuda speculative verifier: negative position %d", pos)
	}
	if len(drafts) == 0 || len(drafts) > dsv4.DefaultSpeculativeLimits.MaxDrafts {
		return fmt.Errorf("cuda speculative verifier: invalid draft count %d", len(drafts))
	}
	for i, draft := range drafts {
		if draft < 0 {
			return fmt.Errorf("cuda speculative verifier: negative draft token %d at row %d", draft, i)
		}
	}
	return nil
}

// cudaSpeculativeVocabSize reads the model-owned vocabulary bound without
// dispatching a speculative row. VerifyBatch is an internal device boundary,
// so it cannot rely on dsv4.Speculator having performed its host-side token
// checks first.
func cudaSpeculativeVocabSize(state *cudaRequestState) (uint64, error) {
	if state == nil || state.runtime == nil || state.runtime.model == nil ||
		len(state.runtime.model.output.embedding.dimensions) < 2 {
		return 0, fmt.Errorf("cuda speculative verifier: missing vocabulary shape: %w", dsv4.ErrUnimplemented)
	}
	vocab := state.runtime.model.output.embedding.dimensions[1]
	if vocab == 0 {
		return 0, fmt.Errorf("cuda speculative verifier: empty vocabulary: %w", dsv4.ErrUnimplemented)
	}
	return vocab, nil
}

func validateCUDASpeculativeBatchVocab(seed int, drafts []int, vocab uint64) error {
	if vocab == 0 {
		return fmt.Errorf("cuda speculative verifier: empty vocabulary")
	}
	if uint64(seed) >= vocab {
		return fmt.Errorf("cuda speculative verifier: seed token %d outside vocabulary size %d", seed, vocab)
	}
	for i, draft := range drafts {
		if uint64(draft) >= vocab {
			return fmt.Errorf("cuda speculative verifier: draft token %d at row %d outside vocabulary size %d", draft, i, vocab)
		}
	}
	return nil
}

func cudaSpeculativeArgmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

func (v *cudaSpeculativeVerifier) CommitPrefix(n int) error {
	if v == nil || v.closed {
		return dsv4.ErrClosed
	}
	if v.state == nil || v.frontier == nil {
		return dsv4.ErrUnimplemented
	}
	if v.failed != nil {
		return v.failed
	}
	if n < 0 || n > len(v.drafts) {
		return fmt.Errorf("cuda speculative verifier: invalid accepted prefix %d", n)
	}
	if err := v.validateRowFrontier(); err != nil {
		return err
	}
	v.rows = nil
	if err := v.state.Restore(v.frontier); err != nil {
		return v.abort(fmt.Errorf("cuda speculative verifier restore frontier: %w", err))
	}
	for i := 0; i < n; i++ {
		if err := v.adapter.Eval(v.state, v.drafts[i], v.pos+i); err != nil {
			return v.abort(fmt.Errorf("cuda speculative verifier replay draft %d: %w", i, err))
		}
	}
	return v.closeFrontier(nil)
}

func (v *cudaSpeculativeVerifier) abort(err error) error {
	if v.frontier != nil {
		if restoreErr := v.state.Restore(v.frontier); restoreErr != nil {
			err = errors.Join(err, dsv4.ErrDesynchronized, fmt.Errorf("restore speculative frontier: %w", restoreErr))
			v.failed = err
		}
	}
	return v.closeFrontier(err)
}

// closeFrontier retains the snapshot on cleanup failure. The returned error
// exposes this verifier as an io.Closer; cleanup can be retried, execution
// cannot. A successful retry does not clear a failed execution disposition.
func (v *cudaSpeculativeVerifier) closeFrontier(cause error) error {
	v.rows = nil
	if v.frontier != nil {
		if err := v.frontier.Close(); err != nil {
			v.failed = errors.Join(v.failed, cause, dsv4.ErrClosed, fmt.Errorf("close speculative frontier: %w", err))
			return &backendCleanupOwner{cause: v.failed, backend: v}
		}
		v.frontier = nil
	}
	v.drafts = nil
	return cause
}

func (v *cudaSpeculativeVerifier) Close() error {
	if v == nil {
		return nil
	}
	v.closed = true
	return v.closeFrontier(nil)
}
