//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaStateAdapter is the checked boundary between the legacy CUDA runtime
// and the request-state contract. openSessionBackend uses it to create one
// request-local runtime around immutable model resources; each adapter state
// owns its mutable arena, stream, graphs, and position fields.
//
// The request runtime borrows immutable model resources from the owner while
// allocating its own mutable arena, stream, graphs, and position fields.
// This keeps the legacy single-request path intact while making ownership
// explicit at the dsv4 boundary.
type cudaStateAdapter struct {
	runtime *cudaRuntime
	closed  bool
}

var _ dsv4.Backend = (*cudaStateAdapter)(nil)
var _ dsv4.IndependentStates = (*cudaStateAdapter)(nil)
var _ dsv4.SpeculativeBackend = (*cudaStateAdapter)(nil)

// NewSpeculativeVerifier returns the rollback-capable verifier. Its default
// remains scalar. DS4_CUDA_SPECULATIVE_GROUPED=1 admits the experimental batch
// readout route for its separate device-state and ordering gate.
func (a *cudaStateAdapter) NewSpeculativeVerifier(state dsv4.State) (dsv4.SpeculativeVerifier, error) {
	if a == nil || a.closed {
		return nil, dsv4.ErrClosed
	}
	s, ok := state.(*cudaRequestState)
	if !ok || s == nil || s.closed || s.owner != a || s.runtime == nil {
		return nil, fmt.Errorf("cuda speculative verifier: state is not owned by backend: %w", dsv4.ErrUnimplemented)
	}
	return &cudaSpeculativeVerifier{adapter: a, state: s, batched: cudaSpeculativeGroupedEnabled()}, nil
}

func (a *cudaStateAdapter) NewState(context int) (dsv4.State, error) {
	if a == nil || a.closed {
		return nil, dsv4.ErrClosed
	}
	if a.runtime == nil {
		return nil, dsv4.ErrUnimplemented
	}
	if err := validateStateContext(context, a.runtime.ctx); err != nil {
		return nil, err
	}
	r, err := a.runtime.newIndependentState(context)
	if err != nil {
		return nil, fmt.Errorf("cuda state: %w", err)
	}
	return &cudaRequestState{runtime: r, owner: a}, nil
}

func (a *cudaStateAdapter) Eval(state dsv4.State, token, position int) error {
	if a == nil || a.closed {
		return dsv4.ErrClosed
	}
	s, ok := state.(*cudaRequestState)
	if !ok || s == nil || s.owner != a {
		return fmt.Errorf("cuda state: %w", dsv4.ErrUnimplemented)
	}
	return s.eval(token, position)
}

func (a *cudaStateAdapter) EvalBatch(state dsv4.State, tokens []int, position int) error {
	if a == nil || a.closed || a.runtime == nil {
		return dsv4.ErrClosed
	}
	s, ok := state.(*cudaRequestState)
	if !ok || s == nil || s.owner != a || s.closed || s.runtime == nil {
		return dsv4.ErrUnimplemented
	}
	// The request-state contract requires a batch to be observationally
	// equivalent to ordered Eval calls. The runtime's optimized prefill path
	// has a separate reduction and route schedule and is not yet qualified for
	// that contract (see the State split receipt). Keep this adapter correct
	// until that path has its own split-invariance gate.
	if s.runtime.power != nil {
		previous := s.runtime.power.prefill
		s.runtime.power.prefill = true
		defer func() { s.runtime.power.prefill = previous }()
	}
	for i, token := range tokens {
		if err := s.eval(token, position+i); err != nil {
			return fmt.Errorf("cuda state batch token %d: %w", i, err)
		}
	}
	return nil
}

func (a *cudaStateAdapter) Logits(state dsv4.State) []float32 {
	if a == nil || a.closed {
		return nil
	}
	s, ok := state.(*cudaRequestState)
	if !ok || s == nil || s.owner != a {
		return nil
	}
	return s.logits()
}

func (a *cudaStateAdapter) Close() error {
	if a == nil || a.closed {
		return nil
	}
	if a.runtime == nil {
		a.closed = true
		return nil
	}
	if err := a.runtime.Close(); err != nil {
		return err
	}
	a.closed = true
	return nil
}

// stateBackend returns the backend adapter used by the session seam.
func (r *cudaRuntime) stateBackend() dsv4.Backend {
	return &cudaStateAdapter{runtime: r}
}

// newIndependentState creates a request-local runtime around the immutable
// model views and compiled kernels already owned by r. It deliberately does
// not call newCUDARuntime: doing so would reopen the model and rebuild all
// load-time artifacts for every request. A shallow foundation copy gives the
// request its own stream while preserving read-only model resources.
func (r *cudaRuntime) newIndependentState(context int) (*cudaRuntime, error) {
	if r != nil && errors.Is(r.failed, dsv4.ErrClosed) {
		return nil, r.failed
	}
	if r != nil && r.cudaFoundation != nil {
		if err := r.checkOpen(); err != nil {
			return nil, err
		}
	}
	if r == nil || r.cudaFoundation == nil || r.model == nil || r.arena == nil {
		return nil, dsv4.ErrUnimplemented
	}
	if err := validateStateContext(context, r.ctx); err != nil {
		return nil, err
	}
	foundation := *r.cudaFoundation
	// A request must not implicitly join another request's captured stream
	// through synchronous copies on the legacy stream. Eval and Reset already
	// explicitly synchronize this request's stream before host reads/writes.
	foundation.workStream = nil
	state := &cudaRuntime{
		power:                   r.power.newRequest(),
		cudaFoundation:          &foundation,
		model:                   r.model,
		moeArtifacts:            r.moeArtifacts,
		denseArtifacts:          r.denseArtifacts,
		denseOutput:             r.denseOutput,
		ctx:                     context,
		borrowedFoundation:      true,
		requestOps:              r.requestOps,
		disableBlockDiagAttnOut: r.disableBlockDiagAttnOut,
		disableSIMDExperts:      r.disableSIMDExperts,
		disableGatherExperts:    r.disableGatherExperts,
		disableGraphs:           r.disableGraphs,
		disableAttentionHMMA:    r.disableAttentionHMMA,
	}
	if foundation.streamingExperts != nil {
		state.streamingRequest = foundation.streamingExperts.coordinator.request()
		state.disableGraphs = true
	}
	stream, err := state.requestOperations().create(foundation.device)
	foundation.workStream = stream
	if err != nil {
		return nil, finishBackendCleanup(fmt.Errorf("create request stream: %w", err), state)
	}
	if err := state.requestOperations().allocate(state); err != nil {
		return nil, finishBackendCleanup(fmt.Errorf("allocate request state: %w", err), state)
	}
	if err := state.requestOperations().hyperconnection(state); err != nil {
		return nil, finishBackendCleanup(fmt.Errorf("allocate request hyperconnection state: %w", err), state)
	}
	state.logits = make([]float32, r.state.logits.Len())
	if err := state.requestOperations().reset(state); err != nil {
		return nil, finishBackendCleanup(fmt.Errorf("reset request state: %w", err), state)
	}
	return state, nil
}

// These operations belong to one runtime. Tests can exercise partial ownership
// without replacing the process's CUDA driver or touching another session.
type cudaRequestOperations struct {
	create          func(*cuda.Device) (*cuda.Stream, error)
	allocate        func(*cudaRuntime) error
	hyperconnection func(*cudaRuntime) error
	reset           func(*cudaRuntime) error
	synchronize     func(*cudaRuntime) error
	destroy         func(*cudaRuntime) error
}

var defaultCUDARequestOperations = cudaRequestOperations{
	create: func(d *cuda.Device) (*cuda.Stream, error) { return d.NewStreamWithFlags(driver.StreamNonBlocking) },
	allocate: func(state *cudaRuntime) error {
		state.arena = cudamem.NewArena(state.device)
		state.allocState()
		return state.arena.Err()
	},
	hyperconnection: func(state *cudaRuntime) error {
		var err error
		state.hcMath, err = newCUDAHC(state.cudaFoundation)
		return err
	},
	reset: func(state *cudaRuntime) error { return state.Reset() },
	synchronize: func(r *cudaRuntime) error {
		if r.workStream == nil {
			return nil
		}
		return syncStream(r.workStream)
	},
	destroy: func(r *cudaRuntime) error { return r.workStream.Close() },
}

func (r *cudaRuntime) requestOperations() *cudaRequestOperations {
	if r.requestOps != nil {
		return r.requestOps
	}
	return &defaultCUDARequestOperations
}

// cudaRequestState is the dsv4 State facade. The wrapped runtime owns every
// mutable KV/compressor/indexer/raw-window/HC/graph allocation; no request
// state aliases another request's arena or stream.
type cudaRequestState struct {
	runtime        *cudaRuntime
	owner          *cudaStateAdapter
	closed         bool
	dspark         *cudaDSparkFeatures
	generation     uint64
	lifecycleEpoch uint64
}

func (s *cudaRequestState) eval(token, position int) (err error) {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if errors.Is(s.runtime.failed, dsv4.ErrClosed) {
		return s.runtime.failed
	}
	if s.dspark != nil {
		if err := s.dspark.begin(position, token); err != nil {
			return err
		}
		s.runtime.dspark = s.dspark
		defer func() {
			s.runtime.dspark = nil
			if err != nil {
				s.invalidateDSpark()
			}
		}()
	}
	if err = s.runtime.Eval(token, position); err != nil {
		_ = s.advanceSpeculativeEpoch()
		return err
	}
	return s.finishDSparkEval()
}

func (s *cudaRequestState) logits() []float32 {
	if s == nil || s.closed || s.runtime == nil || s.runtime.failed != nil {
		return nil
	}
	return s.runtime.Logits()
}

func (s *cudaRequestState) Reset() error {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if errors.Is(s.runtime.failed, dsv4.ErrDesynchronized) {
		return s.runtime.failed
	}
	if err := s.runtime.Reset(); err != nil {
		return err
	}
	s.invalidateDSpark()
	return s.advanceSpeculativeEpoch()
}

func (s *cudaRequestState) Save(w io.Writer) error {
	if s != nil && s.runtime != nil && errors.Is(s.runtime.failed, dsv4.ErrClosed) {
		return s.runtime.failed
	}
	if s != nil && !s.closed && s.runtime != nil && s.dspark != nil {
		return s.saveSupport(w)
	}
	return saveCUDAState(s, w)
}

func (s *cudaRequestState) Load(r io.Reader) error {
	if s != nil && s.runtime != nil && errors.Is(s.runtime.failed, dsv4.ErrClosed) {
		return s.runtime.failed
	}
	if s != nil && !s.closed && s.runtime != nil && s.dspark != nil {
		return s.loadSupport(r)
	}
	if err := loadCUDAState(s, r); err != nil {
		return err
	}
	s.invalidateDSpark()
	return s.completeObservedCacheReplacement(nil)
}

func (s *cudaRequestState) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.invalidateDSpark()
	if s.dspark != nil {
		if err := s.dspark.close(); err != nil {
			return s.finishClose(err)
		}
		s.dspark = nil
	}
	if s.runtime == nil {
		s.closed = true
		return nil
	}
	err := s.runtime.Close()
	return s.finishClose(err)
}

func (s *cudaRequestState) finishClose(err error) error {
	// Retain every failed cleanup owner, including the request's private stream.
	if err != nil {
		s.runtime.failed = errors.Join(err, dsv4.ErrClosed, dsv4.ErrDesynchronized)
		return s.runtime.failed
	}
	s.closed = true
	s.runtime = nil
	return err
}

func cudaStateUnsupported(err error) bool {
	return errors.Is(err, dsv4.ErrUnimplemented)
}
