package cudaworkload

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

// ErrPopMismatch indicates that a correlation pop returned an ID different
// from the one pushed, signaling an unbalanced stack or unexpected thread migration.
var ErrPopMismatch = errors.New("cudaworkload: correlation pop ID mismatch")

// CorrelationHook allows injecting custom correlation push/pop implementations
// for testing or alternative instrumentation providers.
type CorrelationHook interface {
	Push(kind cupti.ExternalCorrelationKind, id uint64) error
	Pop(kind cupti.ExternalCorrelationKind) (uint64, error)
}

type defaultCUPTIHook struct{}

func (defaultCUPTIHook) Push(kind cupti.ExternalCorrelationKind, id uint64) error {
	return cupti.ActivityPushExternalCorrelationId(kind, id)
}

func (defaultCUPTIHook) Pop(kind cupti.ExternalCorrelationKind) (uint64, error) {
	return cupti.ActivityPopExternalCorrelationId(kind)
}

// AnnotatorOption configures an [Annotator].
type AnnotatorOption func(*Annotator)

// WithRunID sets the run identifier for workloads created by the Annotator.
func WithRunID(runID string) AnnotatorOption {
	return func(a *Annotator) {
		a.runID = runID
	}
}

// WithCorrelationKind sets the CUPTI external correlation kind to use for
// native calls. Defaults to cupti.ExternalCorrelationKindCustom0 when enabled.
func WithCorrelationKind(kind cupti.ExternalCorrelationKind) AnnotatorOption {
	return func(a *Annotator) {
		a.corrKind = kind
	}
}

// WithCorrelationHook overrides the push/pop implementation used for native calls.
func WithCorrelationHook(hook CorrelationHook) AnnotatorOption {
	return func(a *Annotator) {
		a.corrHook = hook
	}
}

// WithDefaultLabels sets labels inherited by all operations created by the Annotator.
func WithDefaultLabels(labels map[string]string) AnnotatorOption {
	return func(a *Annotator) {
		if labels != nil {
			a.labels = make(map[string]string, len(labels))
			maps.Copy(a.labels, labels)
		}
	}
}

// Annotator coordinates the allocation of operation identities and context
// propagation for application workloads.
type Annotator struct {
	process  ProcessGenID
	workload string
	runID    string
	labels   map[string]string

	corrKind cupti.ExternalCorrelationKind
	corrHook CorrelationHook
}

// NewAnnotator creates a new Annotator bound to the given process generation.
func NewAnnotator(proc ProcessGenID, workloadName string, opts ...AnnotatorOption) *Annotator {
	a := &Annotator{
		process:  proc,
		workload: workloadName,
		corrKind: cupti.ExternalCorrelationKindCustom0,
		corrHook: defaultCUPTIHook{},
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// ScopeOption configures an operation [Scope].
type ScopeOption func(*Scope)

// WithScopeLabels adds or overrides labels for a specific Scope.
func WithScopeLabels(labels map[string]string) ScopeOption {
	return func(s *Scope) {
		if labels == nil {
			return
		}
		if s.labels == nil {
			s.labels = make(map[string]string, len(labels))
		}
		for k, v := range labels {
			s.labels[k] = v
		}
	}
}

// Scope represents an active operation execution scope.
type Scope struct {
	id        OpID
	parentID  OpID
	annotator *Annotator
	workload  WorkloadIdentity
	name      string
	startTime time.Time
	endTime   time.Time
	labels    map[string]string

	mu    sync.Mutex
	ended bool
}

// ID returns the operation's unique identifier.
func (s *Scope) ID() OpID {
	return s.id
}

// ParentID returns the enclosing parent operation's identifier, or 0 if root.
func (s *Scope) ParentID() OpID {
	return s.parentID
}

// Workload returns a snapshot of the operation's workload identity.
func (s *Scope) Workload() WorkloadIdentity {
	return s.workload.Clone()
}

// Name returns the descriptive operation name.
func (s *Scope) Name() string {
	return s.name
}

// End closes the operation scope.
func (s *Scope) End() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.ended {
		s.ended = true
		s.endTime = time.Now()
	}
}

// NativeCall executes a native CUDA submission function on a pinned OS thread,
// pushing the scope's OpID onto the thread-local CUPTI correlation stack for the
// duration of the call and popping it on completion.
//
// Thread pinning via [runtime.LockOSThread] is active only for the duration of fn,
// ensuring the calling Go goroutine is unpinned immediately upon exit.
// Cleanup is balanced across all return paths, including errors and panics.
func (s *Scope) NativeCall(fn func() error) (err error) {
	if s == nil {
		return errors.New("cudaworkload: nil scope")
	}
	if s.annotator == nil || fn == nil || s.id == 0 {
		return errors.New("cudaworkload: invalid native call")
	}
	hook := s.annotator.corrHook
	kind := s.annotator.corrKind
	if kind < cupti.ExternalCorrelationKindInvalid || kind >= cupti.ExternalCorrelationKindSize {
		return errors.New("cudaworkload: invalid external correlation kind")
	}
	if hook != nil && nilCorrelationHook(hook) {
		return errors.New("cudaworkload: nil correlation hook")
	}
	enableCUPTI := !nilCorrelationHook(hook) && kind != cupti.ExternalCorrelationKindInvalid

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if enableCUPTI {
		if pushErr := hook.Push(kind, uint64(s.id)); pushErr != nil {
			return fmt.Errorf("cudaworkload: push external correlation: %w", pushErr)
		}
		defer func() {
			popped, popErr := hook.Pop(kind)
			if popErr != nil {
				err = errors.Join(err, fmt.Errorf("cudaworkload: pop external correlation: %w", popErr))
			} else if popped != uint64(s.id) {
				err = errors.Join(err, fmt.Errorf("%w: pushed %d, popped %d", ErrPopMismatch, s.id, popped))
			}
		}()
	}

	return fn()
}

// NewSubmission constructs a [HostSubmission] record associated with this scope.
func (s *Scope) NewSubmission(nativeCorr uint32, cs ContextStreamID, graph GraphReplayGeneration, symbol string, evidenceRef string) HostSubmission {
	prov := ProvenanceCUPTI
	if nilCorrelationHook(s.annotator.corrHook) || s.annotator.corrKind <= cupti.ExternalCorrelationKindInvalid || s.annotator.corrKind >= cupti.ExternalCorrelationKindSize {
		prov = ProvenanceUninstrumented
	}
	return HostSubmission{
		Process:             s.workload.Process,
		OpID:                s.id,
		ParentOpID:          s.parentID,
		Workload:            s.workload.Clone(),
		NativeCorrelationID: nativeCorr,
		Provenance:          prov,
		Context:             cs,
		GraphReplay:         graph,
		TID:                 currentTID(),
		TimestampNs:         uint64(time.Now().UnixNano()),
		TimestampClock:      "unix_wall",
		TimestampEpoch:      "unix_epoch",
		Symbol:              symbol,
		EvidenceRef:         evidenceRef,
	}
}

func nilCorrelationHook(hook CorrelationHook) bool {
	if hook == nil {
		return true
	}
	v := reflect.ValueOf(hook)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

// NativeCallWithSubmission executes a native submission function on a pinned OS thread,
// pushing correlation to CUPTI, invoking fn to obtain the driver/runtime correlation ID,
// and returning a populated [HostSubmission].
func (s *Scope) NativeCallWithSubmission(cs ContextStreamID, graph GraphReplayGeneration, symbol string, evidenceRef string, fn func() (uint32, error)) (HostSubmission, error) {
	var sub HostSubmission
	if fn == nil {
		return sub, errors.New("cudaworkload: nil submission callback")
	}

	err := s.NativeCall(func() error {
		var err error
		nativeCorr, err := fn()
		if err == nil {
			sub = s.NewSubmission(nativeCorr, cs, graph, symbol, evidenceRef)
		}
		return err
	})

	if err != nil {
		return HostSubmission{}, err
	}

	return sub, nil
}

type contextKey struct{}

// WithOperation returns a context that carries the given operation Scope.
func WithOperation(ctx context.Context, s *Scope) context.Context {
	return context.WithValue(ctx, contextKey{}, s)
}

// OperationFromContext retrieves the current operation Scope from ctx, if present.
func OperationFromContext(ctx context.Context) (*Scope, bool) {
	s, ok := ctx.Value(contextKey{}).(*Scope)
	return s, ok
}

// Begin initiates a new operation scope. If ctx already contains an active
// operation, the new operation is linked as a nested child.
func (a *Annotator) Begin(ctx context.Context, name string, opts ...ScopeOption) (context.Context, *Scope) {
	newID := nextOperationID()
	var parentID OpID

	labels := make(map[string]string)
	if a.labels != nil {
		maps.Copy(labels, a.labels)
	}

	if parent, ok := OperationFromContext(ctx); ok && parent.workload.Process == a.process {
		parentID = parent.id
		if parent.labels != nil {
			maps.Copy(labels, parent.labels)
		}
	}

	s := &Scope{
		id:        newID,
		parentID:  parentID,
		annotator: a,
		workload: WorkloadIdentity{
			Process:      a.process,
			WorkloadName: a.workload,
			RunID:        a.runID,
			Labels:       labels,
		},
		name:      name,
		startTime: time.Now(),
		labels:    labels,
	}

	for _, opt := range opts {
		opt(s)
	}
	s.workload.Labels = s.labels

	return WithOperation(ctx, s), s
}

// CUPTI external correlation is process-wide. Separate annotators must never
// allocate the same external operation ID in that process.
var processOperationID atomic.Uint64

func nextOperationID() OpID {
	for {
		old := processOperationID.Load()
		if old == ^uint64(0) {
			return 0
		}
		if processOperationID.CompareAndSwap(old, old+1) {
			return OpID(old + 1)
		}
	}
}
