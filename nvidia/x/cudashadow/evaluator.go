package cudashadow

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// Policy defines an observational policy evaluator.
type Policy interface {
	// Info returns the policy's name, version, domain, and seed.
	Info() PolicyInfo
	// Evaluate produces a proposed decision given an observed input.
	Evaluate(ctx context.Context, in Input) (Decision, error)
}

// Evaluator manages shadow policy evaluation, freshness checks, sequence ordering,
// and bounded snapshot recording.
type Evaluator struct {
	mu           sync.Mutex
	evalGate     chan struct{}
	capacity     int
	snapshots    []Snapshot
	head         int
	totalCount   uint64
	lastSequence uint64
	lastNano     int64
	source       *Provenance
	nextID       uint64
	clockFunc    func() int64
}

// NewEvaluator creates a new Evaluator with a bounded snapshot capacity.
// Capacity must be greater than zero.
func NewEvaluator(capacity int) (*Evaluator, error) {
	if capacity <= 0 || capacity > 1<<20 {
		return nil, errors.New("cudashadow: capacity must be between 1 and 1048576")
	}
	return &Evaluator{
		capacity:  capacity,
		evalGate:  make(chan struct{}, 1),
		snapshots: make([]Snapshot, 0, capacity),
		clockFunc: func() int64 { return time.Now().UnixNano() },
	}, nil
}

// SetClock sets a custom clock in the same domain as input timestamps.
// The default clock and TimestampNano use Unix nanoseconds.
func (e *Evaluator) SetClock(fn func() int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if fn != nil {
		e.clockFunc = fn
	}
}

// Evaluate records proposed decisions. The evaluator never executes them.
// Policy implementations are caller code and must themselves be observational.
func (e *Evaluator) Evaluate(ctx context.Context, pol Policy, in Input) (Snapshot, error) {
	if err := in.validateBounds(); err != nil {
		return Snapshot{}, err
	}
	if pol == nil || reflect.ValueOf(pol).Kind() == reflect.Pointer && reflect.ValueOf(pol).IsNil() {
		return Snapshot{}, errors.New("cudashadow: policy cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return Snapshot{}, fmt.Errorf("cudashadow: %w", err)
	}

	select {
	case e.evalGate <- struct{}{}:
		defer func() { <-e.evalGate }()
	case <-ctx.Done():
		return Snapshot{}, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	e.mu.Lock()
	clock := e.clockFunc
	e.mu.Unlock()
	in = cloneInput(in)

	id := fmt.Sprintf("snap-%08d", atomic.AddUint64(&e.nextID, 1))
	evalTime := clock()
	polInfo := clonePolicyInfo(pol.Info())
	if err := polInfo.Validate(); err != nil {
		return Snapshot{}, err
	}

	// 1. Validate mandatory fields
	if err := in.Validate(); err != nil {
		snap := Snapshot{
			ID:         id,
			Provenance: in.Provenance,
			Policy:     polInfo,
			Input:      in,
			Decision: Decision{
				Type:   DecisionReject,
				Reason: fmt.Sprintf("invalid input: %v", err),
			},
			Status: StatusUnsupported,
		}
		e.storeSnapshotLocked(snap)
		return snap, fmt.Errorf("cudashadow: %w", err)
	}

	// 2. Domain compatibility check
	if polInfo.Domain != in.Domain {
		snap := Snapshot{
			ID:         id,
			Provenance: in.Provenance,
			Policy:     polInfo,
			Input:      in,
			Decision: Decision{
				Type:   DecisionReject,
				Reason: fmt.Sprintf("domain mismatch: policy %q handles %s, input is %s", polInfo.Name, polInfo.Domain, in.Domain),
			},
			Status: StatusUnsupported,
		}
		e.storeSnapshotLocked(snap)
		return snap, fmt.Errorf("cudashadow: domain %s unsupported by policy domain %s: %w", in.Domain, polInfo.Domain, cudaerrors.ErrUnsupported)
	}

	// 3. Stale checks: TTL expiration or sequence disorder
	isStale := false
	var staleReason string
	if evalTime < in.Provenance.TimestampNano {
		isStale = true
		staleReason = "input timestamp is in the future"
	}
	if in.Provenance.TTL > 0 && evalTime > in.Provenance.TimestampNano {
		age := time.Duration(evalTime - in.Provenance.TimestampNano)
		if age > in.Provenance.TTL {
			isStale = true
			staleReason = fmt.Sprintf("input age %v exceeds ttl %v", age, in.Provenance.TTL)
		}
	}
	if in.Provenance.Sequence > 0 && e.lastSequence > 0 && in.Provenance.Sequence <= e.lastSequence {
		isStale = true
		staleReason = fmt.Sprintf("out-of-order sequence %d (last seen %d)", in.Provenance.Sequence, e.lastSequence)
	}
	if e.lastNano > 0 && in.Provenance.TimestampNano < e.lastNano {
		isStale = true
		staleReason = "input timestamp moved backwards"
	}
	if e.source != nil && (!sameSource(*e.source, in.Provenance)) {
		return Snapshot{}, errors.New("cudashadow: input source changed; use a new evaluator")
	}

	if isStale {
		snap := Snapshot{
			ID:         id,
			Provenance: in.Provenance,
			Policy:     polInfo,
			Input:      in,
			Decision: Decision{
				Type:   DecisionSkip,
				Reason: fmt.Sprintf("stale input: %s", staleReason),
			},
			Status: StatusStale,
		}
		e.storeSnapshotLocked(snap)
		return snap, nil
	}

	// 4. Fresh and valid: evaluate policy
	dec, err := pol.Evaluate(ctx, cloneInput(in))
	if err == nil {
		err = ctx.Err()
	}
	if err == nil {
		err = dec.validate(in.Domain)
	}
	if err != nil {
		snap := Snapshot{
			ID:         id,
			Provenance: in.Provenance,
			Policy:     polInfo,
			Input:      in,
			Decision: Decision{
				Type:   DecisionReject,
				Reason: fmt.Sprintf("evaluation error: %v", err),
			},
			Status: StatusUnsupported,
		}
		e.storeSnapshotLocked(snap)
		return snap, fmt.Errorf("cudashadow evaluate: %w", err)
	}

	// Update sequence tracker
	if in.Provenance.Sequence > 0 {
		e.lastSequence = in.Provenance.Sequence
	}
	e.lastNano = in.Provenance.TimestampNano
	if e.source == nil {
		source := in.Provenance
		e.source = &source
	}

	snap := Snapshot{
		ID:         id,
		Provenance: in.Provenance,
		Policy:     polInfo,
		Input:      in,
		Decision:   dec,
		Status:     StatusValid,
	}
	e.storeSnapshotLocked(snap)
	return snap, nil
}

// storeSnapshotLocked adds an independent snapshot to the bounded ring buffer.
func (e *Evaluator) storeSnapshotLocked(snap Snapshot) {
	e.mu.Lock()
	defer e.mu.Unlock()
	snap = cloneSnapshot(snap)
	e.totalCount++
	if len(e.snapshots) < e.capacity {
		e.snapshots = append(e.snapshots, snap)
	} else {
		e.snapshots[e.head] = snap
		e.head = (e.head + 1) % e.capacity
	}
}

// Snapshots returns an ordered slice of all currently buffered snapshots.
func (e *Evaluator) Snapshots() []Snapshot {
	e.mu.Lock()
	defer e.mu.Unlock()

	n := len(e.snapshots)
	out := make([]Snapshot, n)
	if n < e.capacity {
		for i := range e.snapshots {
			out[i] = cloneSnapshot(e.snapshots[i])
		}
		return out
	}
	// Ring buffer is wrapped: head points to the oldest entry
	copy(out, e.snapshots[e.head:])
	copy(out[n-e.head:], e.snapshots[:e.head])
	for i := range out {
		out[i] = cloneSnapshot(out[i])
	}
	return out
}

// TotalEvaluated returns the cumulative count of evaluated inputs.
func (e *Evaluator) TotalEvaluated() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.totalCount
}

// RecordOutcome updates a previously recorded snapshot with the observed outcome from baseline execution.
func (e *Evaluator) RecordOutcome(snapshotID string, outcome ObservedOutcome) error {
	if err := outcome.validate(); err != nil {
		return err
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	for i := range e.snapshots {
		if e.snapshots[i].ID == snapshotID {
			if e.snapshots[i].Provenance.Kind == "artifact_description" && !outcome.LatencyUnavailable {
				return errors.New("cudashadow: artifact completion latency unavailable")
			}
			if e.snapshots[i].ObservedOutcome != nil {
				return errors.New("cudashadow: outcome already recorded")
			}
			if outcome.TimestampNano < e.snapshots[i].Provenance.TimestampNano {
				return errors.New("cudashadow: outcome precedes input")
			}
			if e.snapshots[i].Status != StatusValid {
				return errors.New("cudashadow: outcome requires valid snapshot")
			}
			outcome.Details = maps.Clone(outcome.Details)
			e.snapshots[i].ObservedOutcome = &outcome
			return nil
		}
	}
	return fmt.Errorf("cudashadow: snapshot %q not found", snapshotID)
}

// Replay re-evaluates a sequence of recorded snapshots against the policy,
// asserting deterministic reproduction of decisions.
//
// Like WriteHistory, Replay refuses a history with more than 4096 snapshots,
// an invalid snapshot, or a compact encoding over 16 MiB before it copies any
// snapshot or calls the policy. Because every snapshot is validated first,
// an invalid snapshot is reported ahead of a cancellation, policy identity
// mismatch, ordering error or evaluation failure at any step.
func (e *Evaluator) Replay(ctx context.Context, pol Policy, history []Snapshot) ([]Snapshot, error) {
	if pol == nil || reflect.ValueOf(pol).Kind() == reflect.Pointer && reflect.ValueOf(pol).IsNil() {
		return nil, errors.New("cudashadow replay: policy cannot be nil")
	}
	if _, _, err := checkHistory(history, "cudashadow replay step"); err != nil {
		return nil, err
	}
	replayed := make([]Snapshot, 0, len(history))
	var lastSequence uint64
	var previous *Provenance

	for i, snap := range history {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("cudashadow replay canceled at step %d: %w", i, err)
		}

		if !reflect.DeepEqual(pol.Info(), snap.Policy) {
			return nil, fmt.Errorf("cudashadow replay step %d: policy identity mismatch", i)
		}
		if snap.Status == StatusValid && snap.Provenance.Sequence > 0 && lastSequence > 0 && snap.Provenance.Sequence <= lastSequence {
			return nil, errors.New("cudashadow replay: sequence disorder")
		}
		// Re-evaluate input if original status was valid
		if snap.Status == StatusValid {
			if previous != nil && (!sameSource(*previous, snap.Provenance) || previous.TimestampNano > snap.Provenance.TimestampNano) {
				return nil, errors.New("cudashadow replay: source change or timestamp disorder")
			}
			current := snap.Provenance
			previous = &current
			if snap.Provenance.Sequence > 0 {
				lastSequence = snap.Provenance.Sequence
			}
			dec, err := pol.Evaluate(ctx, cloneInput(snap.Input))
			if err != nil {
				return nil, fmt.Errorf("cudashadow replay step %d (id %s): evaluation failed: %w", i, snap.ID, err)
			}
			if !reflect.DeepEqual(dec, snap.Decision) {
				return nil, fmt.Errorf("cudashadow replay step %d (id %s): decision mismatch: recorded %s, got %s",
					i, snap.ID, snap.Decision.Type, dec.Type)
			}
			if !reflect.DeepEqual(dec.Parameters, snap.Decision.Parameters) {
				return nil, fmt.Errorf("cudashadow replay step %d (id %s): parameter mismatch: recorded %v, got %v",
					i, snap.ID, snap.Decision.Parameters, dec.Parameters)
			}
			snapCopy := cloneSnapshot(snap)
			snapCopy.Decision = dec
			replayed = append(replayed, snapCopy)
		} else {
			// Retain non-valid snapshot as recorded
			replayed = append(replayed, cloneSnapshot(snap))
		}
	}

	return replayed, nil
}

func cloneInput(in Input) Input {
	if in.Eviction != nil {
		v := *in.Eviction
		in.Eviction = &v
	}
	if in.Prefetch != nil {
		v := *in.Prefetch
		in.Prefetch = &v
	}
	if in.Scheduling != nil {
		v := *in.Scheduling
		in.Scheduling = &v
	}
	if in.PTX != nil {
		v := *in.PTX
		if v.Geometry != nil {
			g := *v.Geometry
			v.Geometry = &g
		}
		if v.Schema != nil {
			schema := *v.Schema
			schema.Params = append([]cudakernel.Param(nil), schema.Params...)
			v.Schema = &schema
		}
		in.PTX = &v
	}
	return in
}
func cloneSnapshot(s Snapshot) Snapshot {
	s.Input = cloneInput(s.Input)
	s.Policy.Parameters = maps.Clone(s.Policy.Parameters)
	s.Decision.Parameters = maps.Clone(s.Decision.Parameters)
	if s.ObservedOutcome != nil {
		v := *s.ObservedOutcome
		v.Details = maps.Clone(v.Details)
		s.ObservedOutcome = &v
	}
	return s
}

func sameSource(a, b Provenance) bool {
	return a.Kind == b.Kind && a.Source == b.Source && a.DeviceID == b.DeviceID && a.DriverVersion == b.DriverVersion && a.KernelModule == b.KernelModule
}
