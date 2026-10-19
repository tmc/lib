package cudasearch

import (
	"context"
	"errors"
	"io"
	"maps"
	"reflect"
	"strings"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// Implemented only by built-ins with independently owned CPU readback storage.
// Owners returned with an error still need closing. Failed Close retains the
// owner for retry; successful Close ends all use of the returned values.
type ownedExecutableOutputs interface {
	outputsOwned(context.Context) (map[string][]float32, io.Closer, error)
}

// beginAttempt admits the fixed evaluator storage and workload-sized scratch
// before validation or copying. Opaque callbacks remain covered only by their
// declared callback envelope, not by this consumer's logical storage units.
func (e *ExecutableEvaluator) beginAttempt(ctx context.Context) error {
	scope, err := mockMemoryScope(ctx)
	if err != nil {
		return err
	}
	// Fixed scratch covers hash state, fixed-size digest encodings and the
	// evaluator's two-arm scaffolding; map populations are admitted below.
	if err := scope.Reserve(int64(reflect.TypeFor[ExecutableEvaluator]().Size()) + 1024); err != nil {
		scope.Close()
		return err
	}
	// Key formatting, hashing input conversions and dimension formatting are
	// bounded by concrete input populations; scratch may be charged more than
	// once rather than predicting compiler allocation elision.
	if err := bookkeepingSlots[byte](scope, 256*e.Workload.Shape.Rank()); err != nil {
		scope.Close()
		return err
	}
	for _, value := range []string{e.Workload.Name, e.Workload.Layout, string(e.Workload.NumericalPolicy), string(e.Workload.Objective)} {
		if err := scope.Reserve(8 * int64(len(value))); err != nil {
			scope.Close()
			return err
		}
	}
	for _, m := range []map[string]string{e.Workload.InputIdentities, e.Workload.WeightIdentities} {
		if err := reserveBookkeepingMap(scope, m); err != nil {
			scope.Close()
			return err
		}
		if err := bookkeepingSlots[string](scope, len(m)); err != nil {
			scope.Close()
			return err
		}
		for k, v := range m {
			if err := scope.Reserve(8*int64(len(k)) + 8*int64(len(v))); err != nil {
				scope.Close()
				return err
			}
		}
	}
	if err := bookkeepingSlots[map[string]bool](scope, 1); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[struct {
		key   string
		value bool
	}](scope, len(e.Workload.InputNames)); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[map[string][]PreparationPhase](scope, 1); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[struct {
		key   string
		value []PreparationPhase
	}](scope, 2); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[string](scope, len(e.Workload.InputNames)); err != nil {
		scope.Close()
		return err
	}
	for _, name := range e.Workload.InputNames {
		if err := scope.Reserve(8 * int64(len(name))); err != nil {
			scope.Close()
			return err
		}
	}
	// Comparison error construction may include a reference name. Admit that
	// text as well as the map entries before invoking the output comparison.
	for name := range e.References {
		if err := scope.Reserve(8 * int64(len(name))); err != nil {
			scope.Close()
			return err
		}
	}
	if err := reserveBookkeepingMap(scope, e.References); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[string](scope, len(e.References)); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[OutputComparison](scope, len(e.References)); err != nil {
		scope.Close()
		return err
	}
	if err := bookkeepingSlots[*memoryReservation](scope, 1); err != nil {
		scope.Close()
		return err
	}
	e.attemptMemory.Close()
	e.attemptMemory = scope
	e.memoryReservations = make([]*memoryReservation, 0, 1)
	e.Receipts = nil
	e.attemptIdentity = ""
	e.comparisonArm = ""
	e.Comparison = make(map[string]OutputComparison)
	e.Preparation = make(map[string][]PreparationPhase)
	return nil
}

func (e *ExecutableEvaluator) compareExecutable(ctx context.Context, x Executable, candidate bool, a *cudaflightrecorder.CandidateAssociation) (err error) {
	var outputs map[string][]float32
	var owner io.Closer
	if owned, ok := x.(ownedExecutableOutputs); ok {
		outputs, owner, err = owned.outputsOwned(ctx)
	} else {
		outputs, err = x.Outputs(ctx)
	}
	if owner != nil {
		defer func() {
			if closeErr := owner.Close(); closeErr != nil {
				e.pendingOutput = owner
				err = errors.Join(err, closeErr)
			}
		}()
	}
	if err != nil {
		return err
	}
	// The result map and sort keys are admitted before compare allocates them.
	if err := reserveBookkeepingMap(e.attemptMemory, e.References); err != nil {
		return err
	}
	if err := bookkeepingSlots[string](e.attemptMemory, len(e.References)); err != nil {
		return err
	}
	comparison, err := e.Policy.compare(outputs, e.References)
	e.Comparison = comparison
	if candidate && errors.Is(err, ErrVerificationFailed) {
		return e.recordComparisonFailure(ctx, a, outputs, err)
	}
	return err
}

func cloneTrialsWithMemory(in []cudatrial.Receipt, scope *egraphcore.MemoryScope) ([]cudatrial.Receipt, error) {
	if in == nil {
		return nil, nil
	}
	if err := bookkeepingSlots[cudatrial.Receipt](scope, len(in)); err != nil {
		return nil, err
	}
	out := make([]cudatrial.Receipt, len(in))
	for i := range in {
		var err error
		out[i], err = in[i].CloneWithMemory(scope.Reserve)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (e *ExecutableEvaluator) detachCandidate(scope *egraphcore.MemoryScope, c *CandidatePlan) error {
	var err error
	c.Trials, err = cloneTrialsWithMemory(e.Receipts, scope)
	if err != nil {
		return err
	}
	if c.Receipt != nil {
		copy, err := c.Receipt.CloneWithMemory(scope.Reserve)
		if err != nil {
			return err
		}
		c.Receipt = &copy
	}
	if err := reserveBookkeepingMap(scope, e.Comparison); err != nil {
		return err
	}
	c.Comparison = maps.Clone(e.Comparison)
	if err := scope.Reserve(int64(len(c.ID) + len(c.WorkloadKey) + len(c.InputKey))); err != nil {
		return err
	}
	c.ID = strings.Clone(c.ID)
	c.WorkloadKey = strings.Clone(c.WorkloadKey)
	c.InputKey = strings.Clone(c.InputKey)
	c.Preparation, err = clonePreparationWithMemory(scope, c.Preparation)
	return err
}
