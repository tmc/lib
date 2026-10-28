package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// MeasurementContext declares the execution context in which empirical trial
// facts may be reused within one Search call. These strings are caller
// assertions, not device discovery or attestation. TargetIdentity must cover
// the physical device, target ABI, driver/runtime, toolchain and execution
// context generation. ResetIdentity must version ResetState and all state it
// resets. A changed context requires a changed identity before Evaluate.
//
// Empty TargetIdentity, zero evaluator Generation, missing input identities,
// or a nonnil ResetState without ResetIdentity safely disables reuse. The
// epoch belongs to the ExecutableEvaluator passed to Search and ends when
// Search returns; a wrapping Evaluator passed to Search disables reuse. Direct
// Evaluate, held-out Confirm, Evaluate by any other evaluator handed the
// Search context, and Evaluate with a context retained after Search returns
// all measure normally and neither reuse nor extend the Search's facts.
// Reuse never skips extraction, current compilation/cleanup, preparation or
// quarantine checks.
// Receipt provenance marks reused facts; their source timestamps are retained.
type MeasurementContext struct {
	TargetIdentity string
	ResetIdentity  string
}

type measurementMemo struct {
	epoch   *searchEpoch
	ledger  *memoryBudget
	memory  *egraphcore.MemoryScope
	entries map[[32]byte]*measurementRecord
}
type measurementRecord struct {
	memory     *egraphcore.MemoryScope
	trials     []cudatrial.Receipt
	comparison map[string]OutputComparison
}

func (m *measurementMemo) close() {
	if m == nil {
		return
	}
	for _, r := range m.entries {
		r.memory.Close()
	}
	m.entries = nil
	m.memory.Close()
}

// The Search epoch is live until closeSearch retires it, and only the
// evaluator Search was called with joins it. Holding its pointer prevents
// address reuse while this memo is alive. Direct contexts cannot manufacture an epoch by supplying only a
// memory limit, retained contexts cannot revive one, and foreign evaluators
// handed the context neither join it nor charge a memo to its ledger.
func (e *ExecutableEvaluator) measurementAttempt(ctx context.Context, confirmation bool, id string) (*measurementMemo, [32]byte, error) {
	var key [32]byte
	work, _ := ctx.Value(workBudgetKey{}).(*workBudget)
	var epoch *searchEpoch
	var ledger *memoryBudget
	if !confirmation && work != nil && work.epoch != nil && work.epoch.owner == e && !work.epoch.retired.Load() && work.memory == searchMemoryBudget(ctx) {
		epoch, ledger = work.epoch, work.memory
	}
	if e.measurements != nil && e.measurements.epoch != epoch {
		e.measurements.close()
		e.measurements = nil
	}
	c := e.MeasurementContext
	if epoch == nil || e.Generation <= 0 || c.TargetIdentity == "" || e.Workload.ResetState != nil && c.ResetIdentity == "" || len(e.Workload.InputIdentities) == 0 {
		return nil, key, nil
	}
	scratch, err := egraphcore.NewMemoryScope(ledger)
	if err != nil {
		return nil, key, err
	}
	defer scratch.Close()
	if err := scratch.Reserve(1024); err != nil {
		return nil, key, err
	}
	if err := bookkeepingSlots[string](scratch, len(e.References)); err != nil {
		return nil, key, err
	}
	for _, s := range []string{id, e.Baseline.Identity(), c.TargetIdentity, c.ResetIdentity} {
		if err := scratch.Reserve(int64(len(s))); err != nil {
			return nil, key, err
		}
	}
	names := make([]string, 0, len(e.References))
	for name := range e.References {
		if err := scratch.Reserve(int64(len(name))); err != nil {
			return nil, key, err
		}
		names = append(names, name)
	}
	sort.Strings(names)
	h := sha256.New()
	var word [8]byte
	number := func(v uint64) { binary.LittleEndian.PutUint64(word[:], v); h.Write(word[:]) }
	text := func(s string) { number(uint64(len(s))); h.Write([]byte(s)) }
	text("cudasearch-empirical-reuse-v1")
	text(e.Workload.Key())
	text(inputKey(e.Workload))
	text(id)
	text(e.Baseline.Identity())
	text(c.TargetIdentity)
	text(c.ResetIdentity)
	number(uint64(e.Generation))
	text(string(e.TimingKind))
	number(uint64(e.Rounds))
	number(math.Float64bits(e.Policy.Absolute))
	number(math.Float64bits(e.Policy.Relative))
	if e.Policy.NaNsEqual {
		number(1)
	} else {
		number(0)
	}
	if e.Policy.IgnoreZeroSign {
		number(1)
	} else {
		number(0)
	}
	number(uint64(len(names)))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, key, err
		}
		text(name)
		values := e.References[name]
		number(uint64(len(values)))
		for i, v := range values {
			if i%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, key, err
				}
			}
			number(uint64(math.Float32bits(v)))
		}
	}
	copy(key[:], h.Sum(nil))
	if e.measurements == nil {
		m, err := newMeasurementMemo(epoch, ledger)
		if err != nil {
			return nil, key, err
		}
		e.measurements = m
	}
	return e.measurements, key, nil
}

// measurementMemoBytes is the declared storage of an empty memo: its scope,
// the memo, the map header, and the epoch and ledger objects it retains
// after Search releases its own reservations for them.
func measurementMemoBytes() int64 {
	return int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[measurementMemo]().Size() + reflect.TypeFor[map[[32]byte]*measurementRecord]().Size() + reflect.TypeFor[searchEpoch]().Size() + reflect.TypeFor[memoryBudget]().Size())
}

func newMeasurementMemo(epoch *searchEpoch, ledger *memoryBudget) (*measurementMemo, error) {
	memory, err := egraphcore.NewMemoryScope(ledger)
	if err != nil {
		return nil, err
	}
	if err := memory.Reserve(measurementMemoBytes() - memory.Bytes()); err != nil {
		memory.Close()
		return nil, err
	}
	return &measurementMemo{epoch: epoch, ledger: ledger, memory: memory, entries: make(map[[32]byte]*measurementRecord)}, nil
}

func cloneMeasurementComparison(scope *egraphcore.MemoryScope, in map[string]OutputComparison) (map[string]OutputComparison, error) {
	if in == nil {
		return nil, nil
	}
	if err := reserveBookkeepingMap(scope, in); err != nil {
		return nil, err
	}
	for name := range in {
		if err := scope.Reserve(int64(len(name))); err != nil {
			return nil, err
		}
	}
	out := make(map[string]OutputComparison, len(in))
	for name, value := range in {
		out[strings.Clone(name)] = value
	}
	return out, nil
}
func markMeasurement(scope *egraphcore.MemoryScope, trials []cudatrial.Receipt, key [32]byte, c MeasurementContext, reused bool) error {
	if err := scope.Reserve(64); err != nil {
		return err
	}
	source := hex.EncodeToString(key[:])
	kind := "false"
	if reused {
		kind = "true"
	}
	for i := range trials {
		if trials[i].Provenance == nil {
			if err := bookkeepingSlots[map[string]string](scope, 1); err != nil {
				return err
			}
			trials[i].Provenance = make(map[string]string)
		}
		for _, p := range []struct{ name, value string }{{"measurement_reused", kind}, {"measurement_key", source}, {"measurement_target_identity", c.TargetIdentity}, {"measurement_reset_identity", c.ResetIdentity}} {
			if err := bookkeepingSlots[struct{ key, value string }](scope, 1); err != nil {
				return err
			}
			if err := scope.Reserve(int64(len(p.name) + len(p.value))); err != nil {
				return err
			}
			trials[i].Provenance[strings.Clone(p.name)] = strings.Clone(p.value)
		}
	}
	return nil
}
func (e *ExecutableEvaluator) reuseMeasurement(record *measurementRecord, key [32]byte) error {
	trials, err := cloneTrialsWithMemory(record.trials, e.attemptMemory)
	if err != nil {
		return err
	}
	comparison, err := cloneMeasurementComparison(e.attemptMemory, record.comparison)
	if err != nil {
		return err
	}
	if err := markMeasurement(e.attemptMemory, trials, key, e.MeasurementContext, true); err != nil {
		return err
	}
	e.Receipts = trials
	e.Comparison = comparison
	e.comparisonArm = "candidate"
	return nil
}
func (m *measurementMemo) remember(key [32]byte, trials []cudatrial.Receipt, comparison map[string]OutputComparison) error {
	if m.epoch.retired.Load() || m.entries[key] != nil {
		return nil
	}
	// Only complete correctness-qualified successful attempts reach this call,
	// after executable cleanup. Still reject partial/malformed source facts.
	if len(trials) == 0 || len(comparison) == 0 {
		return nil
	}
	for _, r := range trials {
		if !r.Complete || r.Err != nil {
			return nil
		}
	}
	memory, err := egraphcore.NewMemoryScope(m.ledger)
	if err != nil {
		return err
	}
	retained := false
	defer func() {
		if !retained {
			memory.Close()
		}
	}()
	if err := bookkeepingSlots[measurementRecord](memory, 1); err != nil {
		return err
	}
	copyTrials, err := cloneTrialsWithMemory(trials, memory)
	if err != nil {
		return err
	}
	copyComparison, err := cloneMeasurementComparison(memory, comparison)
	if err != nil {
		return err
	}
	if err := bookkeepingSlots[struct {
		key   [32]byte
		value *measurementRecord
	}](m.memory, 1); err != nil {
		return err
	}
	m.entries[key] = &measurementRecord{memory: memory, trials: copyTrials, comparison: copyComparison}
	retained = true
	return nil
}
