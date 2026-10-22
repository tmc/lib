package cudacapture

import (
	"errors"
	"fmt"
	"sync"
)

// SyncMode specifies the synchronization enforcement policy for a [SyncArbiter].
type SyncMode int

const (
	// SyncAllow permits host-blocking CUDA synchronization operations.
	SyncAllow SyncMode = 0
	// SyncWarn records a [SyncViolation] and invokes the optional warning callback,
	// but allows the operation to proceed without returning an error.
	SyncWarn SyncMode = 1
	// SyncReject records a [SyncViolation] and rejects the operation with an error
	// wrapping [ErrForbiddenSync].
	SyncReject SyncMode = 2
)

// String returns the lowercase name of m ("allow", "warn", "reject") or a
// numeric representation for unknown values.
func (m SyncMode) String() string {
	switch m {
	case SyncAllow:
		return "allow"
	case SyncWarn:
		return "warn"
	case SyncReject:
		return "reject"
	default:
		return fmt.Sprintf("SyncMode(%d)", int(m))
	}
}

// ErrForbiddenSync is returned by [SyncArbiter.CheckSync] when a host-blocking
// CUDA synchronization operation is attempted while the effective policy is
// [SyncReject].
var ErrForbiddenSync = errors.New("cudacapture: forbidden host-device synchronization")

// SyncViolation describes a host-blocking synchronization operation observed
// while a [SyncArbiter] was in [SyncWarn] or [SyncReject] mode.
type SyncViolation struct {
	Op           string
	Mode         SyncMode
	Depth        int
	AllowedDepth int
}

type syncScopeKind uint8

const (
	scopeForbid syncScopeKind = iota + 1
	scopeAllow
)

type syncScopeEntry struct {
	id   uint64
	kind syncScopeKind
	mode SyncMode
}

// SyncArbiter arbitrates nested and concurrent synchronization guard scopes
// (analogous to DeepInfer's _SyncDebugArbiter, forbid_cuda_sync, and allow_cuda_sync).
//
// Each [SyncArbiter.Forbid] or [SyncArbiter.Allow] call pushes a uniquely
// identified scope onto the active stack and returns an idempotent restore
// function that removes that exact scope even when concurrent scopes exit out
// of order.
type SyncArbiter struct {
	mu          sync.Mutex
	defaultMode SyncMode
	onWarn      func(SyncViolation)
	nextID      uint64
	active      []syncScopeEntry
	violations  []SyncViolation
}

// NewSyncArbiter constructs a [SyncArbiter] with defaultMode and an optional
// warning callback invoked whenever a violation occurs in [SyncWarn] mode.
func NewSyncArbiter(defaultMode SyncMode, onWarn func(SyncViolation)) *SyncArbiter {
	if defaultMode != SyncAllow && defaultMode != SyncWarn && defaultMode != SyncReject {
		defaultMode = SyncAllow
	}
	return &SyncArbiter{
		defaultMode: defaultMode,
		onWarn:      onWarn,
	}
}

// Forbid pushes a forbidden synchronization scope with mode ([SyncWarn] or
// [SyncReject]) and returns an idempotent restore closure that removes the scope.
// If mode is neither [SyncWarn] nor [SyncReject], [SyncReject] is used.
func (a *SyncArbiter) Forbid(mode SyncMode) (restore func()) {
	if a == nil {
		return func() {}
	}
	if mode != SyncWarn && mode != SyncReject {
		mode = SyncReject
	}
	a.mu.Lock()
	a.nextID++
	id := a.nextID
	a.active = append(a.active, syncScopeEntry{
		id:   id,
		kind: scopeForbid,
		mode: mode,
	})
	a.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			a.removeScope(id)
		})
	}
}

// Allow pushes a temporary allow-override scope and returns an idempotent
// restore closure that removes the scope. While this scope is the innermost
// active scope (or while any active scope on the stack is an allow scope that
// has not been superseded by a more recent inner [SyncArbiter.Forbid]),
// [SyncArbiter.CheckSync] permits synchronization without recording a violation.
func (a *SyncArbiter) Allow() (restore func()) {
	if a == nil {
		return func() {}
	}
	a.mu.Lock()
	a.nextID++
	id := a.nextID
	a.active = append(a.active, syncScopeEntry{
		id:   id,
		kind: scopeAllow,
		mode: SyncAllow,
	})
	a.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			a.removeScope(id)
		})
	}
}

func (a *SyncArbiter) removeScope(id uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := len(a.active) - 1; i >= 0; i-- {
		if a.active[i].id == id {
			a.active = append(a.active[:i], a.active[i+1:]...)
			return
		}
	}
}

func (a *SyncArbiter) effectiveStateLocked() (mode SyncMode, forbidDepth, allowDepth int) {
	for _, s := range a.active {
		if s.kind == scopeForbid {
			forbidDepth++
		} else if s.kind == scopeAllow {
			allowDepth++
		}
	}
	if len(a.active) == 0 {
		return a.defaultMode, 0, 0
	}
	top := a.active[len(a.active)-1]
	if top.kind == scopeAllow {
		return SyncAllow, forbidDepth, allowDepth
	}
	return top.mode, forbidDepth, allowDepth
}

// CheckSync checks whether a host-blocking operation named op (such as
// "cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize",
// "cuMemcpyDtoH", or ".item()") is permitted under the current scope stack.
func (a *SyncArbiter) CheckSync(op string) error {
	if a == nil {
		return nil
	}
	if op == "" {
		op = "unknown_sync"
	}

	a.mu.Lock()
	mode, depth, allowedDepth := a.effectiveStateLocked()
	if mode == SyncAllow {
		a.mu.Unlock()
		return nil
	}
	v := SyncViolation{
		Op:           op,
		Mode:         mode,
		Depth:        depth,
		AllowedDepth: allowedDepth,
	}
	a.violations = append(a.violations, v)
	onWarn := a.onWarn
	a.mu.Unlock()

	if mode == SyncWarn {
		if onWarn != nil {
			onWarn(v)
		}
		return nil
	}
	return fmt.Errorf("cudacapture: sync op %q blocked at depth=%d (allowed_depth=%d): %w",
		op, depth, allowedDepth, ErrForbiddenSync)
}

// Violations returns a copy of all recorded [SyncViolation] events since the
// arbiter was created or last reset.
func (a *SyncArbiter) Violations() []SyncViolation {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.violations) == 0 {
		return nil
	}
	out := make([]SyncViolation, len(a.violations))
	copy(out, a.violations)
	return out
}

// ResetViolations clears all recorded [SyncViolation] events.
func (a *SyncArbiter) ResetViolations() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.violations = nil
}
