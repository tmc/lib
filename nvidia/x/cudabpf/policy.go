package cudabpf

import (
	"errors"
	"fmt"
	"time"
)

// A Reorder is a request to move a root chunk to one end of one of the
// driver's two lists. It is the whole of what this interface can express:
// head and tail are values written into the request, there is no
// insert-after kfunc, and so strict eviction-list ordering is not expressible
// here at all. Rank-and-position reordering is.
type Reorder struct {
	Destination Destination
	Position    Position
}

func (r Reorder) String() string {
	return fmt.Sprintf("%s/%s", r.Destination, r.Position)
}

// Valid reports whether both halves name something the driver defines. A
// policy that returns a Reorder failing this is treated as having failed: the
// driver's own ordering stands and the reason is recorded.
func (r Reorder) Valid() error {
	if !r.Destination.Valid() {
		return fmt.Errorf("cudabpf: invalid reorder destination %d", uint64(r.Destination))
	}
	if !r.Position.Valid() {
		return fmt.Errorf("cudabpf: invalid reorder position %d", uint64(r.Position))
	}
	return nil
}

// A Region is a half-open range of pages to prefetch, matching the first and
// outer members of uvm_bpf_prefetch_decision_t.
type Region struct {
	First uint64
	Outer uint64
}

func (r Region) String() string { return fmt.Sprintf("[%d,%d)", r.First, r.Outer) }

// Valid reports whether the region is non-empty and correctly ordered. An
// empty or inverted region is refused rather than normalized: a handler that
// computed one has a bug, and clamping it would hide the bug behind a
// prefetch that silently does nothing.
func (r Region) Valid() error {
	if r.Outer <= r.First {
		return fmt.Errorf("cudabpf: prefetch region %s is empty or inverted", r)
	}
	return nil
}

// ErrBudgetExceeded is returned by [Budget.Step] when a handler has done more
// work than a fault path permits.
var ErrBudgetExceeded = errors.New("cudabpf: handler budget exceeded")

// A Budget bounds the work a fault-path handler may do. Fault-path handlers
// are bounded work by contract: fixed instruction budget, no unbounded loops,
// no blocking, no allocation. The Budget is the host-testable half of that --
// the BPF verifier enforces the rest inside the kernel -- and a handler that
// exhausts it has its hook disabled rather than being allowed to run long.
//
// The zero Budget permits nothing, so a handler reached without one stops at
// its first step instead of running unbounded.
type Budget struct {
	MaxSteps  int
	used      int
	exhausted bool
}

// NewBudget returns a budget of n steps.
func NewBudget(n int) *Budget { return &Budget{MaxSteps: n} }

// Step charges n units of work and reports [ErrBudgetExceeded] once the total
// passes the maximum. Once exceeded, every later Step also fails: a budget
// does not refill mid-handler.
func (b *Budget) Step(n int) error {
	if n < 0 {
		b.exhausted = true
		return fmt.Errorf("cudabpf: negative budget charge")
	}
	if b.exhausted || b.MaxSteps < 0 || n > b.MaxSteps-b.used {
		b.exhausted = true
		return fmt.Errorf("%w: used %d of %d", ErrBudgetExceeded, b.used, b.MaxSteps)
	}
	b.used += n
	return nil
}

// Used reports the work charged so far.
func (b *Budget) Used() int { return b.used }

// Reset returns the budget to full, for the next hook invocation.
func (b *Budget) Reset() { b.used = 0; b.exhausted = false }

// A MemoryPolicy decides placement. Every method is called on a fault path,
// must charge its work against the budget, and must not block or allocate.
//
// Returning ok false is a first-class answer meaning "no opinion": the
// driver's own behavior stands, and nothing is recorded as a fallback,
// because nothing failed.
type MemoryPolicy interface {
	// BlockAccess runs on gpu_block_access.
	BlockAccess(d *PMMDecision, b *Budget) (r Reorder, ok bool, err error)

	// BlockActivate runs on gpu_block_activate.
	BlockActivate(d *PMMDecision, b *Budget) (r Reorder, ok bool, err error)

	// Prefetch runs on gpu_page_prefetch and gpu_page_prefetch_iter.
	Prefetch(d *PrefetchDecision, b *Budget) (r Region, ok bool, err error)
}

// An Outcome is what happened to one hook invocation. It separates four
// things a single boolean would conflate: whether the policy had an opinion,
// whether that opinion was submitted to the driver, whether the driver
// recorded having taken it, and whether the driver recorded a collision.
//
// DriverAttempted and DriverConflict are read back out of the context after
// the kfunc call. They are the driver's own record, not a Go-side flag, and
// they are the only evidence available that a decision reached the driver at
// all. Neither of them says the driver obeyed it.
type Outcome struct {
	// PartiallySubmitted means an earlier scheduling actuator succeeded before
	// a later call failed. No driver rollback is claimed.
	PartiallySubmitted bool
	Policy             string
	Hook               string
	Consulted          bool // the policy was asked
	HadOpinion         bool // the policy returned one
	Submitted          bool // the kfunc call was made and returned no error

	DriverAttempted bool
	DriverConflict  bool

	// Fallback is set when the driver's default behavior stands because
	// something went wrong rather than because the policy declined.
	Fallback bool
	Reason   string

	Reorder Reorder
	Region  Region

	// HandlerTime is the time spent inside the handler. Timings are
	// reported in separate non-additive fields and an unmeasured one is
	// [Unmeasured], never zero: zero is a real and different answer.
	HandlerTime time.Duration
}

// Unmeasured marks a duration that was not measured. A zero duration means a
// measurement was taken and came out at zero, which is a different claim.
const Unmeasured time.Duration = -1

// Timings separates the four costs a policy path can incur. They are not
// additive -- prefetch worker time overlaps fault handling, transfer overlaps
// compute -- so they are never summed into a single number.
type Timings struct {
	FaultHandling  time.Duration
	PrefetchWorker time.Duration
	Transfer       time.Duration
	GPUCompute     time.Duration
}

// UnmeasuredTimings returns a Timings in which nothing has been measured.
func UnmeasuredTimings() Timings {
	return Timings{Unmeasured, Unmeasured, Unmeasured, Unmeasured}
}

func (o Outcome) String() string {
	switch {
	case o.Fallback:
		return fmt.Sprintf("%s: driver default (%s)", o.Hook, o.Reason)
	case !o.HadOpinion:
		return fmt.Sprintf("%s: no opinion", o.Hook)
	case o.Submitted:
		return fmt.Sprintf("%s: submitted %s (driver attempted=%t conflict=%t)",
			o.Hook, o.Reorder, o.DriverAttempted, o.DriverConflict)
	default:
		return fmt.Sprintf("%s: not submitted (%s)", o.Hook, o.Reason)
	}
}

// applyReorder runs one placement hook end to end: consult the policy,
// validate what it returned, submit it, and read the driver's answer back.
//
// Every failure path lands in the same place -- the driver's own ordering,
// with a recorded reason -- because an undefined placement is the one outcome
// that must not be reachable.
func applyReorder(hook string, fn func(*PMMDecision, *Budget) (Reorder, bool, error),
	d *PMMDecision, b *Budget, k Kfuncs, clock func() time.Time) Outcome {

	out := Outcome{Hook: hook, Consulted: true, HandlerTime: Unmeasured}
	if d == nil {
		out.Fallback = true
		out.Reason = "nil context"
		return out
	}
	var start time.Time
	if clock != nil {
		start = clock()
	}
	r, ok, err := fn(d, b)
	if b != nil && b.exhausted {
		err = ErrBudgetExceeded
	}
	if clock != nil {
		out.HandlerTime = clock().Sub(start)
	}

	switch {
	case err != nil:
		out.Fallback = true
		out.Reason = err.Error()
		return out
	case !ok:
		return out
	}
	out.HadOpinion = true
	out.Reorder = r

	if err := r.Valid(); err != nil {
		out.Fallback = true
		out.Reason = err.Error()
		return out
	}
	if k == nil {
		out.Fallback = true
		out.Reason = "no kfunc actuator"
		return out
	}
	if err := k.RequestReorder(d, r.Destination, r.Position); err != nil {
		out.Fallback = true
		out.Reason = err.Error()
		return out
	}
	out.Submitted = true
	out.DriverAttempted = d.Request.Attempted != 0
	out.DriverConflict = d.Request.Conflict != 0
	if out.DriverConflict {
		// A collision is not a failure of this path: the request was
		// made and the driver recorded that something else got there
		// first. The driver's ordering stands, which is the correct
		// result, so it is recorded as a fallback with the driver's
		// own reason rather than as a success.
		out.Fallback = true
		out.Reason = "driver recorded a conflicting request"
	}
	return out
}
