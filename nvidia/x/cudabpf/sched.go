package cudabpf

import (
	"errors"
	"fmt"
)

// SchedAttributes are the time-slice-group attributes a policy may set on
// task init. Each has a companion Set flag rather than a sentinel value,
// because every numeric value in range is a legitimate request and there is
// no number left over to mean "unset".
type SchedAttributes struct {
	TimesliceUS   uint64
	SetTimeslice  bool
	Interleave    InterleaveLevel
	SetInterleave bool
}

func (a SchedAttributes) String() string {
	switch {
	case a.SetTimeslice && a.SetInterleave:
		return fmt.Sprintf("timeslice %dus, interleave %s", a.TimesliceUS, a.Interleave)
	case a.SetTimeslice:
		return fmt.Sprintf("timeslice %dus", a.TimesliceUS)
	case a.SetInterleave:
		return fmt.Sprintf("interleave %s", a.Interleave)
	}
	return "no attributes set"
}

// Limits are the ranges the driver accepts. They are a property of the
// target. The defaults below are host-model policy limits, not a discovery
// receipt. A consumer must supply driver and tenant limits validated on the
// target rather than treating this example window as an admission result.
type Limits struct {
	MinTimesliceUS uint64
	MaxTimesliceUS uint64
	// MaxInterleave is the highest level the driver defines. There are
	// three, so a request above it is a request for a level that does not
	// exist rather than for more of something.
	MaxInterleave InterleaveLevel
}

// DefaultLimits returns a host-model policy window. It is not a verified
// driver range: no target with the scheduler table has been qualified.
func DefaultLimits() Limits {
	return Limits{MinTimesliceUS: 1, MaxTimesliceUS: 100_000, MaxInterleave: InterleaveHigh}
}

// ErrOutOfRange is returned by [SchedAttributes.Validate].
var ErrOutOfRange = errors.New("cudabpf: scheduling attribute out of range")

// Validate checks every set attribute against the driver's accepted range
// before any kfunc call is made. Validation is not clamping: a request above
// the maximum is refused, and in particular a tenant asking for a longer
// timeslice than it is allowed does not silently receive the maximum.
func (a SchedAttributes) Validate(l Limits) error {
	if l.MinTimesliceUS > l.MaxTimesliceUS || !l.MaxInterleave.Valid() {
		return fmt.Errorf("%w: invalid target limits", ErrOutOfRange)
	}
	if a.SetTimeslice {
		if a.TimesliceUS < l.MinTimesliceUS || a.TimesliceUS > l.MaxTimesliceUS {
			return fmt.Errorf("%w: timeslice %dus outside [%d,%d]",
				ErrOutOfRange, a.TimesliceUS, l.MinTimesliceUS, l.MaxTimesliceUS)
		}
	}
	if a.SetInterleave {
		if !a.Interleave.Valid() || a.Interleave > l.MaxInterleave {
			return fmt.Errorf("%w: interleave level %d above %d",
				ErrOutOfRange, uint32(a.Interleave), uint32(l.MaxInterleave))
		}
	}
	return nil
}

// A SchedPolicy decides scheduling attributes. TaskInit is called on
// on_task_init and Bind on on_bind; both run on a driver path and are bounded
// work under the same rules as a memory hook.
type SchedPolicy interface {
	// TaskInit returns the attributes to set for a newly initialized TSG.
	// The returned attributes are validated before any of them is applied,
	// and a validation failure applies none of them.
	TaskInit(c *TaskInitContext, b *Budget) (SchedAttributes, error)

	// Bind decides whether a channel bind is admitted.
	Bind(c *BindContext, b *Budget) (BindDecision, error)
}

// A BindDecision is the answer to an admission-control hook.
type BindDecision struct {
	Reject bool
	Reason string
}

// BindRejection names the mechanism by which a rejection reaches the driver.
//
// Two forms exist in the interface: the kfunc bpf_nv_gpu_reject_bind, and the
// allow member of nv_gpu_bind_ctx. They are not alternatives. The kfunc is
// non-sleepable and its entire effect is to write the ctx field, so the field
// is what the driver reads and the call is how a verified BPF program is
// permitted to write it. A program that assigns ctx->allow directly is
// writing a struct_ops context member without the kfunc the verifier
// sanctions for it.
//
// This package therefore implements the call form and never writes Allow. The
// determination is drawn from the interface sources, not from a live driver:
// no target available to this lane registers nv_gpu_sched_ops, so it has not
// been observed. See the handoff.
const BindRejection = "bpf_nv_gpu_reject_bind"

// applyTaskInit runs the task-init hook: consult, validate, then apply.
// Validation happens before the first kfunc call. An actuator failure after
// an earlier successful call is reported as PartiallySubmitted; this interface
// cannot guarantee rollback or restoration of driver defaults.
func applyTaskInit(p SchedPolicy, c *TaskInitContext, b *Budget, k Kfuncs, l Limits) (SchedAttributes, Outcome) {
	out := Outcome{Hook: "on_task_init", Consulted: true, HandlerTime: Unmeasured}
	if c == nil {
		out.Fallback, out.Reason = true, "nil context"
		return SchedAttributes{}, out
	}
	a, err := p.TaskInit(c, b)
	if b != nil && b.exhausted {
		err = ErrBudgetExceeded
	}
	if err != nil {
		out.Fallback, out.Reason = true, err.Error()
		return SchedAttributes{}, out
	}
	if !a.SetTimeslice && !a.SetInterleave {
		return a, out
	}
	out.HadOpinion = true
	if err := a.Validate(l); err != nil {
		out.Fallback, out.Reason = true, err.Error()
		return SchedAttributes{}, out
	}
	if k == nil {
		out.Fallback, out.Reason = true, "no kfunc actuator"
		return SchedAttributes{}, out
	}
	if a.SetTimeslice {
		if err := k.SetTimeslice(c, a.TimesliceUS); err != nil {
			out.Fallback, out.Reason = true, err.Error()
			return SchedAttributes{}, out
		}
	}
	if a.SetInterleave {
		if err := k.SetInterleave(c, a.Interleave); err != nil {
			out.PartiallySubmitted = a.SetTimeslice
			out.Submitted = a.SetTimeslice
			out.Fallback, out.Reason = true, err.Error()
			return SchedAttributes{}, out
		}
	}
	out.Submitted = true
	return a, out
}

// applyBind runs the admission hook.
func applyBind(p SchedPolicy, c *BindContext, b *Budget, k Kfuncs) (BindDecision, Outcome) {
	out := Outcome{Hook: "on_bind", Consulted: true, HandlerTime: Unmeasured}
	if c == nil {
		out.Fallback, out.Reason = true, "nil context"
		return BindDecision{}, out
	}
	d, err := p.Bind(c, b)
	if b != nil && b.exhausted {
		err = ErrBudgetExceeded
	}
	if err != nil {
		out.Fallback, out.Reason = true, err.Error()
		return BindDecision{}, out
	}
	if !d.Reject {
		return d, out
	}
	out.HadOpinion = true
	if k == nil {
		out.Fallback, out.Reason = true, "no kfunc actuator"
		return BindDecision{}, out
	}
	if err := k.RejectBind(c); err != nil {
		out.Fallback, out.Reason = true, err.Error()
		return BindDecision{}, out
	}
	out.Submitted = true
	out.Reason = d.Reason
	return d, out
}
