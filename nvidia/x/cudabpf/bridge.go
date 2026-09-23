package cudabpf

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ErrDisabled is returned when a hook or the whole bridge is switched off.
// Attachment is opt-in per process and reversible, so being off is a normal
// state and not a failure.
var ErrDisabled = errors.New("cudabpf: hook disabled")

// ErrNotAttached is returned when an operation needs a live attachment.
var ErrNotAttached = errors.New("cudabpf: not attached")

// Config configures a [Bridge].
type Config struct {
	// Limits are the driver's accepted attribute ranges.
	Limits Limits
	// Budget is the per-invocation step budget for a fault-path handler.
	// A handler that exceeds it has its hook disabled.
	Budget int
	// Observer receives bounded records. It may be nil.
	Observer Observer
	// Clock measures handler time. Nil means handler time is
	// [Unmeasured] rather than zero.
	Clock func() time.Time
}

// A Bridge attaches policy to a target and routes hook invocations to it.
//
// The bridge never takes eviction authority from the driver. Every path
// through it ends either in a submitted request the driver may ignore, or in
// the driver's own behavior with a recorded reason. There is no path that
// leaves a placement undefined.
type Bridge struct {
	cfg    Config
	loader Loader
	kfuncs Kfuncs

	mu       sync.Mutex
	report   *Report
	object   Object
	links    map[string]Link // table -> link
	disabled map[string]string
	off      bool
	mem      MemoryPolicy
	sched    SchedPolicy
	outcomes []Outcome
}

// NewBridge returns a bridge over a loader and a kfunc actuator. Neither is
// used until [Bridge.Attach] is called, and a bridge with nil policies
// attaches nothing.
func NewBridge(l Loader, k Kfuncs, cfg Config) *Bridge {
	if cfg.Limits == (Limits{}) {
		cfg.Limits = DefaultLimits()
	}
	return &Bridge{
		cfg:      cfg,
		loader:   l,
		kfuncs:   k,
		links:    make(map[string]Link),
		disabled: make(map[string]string),
	}
}

// SetMemoryPolicy installs the placement policy.
func (br *Bridge) SetMemoryPolicy(p MemoryPolicy) {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.mem = p
}

// SetSchedPolicy installs the scheduling policy.
func (br *Bridge) SetSchedPolicy(p SchedPolicy) {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.sched = p
}

// Discover runs capability discovery and caches the report.
func (br *Bridge) Discover(ctx context.Context) (*Report, error) {
	r, err := Discover(ctx, br.loader)
	if err != nil {
		return nil, err
	}
	br.mu.Lock()
	br.report = r
	br.mu.Unlock()
	return r, nil
}

// Report returns the cached discovery report, or nil.
func (br *Bridge) Report() *Report {
	br.mu.Lock()
	defer br.mu.Unlock()
	return br.report
}

// Attach loads an object and registers one struct_ops map.
//
// Attaching a table that is already attached is not an error and does not
// register a second map: the existing link is returned. Two processes racing
// to attach is a different case, and one the kernel arbitrates -- the second
// registration fails at the loader and the error is returned unwrapped.
func (br *Bridge) Attach(ctx context.Context, obj ObjectSpec, spec StructOpsSpec) (Link, error) {
	if err := spec.Validate(obj); err != nil {
		return nil, err
	}

	br.mu.Lock()
	if br.off {
		br.mu.Unlock()
		return nil, fmt.Errorf("%w: bridge", ErrDisabled)
	}
	if l, ok := br.links[spec.Table]; ok {
		br.mu.Unlock()
		return l, nil
	}
	report := br.report
	br.mu.Unlock()

	if report == nil {
		var err error
		if report, err = br.Discover(ctx); err != nil {
			return nil, err
		}
	}
	if report.LayoutError != nil {
		return nil, fmt.Errorf("cudabpf: refusing to attach: %w", report.LayoutError)
	}
	for member := range spec.Programs {
		if !hookAvailable(report, spec.Table, member) {
			return nil, fmt.Errorf("cudabpf: %s.%s is not available on this target: %s",
				spec.Table, member, hookReason(report, spec.Table, member))
		}
	}
	for _, name := range spec.RequiredKfuncs() {
		k, ok := report.Kfunc(name)
		if !ok || !k.Available {
			return nil, fmt.Errorf("cudabpf: kfunc %s is not available: %s", name, kfuncReason(report, name))
		}
	}

	o, err := br.loader.Load(ctx, obj)
	if err != nil {
		return nil, fmt.Errorf("cudabpf: loading %s: %w", obj.Name, err)
	}
	link, err := o.AttachStructOps(ctx, spec)
	if err != nil {
		o.Close()
		return nil, fmt.Errorf("cudabpf: attaching %s: %w", spec.Table, err)
	}

	br.mu.Lock()
	defer br.mu.Unlock()
	if existing, ok := br.links[spec.Table]; ok {
		// Lost a race with another Attach in this process. Keep the
		// first and undo this one rather than leaking a registration.
		link.Detach()
		o.Close()
		return existing, nil
	}
	br.links[spec.Table] = link
	br.object = o
	return link, nil
}

// Detach removes one table's attachment. Detaching what is not attached
// succeeds: a disable switch must work whatever state it finds.
func (br *Bridge) Detach(table string) error {
	br.mu.Lock()
	link, ok := br.links[table]
	delete(br.links, table)
	br.mu.Unlock()
	if !ok {
		return nil
	}
	return link.Detach()
}

// Close detaches everything and releases the object. It is the process-level
// off switch, and after it the driver's default behavior is what runs.
func (br *Bridge) Close() error {
	br.mu.Lock()
	links := br.links
	br.links = make(map[string]Link)
	obj := br.object
	br.object = nil
	br.off = true
	br.mu.Unlock()

	var err error
	for _, name := range sortedLinkNames(links) {
		if derr := links[name].Detach(); derr != nil && err == nil {
			err = derr
		}
	}
	if obj != nil {
		if cerr := obj.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}
	return err
}

// Reattach re-registers a table whose link is stale because the driver
// re-registered its tables. A generation change invalidates the link but not
// the policy, so the old link is detached and a new one taken.
func (br *Bridge) Reattach(ctx context.Context, obj ObjectSpec, spec StructOpsSpec, generation uint64) (Link, error) {
	br.mu.Lock()
	link, ok := br.links[spec.Table]
	br.mu.Unlock()
	if ok && link.Generation() == generation {
		return link, nil
	}
	if err := br.Detach(spec.Table); err != nil {
		return nil, err
	}
	br.mu.Lock()
	br.report = nil // the tables were re-registered; rediscover
	br.mu.Unlock()
	return br.Attach(ctx, obj, spec)
}

// DisableHook switches one hook off with a recorded reason. The driver's
// default behavior takes over immediately; the attachment is left in place so
// the hook can be re-enabled without a reload.
func (br *Bridge) DisableHook(member, reason string) {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.disabled[member] = reason
}

// EnableHook clears a disable.
func (br *Bridge) EnableHook(member string) {
	br.mu.Lock()
	defer br.mu.Unlock()
	delete(br.disabled, member)
}

// HookDisabled reports whether a hook is off, and why.
func (br *Bridge) HookDisabled(member string) (string, bool) {
	br.mu.Lock()
	defer br.mu.Unlock()
	if br.off {
		return "bridge disabled", true
	}
	r, ok := br.disabled[member]
	return r, ok
}

// Disable switches the whole process off.
func (br *Bridge) Disable() {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.off = true
}

// Enable switches the process back on.
func (br *Bridge) Enable() {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.off = false
}

// Outcomes returns the outcomes recorded so far, oldest first.
func (br *Bridge) Outcomes() []Outcome {
	br.mu.Lock()
	defer br.mu.Unlock()
	return append([]Outcome(nil), br.outcomes...)
}

// budget returns a fresh budget for one invocation.
func (br *Bridge) budget() *Budget {
	n := br.cfg.Budget
	if n <= 0 {
		n = 64
	}
	return NewBudget(n)
}

// record files an outcome and, when a handler blew its budget, disables the
// hook that did it. A handler that cannot finish inside a fault path must
// stop being called, not be called again and fail again.
func (br *Bridge) record(out Outcome) Outcome {
	br.mu.Lock()
	br.outcomes = append(br.outcomes, out)
	if out.Fallback && errorsIsBudget(out.Reason) {
		br.disabled[out.Hook] = out.Reason
	}
	obs := br.cfg.Observer
	br.mu.Unlock()
	if obs != nil {
		obs.Observe(out)
	}
	return out
}

// OnBlockAccess routes a gpu_block_access invocation.
func (br *Bridge) OnBlockAccess(d *PMMDecision) Outcome {
	return br.routeMemory("gpu_block_access", d)
}

// OnBlockActivate routes a gpu_block_activate invocation.
func (br *Bridge) OnBlockActivate(d *PMMDecision) Outcome {
	return br.routeMemory("gpu_block_activate", d)
}

func (br *Bridge) routeMemory(hook string, d *PMMDecision) Outcome {
	br.mu.Lock()
	p, off, reason := br.mem, br.off, br.disabled[hook]
	br.mu.Unlock()

	if off {
		return br.record(Outcome{Hook: hook, Fallback: true, Reason: "bridge disabled", HandlerTime: Unmeasured})
	}
	if reason != "" {
		return br.record(Outcome{Hook: hook, Fallback: true, Reason: reason, HandlerTime: Unmeasured})
	}
	if p == nil {
		return br.record(Outcome{Hook: hook, HandlerTime: Unmeasured})
	}
	fn := p.BlockAccess
	if hook == "gpu_block_activate" {
		fn = p.BlockActivate
	}
	return br.record(applyReorder(hook, fn, d, br.budget(), br.kfuncs, br.cfg.Clock))
}

// OnTaskInit routes an on_task_init invocation.
func (br *Bridge) OnTaskInit(c *TaskInitContext) (SchedAttributes, Outcome) {
	br.mu.Lock()
	p, off, reason, limits := br.sched, br.off, br.disabled["on_task_init"], br.cfg.Limits
	br.mu.Unlock()

	switch {
	case off:
		return SchedAttributes{}, br.record(Outcome{Hook: "on_task_init", Fallback: true, Reason: "bridge disabled", HandlerTime: Unmeasured})
	case reason != "":
		return SchedAttributes{}, br.record(Outcome{Hook: "on_task_init", Fallback: true, Reason: reason, HandlerTime: Unmeasured})
	case p == nil:
		return SchedAttributes{}, br.record(Outcome{Hook: "on_task_init", HandlerTime: Unmeasured})
	}
	a, out := applyTaskInit(p, c, br.budget(), br.kfuncs, limits)
	return a, br.record(out)
}

// OnBind routes an on_bind invocation.
func (br *Bridge) OnBind(c *BindContext) (BindDecision, Outcome) {
	br.mu.Lock()
	p, off, reason := br.sched, br.off, br.disabled["on_bind"]
	br.mu.Unlock()

	switch {
	case off:
		return BindDecision{}, br.record(Outcome{Hook: "on_bind", Fallback: true, Reason: "bridge disabled", HandlerTime: Unmeasured})
	case reason != "":
		return BindDecision{}, br.record(Outcome{Hook: "on_bind", Fallback: true, Reason: reason, HandlerTime: Unmeasured})
	case p == nil:
		return BindDecision{}, br.record(Outcome{Hook: "on_bind", HandlerTime: Unmeasured})
	}
	d, out := applyBind(p, c, br.budget(), br.kfuncs)
	return d, br.record(out)
}

func hookAvailable(r *Report, table, member string) bool {
	for _, h := range r.Hooks {
		if h.Table == table && h.Member == member {
			return h.Available
		}
	}
	return false
}

func hookReason(r *Report, table, member string) string {
	for _, h := range r.Hooks {
		if h.Table == table && h.Member == member {
			if h.Reason == "" {
				return "no reason recorded"
			}
			return h.Reason
		}
	}
	return "not a member of " + table
}

func kfuncReason(r *Report, name string) string {
	if k, ok := r.Kfunc(name); ok && k.Reason != "" {
		return k.Reason
	}
	return "absent"
}

func sortedLinkNames(m map[string]Link) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// errorsIsBudget reports whether a recorded reason came from the budget. The
// reason is a string by the time it reaches here because it crossed an
// Outcome, so the check is on the sentinel's text.
func errorsIsBudget(reason string) bool {
	return len(reason) >= len(ErrBudgetExceeded.Error()) &&
		reason[:len(ErrBudgetExceeded.Error())] == ErrBudgetExceeded.Error()
}
