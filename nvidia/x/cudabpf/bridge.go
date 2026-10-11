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
	// MaxOutcomes bounds retained host observations; zero uses 256.
	MaxOutcomes int
	// Clock measures handler time. Nil means handler time is
	// [Unmeasured] rather than zero.
	Clock func() time.Time
}

// A Bridge owns compiled-object attachments and evaluates host policy models.
// SetMemoryPolicy and SetSchedPolicy do not compile Go callbacks into BPF or
// alter the supplied object. On* methods are explicit host-model invocations;
// kernel hooks run the compiled object programs, not these Go methods.
//
// The bridge never takes eviction authority from the driver. Every path
// through it ends either in a submitted request the driver may ignore, or in
// the driver's own behavior with a recorded reason. There is no path that
// leaves a placement undefined.
type Bridge struct {
	cfg    Config
	loader Loader
	kfuncs Kfuncs

	lifeMu   sync.Mutex // serializes attachment ownership and disable
	mu       sync.Mutex
	report   *Report
	objects  map[string]Object
	links    map[string]Link // table -> link
	disabled map[string]string
	off      bool
	mem      MemoryPolicy
	sched    SchedPolicy
	outcomes []Outcome
}

// NewBridge returns an attachment owner and host-model evaluator. Policies
// are used only by the On* methods. Attach loads the supplied compiled object
// regardless of which host policies are installed.
func NewBridge(l Loader, k Kfuncs, cfg Config) *Bridge {
	if cfg.Limits == (Limits{}) {
		cfg.Limits = DefaultLimits()
	}
	return &Bridge{
		cfg:      cfg,
		loader:   l,
		kfuncs:   k,
		links:    make(map[string]Link),
		objects:  make(map[string]Object),
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
	br.report = cloneReport(r)
	br.mu.Unlock()
	return r, nil
}

// Report returns the cached discovery report, or nil.
func (br *Bridge) Report() *Report {
	br.mu.Lock()
	defer br.mu.Unlock()
	return cloneReport(br.report)
}

// Attach loads an object and registers one struct_ops map.
//
// Attaching a table that is already attached is not an error and does not
// register a second map: the existing link is returned. Two processes racing
// to attach is a different case, and one the kernel arbitrates -- the second
// registration fails at the loader and the error is returned unwrapped.
func (br *Bridge) Attach(ctx context.Context, obj ObjectSpec, spec StructOpsSpec) (Link, error) {
	br.lifeMu.Lock()
	defer br.lifeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
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
	if br.objects[spec.Table] != nil {
		br.mu.Unlock()
		return nil, fmt.Errorf("cudabpf: cleanup pending for %s", spec.Table)
	}
	report := (*Report)(nil)
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
	if o == nil {
		return nil, fmt.Errorf("cudabpf: loader returned nil object")
	}
	link, err := o.AttachStructOps(ctx, spec)
	if err != nil {
		cerr := o.Close()
		if cerr != nil {
			br.mu.Lock()
			br.objects[spec.Table] = o
			br.mu.Unlock()
		}
		return nil, errors.Join(fmt.Errorf("cudabpf: attaching %s: %w", spec.Table, err), cerr)
	}

	if link == nil {
		cerr := o.Close()
		if cerr != nil {
			br.mu.Lock()
			br.objects[spec.Table] = o
			br.mu.Unlock()
		}
		return nil, errors.Join(fmt.Errorf("cudabpf: loader returned nil link"), cerr)
	}
	br.mu.Lock()
	br.links[spec.Table] = link
	br.objects[spec.Table] = o
	br.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, br.detach(spec.Table))
	}
	return link, nil
}

// Detach removes one table's attachment. Detaching what is not attached
// succeeds: a disable switch must work whatever state it finds.
func (br *Bridge) Detach(table string) error {
	br.lifeMu.Lock()
	defer br.lifeMu.Unlock()
	return br.detach(table)
}

func (br *Bridge) detach(table string) error {
	br.mu.Lock()
	link, obj := br.links[table], br.objects[table]
	br.mu.Unlock()
	if link != nil {
		if err := link.Detach(); err != nil {
			return err
		}
		br.mu.Lock()
		delete(br.links, table)
		br.mu.Unlock()
	}
	if obj != nil {
		if err := obj.Close(); err != nil {
			return err
		}
		br.mu.Lock()
		delete(br.objects, table)
		br.mu.Unlock()
	}
	return nil
}

// Close disables the bridge and detaches all owned links. Failed detach or
// close operations retain ownership so a later Close can retry cleanup.
func (br *Bridge) Close() error {
	br.lifeMu.Lock()
	defer br.lifeMu.Unlock()
	br.mu.Lock()
	br.off = true
	names := make(map[string]Link)
	for name := range br.objects {
		names[name] = br.links[name]
	}
	br.mu.Unlock()
	var errs []error
	for _, name := range sortedLinkNames(names) {
		if err := br.detach(name); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Reattach re-registers a table whose link is stale because the driver
// re-registered its tables. A generation change invalidates the link but not
// the policy, so the old link is detached and a new one taken.
func (br *Bridge) Reattach(ctx context.Context, obj ObjectSpec, spec StructOpsSpec, generation uint64) (Link, error) {
	if err := spec.Validate(obj); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
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
	fresh, err := br.Attach(ctx, obj, spec)
	if err != nil {
		return nil, err
	}
	if fresh.Generation() != generation {
		return nil, errors.Join(fmt.Errorf("cudabpf: driver generation is %d, want %d", fresh.Generation(), generation), br.Detach(spec.Table))
	}
	return fresh, nil
}

// DisableHook switches one hook off with a recorded reason. The driver's
// default behavior takes over immediately; the attachment is left in place so
// the hook can be re-enabled without a reload.
func (br *Bridge) DisableHook(member, reason string) {
	br.mu.Lock()
	defer br.mu.Unlock()
	if reason == "" {
		reason = "operator disabled hook"
	}
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

// Disable switches host-model evaluation and new attachment requests off.
// Existing kernel links remain attached until Detach or Close.
func (br *Bridge) Disable() {
	br.lifeMu.Lock()
	defer br.lifeMu.Unlock()
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
	limit := br.cfg.MaxOutcomes
	if limit <= 0 {
		limit = 256
	}
	if len(br.outcomes) == limit {
		copy(br.outcomes, br.outcomes[1:])
		br.outcomes = br.outcomes[:limit-1]
	}
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

func cloneReport(r *Report) *Report {
	if r == nil {
		return nil
	}
	c := *r
	c.Hooks = append([]HookStatus(nil), r.Hooks...)
	c.Types = append([]TypeStatus(nil), r.Types...)
	c.Kfuncs = append([]KfuncStatus(nil), r.Kfuncs...)
	for i := range c.Kfuncs {
		c.Kfuncs[i].Signature.Params = append([]string(nil), r.Kfuncs[i].Signature.Params...)
	}
	c.Capabilities.StructOpsTables = append([]string(nil), r.Capabilities.StructOpsTables...)
	c.Capabilities.KernelConfig = make(map[string]string, len(r.Capabilities.KernelConfig))
	for k, v := range r.Capabilities.KernelConfig {
		c.Capabilities.KernelConfig[k] = v
	}
	return &c
}
