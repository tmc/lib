package cudabpf

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// A TypeAbsentError reports that BTF does not define a type. It is the
// expected answer on every target without the out-of-tree policy module, so
// it is a distinct error rather than a generic failure: callers branch on it.
type TypeAbsentError struct{ Name string }

func (e *TypeAbsentError) Error() string { return "cudabpf: BTF has no type " + e.Name }

// A KfuncAbsentError reports that BTF does not define a kfunc.
type KfuncAbsentError struct{ Name string }

func (e *KfuncAbsentError) Error() string { return "cudabpf: BTF has no kfunc " + e.Name }

// ErrNoBTF is returned by a loader on a target with no type information.
// Without BTF nothing here can be attached, because every identifier would be
// an unverified guess.
var ErrNoBTF = errors.New("cudabpf: target has no BTF")

// Capabilities is the P0 identity of a target: enough to decide what can be
// attached, and enough that a later claim about kernel behavior can name the
// machine it was made on.
//
// The string fields are recorded verbatim from the target. An unknown one is
// empty, and empty means unknown -- it is never filled with a plausible value.
type Capabilities struct {
	Kernel       string // uname -r
	KernelConfig map[string]string
	Driver       string // e.g. "NVIDIA UNIX Open Kernel Module 580.173.02"
	GPU          string // e.g. "NVIDIA GB10"
	ComputeCap   string // e.g. "12.1"
	CUDA         string

	BTFAvailable bool
	// UnprivilegedBPFKnown distinguishes "permitted" from "not
	// determined". Without it, false would mean both.
	UnprivilegedBPFKnown   bool
	UnprivilegedBPFAllowed bool

	StructOpsTables []string
}

func (c Capabilities) String() string {
	unpriv := "unknown"
	if c.UnprivilegedBPFKnown {
		unpriv = fmt.Sprintf("%t", c.UnprivilegedBPFAllowed)
	}
	return fmt.Sprintf("kernel %s, driver %s, gpu %s (cc %s), btf %t, unprivileged bpf %s, %d struct_ops tables",
		orUnknown(c.Kernel), orUnknown(c.Driver), orUnknown(c.GPU), orUnknown(c.ComputeCap),
		c.BTFAvailable, unpriv, len(c.StructOpsTables))
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

// A HookStatus is one hook's availability on a target. Available false with a
// Reason is the normal result: these tables are not mainline, so on an
// ordinary kernel every hook is absent.
type HookStatus struct {
	Table     string
	Member    string
	Available bool
	Reason    string
}

func (h HookStatus) String() string {
	if h.Available {
		return h.Table + "." + h.Member + ": available"
	}
	return h.Table + "." + h.Member + ": absent (" + h.Reason + ")"
}

// A KfuncStatus is one kfunc's availability and, when present, the signature
// BTF reports for it. A kfunc present but unannotated cannot be called from a
// BPF program, so it is recorded as unavailable with that as the reason.
type KfuncStatus struct {
	Name      string
	Available bool
	Reason    string
	Signature KfuncSignature
}

// A TypeStatus is one context type's availability and whether its layout
// matched. Layout mismatch is not absence: the type exists and disagrees,
// which is a harder failure than not being there.
type TypeStatus struct {
	Name      string
	Available bool
	Reason    string
}

// A Report is the outcome of [Discover]: the capability matrix a handoff
// quotes and an attach path consults.
type Report struct {
	Capabilities Capabilities
	Hooks        []HookStatus
	Kfuncs       []KfuncStatus
	Types        []TypeStatus
	// LayoutError is set when a type BTF defines disagrees with the
	// layout this package requires. It is fatal for attach even when
	// every hook is present.
	LayoutError error
}

// AnyHookAvailable reports whether any hook can be attached at all.
func (r *Report) AnyHookAvailable() bool {
	for _, h := range r.Hooks {
		if h.Available {
			return true
		}
	}
	return false
}

// TableAvailable reports whether every member of a table is attachable.
func (r *Report) TableAvailable(table string) bool {
	found := false
	for _, h := range r.Hooks {
		if h.Table != table {
			continue
		}
		found = true
		if !h.Available {
			return false
		}
	}
	return found
}

// Kfunc returns the status recorded for a named kfunc.
func (r *Report) Kfunc(name string) (KfuncStatus, bool) {
	for _, k := range r.Kfuncs {
		if k.Name == name {
			return k, true
		}
	}
	return KfuncStatus{}, false
}

func (r *Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", r.Capabilities)
	if r.LayoutError != nil {
		fmt.Fprintf(&b, "layout: %v\n", r.LayoutError)
	}
	avail := 0
	for _, h := range r.Hooks {
		if h.Available {
			avail++
		}
	}
	fmt.Fprintf(&b, "hooks: %d of %d available\n", avail, len(r.Hooks))
	for _, h := range r.Hooks {
		fmt.Fprintf(&b, "  %s\n", h)
	}
	for _, k := range r.Kfuncs {
		if k.Available {
			fmt.Fprintf(&b, "  kfunc %s\n", k.Signature)
		} else {
			fmt.Fprintf(&b, "  kfunc %s: absent (%s)\n", k.Name, k.Reason)
		}
	}
	return b.String()
}

// inventoryKfuncs are the kfuncs the two tables actuate through, with the
// signature each is required to have. A kfunc BTF reports with a different
// shape is treated as a different kfunc that happens to share a name.
func inventoryKfuncs() []KfuncSignature {
	return []KfuncSignature{
		{Name: "bpf_gpu_set_prefetch_region", Params: []string{"uvm_bpf_prefetch_decision_t*", "u64", "u64"}, Returns: "int"},
		{Name: "bpf_gpu_request_reorder", Params: []string{"uvm_bpf_pmm_decision_ctx_t*", "u64", "u64"}, Returns: "int"},
		{Name: "bpf_gpu_migrate_range", Params: []string{"u64", "u64", "u64"}, Returns: "int"},
		{Name: "bpf_gpu_strstr", Params: []string{"char*", "u32", "char*", "u32"}, Returns: "int"},
		{Name: "bpf_nv_gpu_set_timeslice", Params: []string{"nv_gpu_task_init_ctx*", "u64"}, Returns: "int"},
		{Name: "bpf_nv_gpu_set_interleave", Params: []string{"nv_gpu_task_init_ctx*", "u32"}, Returns: "int"},
		{Name: "bpf_nv_gpu_override_timeslice", Params: []string{"nv_gpu_timeslice_control_ctx*", "u64"}, Returns: "int"},
		{Name: "bpf_nv_gpu_preempt_tsg", Params: []string{"u32", "u32"}, Returns: "int"},
		{Name: "bpf_nv_gpu_reject_bind", Params: []string{"nv_gpu_bind_ctx*"}, Returns: "void"},
	}
}

// Discover records what a target actually provides. It never reports a hook
// as available on the strength of an identifier appearing in a header: a hook
// is available only when BTF defines its table, its context types match the
// required layout, and the kfuncs it actuates through are present and
// annotated.
//
// Discovery is not an error when nothing is available. A target with no
// policy module returns a report in which every hook is absent, which is the
// expected result and the one an unpatched kernel must produce.
func Discover(ctx context.Context, l Loader) (*Report, error) {
	if l == nil {
		return nil, fmt.Errorf("cudabpf: nil loader")
	}
	caps, err := l.Capabilities(ctx)
	if err != nil {
		return nil, fmt.Errorf("cudabpf: reading capabilities: %w", err)
	}
	r := &Report{Capabilities: caps}

	b, err := l.BTF(ctx)
	if errors.Is(err, ErrNoBTF) || (err == nil && b == nil) {
		r.Capabilities.BTFAvailable = false
		r.Hooks = allAbsent("no BTF on target")
		for _, k := range inventoryKfuncs() {
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{Name: k.Name, Reason: "no BTF on target"})
		}
		for _, t := range InventoryLayouts() {
			r.Types = append(r.Types, TypeStatus{Name: t.Name, Reason: "no BTF on target"})
		}
		return r, nil
	}
	if err != nil {
		return nil, fmt.Errorf("cudabpf: reading BTF: %w", err)
	}
	r.Capabilities.BTFAvailable = true

	tables, err := b.StructOpsTables()
	if err != nil {
		return nil, fmt.Errorf("cudabpf: reading struct_ops registration: %w", err)
	}
	r.Capabilities.StructOpsTables = append([]string(nil), tables...)

	// Types first: a table whose context types are absent or disagree
	// cannot be attached however present the table itself is.
	absentTypes := make(map[string]bool)
	for _, want := range InventoryLayouts() {
		_, terr := b.TypeLayout(want.Name)
		var missing *TypeAbsentError
		switch {
		case errors.As(terr, &missing):
			absentTypes[want.Name] = true
			r.Types = append(r.Types, TypeStatus{Name: want.Name, Reason: "absent from BTF"})
		case terr != nil:
			absentTypes[want.Name] = true
			r.Types = append(r.Types, TypeStatus{Name: want.Name, Reason: terr.Error()})
		default:
			r.Types = append(r.Types, TypeStatus{Name: want.Name, Available: true})
		}
	}
	if _, err := CheckLayouts(b); err != nil {
		var mm *LayoutMismatch
		if errors.As(err, &mm) {
			r.LayoutError = err
			absentTypes[mm.Type] = true
			for i := range r.Types {
				if r.Types[i].Name == mm.Type {
					r.Types[i].Available = false
					r.Types[i].Reason = mm.Error()
				}
			}
		} else {
			return nil, err
		}
	}

	// Kfuncs.
	haveKfunc := make(map[string]bool)
	for _, want := range inventoryKfuncs() {
		got, kerr := b.KfuncSignature(want.Name)
		var missing *KfuncAbsentError
		switch {
		case errors.As(kerr, &missing):
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{Name: want.Name, Reason: "absent from BTF"})
		case kerr != nil:
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{Name: want.Name, Reason: kerr.Error()})
		case !got.Annotated:
			// Present in BTF but not registered as callable. A program
			// that calls it is rejected by the verifier, so treating it
			// as available would move the failure into the load path.
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{
				Name: want.Name, Reason: "not BTF-annotated as a kfunc", Signature: got,
			})
		case !sameSignature(want, got):
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{
				Name:      want.Name,
				Reason:    fmt.Sprintf("signature is %s, want %s", got, want),
				Signature: got,
			})
		default:
			haveKfunc[want.Name] = true
			r.Kfuncs = append(r.Kfuncs, KfuncStatus{Name: want.Name, Available: true, Signature: got})
		}
	}

	// Tables and their members.
	registered := make(map[string]bool, len(r.Capabilities.StructOpsTables))
	for _, t := range r.Capabilities.StructOpsTables {
		registered[t] = true
	}
	for _, h := range allHooks() {
		st := HookStatus{Table: h.Table, Member: h.Member}
		switch {
		case !registered[h.Table]:
			st.Reason = "struct_ops table " + h.Table + " is not registered"
		case len(missingOf(h.Types, absentTypes)) > 0:
			st.Reason = "context types absent or mismatched: " + strings.Join(missingOf(h.Types, absentTypes), ", ")
		case len(missingKfuncs(h.Kfuncs, haveKfunc)) > 0:
			st.Reason = "kfuncs unavailable: " + strings.Join(missingKfuncs(h.Kfuncs, haveKfunc), ", ")
		default:
			if err := b.ValidateStructOpsMember(h.Table, h.Member); err != nil {
				st.Reason = "callback validation: " + err.Error()
			} else {
				st.Available = true
			}
		}
		r.Hooks = append(r.Hooks, st)
	}
	return r, nil
}

// hookRequirement is what one member needs in order to be attachable.
type hookRequirement struct {
	Table  string
	Member string
	Types  []string
	Kfuncs []string
}

func allHooks() []hookRequirement {
	return []hookRequirement{
		{MemOpsTable, "gpu_test_trigger", nil, nil},
		{MemOpsTable, "gpu_page_prefetch", []string{"uvm_bpf_prefetch_decision_t"}, []string{"bpf_gpu_set_prefetch_region"}},
		{MemOpsTable, "gpu_page_prefetch_iter", []string{"uvm_bpf_prefetch_decision_t"}, []string{"bpf_gpu_set_prefetch_region"}},
		{MemOpsTable, "gpu_block_activate", []string{"uvm_bpf_pmm_decision_ctx_t", "nv_gpu_pmm_request_t"}, []string{"bpf_gpu_request_reorder"}},
		{MemOpsTable, "gpu_block_access", []string{"uvm_bpf_pmm_decision_ctx_t", "nv_gpu_pmm_request_t"}, []string{"bpf_gpu_request_reorder"}},
		{MemOpsTable, "gpu_evict_prepare", []string{"uvm_bpf_pmm_decision_ctx_t"}, nil},
		{SchedOpsTable, "on_task_init", []string{"nv_gpu_task_init_ctx"}, []string{"bpf_nv_gpu_set_timeslice", "bpf_nv_gpu_set_interleave"}},
		{SchedOpsTable, "on_bind", []string{"nv_gpu_bind_ctx"}, []string{"bpf_nv_gpu_reject_bind"}},
		{SchedOpsTable, "on_task_destroy", []string{"nv_gpu_task_destroy_ctx"}, nil},
		{SchedOpsTable, "on_timeslice_control", []string{"nv_gpu_timeslice_control_ctx"}, []string{"bpf_nv_gpu_override_timeslice"}},
	}
}

func allAbsent(reason string) []HookStatus {
	hooks := allHooks()
	out := make([]HookStatus, 0, len(hooks))
	for _, h := range hooks {
		out = append(out, HookStatus{Table: h.Table, Member: h.Member, Reason: reason})
	}
	return out
}

func missingOf(want []string, absent map[string]bool) []string {
	var out []string
	for _, t := range want {
		if absent[t] {
			out = append(out, t)
		}
	}
	sort.Strings(out)
	return out
}

func missingKfuncs(want []string, have map[string]bool) []string {
	var out []string
	for _, k := range want {
		if !have[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func sameSignature(want, got KfuncSignature) bool {
	if want.Name != got.Name || want.Returns != got.Returns || len(want.Params) != len(got.Params) {
		return false
	}
	for i := range want.Params {
		if want.Params[i] != got.Params[i] {
			return false
		}
	}
	return true
}
