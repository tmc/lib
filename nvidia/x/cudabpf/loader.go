package cudabpf

import (
	"context"
	"fmt"
	"sort"
)

// The module boundary. Everything below is an interface because the
// implementation is generic Linux plumbing -- bpf(2), BTF parsing, map and
// program objects, struct_ops registration, link lifetime -- and none of it is
// NVIDIA-specific. It lives in a sibling module that this package must not
// import, so that the policy layer builds and tests on a host with no kernel.

// A BTF is a target's type information. It answers questions about the types
// and kfuncs a driver actually exports, and it is the only authority: an
// identifier from a header, a proposal or [InventoryLayouts] is a hypothesis
// until BTF confirms it.
type BTF interface {
	// TypeLayout returns the size and member offsets of a named type.
	// A type the target does not define is reported with a
	// [TypeAbsentError], not a zero layout.
	TypeLayout(name string) (TypeLayout, error)

	// KfuncSignature returns the signature of a named BPF kfunc. A kfunc
	// the target does not define is reported with a [KfuncAbsentError].
	KfuncSignature(name string) (KfuncSignature, error)

	// StructOpsTables lists the struct_ops table types the kernel has
	// registered. On a target with no GPU policy module this contains the
	// ordinary kernel tables and neither of the two this package wants.
	StructOpsTables() ([]string, error)

	// ValidateStructOpsMember checks that the registered table contains the
	// requested callback and its prototype matches the driver ABI. A BTF
	// struct name alone does not establish registration or verifier support.
	ValidateStructOpsMember(table, member string) error
}

// A KfuncSignature is a kfunc's name and parameter types as BTF reports them.
// Params holds BTF type names in declaration order; a pointer parameter is
// spelled with a trailing "*", matching how the driver's headers declare it.
type KfuncSignature struct {
	Name      string
	Params    []string
	Returns   string
	Annotated bool // BTF_ID_FLAGS present, i.e. callable from a BPF program
}

func (s KfuncSignature) String() string {
	return fmt.Sprintf("%s(%s) %s", s.Name, joinComma(s.Params), s.Returns)
}

// A Loader loads and attaches BPF programs. It is satisfied by a consumer in
// a module that may depend on libbpf or cilium/ebpf; nothing here does.
type Loader interface {
	// Capabilities reports what the target provides: kernel, driver and
	// GPU identity, BTF presence, registered struct_ops tables, and
	// whether unprivileged BPF is permitted.
	Capabilities(ctx context.Context) (Capabilities, error)

	// BTF returns the target's type information, or an error if the
	// target has none.
	BTF(ctx context.Context) (BTF, error)

	// Load loads the programs and maps of one object without attaching
	// anything.
	Load(ctx context.Context, spec ObjectSpec) (Object, error)
}

// An Object is a loaded but unattached set of programs and maps.
type Object interface {
	// AttachStructOps registers a struct_ops map against a table and
	// returns a link whose lifetime owns the attachment.
	AttachStructOps(ctx context.Context, spec StructOpsSpec) (Link, error)

	// Close releases the programs and maps. Closing an object does not
	// detach links taken from it; a link outlives its object.
	Close() error
}

// A Link is one live attachment. Detaching is the only way to stop a policy
// from running, and it must always be available: a bridge that cannot detach
// cannot honor its disable switch.
type Link interface {
	// Generation is the driver generation the link was made against. A
	// change means the tables were re-registered and the link is stale.
	Generation() uint64

	// Detach removes the attachment. Detach is idempotent.
	Detach() error
}

// A Kfuncs actuates a policy decision. On a real target these are BPF kfunc
// calls made from inside a handler; the interface exists so the decision logic
// can be exercised with no kernel, and so an unannotated or absent kfunc is a
// refusal at one place rather than a silent no-op at many.
type Kfuncs interface {
	// RequestReorder is bpf_gpu_request_reorder. It submits destination
	// and position; it does not report whether the driver obeyed them.
	// The answer, such as it is, arrives in ctx.Request.Attempted and
	// ctx.Request.Conflict, which the driver writes.
	RequestReorder(ctx *PMMDecision, dst Destination, pos Position) error

	// SetPrefetchRegion is bpf_gpu_set_prefetch_region.
	SetPrefetchRegion(d *PrefetchDecision, first, outer uint64) error

	// MigrateRange is bpf_gpu_migrate_range.
	MigrateRange(vaSpace, addr, length uint64) error

	// SetTimeslice is bpf_nv_gpu_set_timeslice.
	SetTimeslice(ctx *TaskInitContext, us uint64) error

	// SetInterleave is bpf_nv_gpu_set_interleave.
	SetInterleave(ctx *TaskInitContext, level InterleaveLevel) error

	// RejectBind is bpf_nv_gpu_reject_bind. It is non-sleepable and its
	// whole effect is to clear ctx.Allow; see [BindRejection] for why the
	// call form and not the field write is the supported path.
	RejectBind(ctx *BindContext) error
}

// A ProgramSpec is one BPF program within an object.
//
// Section and Member are separate fields and are never derived from one
// another. In a struct_ops object the member a program binds to comes from the
// .struct_ops map initializer, not from the text after "struct_ops/" in the
// section name. The two agree for the memory table and disagree for at least
// one real scheduler object, where programs named gp_task_init, gp_bind,
// gp_destroy and gp_timeslice_control carry sections of those names while
// binding to members on_task_init, on_bind, on_task_destroy and
// on_timeslice_control. A loader that derives the member from the section
// attaches the wrong slot or none at all.
type ProgramSpec struct {
	Name    string // symbol name of the program
	Section string // SEC() string, informational
	Member  string // struct_ops member this program binds to
}

// A StructOpsSpec is a request to register one struct_ops map against one
// table. Programs maps member name to program name, reproducing the .struct_ops
// initializer.
type StructOpsSpec struct {
	Table    string // BTF name of the table type, e.g. "gpu_mem_ops"
	MapName  string
	Programs map[string]string // member -> program name
}

// An ObjectSpec is a request to load one object.
type ObjectSpec struct {
	Name     string
	Programs []ProgramSpec
	// Bytes is the compiled object. It is opaque here: authoring and
	// compiling programs is this lane's work, but the encoding of an ELF
	// is the loader's.
	Bytes []byte
}

// program returns the spec for a named program.
func (o ObjectSpec) program(name string) (ProgramSpec, bool) {
	for _, p := range o.Programs {
		if p.Name == name {
			return p, true
		}
	}
	return ProgramSpec{}, false
}

// The two tables this package knows how to drive, and their members in
// declaration order. Member order is the slot order the kernel assigns, so it
// is part of the contract and not a presentation choice.
const (
	MemOpsTable   = "gpu_mem_ops"
	SchedOpsTable = "nv_gpu_sched_ops"
)

// MemOpsMembers are the six members of struct gpu_mem_ops.
func MemOpsMembers() []string {
	return []string{
		"gpu_test_trigger",
		"gpu_page_prefetch",
		"gpu_page_prefetch_iter",
		"gpu_block_activate",
		"gpu_block_access",
		"gpu_evict_prepare",
	}
}

// SchedOpsMembers are the four members of struct nv_gpu_sched_ops.
func SchedOpsMembers() []string {
	return []string{"on_task_init", "on_bind", "on_task_destroy", "on_timeslice_control"}
}

// Validate checks a struct_ops request against the table it names before any
// of it reaches the kernel: the table must be one this package knows, every
// member must exist on it, and every named program must exist in the object.
//
// It deliberately does not check that Section agrees with Member. They are
// allowed to disagree, and a validator that required agreement would reject a
// correct scheduler object.
func (s StructOpsSpec) Validate(obj ObjectSpec) error {
	if s.MapName == "" {
		return fmt.Errorf("cudabpf: empty struct_ops map name")
	}
	seen := make(map[string]bool)
	for _, p := range obj.Programs {
		if p.Name == "" || seen[p.Name] {
			return fmt.Errorf("cudabpf: empty or duplicate program name %q", p.Name)
		}
		seen[p.Name] = true
	}
	var members []string
	switch s.Table {
	case MemOpsTable:
		members = MemOpsMembers()
	case SchedOpsTable:
		members = SchedOpsMembers()
	default:
		return fmt.Errorf("cudabpf: unknown struct_ops table %q", s.Table)
	}
	known := make(map[string]bool, len(members))
	for _, m := range members {
		known[m] = true
	}
	if len(s.Programs) == 0 {
		return fmt.Errorf("cudabpf: struct_ops %q names no programs", s.Table)
	}
	for _, m := range sortedKeys(s.Programs) {
		if !known[m] {
			return fmt.Errorf("cudabpf: %s has no member %q", s.Table, m)
		}
		name := s.Programs[m]
		if p, ok := obj.program(name); !ok {
			return fmt.Errorf("cudabpf: %s.%s names program %q, which the object does not define",
				s.Table, m, name)
		} else if p.Member != m {
			return fmt.Errorf("cudabpf: program %q binds %q, not %q", name, p.Member, m)
		}
	}
	return nil
}

// RequiredKfuncs returns the kfuncs a struct_ops request will actuate through,
// so they can be checked against BTF before the object is loaded rather than
// discovered missing inside a fault handler.
func (s StructOpsSpec) RequiredKfuncs() []string {
	var out []string
	for _, m := range sortedKeys(s.Programs) {
		switch m {
		case "gpu_block_access", "gpu_block_activate":
			out = append(out, "bpf_gpu_request_reorder")
		case "gpu_page_prefetch", "gpu_page_prefetch_iter":
			out = append(out, "bpf_gpu_set_prefetch_region")
		case "on_task_init":
			out = append(out, "bpf_nv_gpu_set_timeslice", "bpf_nv_gpu_set_interleave")
		case "on_bind":
			out = append(out, "bpf_nv_gpu_reject_bind")
		case "on_timeslice_control":
			out = append(out, "bpf_nv_gpu_override_timeslice")
		}
	}
	return dedupe(out)
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func dedupe(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := in[:0]
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func joinComma(s []string) string {
	out := ""
	for i, p := range s {
		if i > 0 {
			out += ", "
		}
		out += p
	}
	return out
}
