package cudabpf

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"unsafe"
)

// The hook context structs. Field order, types and padding reproduce the C
// declarations in the patched driver's headers; the layout is asserted rather
// than assumed, both against the Go compiler's own arrangement (see
// [GoLayouts]) and against the target's BTF (see [CheckLayouts]).
//
// Pointer-valued members are carried as uint64 rather than unsafe.Pointer.
// Nothing in this package dereferences them: they identify a kernel object to
// the kfunc that is passed the whole context back, and a Go pointer type would
// invite a dereference the boundaries forbid.

// PrefetchDecision is uvm_bpf_prefetch_decision_t. Attempted and Conflict are
// the driver's own record of whether a decision was submitted and whether it
// collided with another; a handler reads them and must not write them.
type PrefetchDecision struct {
	Attempted uint8
	Conflict  uint8
	First     uint64
	Outer     uint64
}

// PMMSnapshot is nv_gpu_pmm_snapshot_t, the driver's view of a root chunk at
// the moment the hook fired.
type PMMSnapshot struct {
	OwnerID    uint64
	RootID     uint64
	Generation uint64
	Source     uint32
}

// PMMRequest is nv_gpu_pmm_request_t. Destination and Position carry the
// policy's request; Attempted and Conflict are the driver's answer to it.
type PMMRequest struct {
	Attempted   uint8
	Conflict    uint8
	Destination uint64
	Position    uint64
}

// PMMDecision is uvm_bpf_pmm_decision_ctx_t, the context passed to
// gpu_block_access and gpu_block_activate.
type PMMDecision struct {
	PMM       uint64
	RootChunk uint64
	Observed  PMMSnapshot
	Request   PMMRequest
}

// TaskInitContext is struct nv_gpu_task_init_ctx.
type TaskInitContext struct {
	TSGID             uint64
	EngineType        uint32
	DefaultTimeslice  uint64
	DefaultInterleave uint32
	RunlistID         uint32
}

// BindContext is struct nv_gpu_bind_ctx. Allow is written by the driver in
// response to bpf_nv_gpu_reject_bind; see [BindRejection].
type BindContext struct {
	TSGID           uint64
	RunlistID       uint32
	ChannelCount    uint32
	TimesliceUS     uint64
	InterleaveLevel uint32
	Allow           uint32
}

// TaskDestroyContext is struct nv_gpu_task_destroy_ctx.
type TaskDestroyContext struct {
	TSGID      uint64
	RunlistID  uint32
	EngineType uint32
}

// TimesliceControlContext is struct nv_gpu_timeslice_control_ctx.
type TimesliceControlContext struct {
	TSGID                uint64
	RequestedTimesliceUS uint64
	EngineType           uint32
	RunlistID            uint32
	HClient              uint32
	HTSG                 uint32
	GPUInstance          uint32
	Phase                uint32
}

// Destination and Position are the values a reordering policy writes into
// PMMRequest. Head and tail are values here, not calls: there is no
// bpf_gpu_move_head or bpf_gpu_move_tail, and a strict insert-after ordering
// is not expressible against this interface at all.
type (
	// Destination selects which of the driver's two chunk lists the request
	// names.
	Destination uint64
	// Position selects an end of that list. Only the two ends are
	// addressable; an arbitrary rank is not.
	Position uint64
)

// Destination and Position values, as enum nv_gpu_pmm_destination and
// enum nv_gpu_pmm_position. Zero is not a valid value of either, which is
// what makes an unset request distinguishable from a request for the head.
const (
	DestinationUnset  Destination = 0
	DestinationUsed   Destination = 1
	DestinationUnused Destination = 2

	PositionUnset Position = 0
	PositionHead  Position = 1
	PositionTail  Position = 2
)

func (d Destination) String() string {
	switch d {
	case DestinationUsed:
		return "used"
	case DestinationUnused:
		return "unused"
	case DestinationUnset:
		return "unset"
	}
	return fmt.Sprintf("Destination(%d)", uint64(d))
}

func (p Position) String() string {
	switch p {
	case PositionHead:
		return "head"
	case PositionTail:
		return "tail"
	case PositionUnset:
		return "unset"
	}
	return fmt.Sprintf("Position(%d)", uint64(p))
}

// Valid reports whether d names one of the driver's lists.
func (d Destination) Valid() bool { return d == DestinationUsed || d == DestinationUnused }

// Valid reports whether p names one of the two addressable ends.
func (p Position) Valid() bool { return p == PositionHead || p == PositionTail }

// Engine types, as NV_ENGINE_TYPE_*.
const (
	EngineGraphics uint32 = 0
	EngineCopy     uint32 = 1
	EngineNVDEC    uint32 = 2
	EngineNVENC    uint32 = 3
	EngineNVJPEG   uint32 = 4
)

// InterleaveLevel is a runlist interleave level, as NV_INTERLEAVE_LEVEL_*.
type InterleaveLevel uint32

// The interleave levels the driver defines. There are exactly three; a value
// outside them is rejected by [SchedAttributes.Validate] rather than clamped.
const (
	InterleaveLow    InterleaveLevel = 0
	InterleaveMedium InterleaveLevel = 1
	InterleaveHigh   InterleaveLevel = 2
)

func (l InterleaveLevel) String() string {
	switch l {
	case InterleaveLow:
		return "low"
	case InterleaveMedium:
		return "medium"
	case InterleaveHigh:
		return "high"
	}
	return fmt.Sprintf("InterleaveLevel(%d)", uint32(l))
}

// Valid reports whether l is one of the three defined levels.
func (l InterleaveLevel) Valid() bool { return l <= InterleaveHigh }

// FieldLayout is the offset of one named member within a type.
type FieldLayout struct {
	Name   string
	Offset uintptr
}

// TypeLayout is the size of a context type and the offsets of the members
// whose placement is load-bearing. It is deliberately not a full description:
// only members the driver and the policy layer both address are listed, since
// those are the ones a drift would break.
type TypeLayout struct {
	Name   string
	Size   uintptr
	Fields []FieldLayout
}

func (t TypeLayout) String() string {
	parts := make([]string, 0, len(t.Fields))
	for _, f := range t.Fields {
		parts = append(parts, fmt.Sprintf("%s@%d", f.Name, f.Offset))
	}
	return fmt.Sprintf("%s{size %d; %s}", t.Name, t.Size, strings.Join(parts, " "))
}

// field returns the offset of the named member, and whether it is listed.
func (t TypeLayout) field(name string) (uintptr, bool) {
	for _, f := range t.Fields {
		if f.Name == name {
			return f.Offset, true
		}
	}
	return 0, false
}

// InventoryLayouts returns the layout every context type is required to have,
// transcribed from the _Static_assert declarations in the driver's own
// headers. These are constants in the source sense -- they are what the C side
// enforces at compile time -- and they are held separately from [GoLayouts] on
// purpose: a Go struct edited until the two agree has been edited until the
// test passes, so the numbers a test compares against must not be derived from
// the structs under test.
func InventoryLayouts() []TypeLayout {
	return []TypeLayout{
		{Name: "uvm_bpf_prefetch_decision_t", Size: 24, Fields: []FieldLayout{
			{"attempted", 0}, {"conflict", 1}, {"first", 8}, {"outer", 16},
		}},
		{Name: "uvm_bpf_pmm_decision_ctx_t", Size: 72, Fields: []FieldLayout{
			{"pmm", 0}, {"root_chunk", 8}, {"observed", 16}, {"request", 48},
		}},
		{Name: "nv_gpu_pmm_snapshot_t", Size: 32, Fields: []FieldLayout{
			{"owner_id", 0}, {"root_id", 8}, {"generation", 16}, {"source", 24},
		}},
		{Name: "nv_gpu_pmm_request_t", Size: 24, Fields: []FieldLayout{
			{"attempted", 0}, {"conflict", 1}, {"destination", 8}, {"position", 16},
		}},
		{Name: "nv_gpu_task_init_ctx", Size: 32, Fields: []FieldLayout{
			{"tsg_id", 0}, {"engine_type", 8}, {"default_timeslice", 16},
			{"default_interleave", 24}, {"runlist_id", 28},
		}},
		{Name: "nv_gpu_bind_ctx", Size: 32, Fields: []FieldLayout{
			{"tsg_id", 0}, {"runlist_id", 8}, {"channel_count", 12},
			{"timeslice_us", 16}, {"interleave_level", 24}, {"allow", 28},
		}},
		{Name: "nv_gpu_task_destroy_ctx", Size: 16, Fields: []FieldLayout{
			{"tsg_id", 0}, {"runlist_id", 8}, {"engine_type", 12},
		}},
		{Name: "nv_gpu_timeslice_control_ctx", Size: 40, Fields: []FieldLayout{
			{"tsg_id", 0}, {"requested_timeslice_us", 8}, {"engine_type", 16},
			{"runlist_id", 20}, {"hclient", 24}, {"htsg", 28},
			{"gpu_instance", 32}, {"phase", 36},
		}},
	}
}

// goType maps each C type name to the Go declaration that stands in for it,
// and each C member name to the Go field that carries it.
var goTypes = map[string]struct {
	typ     reflect.Type
	members map[string]string
}{
	"uvm_bpf_prefetch_decision_t": {reflect.TypeOf(PrefetchDecision{}), map[string]string{
		"attempted": "Attempted", "conflict": "Conflict", "first": "First", "outer": "Outer",
	}},
	"uvm_bpf_pmm_decision_ctx_t": {reflect.TypeOf(PMMDecision{}), map[string]string{
		"pmm": "PMM", "root_chunk": "RootChunk", "observed": "Observed", "request": "Request",
	}},
	"nv_gpu_pmm_snapshot_t": {reflect.TypeOf(PMMSnapshot{}), map[string]string{
		"owner_id": "OwnerID", "root_id": "RootID", "generation": "Generation", "source": "Source",
	}},
	"nv_gpu_pmm_request_t": {reflect.TypeOf(PMMRequest{}), map[string]string{
		"attempted": "Attempted", "conflict": "Conflict",
		"destination": "Destination", "position": "Position",
	}},
	"nv_gpu_task_init_ctx": {reflect.TypeOf(TaskInitContext{}), map[string]string{
		"tsg_id": "TSGID", "engine_type": "EngineType", "default_timeslice": "DefaultTimeslice",
		"default_interleave": "DefaultInterleave", "runlist_id": "RunlistID",
	}},
	"nv_gpu_bind_ctx": {reflect.TypeOf(BindContext{}), map[string]string{
		"tsg_id": "TSGID", "runlist_id": "RunlistID", "channel_count": "ChannelCount",
		"timeslice_us": "TimesliceUS", "interleave_level": "InterleaveLevel", "allow": "Allow",
	}},
	"nv_gpu_task_destroy_ctx": {reflect.TypeOf(TaskDestroyContext{}), map[string]string{
		"tsg_id": "TSGID", "runlist_id": "RunlistID", "engine_type": "EngineType",
	}},
	"nv_gpu_timeslice_control_ctx": {reflect.TypeOf(TimesliceControlContext{}), map[string]string{
		"tsg_id": "TSGID", "requested_timeslice_us": "RequestedTimesliceUS",
		"engine_type": "EngineType", "runlist_id": "RunlistID", "hclient": "HClient",
		"htsg": "HTSG", "gpu_instance": "GPUInstance", "phase": "Phase",
	}},
}

// GoLayouts returns the layout the Go declarations in this package actually
// have, as the compiler arranged them. Compare it with [InventoryLayouts] to
// find a Go struct that has drifted from the C one, and with a target's BTF to
// find a driver that has drifted from both.
func GoLayouts() []TypeLayout {
	names := make([]string, 0, len(goTypes))
	for name := range goTypes {
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]TypeLayout, 0, len(names))
	for _, name := range names {
		g := goTypes[name]
		l := TypeLayout{Name: name, Size: g.typ.Size()}
		for c, goName := range g.members {
			f, ok := g.typ.FieldByName(goName)
			if !ok {
				panic("cudabpf: no Go field " + goName + " on " + name)
			}
			l.Fields = append(l.Fields, FieldLayout{Name: c, Offset: f.Offset})
		}
		sort.Slice(l.Fields, func(i, j int) bool { return l.Fields[i].Offset < l.Fields[j].Offset })
		out = append(out, l)
	}
	return out
}

// A LayoutMismatch is a disagreement between a layout this package requires
// and one a target reports. It is an error, never a warning: a context struct
// read at the wrong offset yields a plausible number from the wrong field, and
// nothing downstream can tell that it did.
type LayoutMismatch struct {
	Type   string
	Member string // empty when the size itself disagrees
	Want   uintptr
	Got    uintptr
	Source string // where Got came from, e.g. "BTF"
}

func (e *LayoutMismatch) Error() string {
	if e.Member == "" {
		return fmt.Sprintf("cudabpf: %s size is %d in %s, want %d", e.Type, e.Got, e.Source, e.Want)
	}
	return fmt.Sprintf("cudabpf: %s.%s is at offset %d in %s, want %d",
		e.Type, e.Member, e.Got, e.Source, e.Want)
}

// CheckLayouts compares every inventory layout against the target's BTF.
//
// A type BTF does not know is reported through [TypeAbsentError], which is a
// capability result rather than a mismatch: on an unpatched kernel every one
// of these types is absent and that is the expected answer. A type BTF does
// know but lays out differently is a [LayoutMismatch], and the caller must not
// attach.
func CheckLayouts(b BTF) (absent []string, err error) {
	for _, want := range InventoryLayouts() {
		got, gerr := b.TypeLayout(want.Name)
		var missing *TypeAbsentError
		if errors.As(gerr, &missing) {
			absent = append(absent, want.Name)
			continue
		}
		if gerr != nil {
			return absent, fmt.Errorf("cudabpf: reading BTF for %s: %w", want.Name, gerr)
		}
		if got.Size != want.Size {
			return absent, &LayoutMismatch{Type: want.Name, Want: want.Size, Got: got.Size, Source: "BTF"}
		}
		for _, f := range want.Fields {
			off, ok := got.field(f.Name)
			if !ok {
				return absent, &LayoutMismatch{
					Type: want.Name, Member: f.Name, Want: f.Offset, Got: ^uintptr(0), Source: "BTF",
				}
			}
			if off != f.Offset {
				return absent, &LayoutMismatch{
					Type: want.Name, Member: f.Name, Want: f.Offset, Got: off, Source: "BTF",
				}
			}
		}
	}
	return absent, nil
}

// CheckGoLayouts reports the first disagreement between the Go declarations
// in this package and [InventoryLayouts]. It exists so a struct edit that
// silently changes an offset is caught on a host with no BTF at all, and so a
// consumer can assert the same thing without a target.
func CheckGoLayouts() error {
	have := make(map[string]TypeLayout)
	for _, l := range GoLayouts() {
		have[l.Name] = l
	}
	for _, want := range InventoryLayouts() {
		got, ok := have[want.Name]
		if !ok {
			return fmt.Errorf("cudabpf: no Go declaration for %s", want.Name)
		}
		if got.Size != want.Size {
			return &LayoutMismatch{Type: want.Name, Want: want.Size, Got: got.Size, Source: "Go"}
		}
		for _, f := range want.Fields {
			off, ok := got.field(f.Name)
			if !ok {
				return fmt.Errorf("cudabpf: Go %s has no member %s", want.Name, f.Name)
			}
			if off != f.Offset {
				return &LayoutMismatch{
					Type: want.Name, Member: f.Name, Want: f.Offset, Got: off, Source: "Go",
				}
			}
		}
	}
	return nil
}

// Compile-time reassurance that the two structs whose sizes the C side asserts
// most loudly are the sizes the C side asserts. The full comparison is
// checkGoLayouts; these two catch the common case at build time.
const (
	_ = -uint(unsafe.Sizeof(PrefetchDecision{}) - 24)
	_ = -uint(unsafe.Sizeof(PMMDecision{}) - 72)
)
