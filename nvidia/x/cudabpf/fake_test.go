package cudabpf_test

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudabpf"
)

// The fake target. It stands in for a patched kernel: it answers BTF
// questions, accepts struct_ops registrations and records kfunc calls. It is
// deliberately configurable in the ways a real target varies -- a table
// missing, a type at the wrong offset, a kfunc present but unannotated -- so
// that absence and disagreement are exercised rather than assumed.

type fakeBTF struct {
	types     map[string]cudabpf.TypeLayout
	kfuncs    map[string]cudabpf.KfuncSignature
	tables    []string
	err       error
	memberErr error
	tableErr  error
}

func (b *fakeBTF) TypeLayout(name string) (cudabpf.TypeLayout, error) {
	if b.err != nil {
		return cudabpf.TypeLayout{}, b.err
	}
	t, ok := b.types[name]
	if !ok {
		return cudabpf.TypeLayout{}, &cudabpf.TypeAbsentError{Name: name}
	}
	return t, nil
}

func (b *fakeBTF) KfuncSignature(name string) (cudabpf.KfuncSignature, error) {
	k, ok := b.kfuncs[name]
	if !ok {
		return cudabpf.KfuncSignature{}, &cudabpf.KfuncAbsentError{Name: name}
	}
	return k, nil
}

func (b *fakeBTF) ValidateStructOpsMember(table, member string) error { return b.memberErr }

func (b *fakeBTF) StructOpsTables() ([]string, error) { return b.tables, b.tableErr }

// patchedBTF returns a BTF describing a target that has the policy module:
// every inventory type at its required layout, every kfunc annotated with the
// required signature, and both tables registered.
func patchedBTF() *fakeBTF {
	types := make(map[string]cudabpf.TypeLayout)
	for _, l := range cudabpf.InventoryLayouts() {
		types[l.Name] = l
	}
	kfuncs := make(map[string]cudabpf.KfuncSignature)
	for _, k := range exportedKfuncs() {
		k.Annotated = true
		kfuncs[k.Name] = k
	}
	return &fakeBTF{
		types:  types,
		kfuncs: kfuncs,
		tables: []string{cudabpf.MemOpsTable, cudabpf.SchedOpsTable, "tcp_congestion_ops"},
	}
}

// stockBTF returns a BTF describing an ordinary kernel: generic struct_ops
// infrastructure, none of the GPU policy tables, types, or kfuncs. This is
// what the authorized DGX actually reports.
func stockBTF() *fakeBTF {
	return &fakeBTF{
		types:  map[string]cudabpf.TypeLayout{},
		kfuncs: map[string]cudabpf.KfuncSignature{},
		tables: []string{"tcp_congestion_ops", "sched_ext_ops", "bpf_dummy_ops", "hid_bpf_ops"},
	}
}

// exportedKfuncs mirrors the signatures the driver's headers declare. It is
// written out here rather than read from the package so that a test does not
// compare the package against itself.
func exportedKfuncs() []cudabpf.KfuncSignature {
	return []cudabpf.KfuncSignature{
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

type fakeLink struct {
	table      string
	generation uint64
	detached   bool
	owner      *fakeLoader
}

func (l *fakeLink) Generation() uint64 { return l.generation }

func (l *fakeLink) Detach() error {
	if l.detached {
		return nil // idempotent
	}
	l.detached = true
	if l.owner != nil {
		l.owner.mu.Lock()
		delete(l.owner.registered, l.table)
		l.owner.detaches++
		l.owner.mu.Unlock()
	}
	return nil
}

type fakeObject struct {
	spec   cudabpf.ObjectSpec
	loader *fakeLoader
	closed bool
}

func (o *fakeObject) AttachStructOps(ctx context.Context, spec cudabpf.StructOpsSpec) (cudabpf.Link, error) {
	if err := spec.Validate(o.spec); err != nil {
		return nil, err
	}
	o.loader.mu.Lock()
	defer o.loader.mu.Unlock()
	if o.loader.attachErr != nil {
		return nil, o.loader.attachErr
	}
	if o.loader.registered[spec.Table] {
		// This is what a second process racing to register the same
		// table gets from the kernel.
		return nil, fmt.Errorf("struct_ops %s already registered: %w", spec.Table, errEEXIST)
	}
	o.loader.registered[spec.Table] = true
	o.loader.attaches++
	// The member binding is taken from the map, never derived from a
	// program's section name. Recording it here is how the asymmetry test
	// observes which slot was actually filled.
	bound := make(map[string]string, len(spec.Programs))
	for member, prog := range spec.Programs {
		bound[member] = prog
	}
	o.loader.bound[spec.Table] = bound
	return &fakeLink{table: spec.Table, generation: o.loader.generation, owner: o.loader}, nil
}

func (o *fakeObject) Close() error {
	o.loader.mu.Lock()
	defer o.loader.mu.Unlock()
	o.closed = true
	o.loader.closes++
	return nil
}

var errEEXIST = errors.New("file exists")

type fakeLoader struct {
	mu         sync.Mutex
	caps       cudabpf.Capabilities
	btf        cudabpf.BTF
	btfErr     error
	loadErr    error
	attachErr  error
	generation uint64
	registered map[string]bool
	bound      map[string]map[string]string
	attaches   int
	detaches   int
	loads      int
	closes     int
}

func newFakeLoader(btf cudabpf.BTF) *fakeLoader {
	return &fakeLoader{
		caps: cudabpf.Capabilities{
			Kernel:                 "6.17.0-1031-nvidia",
			Driver:                 "NVIDIA UNIX Open Kernel Module 580.173.02",
			GPU:                    "NVIDIA GB10",
			ComputeCap:             "12.1",
			CUDA:                   "13.4",
			BTFAvailable:           btf != nil,
			UnprivilegedBPFKnown:   true,
			UnprivilegedBPFAllowed: false,
		},
		btf:        btf,
		registered: make(map[string]bool),
		bound:      make(map[string]map[string]string),
	}
}

func (l *fakeLoader) Capabilities(context.Context) (cudabpf.Capabilities, error) {
	return l.caps, nil
}

func (l *fakeLoader) BTF(context.Context) (cudabpf.BTF, error) {
	if l.btfErr != nil {
		return nil, l.btfErr
	}
	return l.btf, nil
}

func (l *fakeLoader) Load(_ context.Context, spec cudabpf.ObjectSpec) (cudabpf.Object, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.loadErr != nil {
		return nil, l.loadErr
	}
	l.loads++
	return &fakeObject{spec: spec, loader: l}, nil
}

// fakeKfuncs records every actuation and simulates the driver's answer.
type fakeKfuncs struct {
	mu sync.Mutex

	reorders   []reorderCall
	prefetches []prefetchCall
	timeslices []uint64
	interleave []cudabpf.InterleaveLevel
	rejects    int
	migrations int

	// reorderErr, when set, is returned instead of submitting.
	reorderErr   error
	prefetchErr  error
	timesliceErr error

	// conflict makes the driver record a collision on the next reorder,
	// as a contended chunk would.
	conflict bool
	// ignore makes the driver record nothing at all, as it may for an
	// advisory request it declines to act on.
	ignore bool
}

type reorderCall struct {
	dst cudabpf.Destination
	pos cudabpf.Position
}

type prefetchCall struct {
	first, outer uint64
}

func (k *fakeKfuncs) RequestReorder(ctx *cudabpf.PMMDecision, dst cudabpf.Destination, pos cudabpf.Position) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.reorderErr != nil {
		return k.reorderErr
	}
	k.reorders = append(k.reorders, reorderCall{dst, pos})
	if ctx == nil || k.ignore {
		return nil
	}
	// The driver writes the request through and records that it took it.
	ctx.Request.Destination = uint64(dst)
	ctx.Request.Position = uint64(pos)
	ctx.Request.Attempted = 1
	if k.conflict {
		ctx.Request.Conflict = 1
	}
	return nil
}

func (k *fakeKfuncs) SetPrefetchRegion(d *cudabpf.PrefetchDecision, first, outer uint64) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.prefetchErr != nil {
		return k.prefetchErr
	}
	k.prefetches = append(k.prefetches, prefetchCall{first, outer})
	if d != nil {
		d.First, d.Outer, d.Attempted = first, outer, 1
	}
	return nil
}

func (k *fakeKfuncs) MigrateRange(vaSpace, addr, length uint64) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.migrations++
	return nil
}

func (k *fakeKfuncs) SetTimeslice(_ *cudabpf.TaskInitContext, us uint64) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.timesliceErr != nil {
		return k.timesliceErr
	}
	k.timeslices = append(k.timeslices, us)
	return nil
}

func (k *fakeKfuncs) SetInterleave(_ *cudabpf.TaskInitContext, l cudabpf.InterleaveLevel) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.interleave = append(k.interleave, l)
	return nil
}

func (k *fakeKfuncs) RejectBind(ctx *cudabpf.BindContext) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.rejects++
	// The kfunc is non-sleepable and its whole effect is to clear the
	// context's allow member. The driver does this, not the policy layer.
	if ctx != nil {
		ctx.Allow = 0
	}
	return nil
}

func (k *fakeKfuncs) counts() (reorders, prefetches, rejects int) {
	k.mu.Lock()
	defer k.mu.Unlock()
	return len(k.reorders), len(k.prefetches), k.rejects
}

// staticPolicy is a MemoryPolicy that returns whatever it was configured with.
type staticPolicy struct {
	reorder reorderOpinion
	err     error
	steps   int
}

// reorderOpinion carries the reorder and whether there is an opinion at all.
type reorderOpinion struct {
	R  cudabpf.Reorder
	Ok bool
}

func (p *staticPolicy) charge(b *cudabpf.Budget) error {
	n := p.steps
	if n == 0 {
		n = 1
	}
	if b == nil {
		return nil
	}
	return b.Step(n)
}

func (p *staticPolicy) BlockAccess(_ *cudabpf.PMMDecision, b *cudabpf.Budget) (cudabpf.Reorder, bool, error) {
	if err := p.charge(b); err != nil {
		return cudabpf.Reorder{}, false, err
	}
	return p.reorder.R, p.reorder.Ok, p.err
}

func (p *staticPolicy) BlockActivate(d *cudabpf.PMMDecision, b *cudabpf.Budget) (cudabpf.Reorder, bool, error) {
	return p.BlockAccess(d, b)
}

func (p *staticPolicy) Prefetch(_ *cudabpf.PrefetchDecision, b *cudabpf.Budget) (cudabpf.Region, bool, error) {
	if err := p.charge(b); err != nil {
		return cudabpf.Region{}, false, err
	}
	return cudabpf.Region{}, false, nil
}

// tenantPolicy assigns scheduling attributes by a label on the TSG, which is
// how a multi-tenant caller would use this layer.
type tenantPolicy struct {
	byTSG  map[uint64]cudabpf.SchedAttributes
	reject map[uint64]string
}

func (p *tenantPolicy) TaskInit(c *cudabpf.TaskInitContext, b *cudabpf.Budget) (cudabpf.SchedAttributes, error) {
	if b != nil {
		if err := b.Step(1); err != nil {
			return cudabpf.SchedAttributes{}, err
		}
	}
	return p.byTSG[c.TSGID], nil
}

func (p *tenantPolicy) Bind(c *cudabpf.BindContext, b *cudabpf.Budget) (cudabpf.BindDecision, error) {
	if b != nil {
		if err := b.Step(1); err != nil {
			return cudabpf.BindDecision{}, err
		}
	}
	if reason, ok := p.reject[c.TSGID]; ok {
		return cudabpf.BindDecision{Reject: true, Reason: reason}, nil
	}
	return cudabpf.BindDecision{}, nil
}

// memObject is a minimal object spec for the memory table.
func memObject() cudabpf.ObjectSpec {
	return cudabpf.ObjectSpec{
		Name: "gpu_mem_policy.bpf.o",
		Programs: []cudabpf.ProgramSpec{
			{Name: "on_block_access", Section: "struct_ops/gpu_block_access", Member: "gpu_block_access"},
			{Name: "on_block_activate", Section: "struct_ops/gpu_block_activate", Member: "gpu_block_activate"},
		},
	}
}

func memSpec() cudabpf.StructOpsSpec {
	return cudabpf.StructOpsSpec{
		Table:   cudabpf.MemOpsTable,
		MapName: "gpu_mem_policy",
		Programs: map[string]string{
			"gpu_block_access":   "on_block_access",
			"gpu_block_activate": "on_block_activate",
		},
	}
}
