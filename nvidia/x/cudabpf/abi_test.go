package cudabpf_test

import (
	"context"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabpf"
)

// TestContextStructSizesAndOffsetsMatchBTF checks the six numbers the C side
// asserts with _Static_assert, plus the scheduler sizes its own header
// asserts, against both the Go declarations and a target's BTF.
func TestContextStructSizesAndOffsetsMatchBTF(t *testing.T) {
	// The numbers are written out here rather than taken from the
	// package, so that a struct edited until the package agrees with
	// itself still fails.
	want := map[string]struct {
		size   uintptr
		fields map[string]uintptr
	}{
		"uvm_bpf_prefetch_decision_t":  {24, map[string]uintptr{"first": 8, "outer": 16}},
		"uvm_bpf_pmm_decision_ctx_t":   {72, map[string]uintptr{"observed": 16, "request": 48}},
		"nv_gpu_task_init_ctx":         {32, map[string]uintptr{"engine_type": 8, "default_timeslice": 16, "runlist_id": 28}},
		"nv_gpu_timeslice_control_ctx": {40, nil},
	}

	have := make(map[string]cudabpf.TypeLayout)
	for _, l := range cudabpf.GoLayouts() {
		have[l.Name] = l
	}
	for name, w := range want {
		got, ok := have[name]
		if !ok {
			t.Errorf("no Go declaration for %s", name)
			continue
		}
		if got.Size != w.size {
			t.Errorf("Go sizeof(%s) = %d, want %d", name, got.Size, w.size)
		}
		for member, off := range w.fields {
			found := false
			for _, f := range got.Fields {
				if f.Name != member {
					continue
				}
				found = true
				if f.Offset != off {
					t.Errorf("Go %s.%s at offset %d, want %d", name, member, f.Offset, off)
				}
			}
			if !found {
				t.Errorf("Go %s has no member %s", name, member)
			}
		}
	}

	if err := cudabpf.CheckGoLayouts(); err != nil {
		t.Errorf("CheckGoLayouts: %v", err)
	}

	absent, err := cudabpf.CheckLayouts(patchedBTF())
	if err != nil {
		t.Fatalf("CheckLayouts against a matching target: %v", err)
	}
	if len(absent) != 0 {
		t.Errorf("types absent from a patched target: %v", absent)
	}
}

// TestBTFTypeMismatchRejected covers the two ways BTF can disagree: a wrong
// size, and the subtler case of a right size with a wrong field offset. The
// second is the one that matters, because a size check alone passes it and
// every field read afterwards returns a plausible number from the wrong
// member.
func TestBTFTypeMismatchRejected(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*fakeBTF)
		member string
	}{
		{
			name: "wrong size",
			mutate: func(b *fakeBTF) {
				l := b.types["uvm_bpf_pmm_decision_ctx_t"]
				l.Size = 64
				b.types["uvm_bpf_pmm_decision_ctx_t"] = l
			},
		},
		{
			name:   "right size, wrong offset",
			member: "request",
			mutate: func(b *fakeBTF) {
				l := b.types["uvm_bpf_pmm_decision_ctx_t"]
				fields := append([]cudabpf.FieldLayout(nil), l.Fields...)
				for i := range fields {
					if fields[i].Name == "request" {
						fields[i].Offset = 40
					}
				}
				l.Fields = fields
				b.types["uvm_bpf_pmm_decision_ctx_t"] = l
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := patchedBTF()
			tt.mutate(b)
			_, err := cudabpf.CheckLayouts(b)
			var mm *cudabpf.LayoutMismatch
			if !asLayoutMismatch(err, &mm) {
				t.Fatalf("CheckLayouts = %v, want a *LayoutMismatch", err)
			}
			if mm.Type != "uvm_bpf_pmm_decision_ctx_t" {
				t.Errorf("mismatch on %s, want uvm_bpf_pmm_decision_ctx_t", mm.Type)
			}
			if mm.Member != tt.member {
				t.Errorf("mismatch member = %q, want %q", mm.Member, tt.member)
			}

			// A mismatch must also stop an attach, not merely be
			// reported.
			l := newFakeLoader(b)
			br := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
			if _, err := br.Attach(context.Background(), memObject(), memSpec()); err == nil {
				t.Fatal("attached despite a BTF layout mismatch")
			}
		})
	}
}

// TestCapabilityDiscoveryReportsAbsentHooks is the expected result on every
// ordinary kernel, including the authorized DGX: the generic struct_ops
// infrastructure is there, the GPU policy tables are not, and discovery says
// so without erroring.
func TestCapabilityDiscoveryReportsAbsentHooks(t *testing.T) {
	r, err := cudabpf.Discover(context.Background(), newFakeLoader(stockBTF()))
	if err != nil {
		t.Fatalf("Discover on an unpatched target: %v", err)
	}
	if r.AnyHookAvailable() {
		t.Fatal("an unpatched target reported an available hook")
	}
	if len(r.Hooks) != 10 {
		t.Errorf("reported %d hooks, want 10 (6 memory + 4 scheduling)", len(r.Hooks))
	}
	for _, h := range r.Hooks {
		if h.Reason == "" {
			t.Errorf("%s.%s is absent with no reason recorded", h.Table, h.Member)
		}
	}
	for _, k := range r.Kfuncs {
		if k.Available {
			t.Errorf("kfunc %s reported available on an unpatched target", k.Name)
		}
	}
	for _, ty := range r.Types {
		if ty.Available {
			t.Errorf("type %s reported available on an unpatched target", ty.Name)
		}
	}
	// Identity must survive discovery: a later claim has to be able to
	// name the machine.
	if r.Capabilities.Kernel == "" || r.Capabilities.GPU == "" {
		t.Errorf("capabilities lost identity: %+v", r.Capabilities)
	}

	// And a patched target must come out the other way, or the test
	// above proves only that the fake is empty.
	full, err := cudabpf.Discover(context.Background(), newFakeLoader(patchedBTF()))
	if err != nil {
		t.Fatalf("Discover on a patched target: %v", err)
	}
	if !full.TableAvailable(cudabpf.MemOpsTable) || !full.TableAvailable(cudabpf.SchedOpsTable) {
		t.Fatalf("patched target reported a table unavailable:\n%s", full)
	}
}

// TestNoBTFReportsEverythingAbsent covers the stale or missing BTF control: a
// target with no type information can attach nothing, and says so rather than
// falling through to the header constants.
func TestNoBTFReportsEverythingAbsent(t *testing.T) {
	l := newFakeLoader(nil)
	l.btfErr = cudabpf.ErrNoBTF
	r, err := cudabpf.Discover(context.Background(), l)
	if err != nil {
		t.Fatalf("Discover with no BTF: %v", err)
	}
	if r.Capabilities.BTFAvailable {
		t.Error("BTFAvailable true with no BTF")
	}
	if r.AnyHookAvailable() {
		t.Error("a hook was available with no BTF to verify it")
	}
	for _, h := range r.Hooks {
		if h.Reason != "no BTF on target" {
			t.Errorf("%s.%s reason = %q", h.Table, h.Member, h.Reason)
		}
	}
}

// TestUnannotatedKfuncRejected covers a kfunc that exists in BTF but was
// never registered as callable. Treating it as available would move the
// failure from discovery into the verifier, where a handler has already been
// written against it.
func TestUnannotatedKfuncRejected(t *testing.T) {
	b := patchedBTF()
	k := b.kfuncs["bpf_gpu_request_reorder"]
	k.Annotated = false
	b.kfuncs["bpf_gpu_request_reorder"] = k

	r, err := cudabpf.Discover(context.Background(), newFakeLoader(b))
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	st, ok := r.Kfunc("bpf_gpu_request_reorder")
	if !ok {
		t.Fatal("no status for bpf_gpu_request_reorder")
	}
	if st.Available {
		t.Fatal("an unannotated kfunc was reported available")
	}
	if !strings.Contains(st.Reason, "annotated") {
		t.Errorf("reason = %q, want it to name the annotation", st.Reason)
	}
	// The two hooks that actuate through it must go with it.
	for _, m := range []string{"gpu_block_access", "gpu_block_activate"} {
		for _, h := range r.Hooks {
			if h.Member == m && h.Available {
				t.Errorf("%s available though its only actuator is not callable", m)
			}
		}
	}
}

// TestKfuncSignatureMismatchRejected covers drift: a kfunc that kept its name
// and changed its shape is a different kfunc.
func TestKfuncSignatureMismatchRejected(t *testing.T) {
	b := patchedBTF()
	k := b.kfuncs["bpf_nv_gpu_set_timeslice"]
	k.Params = []string{"nv_gpu_task_init_ctx*", "u32"} // was u64
	b.kfuncs["bpf_nv_gpu_set_timeslice"] = k

	r, err := cudabpf.Discover(context.Background(), newFakeLoader(b))
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	st, _ := r.Kfunc("bpf_nv_gpu_set_timeslice")
	if st.Available {
		t.Fatal("a kfunc with a changed signature was reported available")
	}
	if !strings.Contains(st.Reason, "signature") {
		t.Errorf("reason = %q, want it to name the signature", st.Reason)
	}
}

// TestAbsentKfuncNotSynthesized covers the names the proposal had and the
// interface does not. bpf_gpu_move_head and bpf_gpu_move_tail do not exist,
// and neither does bpf_gpu_block_insert_after -- so this package must not
// export anything that looks like a call to them, and an attach must refuse
// rather than emulate.
func TestAbsentKfuncNotSynthesized(t *testing.T) {
	for _, name := range []string{
		"bpf_gpu_move_head", "bpf_gpu_move_tail", "bpf_gpu_set_attr",
		"bpf_gpu_block_insert_after", "bpf_gpu_reject_bind",
		"BPF_PROG_TYPE_GPU_MEM", "BPF_PROG_TYPE_GPU_SCHED",
		"gdrv_mem_access_ctx_t", "gdrv_mem_prefetch_ctx_t", "gsched_queue_ctx_t",
	} {
		if found := grepPackage(t, name); len(found) > 0 {
			t.Errorf("%s, which does not exist in the interface, appears in %v", name, found)
		}
	}

	// And an attach against a target that lacks the actuator refuses by
	// name rather than proceeding.
	b := patchedBTF()
	delete(b.kfuncs, "bpf_gpu_request_reorder")
	br := cudabpf.NewBridge(newFakeLoader(b), &fakeKfuncs{}, cudabpf.Config{})
	_, err := br.Attach(context.Background(), memObject(), memSpec())
	if err == nil {
		t.Fatal("attached with no reorder kfunc on the target")
	}
	if !strings.Contains(err.Error(), "bpf_gpu_request_reorder") {
		t.Errorf("error = %v, want it to name the missing kfunc", err)
	}
}

// TestStructOpsProgramTypeIsGeneric covers the proposal's biggest correction.
// Both tables are ordinary BPF_PROG_TYPE_STRUCT_OPS programs; a loader that
// asks the kernel for a GPU-specific program type fails on every kernel,
// including one where the hooks are present. Nothing in this package may name
// such a type, and the attach path must carry no program type at all --
// choosing one is the loader's business.
func TestStructOpsProgramTypeIsGeneric(t *testing.T) {
	for _, name := range []string{"BPF_PROG_TYPE_GPU", "ProgTypeGPU", "GPUMemProgType", "GPUSchedProgType"} {
		if found := grepPackage(t, name); len(found) > 0 {
			t.Errorf("%s appears in %v; both tables are generic struct_ops", name, found)
		}
	}
	if grep := grepPackage(t, "ProgramType"); len(grep) > 0 {
		t.Errorf("the attach path names a program type in %v; that is the loader's choice", grep)
	}
}

// TestSchedulerSectionNameNotUsedAsMemberName is the asymmetry test.
//
// In a struct_ops object the member a program binds to comes from the
// .struct_ops map initializer, not from the section name. The memory table's
// sections happen to match its members; at least one real scheduler object
// declares programs gp_task_init, gp_bind, gp_destroy and
// gp_timeslice_control with sections of those names while binding them to
// members on_task_init, on_bind, on_task_destroy and on_timeslice_control. A
// loader that derives the member from the section attaches the wrong slot or
// none at all.
func TestSchedulerSectionNameNotUsedAsMemberName(t *testing.T) {
	obj := cudabpf.ObjectSpec{
		Name: "gpreempt_policy.bpf.o",
		Programs: []cudabpf.ProgramSpec{
			{Name: "gp_task_init", Section: "struct_ops/gp_task_init", Member: "on_task_init"},
			{Name: "gp_bind", Section: "struct_ops/gp_bind", Member: "on_bind"},
			{Name: "gp_destroy", Section: "struct_ops/gp_destroy", Member: "on_task_destroy"},
			{Name: "gp_timeslice_control", Section: "struct_ops/gp_timeslice_control", Member: "on_timeslice_control"},
		},
	}
	spec := cudabpf.StructOpsSpec{
		Table:   cudabpf.SchedOpsTable,
		MapName: "gpreempt_ops",
		Programs: map[string]string{
			"on_task_init":         "gp_task_init",
			"on_bind":              "gp_bind",
			"on_task_destroy":      "gp_destroy",
			"on_timeslice_control": "gp_timeslice_control",
		},
	}
	if err := spec.Validate(obj); err != nil {
		t.Fatalf("a scheduler object whose sections disagree with its members was rejected: %v", err)
	}

	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
	if _, err := br.Attach(context.Background(), obj, spec); err != nil {
		t.Fatalf("Attach: %v", err)
	}
	bound := l.bound[cudabpf.SchedOpsTable]
	for member, prog := range spec.Programs {
		if bound[member] != prog {
			t.Errorf("member %s bound to %q, want %q", member, bound[member], prog)
		}
	}
	// The failure being guarded against: deriving the member from the
	// section would have produced these four names, none of which is a
	// member of the table.
	for _, derived := range []string{"gp_task_init", "gp_bind", "gp_destroy", "gp_timeslice_control"} {
		if _, ok := bound[derived]; ok {
			t.Errorf("a section-derived name %q was used as a member", derived)
		}
		bad := cudabpf.StructOpsSpec{
			Table: cudabpf.SchedOpsTable, MapName: "x",
			Programs: map[string]string{derived: derived},
		}
		if err := bad.Validate(obj); err == nil {
			t.Errorf("%q was accepted as a member of %s", derived, cudabpf.SchedOpsTable)
		}
	}

	// The memory table is the control: there the two do agree, and the
	// binding still comes from the map.
	for _, p := range memObject().Programs {
		if got := strings.TrimPrefix(p.Section, "struct_ops/"); got != p.Member {
			t.Errorf("memory program %s: section %q, member %q; they are expected to agree",
				p.Name, p.Section, p.Member)
		}
	}
}

// TestPolicyLayerHasNoLoaderDependency is the module-boundary test. The
// policy layer must be buildable and testable on a host that cannot load BPF,
// so it may not import the sibling loader module, a BPF library, or cgo.
func TestPolicyLayerHasNoLoaderDependency(t *testing.T) {
	forbidden := []string{"ebpfhost", "cilium/ebpf", "libbpf", "C", "golang.org/x/sys/unix"}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parsing package: %v", err)
	}
	seen := 0
	for _, pkg := range pkgs {
		for name, f := range pkg.Files {
			seen++
			for _, imp := range f.Imports {
				path, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatalf("%s: bad import %s", name, imp.Path.Value)
				}
				for _, bad := range forbidden {
					if path == bad || strings.Contains(path, bad) {
						t.Errorf("%s imports %q", filepath.Base(name), path)
					}
				}
			}
		}
	}
	if seen == 0 {
		t.Fatal("parsed no files; the test would pass vacuously")
	}
}

// grepPackage returns the non-test files containing s.
func grepPackage(t *testing.T, s string) []string {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading package dir: %v", err)
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		// Skip the doc comment lines that discuss what does not exist.
		for _, line := range strings.Split(string(b), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "*") {
				continue
			}
			if strings.Contains(line, s) {
				out = append(out, name)
				break
			}
		}
	}
	return out
}

func asLayoutMismatch(err error, target **cudabpf.LayoutMismatch) bool {
	for err != nil {
		if m, ok := err.(*cudabpf.LayoutMismatch); ok {
			*target = m
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}
