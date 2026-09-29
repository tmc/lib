package nvvmx

import (
	"bytes"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

func TestAddrSpace_InfersSharedSlot(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_shared(ptr addrspace(1) %out) {
entry:
  %slot = alloca ptr, align 8
  br label %body

body:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %val = load float, ptr %p, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["@kernel_shared/%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report: %+v", report.Slots)
	}
	if info.State != StateShared {
		t.Errorf("slot state = %v, want StateShared", info.State)
	}
	if info.Effective != AddrSpaceShared {
		t.Errorf("effective address space = %v, want AddrSpaceShared", info.Effective)
	}

	outStr := string(rewritten)
	if !strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("expected %%slot to be declared in addrspace(3), got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "load ptr addrspace(3), ptr %slot") {
		t.Errorf("expected load to read ptr addrspace(3), got:\n%s", outStr)
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestAddrSpace_MixedStaysGeneric(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_mixed(ptr addrspace(1) %gptr, i1 %cond) {
entry:
  %slot = alloca ptr, align 8
  br i1 %cond, label %b1, label %b2

b2:
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  br label %exit

b1:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  br label %exit

exit:
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["@kernel_mixed/%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if info.State != StateGenericRequired {
		t.Errorf("slot state = %v, want StateGenericRequired", info.State)
	}
	if info.Effective != AddrSpaceGeneric {
		t.Errorf("effective address space = %v, want AddrSpaceGeneric", info.Effective)
	}

	outStr := string(rewritten)
	if strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("slot %%slot was erroneously declared addrspace(3)")
	}
	if !strings.Contains(outStr, "%slot = alloca ptr, align 8") {
		t.Errorf("slot %%slot should remain generic ptr, got:\n%s", outStr)
	}
}

func TestAddrSpace_ExplicitDeclarationWins(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_explicit() {
entry:
  %slot = alloca ptr addrspace(1), align 8
  br label %body

body:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["@kernel_explicit/%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if !info.Explicit {
		t.Errorf("slot %%slot should be marked Explicit")
	}
	if info.Effective != AddrSpaceGlobal {
		t.Errorf("effective address space = %v, want AddrSpaceGlobal (explicit wins)", info.Effective)
	}

	outStr := string(rewritten)
	if !strings.Contains(outStr, "%slot = alloca ptr addrspace(1)") {
		t.Errorf("explicit declaration overwritten, got:\n%s", outStr)
	}
}

func TestAddrSpace_FixedPointTerminates(t *testing.T) {
	// Loop-carried dependencies between 3 slots:
	// slot1 -> slot2 -> slot3 -> slot1
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_loop(i1 %cond) {
entry:
  %slot1 = alloca ptr, align 8
  %slot2 = alloca ptr, align 8
  %slot3 = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot1, align 8
  br label %loop

loop:
  %v1 = load ptr, ptr %slot1, align 8
  store ptr %v1, ptr %slot2, align 8
  %v2 = load ptr, ptr %slot2, align 8
  store ptr %v2, ptr %slot3, align 8
  %v3 = load ptr, ptr %slot3, align 8
  store ptr %v3, ptr %slot1, align 8
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	numSlots := len(report.Slots)
	maxAllowedIterations := 2*numSlots + 1

	if report.Iterations > maxAllowedIterations {
		t.Errorf("iterations = %d, exceeded bound 2*N+1 (%d)", report.Iterations, maxAllowedIterations)
	}

	for _, name := range []string{"%slot1", "%slot2", "%slot3"} {
		info, ok := report.Slots["@kernel_loop/"+name]
		if !ok {
			t.Errorf("slot %s missing from report", name)
			continue
		}
		if info.State != StateShared {
			t.Errorf("slot %s state = %v, want StateShared", name, info.State)
		}
		if info.Effective != AddrSpaceShared {
			t.Errorf("slot %s effective = %v, want AddrSpaceShared", name, info.Effective)
		}
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestAddrSpace_ClusterPointerNotInferred(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

define void @kernel_cluster(ptr addrspace(7) %cluster_ptr) {
entry:
  %slot = alloca ptr, align 8
  br label %body

body:
  store ptr addrspace(7) %cluster_ptr, ptr %slot, align 8
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["@kernel_cluster/%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if info.State != StateGenericRequired {
		t.Errorf("cluster pointer must not infer shared: state = %v, want StateGenericRequired", info.State)
	}
	if info.Effective != AddrSpaceGeneric {
		t.Errorf("effective = %v, want AddrSpaceGeneric", info.Effective)
	}

	outStr := string(rewritten)
	if strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("cluster pointer erroneously declared addrspace(3)")
	}
}

func TestAddrSpace_UseBeforeDisqualifyingWrite(t *testing.T) {
	ir := `define void @kernel_use_first(ptr addrspace(1) %gptr) {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  ret void
}
@smem = internal addrspace(3) global [4 x i8] undef
`
	_, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	if got := report.Slots["@kernel_use_first/%slot"].State; got != StateGenericRequired {
		t.Fatalf("slot state = %v, want generic-required", got)
	}
	if got := report.Slots["@kernel_use_first/%slot"].Effective; got != AddrSpaceGeneric {
		t.Fatalf("effective address space = %d, want generic", got)
	}
}

func TestAddrSpace_MixedStaysGenericForUntrackedWrites(t *testing.T) {
	base := `@smem = internal addrspace(3) global [4 x i8] undef
declare void @consume(ptr)
define void @kernel(ptr addrspace(1) %gptr, ptr %unknown) {
entry:
  %slot = alloca ptr, align 8
  %dest = alloca ptr, align 8
  %src = alloca ptr, align 8
  store ptr addrspace(1) %gptr, ptr %src, align 8
  %loaded = load ptr, ptr %src, align 8
  %cast = addrspacecast ptr addrspace(1) %gptr to ptr addrspace(3)
  %gep = getelementptr i8, ptr %unknown, i64 0
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !0, i64 0}
`
	tests := []struct {
		name, line string
	}{
		{"metadata", "store ptr addrspace(1) %gptr, ptr %slot, align 8, !tbaa !1"},
		{"atomic store", "store atomic ptr addrspace(1) %gptr, ptr %slot seq_cst, align 8"},
		{"escape to call", "call void @consume(ptr %slot)"},
		{"atomicrmw", "%old = atomicrmw xchg ptr %slot, ptr addrspace(3) @smem seq_cst"},
		{"cmpxchg", "%pair = cmpxchg ptr %slot, ptr addrspace(3) @smem, ptr addrspace(3) @smem seq_cst seq_cst"},
		{"memcpy destination", "call void @llvm.memcpy.p0.p0.i64(ptr %slot, ptr %unknown, i64 8, i1 false)"},
		{"unknown stored value", "store ptr %unknown, ptr %slot, align 8"},
		{"loaded from generic-required slot", "store ptr %loaded, ptr %slot, align 8"},
		{"register cast from non-shared", "store ptr addrspace(3) %cast, ptr %slot, align 8"},
		{"gep of non-shared base", "store ptr %gep, ptr %slot, align 8"},
		{"slot address escapes to memory", "store ptr %slot, ptr %dest, align 8"},
		{"atomic load", "%old = load atomic ptr, ptr %slot seq_cst, align 8"},
	}
	baseIR, _, err := InferAddressSpaces([]byte(base))
	if err != nil {
		t.Fatal(err)
	}
	verifyIRWithClang(t, []byte(base))
	if !strings.Contains(string(baseIR), "%slot = alloca ptr addrspace(3)") {
		t.Fatalf("control base did not infer shared:\n%s", baseIR)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := strings.Replace(base, "  ret void", "  "+tt.line+"\n  ret void", 1)
			verifyIRWithClang(t, []byte(ir))
			_, report, err := InferAddressSpaces([]byte(ir))
			if err != nil {
				t.Fatal(err)
			}
			if got := report.Slots["@kernel/%slot"].State; got != StateGenericRequired {
				t.Fatalf("slot state = %v, want generic-required", got)
			}
		})
	}
}

func TestAddrSpace_ClusterValuesStayGeneric(t *testing.T) {
	for _, value := range []string{
		"store ptr addrspace(7) %cluster, ptr %slot, align 8",
		"%cast = addrspacecast ptr addrspace(7) %cluster to ptr addrspace(3)\n  store ptr addrspace(3) %cast, ptr %slot, align 8",
		"%cluster.result = call ptr addrspace(3) @mapa_shared_cluster(ptr addrspace(3) %p, i32 %rank)\n  store ptr addrspace(3) %cluster.result, ptr %slot, align 8",
	} {
		ir := `declare ptr addrspace(3) @mapa_shared_cluster(ptr addrspace(3), i32)
define void @kernel(ptr addrspace(7) %cluster, ptr addrspace(3) %p, i32 %rank) {
entry:
  %slot = alloca ptr, align 8
  ` + value + `
  ret void
}
`
		verifyIRWithClang(t, []byte(ir))
		_, report, err := InferAddressSpaces([]byte(ir))
		if err != nil {
			t.Fatal(err)
		}
		if got := report.Slots["@kernel/%slot"].State; got != StateGenericRequired {
			t.Errorf("value %q: state = %v, want generic-required", value, got)
		}
	}
}

func TestAddrSpace_BlockOrderIndependent(t *testing.T) {
	body := func(blocks string) []byte {
		return []byte("@smem = internal addrspace(3) global [4 x i8] undef\n" +
			"define void @kernel(ptr addrspace(1) %gptr, i1 %cond) {\n" +
			"entry:\n  %slot = alloca ptr, align 8\n  br i1 %cond, label %init, label %use\n" + blocks + "}\n")
	}
	blocks := []string{
		"init:\n  store ptr addrspace(3) @smem, ptr %slot, align 8\n  br label %exit\nuse:\n  store ptr addrspace(1) %gptr, ptr %slot, align 8\n  br label %exit\nexit:\n  ret void\n",
		"use:\n  store ptr addrspace(1) %gptr, ptr %slot, align 8\n  br label %exit\ninit:\n  store ptr addrspace(3) @smem, ptr %slot, align 8\n  br label %exit\nexit:\n  ret void\n",
	}
	for i, block := range blocks {
		_, report, err := InferAddressSpaces(body(block))
		if err != nil {
			t.Fatal(err)
		}
		if got := report.Slots["@kernel/%slot"].State; got != StateGenericRequired {
			t.Errorf("block order %d: state = %v, want generic-required", i, got)
		}
	}
}

func TestAddrSpace_FixedPointNeedsMultiplePasses(t *testing.T) {
	ir := `@smem = internal addrspace(3) global [4 x i8] undef
define void @kernel(i1 %again) {
entry:
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %c = alloca ptr, align 8
  br label %loop
loop:
  %vc = load ptr, ptr %c, align 8
  store ptr %vc, ptr %b, align 8
  %vb = load ptr, ptr %b, align 8
  store ptr %vb, ptr %a, align 8
  store ptr addrspace(3) @smem, ptr %c, align 8
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	if report.Iterations < 3 || report.Iterations > 2*len(report.Slots)+1 {
		t.Fatalf("iterations = %d, want >2 and <=2N+1", report.Iterations)
	}
	for _, name := range []string{"@kernel/%a", "@kernel/%b", "@kernel/%c"} {
		if got := report.Slots[name].State; got != StateShared {
			t.Errorf("%s state = %v, want shared", name, got)
		}
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestInferAddressSpaces_LongLine(t *testing.T) {
	comment := ";" + strings.Repeat("x", 140_000) + "\r\n"
	ir := []byte(comment + "@smem = internal addrspace(3) global i32 undef\r\n" +
		"define void @kernel() {\r\nentry:\r\n" +
		"  %slot = alloca ptr, align 8\r\n" +
		"  store ptr addrspace(3) @smem, ptr %slot, align 8\r\n" +
		"  ret void\r\n}\r\n")
	got, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	want := bytes.Replace(ir, []byte("  %slot = alloca ptr, align 8\r\n"), []byte("  %slot = alloca ptr addrspace(3), align 8\r\n"), 1)
	if !bytes.Equal(got, want) {
		t.Fatalf("long-line output differs beyond the inferred alloca:\n got %d bytes\nwant %d bytes", len(got), len(want))
	}
	verifyIRWithClang(t, got)
	assertIndependentSlotTypes(t, got, report)
}

func TestInferAddressSpaces_RefusesBitcode(t *testing.T) {
	if _, _, err := InferAddressSpaces([]byte{'B', 'C', 0xc0, 0xde, 0, 0}); err == nil {
		t.Fatal("InferAddressSpaces accepted bitcode")
	}
}

func TestAddrSpace_ReportUsesFunctionAndSlotKey(t *testing.T) {
	ir := `@smem = internal addrspace(3) global [4 x i8] undef
define void @first() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
define void @second(ptr addrspace(1) %gptr) {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@first/%slot"].State != StateShared || report.Slots["@second/%slot"].State != StateGenericRequired {
		t.Fatalf("per-function slot states = %+v", report.Slots)
	}
	if !reflect.DeepEqual(report.RewrittenSlots, []string{"@first/%slot"}) {
		t.Fatalf("rewritten slots = %v, want [@first/%%slot]", report.RewrittenSlots)
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestAddrSpace_OutputIsValidIR(t *testing.T) {
	ir := []byte(`target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"
@smem = internal addrspace(3) global [4 x i8] undef
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %value = load i8, ptr %p, align 1
  ret void
}
`)
	rewritten, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func verifyIRWithClang(t *testing.T, ir []byte) {
	t.Helper()
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Log("clang is not on PATH; skipping only IR verification")
		return
	}
	cmd := exec.Command(clang, "-x", "ir", "-S", "-emit-llvm", "-o", "/dev/null", "-")
	cmd.Stdin = bytes.NewReader(ir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("clang rejected rewritten IR: %v\n%s\n%s", err, out, ir)
	}
}

func TestInferAddressSpacesRejectsUnterminatedFunction(t *testing.T) {
	ir := []byte("define void @broken() {\nentry:\n  ret void\n")
	if _, _, err := InferAddressSpaces(ir); err == nil {
		t.Fatal("unterminated function accepted")
	}
}

func TestAddrSpace_GlobalQualifierDoesNotComeFromElementType(t *testing.T) {
	tests := []struct{ name, global, store string }{
		{"addrspace one global with shared pointer element", "@gp = internal addrspace(1) global ptr addrspace(3) null, align 8", "store ptr addrspace(1) @gp, ptr %slot, align 8"},
		{"generic global with shared pointer initializer", "@tbl = internal global ptr addrspace(3) @smem, align 8", "store ptr @tbl, ptr %slot, align 8"},
		{"generic global with address-space element and section", "@tbl = internal global ptr addrspace(3) @smem, section \"global\"", "store ptr @tbl, ptr %slot, align 8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := "@smem = internal addrspace(3) global i8 0\n" + tt.global + `
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  ` + tt.store + `
  ret void
}
`
			rewritten, report, err := InferAddressSpaces([]byte(ir))
			if err != nil {
				t.Fatal(err)
			}
			if got := report.Slots["@kernel/%slot"].State; got != StateGenericRequired {
				t.Fatalf("state=%v, want generic-required", got)
			}
			if strings.Contains(string(rewritten), "%slot = alloca ptr addrspace(3)") {
				t.Fatalf("non-shared global inferred shared:\n%s", rewritten)
			}
		})
	}
}

func TestAddrSpace_GlobalQualifierUsesFirstGlobalKeyword(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0
@g = internal global ptr addrspace(3) @smem, section "global"
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr @g, ptr %slot, align 8
  ret void
}
`)
	got, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].State != StateGenericRequired || !bytes.Equal(got, ir) {
		t.Fatalf("generic global with shared pointer element was inferred shared: state=%v\n%s", report.Slots["@kernel/%slot"].State, got)
	}
}

func TestAddrSpace_TypedPointerSlotUsesConsistentTypes(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i32 0
define void @kernel() {
entry:
  %slot = alloca i32*, align 8
  store i32 addrspace(3)* @smem, i32** %slot, align 8
  %p = load i32*, i32** %slot, align 8
  %v = load i32, i32* %p, align 4
  ret void
}
`)
	rewritten, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].State != StateShared {
		t.Fatalf("typed slot not inferred: %+v", report.Slots["@kernel/%slot"])
	}
	if !bytes.Contains(rewritten, []byte("%slot = alloca i32 addrspace(3)*, align 8")) {
		t.Fatalf("typed-pointer alloca was not rewritten:\n%s", rewritten)
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestAddrSpace_OpaqueInlineCastIsRemovedWhenNarrowing(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspacecast (ptr addrspace(3) @smem to ptr), ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %v = load i8, ptr %p, align 1
  ret void
}
`)
	verifyIRWithClang(t, ir)
	got, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatalf("inline-cast slot did not narrow: %+v", report.Slots["@kernel/%slot"])
	}
	if !bytes.Contains(got, []byte("store ptr addrspace(3) @smem, ptr %slot, align 8")) {
		t.Fatalf("inline cast was not removed from the stored value:\n%s", got)
	}
	verifyIRWithClang(t, got)
}

func TestInferAddressSpaces_RejectsCommentAfterClosingBrace(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
} ; end
`)
	if _, _, err := InferAddressSpaces(ir); err == nil || !strings.Contains(err.Error(), "unterminated function") {
		t.Fatalf("commented closing brace error = %v, want refusal", err)
	}
}

func TestAddrSpace_ExampleOutputIsValidIR(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global [16 x float] undef, align 4
define void @example() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`)
	rewritten, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	verifyIRWithClang(t, rewritten)
	assertIndependentSlotTypes(t, rewritten, report)
}

func TestAddrSpace_LoadedValueUsedAsPointerAndStoredValueStaysGeneric(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0, align 1
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  store ptr %p, ptr %p, align 8
  ret void
}
`)
	verifyIRWithClang(t, ir)
	rewritten, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].State != StateGenericRequired {
		t.Fatalf("slot state = %v, want generic-required", report.Slots["@kernel/%slot"].State)
	}
	if !bytes.Equal(rewritten, ir) {
		t.Fatalf("slot with a stored-value escape was rewritten:\n%s", rewritten)
	}
	verifyIRWithClang(t, rewritten)
}

func TestAddrSpace_UnrewritableSupportedFormsStayValidAndGeneric(t *testing.T) {
	tests := []struct {
		name     string
		slotType string
		store    string
		load     string
		use      string
	}{
		{
			name:     "opaque store addrspace zero",
			slotType: "ptr",
			store:    "store ptr addrspace(3) @smem, ptr %slot, align 8",
			load:     "%p = load ptr, ptr %slot, align 8",
			use:      "store i8 7, ptr addrspace(0) %p, align 1",
		},
		{
			name:     "opaque load addrspace zero",
			slotType: "ptr",
			store:    "store ptr addrspace(3) @smem, ptr %slot, align 8",
			load:     "%p = load ptr, ptr %slot, align 8",
			use:      "%value = load i8, ptr addrspace(0) %p, align 1",
		},
		{
			name:     "typed store addrspace zero",
			slotType: "i8*",
			store:    "store i8 addrspace(3)* @smem, i8** %slot, align 8",
			load:     "%p = load i8*, i8** %slot, align 8",
			use:      "store i8 7, i8 addrspace(0)* %p, align 1",
		},
		{
			name:     "typed load addrspace zero",
			slotType: "i8*",
			store:    "store i8 addrspace(3)* @smem, i8** %slot, align 8",
			load:     "%p = load i8*, i8** %slot, align 8",
			use:      "%value = load i8, i8 addrspace(0)* %p, align 1",
		},
		{
			name:     "extra pointer whitespace",
			slotType: "ptr",
			store:    "store ptr addrspace(3) @smem, ptr %slot, align 8",
			load:     "%p = load ptr, ptr %slot, align 8",
			use:      "store i8 7, ptr  %p, align 1",
		},
		{
			name:     "extra whitespace after load",
			slotType: "ptr",
			store:    "store ptr addrspace(3) @smem, ptr %slot, align 8",
			load:     "%p = load  ptr, ptr %slot, align 8",
			use:      "store i8 7, ptr %p, align 1",
		},
		{
			name:     "extra whitespace before load comma",
			slotType: "ptr",
			store:    "store ptr addrspace(3) @smem, ptr %slot, align 8",
			load:     "%p = load ptr , ptr %slot, align 8",
			use:      "store i8 7, ptr %p, align 1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allocType := tt.slotType
			irText := strings.Join([]string{
				`target datalayout = "e-p:64:64:64-p3:32:32:32"`,
				`@smem = internal addrspace(3) global i8 0, align 1`,
				`define void @kernel() {`,
				`entry:`,
				`  %slot = alloca SLOT_TYPE, align 8`,
				`  STORE_LINE`,
				`  LOAD_LINE`,
				`  USE_LINE`,
				`  ret void`,
				`}`,
			}, "\n") + "\n"
			ir := []byte(strings.NewReplacer(
				"SLOT_TYPE", allocType,
				"STORE_LINE", tt.store,
				"LOAD_LINE", tt.load,
				"USE_LINE", tt.use,
			).Replace(irText))
			verifyIRWithClang(t, ir)
			rewritten, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			info := report.Slots["@kernel/%slot"]
			if info.State == StateShared {
				if bytes.Equal(rewritten, ir) || len(report.RewrittenSlots) == 0 {
					t.Fatalf("reported a narrowed slot without rewriting it:\n%s", rewritten)
				}
			} else if info.State != StateGenericRequired || !bytes.Equal(rewritten, ir) {
				t.Fatalf("unrewritable input did not fail closed: state=%v\n%s", info.State, rewritten)
			}
			verifyIRWithClang(t, rewritten)
		})
	}
}
