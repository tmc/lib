package nvvmx

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"
)

func TestAddrSpace_InfersSharedSlot(t *testing.T) {
	if !supportedSlotReference("store ptr addrspace(3) @smem, ptr %slot, align 8", "%slot") {
		t.Fatal("direct store not recognized")
	}
	if !supportedSlotReference("%p = load ptr, ptr %slot, align 8", "%slot") {
		t.Fatal("direct load not recognized")
	}
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
	_, report, err := InferAddressSpaces([]byte(ir))
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
define void @kernel(ptr addrspace(1) %gptr) {
entry:
  %slot = alloca ptr, align 8
  %gep = getelementptr ptr, ptr %slot, i64 0
  %cast = addrspacecast ptr %slot to ptr addrspace(3)
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %LOAD
  %WRITE
  ret void
}
`
	tests := []struct {
		name, line string
	}{
		{"metadata", "store ptr addrspace(1) %gptr, ptr %slot, !tbaa !1"},
		{"atomic store", "store atomic ptr addrspace(1) %gptr, ptr %slot seq_cst, align 8"},
		{"store through gep", "store ptr addrspace(3) @smem, ptr %gep, align 8"},
		{"escape to call", "call void @consume(ptr %slot)"},
		{"atomicrmw", "%old = atomicrmw xchg ptr %slot, ptr addrspace(3) @smem seq_cst"},
		{"cmpxchg", "%pair = cmpxchg ptr %slot, ptr addrspace(3) @smem, ptr addrspace(3) @smem seq_cst seq_cst"},
		{"memcpy destination", "call void @llvm.memcpy.p0.p0.i64(ptr %slot, ptr %gptr, i64 8, i1 false)"},
		{"unknown stored value", "store ptr %unknown, ptr %slot, align 8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := strings.Replace(base, "%LOAD", "", 1)
			ir = strings.Replace(ir, "%WRITE", tt.line, 1)
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
		"store ptr addrspace(3) addrspacecast (ptr addrspace(7) %cluster to ptr addrspace(3)), ptr %slot, align 8",
		"%cluster = call ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p, i32 %rank)\n  store ptr addrspace(3) %cluster, ptr %slot, align 8",
	} {
		ir := `define void @kernel(ptr addrspace(7) %cluster, ptr addrspace(3) %p, i32 %rank) {
entry:
  %slot = alloca ptr, align 8
  ` + value + `
  ret void
}
`
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
	_, report, err := InferAddressSpaces([]byte(ir))
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
}

func TestInferAddressSpaces_LongLine(t *testing.T) {
	comment := ";" + strings.Repeat("x", 140_000) + "\r\n"
	ir := []byte(comment + "@smem = internal addrspace(3) global i32 undef\r\n" +
		"define void @kernel() {\r\nentry:\r\n" +
		"  %slot = alloca ptr, align 8\r\n" +
		"  store ptr addrspace(3) @smem, ptr %slot, align 8\r\n" +
		"  %p = load ptr, ptr %slot, align 8\r\n  ret void\r\n}\r\n")
	got, _, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(got, []byte(comment)) || !bytes.Contains(got, []byte("  %slot = alloca ptr addrspace(3), align 8\r\n")) || !bytes.HasSuffix(got, []byte("}\r\n")) {
		t.Fatalf("long line or CRLF boundaries were not preserved (got %d bytes, input %d)", len(got), len(ir))
	}
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
	_, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@first/%slot"].State != StateShared || report.Slots["@second/%slot"].State != StateGenericRequired {
		t.Fatalf("per-function slot states = %+v", report.Slots)
	}
}

func TestAddrSpace_OutputIsValidIR(t *testing.T) {
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang is not on PATH")
	}
	ir := []byte(`target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"
@smem = internal addrspace(3) global [4 x i8] undef
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  ret void
}
`)
	rewritten, _, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(clang, "-x", "ir", "-S", "-emit-llvm", "-o", "/dev/null", "-")
	cmd.Stdin = bytes.NewReader(rewritten)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("clang rejected rewritten IR: %v\n%s\n%s", err, out, rewritten)
	}
}
