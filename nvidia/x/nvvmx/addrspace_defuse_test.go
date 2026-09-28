package nvvmx

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestAddrSpace_NestedTypesAndUnlistedValuesStayGeneric(t *testing.T) {
	tests := []struct{ name, globals, args, body string }{
		{"nested type on generic global", `@tbl2 = internal global {ptr addrspace(3), i32} zeroinitializer, align 8
`, "", `  store ptr getelementptr ({ptr addrspace(3), i32}, ptr @tbl2, i32 0, i32 1), ptr %slot, align 8`},
		{"quoted name contains address-space words", `@"global addrspace(3)" = internal global i8 0
`, "", `  store ptr @"global addrspace(3)", ptr %slot, align 8`},
		{"nested type on generic GEP base", "", "ptr %gp", `  %g = getelementptr {i32, ptr addrspace(3)}, ptr %gp, i32 0, i32 1
  store ptr %g, ptr %slot, align 8`},
		{"unlisted select definition", `@smem = internal addrspace(3) global i8 0
`, "", `  %selected = select i1 true, ptr addrspace(3) @smem, ptr addrspace(3) @smem
  store ptr addrspace(3) %selected, ptr %slot, align 8`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(tt.globals + "define void @kernel(" + tt.args + ") {\nentry:\n  %slot = alloca ptr, align 8\n" + tt.body + "\n  ret void\n}\n")
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
				t.Fatalf("slot narrowed: %+v", report.Slots["@kernel/%slot"])
			}
			if bytes.Contains(got, []byte("alloca ptr addrspace(3)")) {
				t.Fatalf("generic value was narrowed:\n%s", got)
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_AddressSpaceABFixtureHasIndependentTypes(t *testing.T) {
	before, err := os.ReadFile("../cudalowering/testdata/addrspace_shared.ll")
	if err != nil {
		t.Fatal(err)
	}
	got, report, err := InferAddressSpaces(before)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatalf("A/B slot was not narrowed: %+v", report.Slots["@kernel/%slot"])
	}
	verifyIRWithClang(t, before)
	verifyIRWithClang(t, got)
	assertIndependentSlotTypes(t, got, report)
}

func TestAddrSpace_LoadedValuesEscapeFailClosed(t *testing.T) {
	tests := []struct {
		name       string
		tail       string
		wantShared bool
	}{
		{"supported load", `  %value = load i8, ptr %p, align 1`, true},
		{"caller memory", `  store ptr %p, ptr %out, align 8`, false},
		{"generic slot", `  store ptr %unknown, ptr %other, align 8
  store ptr %p, ptr %other, align 8
		%q = load ptr, ptr %other, align 8`, false},
		{"call", `  call void @consume(ptr %p)`, false},
		{"GEP", `  %q = getelementptr i8, ptr %p, i64 1`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(`@smem = internal addrspace(3) global [8 x i8] undef
declare void @consume(ptr)
define void @kernel(ptr %out, ptr %unknown) {
entry:
  %slot = alloca ptr, align 8
  %other = alloca ptr, align 8
  store ptr %unknown, ptr %other, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
` + tt.tail + `
  ret void
}
`)
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			want := AddrSpaceGeneric
			if tt.wantShared {
				want = AddrSpaceShared
			}
			if got := report.Slots["@kernel/%slot"].Effective; got != want {
				t.Fatalf("effective address space=%v, want %v: %+v", got, want, report.Slots["@kernel/%slot"])
			}
			verifyIRWithClang(t, got)
		})
	}
	typed := []byte(`@smem = internal addrspace(3) global i8 0
declare void @consume(i8*)
define void @kernel() {
entry:
  %slot = alloca i8*, align 8
  store i8 addrspace(3)* @smem, i8** %slot, align 8
  %p = load i8*, i8** %slot, align 8
  call void @consume(i8* %p)
  ret void
}
`)
	verifyIRWithClang(t, typed)
	got, report, err := InferAddressSpaces(typed)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
		t.Fatalf("typed escape narrowed: %+v", report.Slots["@kernel/%slot"])
	}
	verifyIRWithClang(t, got)
}

func TestAddrSpace_GenericRegisterCastsDoNotNarrowSlots(t *testing.T) {
	tests := []struct {
		name string
		ir   string
	}{
		{"cast result", `@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  %c = addrspacecast ptr addrspace(3) @smem to ptr
  store ptr %c, ptr %slot, align 8
  ret void
}`},
		{"cast GEP result", `@smem = internal addrspace(3) global [8 x i8] undef
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  %c = addrspacecast ptr addrspace(3) @smem to ptr
  %g = getelementptr i8, ptr %c, i64 4
  store ptr %g, ptr %slot, align 8
  ret void
}`},
		{"parameter cast result", `define void @kernel(ptr addrspace(3) %s) {
entry:
  %slot = alloca ptr, align 8
  %c = addrspacecast ptr addrspace(3) %s to ptr
  store ptr %c, ptr %slot, align 8
  ret void
}`},
		{"typed pointer cast result", `@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca i8*, align 8
  %c = addrspacecast i8 addrspace(3)* @smem to i8*
  store i8* %c, i8** %slot, align 8
  ret void
}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(tt.ir + "\n")
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
				t.Fatalf("generic register cast narrowed the slot: %+v", report.Slots["@kernel/%slot"])
			}
			if bytes.Contains(got, []byte("alloca ptr addrspace(3)")) || bytes.Contains(got, []byte("alloca i8 addrspace(3)*")) {
				t.Fatalf("generic register cast produced a narrowed slot:\n%s", got)
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_ParameterAddressSpaceIgnoresTypeAttributes(t *testing.T) {
	tests := []struct{ name, attribute string }{
		{"byval", `byval({ ptr addrspace(3), i32 }) align 8`},
		{"sret", `sret(ptr addrspace(3))`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(`define void @kernel(ptr ` + tt.attribute + ` %params) {
entry:
  %slot = alloca ptr, align 8
  store ptr %params, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %value = load i8, ptr %p, align 1
  ret void
}
`)
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
				t.Fatalf("nested attribute type narrowed generic parameter: %+v", report.Slots["@kernel/%slot"])
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_DependencySlotsDemoteTogether(t *testing.T) {
	storeOrders := []struct{ name, stores string }{
		{"shared then generic", `  store ptr %va, ptr %d, align 8
  store ptr %vb, ptr %d, align 8`},
		{"generic then shared", `  store ptr %vb, ptr %d, align 8
  store ptr %va, ptr %d, align 8`},
	}
	for _, tt := range storeOrders {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %d = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %a, align 8
  %va = load ptr, ptr %a, align 8
  %vb = load ptr, ptr %b, align 8
` + tt.stores + `
  ret void
}
`)
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"%a", "%b", "%d"} {
				if info := report.Slots["@kernel/"+name]; info.Effective != AddrSpaceGeneric {
					t.Errorf("%s was partially narrowed: %+v", name, info)
				}
			}
			if strings.Contains(string(got), "alloca ptr addrspace(3)") || strings.Contains(string(got), "store ptr addrspace(3) %v") {
				t.Fatalf("partial slot demotion left mismatched types:\n%s", got)
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_StoreThroughLoadedValueRewritesPointerType(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  store i8 0, ptr %p, align 1
  ret void
}
`)
	verifyIRWithClang(t, ir)
	got, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatalf("supported store through loaded value did not narrow slot: %+v", report.Slots["@kernel/%slot"])
	}
	if !bytes.Contains(got, []byte("store i8 0, ptr addrspace(3) %p, align 1")) {
		t.Fatalf("store pointer type was not rewritten:\n%s", got)
	}
	verifyIRWithClang(t, got)
	assertIndependentSlotTypes(t, got, report)
}

func TestAddrSpace_LoadedUsesAreCollectedAcrossTextOrder(t *testing.T) {
	tests := []struct{ name, body string }{
		{"phi precedes load", `entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  br label %loop
loop:
  %p = phi ptr [ null, %entry ], [ %v, %loop ]
  %v = load ptr, ptr %slot, align 8
  br i1 %again, label %loop, label %exit
exit:
  ret void`},
		{"GEP use precedes dominating load textually", `entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  br label %def
use:
  %g = getelementptr i8, ptr %v, i64 1
  %x = load i8, ptr %g, align 1
  ret void
def:
  %v = load ptr, ptr %slot, align 8
  br label %use`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte("@smem = internal addrspace(3) global [8 x i8] undef\ndefine void @kernel(i1 %again) {\n" + tt.body + "\n}\n")
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
				t.Fatalf("forward use escaped analysis: %+v", report.Slots["@kernel/%slot"])
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_UnrewrittenAccessKeepsSlotGeneric(t *testing.T) {
	tests := []struct{ name, access string }{
		{"wide integer load", `  %x = load i64, ptr %slot, align 8`},
		{"integer store", `  store i64 7, ptr %slot, align 8`},
		{"other address space load", `  %x = load ptr addrspace(1), ptr %slot, align 8`},
		{"volatile load", `  %x = load volatile ptr, ptr %slot, align 8`},
		{"volatile store", `  store volatile ptr addrspace(3) @smem, ptr %slot, align 8`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
` + tt.access + `
  ret void
}
`)
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
				t.Fatalf("unrewritten access narrowed slot: %+v", report.Slots["@kernel/%slot"])
			}
			verifyIRWithClang(t, got)
		})
	}
}

func TestAddrSpace_InlineCastsAndWholeNameUses(t *testing.T) {
	for _, global := range []string{"@smem", `@"s m"`, "@s-m"} {
		t.Run(global, func(t *testing.T) {
			ir := []byte(global + " = internal addrspace(3) global i8 0\ndefine void @kernel() {\nentry:\n  %slot = alloca ptr, align 8\n  store ptr addrspacecast (ptr addrspace(3) " + global + " to ptr), ptr %slot, align 8\n  %p = load ptr, ptr %slot, align 8\n  %v = load i8, ptr %p, align 1\n  ret void\n}\n")
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
				t.Fatalf("inline cast did not prove shared: %+v", report.Slots["@kernel/%slot"])
			}
			verifyIRWithClang(t, got)
			assertIndependentSlotTypes(t, got, report)
			if strings.Contains(string(got), "addrspacecast") {
				t.Fatalf("inline cast not stripped:\n%s", got)
			}
		})
	}
	constantGEP := []byte(`@smem = internal addrspace(3) global [4 x i8] undef
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspacecast (ptr addrspace(3) getelementptr inbounds ([4 x i8], ptr addrspace(3) @smem, i64 0, i64 1) to ptr), ptr %slot, align 8
  ret void
}
`)
	verifyIRWithClang(t, constantGEP)
	got, report, err := InferAddressSpaces(constantGEP)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceGeneric {
		t.Fatalf("unsupported constant expression narrowed slot: %+v", report.Slots["@kernel/%slot"])
	}
	verifyIRWithClang(t, got)

	nameCase := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel(ptr addrspace(1) %p.addr) {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %a = load i8, ptr %p, align 1
  %b = load i8, ptr addrspace(1) %p.addr, align 1
  ret void
}
`)
	verifyIRWithClang(t, nameCase)
	got, report, err = InferAddressSpaces(nameCase)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatalf("unrelated dotted identifier escaped: %+v", report.Slots["@kernel/%slot"])
	}
	if !strings.Contains(string(got), "%b = load i8, ptr addrspace(1) %p.addr") {
		t.Fatalf("nearby identifier was rewritten:\n%s", got)
	}
	verifyIRWithClang(t, got)
	assertIndependentSlotTypes(t, got, report)
}

func TestAddrSpace_TypedInlineCastUsesConsistentAccessTypes(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global i8 0
define void @kernel() {
entry:
  %slot = alloca i8*, align 8
  store i8* addrspacecast (i8 addrspace(3)* @smem to i8*), i8** %slot, align 8
  %p = load i8*, i8** %slot, align 8
  %v = load i8, i8* %p, align 1
  ret void
}

`)
	verifyIRWithClang(t, ir)
	got, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	if report.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatalf("typed slot not narrowed: %+v", report.Slots["@kernel/%slot"])
	}
	for _, want := range []string{
		"%slot = alloca i8 addrspace(3)*, align 8",
		"store i8 addrspace(3)* @smem, i8 addrspace(3)** %slot, align 8",
		"%p = load i8 addrspace(3)*, i8 addrspace(3)** %slot, align 8",
		"%v = load i8, i8 addrspace(3)* %p, align 1",
	} {
		if !strings.Contains(string(got), want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}
	verifyIRWithClang(t, got)
	assertIndependentSlotTypes(t, got, report)
}

func TestAddrSpace_LLVMIdentifiersAreWholeTokens(t *testing.T) {
	for _, pair := range [][2]string{{"%slot-name", "%p.addr"}, {"%slot$cash", "%-p"}, {`%"slot name"`, "%p$x"}} {
		slot, value := pair[0], pair[1]
		t.Run(slot+"/"+value, func(t *testing.T) {
			ir := []byte("@smem = internal addrspace(3) global i8 0\ndefine void @kernel() {\nentry:\n  " + slot + " = alloca ptr, align 8\n  store ptr addrspace(3) @smem, ptr " + slot + ", align 8\n  " + value + " = load ptr, ptr " + slot + ", align 8\n  %nearby = load i8, ptr " + value + ", align 1\n  ret void\n}\n")
			verifyIRWithClang(t, ir)
			got, report, err := InferAddressSpaces(ir)
			if err != nil {
				t.Fatal(err)
			}
			if report.Slots["@kernel/"+slot].Effective != AddrSpaceShared {
				t.Fatalf("slot did not narrow: %+v", report.Slots)
			}
			if !bytes.Contains(got, []byte("load i8, ptr addrspace(3) "+value)) {
				t.Fatalf("loaded identifier was not rewritten exactly:\n%s", got)
			}
			verifyIRWithClang(t, got)
			assertIndependentSlotTypes(t, got, report)
		})
	}
}

func assertIndependentSlotTypes(t *testing.T, ir []byte, report *InferenceReport) {
	t.Helper()
	for key, info := range report.Slots {
		if info.Effective != AddrSpaceShared {
			continue
		}
		slash := strings.LastIndex(key, "/")
		function, name := key[:slash], key[slash+1:]
		inFunction := false
		for _, line := range strings.Split(string(ir), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "define ") {
				inFunction = testHasIdentifier(line, function)
				continue
			}
			if inFunction && trimmed == "}" {
				inFunction = false
				continue
			}
			if !inFunction {
				continue
			}
			if !testHasIdentifier(line, name) || strings.Contains(line, "= alloca ") {
				continue
			}
			if strings.HasPrefix(trimmed, "store ") {
				parts := testSplitOperands(strings.TrimSpace(strings.TrimPrefix(trimmed, "store ")))
				if len(parts) < 2 || !strings.Contains(parts[0], "addrspace(3)") {
					t.Errorf("store to %s is not shared-typed: %s", name, line)
				}
				continue
			}
			if strings.Contains(trimmed, " = load ") {
				parts := testSplitOperands(strings.TrimSpace(strings.SplitN(trimmed, " = load ", 2)[1]))
				if len(parts) < 2 || !strings.Contains(parts[0], "addrspace(3)") {
					t.Errorf("load from %s is not shared-typed: %s", name, line)
				}
				continue
			}
			t.Errorf("unexpected use of rewritten slot %s: %s", name, line)
		}
	}
}

// This small scanner is intentionally separate from the inference lexer.
func testHasIdentifier(line, want string) bool {
	for i := 0; i < len(line); i++ {
		if line[i] != '%' && line[i] != '@' {
			continue
		}
		start := i
		i++
		if i < len(line) && line[i] == '"' {
			i++
			for i < len(line) && line[i] != '"' {
				if line[i] == '\\' {
					i++
				}
				i++
			}
			if i < len(line) {
				i++
			}
		} else {
			for i < len(line) && (line[i] >= 'a' && line[i] <= 'z' || line[i] >= 'A' && line[i] <= 'Z' || line[i] >= '0' && line[i] <= '9' || strings.ContainsRune("_.$-", rune(line[i]))) {
				i++
			}
		}
		if line[start:i] == want {
			return true
		}
		i--
	}
	return false
}

func testSplitOperands(s string) []string {
	var parts []string
	start, depth := 0, 0
	quoted, escaped := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		switch c {
		case '(', '[', '{', '<':
			depth++
		case ')', ']', '}', '>':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				parts = append(parts, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	return append(parts, strings.TrimSpace(s[start:]))
}
