package cudalowering

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

// Authored debug-metadata and compiler-text fixtures. No compiler was run to
// generate them and they are not a native metadata-format qualification.
const slotIR = `@ga = internal addrspace(3) global i32 undef, align 4
@gb = internal addrspace(3) global i32 undef, align 4
define void @kernel() !dbg !2 {
entry:
 %a = alloca ptr, align 8
 %b = alloca ptr, align 8
 store ptr addrspacecast (ptr addrspace(3) @ga to ptr), ptr %a, align 8
 store ptr addrspacecast (ptr addrspace(3) @gb to ptr), ptr %b, align 8
 %pa = load ptr, ptr %a, align 8
 %pb = load ptr, ptr %b, align 8
 %va = load i32, ptr %pa, align 4, !dbg !10
 %vb = load i32, ptr %pb, align 4, !dbg !20
 ret void
}
!1 = !DIFile(filename: "slots.ll", directory: "/fixture")
!2 = distinct !DISubprogram(name: "kernel", scope: !1, file: !1, line: 1)
!10 = !DILocation(line: 10, column: 1, scope: !2)
!20 = !DILocation(line: 20, column: 1, scope: !2)
`
const slotPTX = `.version 8.0
.target sm_121
.address_size 64
.file 1 "/fixture/slots.ll"
.visible .entry kernel() {
 .loc 1 5 1
 st.shared.u32 [%r0], %r9;
 .loc 1 20 1
 ld.u32 %r8, [%rd2];
 .loc 1 10 1
 ld.shared.u32 %r11, [%r10];
 ret;
}
`

func slotSASS(ptx string) string {
	line := func(op string) int {
		for i, s := range strings.Split(ptx, "\n") {
			if strings.Contains(s, op) {
				return i + 1
			}
		}
		panic("fixture opcode missing")
	}
	return fmt.Sprintf(`code for sm_121
Function : kernel
/*0000*/ LDCU.64 UR4, c[0x0][0x358];
/*0010*/ MOV R0, 0x400;
/*0020*/ S2R R5, SR_SWINHI;
/*0030*/ MOV R2, R0;
/*0040*/ MOV R3, R5;
//## File ".nv_debug_ptx_txt", line %d
/*0050*/ STS [R0], R9;
//## File ".nv_debug_ptx_txt", line %d
/*0060*/ LD.E R8, desc[UR4][R2.64];
/*0070*/ MOV R10, 0x500;
//## File ".nv_debug_ptx_txt", line %d
/*0080*/ LDS R11, [R10];
/*0090*/ EXIT;
`, line("st.shared"), line("ld.u32"), line("ld.shared"))
}
func inferSlots(t *testing.T, ir string) *nvvmx.InferenceReport {
	t.Helper()
	_, report, err := nvvmx.InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	return report
}
func TestSlotProvenanceAttribution(t *testing.T) {
	if driver.Loaded() {
		t.Fatal("driver loaded before host controls")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("driver loaded")
		}
	})
	inferred := inferSlots(t, slotIR)
	for _, id := range []string{"@kernel/%a", "@kernel/%b"} {
		if inferred.Slots[id].Effective != nvvmx.AddrSpaceShared {
			t.Fatal("fixture not actually inferred", inferred.Slots)
		}
	}
	report, err := VerifyInferredSharedConversions([]byte(slotIR), inferred, []byte(slotPTX), []byte(slotSASS(slotPTX)))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Divergences) != 1 || report.Divergences[0].Slot != "@kernel/%b" || report.Divergences[0].SASSAddress != 0x60 || report.Divergences[0].IRLine != 12 {
		t.Fatalf("not B-only: %+v", report.Divergences)
	}
	if len(report.Slots) != 2 || !report.Slots[0].Verified || report.Slots[1].Verified || len(report.Slots[0].Uses[0].SharedPCs) != 1 {
		t.Fatalf("coverage: %+v", report.Slots)
	}
	if report.Err() == nil {
		t.Fatal("conversion finding lost existing Report.Err vocabulary")
	}
	// Only the observed compiler source association changes. The actual inference
	// object, slot names, registers and mechanism PCs remain identical.
	moved := strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(slotPTX, ".loc 1 20 1", ".loc 1 999 1"), ".loc 1 10 1", ".loc 1 20 1"), ".loc 1 999 1", ".loc 1 10 1")
	r, err := VerifyInferredSharedConversions([]byte(slotIR), inferred, []byte(moved), []byte(slotSASS(moved)))
	if err != nil || len(r.Divergences) != 1 || r.Divergences[0].Slot != "@kernel/%a" {
		t.Fatalf("association did not move: %+v %v", r, err)
	}
}
func TestSlotProvenanceDirectCoverage(t *testing.T) {
	ptx := strings.Replace(slotPTX, "ld.u32 %r8", "ld.shared.u32 %r8", 1)
	sass := strings.Replace(slotSASS(slotPTX), "LD.E R8, desc[UR4][R2.64]", "LDS R8, [R0]", 1)
	report, err := VerifyInferredSharedConversions([]byte(slotIR), inferSlots(t, slotIR), []byte(ptx), []byte(sass))
	if err != nil || report.Err() != nil {
		t.Fatal(report, err)
	}
	for _, slot := range report.Slots {
		if !slot.Verified || slot.OriginalSHA256 == "" || slot.PreparedSHA256 == "" || slot.PTXSHA256 == "" || slot.SASSSHA256 == "" {
			t.Fatalf("unverified origin: %+v", slot)
		}
	}
}

// slotDirectSASS pairs the two authored terminal memory statements with their
// physical PTX lines. It does not derive a compiler or device observation.
func slotDirectSASS(ptx, operation string) string {
	lineA, lineB := 0, 0
	for i, line := range strings.Split(ptx, "\n") {
		if strings.Contains(line, "%r11") {
			lineA = i + 1
		}
		if strings.Contains(line, "%r8") {
			lineB = i + 1
		}
	}
	return fmt.Sprintf("code for sm_121\nFunction : kernel\n/*0000*/ MOV R0, 0x400;\n//## File \".nv_debug_ptx_txt\", line %d\n/*0010*/ %s;\n//## File \".nv_debug_ptx_txt\", line %d\n/*0020*/ LDS R8, [R0];\n/*0030*/ EXIT;\n", lineA, operation, lineB)
}

func TestSlotProvenanceMemoryDirection(t *testing.T) {
	for _, terminal := range []string{"load", "store"} {
		for _, observed := range []string{"load", "store"} {
			t.Run(terminal+"/"+observed, func(t *testing.T) {
				ir := slotIR
				ptx := strings.Replace(slotPTX, "ld.u32 %r8", "ld.shared.u32 %r8", 1)
				if terminal == "store" {
					ir = strings.Replace(ir, "%va = load i32, ptr %pa", "store i32 7, ptr %pa", 1)
					ptx = strings.Replace(ptx, "ld.shared.u32 %r11, [%r10]", "st.shared.u32 [%r10], %r11", 1)
				}
				op := "LDS R11, [R0]"
				if observed == "store" {
					op = "STS [R0], R11"
				}
				r, err := VerifyInferredSharedConversions([]byte(ir), inferSlots(t, ir), []byte(ptx), []byte(slotDirectSASS(ptx, op)))
				if terminal != observed {
					if err == nil || !strings.Contains(err.Error(), "SASS memory operation conflicts with IR use") || r == nil {
						t.Fatalf("opposite SASS operation lacks direction refusal: %+v %v", r, err)
					}
					for _, slot := range r.Slots {
						if slot.Verified {
							t.Fatalf("unavailable direction marked verified: %+v", slot)
						}
					}
					return
				}
				if err != nil || r.Err() != nil || len(r.Slots) != 2 || !r.Slots[0].Verified || !r.Slots[1].Verified {
					t.Fatalf("matching operations refused: %+v %v", r, err)
				}
			})
		}
	}
}

func TestSlotProvenancePTXConversionLocation(t *testing.T) {
	for _, tt := range []struct {
		name, suffix string
		direct       bool
	}{
		{"direct", "", true},
		{"function", ", function_name .debug_str+0", false},
		{"inline", ", inlined_at 1 90 1", false},
		{"function and inline", ", function_name .debug_str+0, inlined_at 1 90 1", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ptx := strings.Replace(slotPTX, "ld.u32 %r8", "ld.shared.u32 %r8", 1)
			prefix := ".loc 1 10 1" + tt.suffix + "\n cvta.to.shared.u64 %rd20, %rd21;\n "
			ptx = strings.Replace(ptx, ".loc 1 10 1", prefix+".loc 1 10 1", 1)
			parsed, err := cudaptx.Parse(strings.NewReader(ptx))
			if err != nil {
				t.Fatal("invalid authored fixture", err)
			}
			if len(parsed.Kernels) != 1 {
				t.Fatalf("authored fixture kernels: %d", len(parsed.Kernels))
			}
			found := false
			for _, inst := range parsed.Kernels[0].Instructions {
				if inst.Mnemonic == "cvta" {
					found = true
					if (len(inst.InlinedAt) == 0 && inst.Loc.FunctionName == "") != tt.direct {
						t.Fatalf("location fixture did not parse as intended: %+v", inst)
					}
				}
			}
			if !found {
				t.Fatal("missing conversion fixture")
			}
			r, err := VerifyInferredSharedConversions([]byte(slotIR), inferSlots(t, slotIR), []byte(ptx), []byte(slotDirectSASS(ptx, "LDS R11, [R0]")))
			if tt.direct {
				if err != nil || len(r.Divergences) != 1 || r.Divergences[0].Slot != "@kernel/%a" || r.Slots[0].Verified || !r.Slots[1].Verified {
					t.Fatalf("direct conversion lost: %+v %v", r, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "lacks a direct source location") || r == nil || len(r.Divergences) != 0 {
				t.Fatalf("unsupported conversion location attributed: %+v %v", r, err)
			}
			for _, slot := range r.Slots {
				if slot.Verified {
					t.Fatalf("unsupported location marked verified: %+v", slot)
				}
			}
		})
	}
}

func TestSlotProvenanceRefusals(t *testing.T) {
	for _, tt := range []struct{ name, ir, ptx, sass string }{
		{"ambiguous physical PTX line", slotIR, strings.Replace(slotPTX, "ld.u32 %r8, [%rd2];", "ld.u32 %r8, [%rd2]; ld.u32 %r13, [%rd3];", 1), slotSASS(slotPTX)},
		{"missing PTX line", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "line 9", "line 0", 1)},
		{"out of range", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "line 9", "line 9999", 1)},
		{"elided A", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "LDS R11, [R10]", "NOP", 1)},
		{"elided B", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "LD.E R8, desc[UR4][R2.64]", "NOP", 1)},
		{"unknown mechanism", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "SR_SWINHI", "SR_UNKNOWN", 1)},
		{"clobbered low", slotIR, slotPTX, strings.Replace(slotSASS(slotPTX), "MOV R2, R0", "MOV R2, 0x700", 1)},
		{"ambiguous location", strings.Replace(slotIR, "line: 20, column: 1", "line: 10, column: 1", 1), slotPTX, slotSASS(slotPTX)},
		{"intermediate is not source", slotIR, strings.ReplaceAll(slotPTX, ".loc ", ".loc_intermediate "), slotSASS(slotPTX)},
		{"wrong target", slotIR, strings.Replace(slotPTX, "sm_121", "sm_100", 1), slotSASS(slotPTX)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := VerifyInferredSharedConversions([]byte(tt.ir), inferSlots(t, tt.ir), []byte(tt.ptx), []byte(tt.sass))
			if err == nil {
				t.Fatalf("accepted unsupported attribution: %+v", r)
			}
			if r != nil {
				for _, s := range r.Slots {
					if s.Verified {
						t.Fatal("unavailable analysis marked clean")
					}
				}
			}
		})
	}
	t.Run("wrong input", func(t *testing.T) {
		if _, err := VerifyInferredSharedConversions([]byte(slotIR+"\n"), inferSlots(t, slotIR), []byte(slotPTX), []byte(slotSASS(slotPTX))); err == nil {
			t.Fatal("wrong original accepted")
		}
	})
	t.Run("forged report", func(t *testing.T) {
		if _, err := VerifyInferredSharedConversions([]byte(slotIR), &nvvmx.InferenceReport{}, []byte(slotPTX), []byte(slotSASS(slotPTX))); err == nil {
			t.Fatal("caller report accepted")
		}
	})
	t.Run("changed labels", func(t *testing.T) {
		r := inferSlots(t, slotIR)
		r.RewrittenSlots[0] = "@kernel/%ghost"
		if _, err := VerifyInferredSharedConversions([]byte(slotIR), r, []byte(slotPTX), []byte(slotSASS(slotPTX))); err == nil {
			t.Fatal("modified report accepted")
		}
	})
	t.Run("zero compiled result", func(t *testing.T) {
		if _, err := VerifyCompiledSharedConversions(&nvvmx.Result{PTX: []byte(slotPTX)}, []byte(slotSASS(slotPTX))); err == nil {
			t.Fatal("fabricated compiled result accepted")
		}
	})
}
func TestSlotProvenanceUnsupportedMetadata(t *testing.T) {
	for _, suffix := range []string{", !tbaa !10", ", !dbg !999", ", !dbg !10, !nontemporal !20"} {
		ir := strings.Replace(slotIR, ", !dbg !10", suffix, 1)
		result, report, err := nvvmx.InferAddressSpaces([]byte(ir))
		if err != nil {
			t.Fatal(err)
		}
		if report.Slots["@kernel/%a"].Effective != nvvmx.AddrSpaceGeneric {
			t.Fatalf("unsupported metadata inferred: %s", suffix)
		}
		if !bytes.Contains(result, []byte(suffix)) {
			t.Fatal("unsupported attachment removed")
		}
	}
}

func TestSlotProvenanceDebugRefusals(t *testing.T) {
	for _, tt := range []struct{ name, old, new string }{
		{"cyclic scope", "scope: !2)", "scope: !10)"},
		{"duplicate node", "!10 = !DILocation(line: 10, column: 1, scope: !2)", "!10 = !DILocation(line: 10, column: 1, scope: !2)\n!10 = !DILocation(line: 99, column: 1, scope: !2)"},
		{"unknown inline field", "line: 10, column: 1, scope: !2)", "line: 10, column: 1, scope: !2, inlinedAt: !20)"},
		{"wrong owner", "name: \"kernel\"", "name: \"other\""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ir := strings.Replace(slotIR, tt.old, tt.new, 1)
			if ir == slotIR {
				t.Fatal("ineffective fixture mutation")
			}
			_, r, err := nvvmx.InferAddressSpaces([]byte(ir))
			if err != nil {
				t.Fatal(err)
			}
			if r.Slots["@kernel/%a"].Effective != nvvmx.AddrSpaceGeneric {
				t.Fatal("invalid debug attachment enabled inference")
			}
		})
	}
	t.Run("node budget before mutation", func(t *testing.T) {
		var more strings.Builder
		for i := 100; i < 4300; i++ {
			fmt.Fprintf(&more, "!%d = !DILocation(line: 1, column: 1, scope: !2)\n", i)
		}
		ir := slotIR + more.String()
		got, r, err := nvvmx.InferAddressSpaces([]byte(ir))
		if err != nil {
			t.Fatal(err)
		}
		if r.Slots["@kernel/%a"].Effective != nvvmx.AddrSpaceGeneric || !bytes.Contains(got, []byte(", !dbg !10")) {
			t.Fatal("budget failure stripped debug metadata and inferred")
		}
	})
	t.Run("unrelated instruction shares location", func(t *testing.T) {
		ir := strings.Replace(slotIR, " %vb =", " %other = load i32, ptr addrspace(3) @ga, align 4, !dbg !10\n %vb =", 1)
		r, err := VerifyInferredSharedConversions([]byte(ir), inferSlots(t, ir), []byte(slotPTX), []byte(slotSASS(slotPTX)))
		if err == nil {
			t.Fatalf("shared location became proof: %+v", r)
		}
	})
}

func TestSlotProvenanceFunctionScope(t *testing.T) {
	second := strings.Join(strings.Split(slotIR, "\n")[2:], "\n")
	second = strings.ReplaceAll(second, "kernel", "second")
	second = strings.NewReplacer("!20", "!120", "!10", "!110", "!2", "!102", "!1", "!101").Replace(second)
	ir := slotIR + second
	ptx := slotPTX + strings.Replace(strings.Join(strings.Split(slotPTX, "\n")[4:], "\n"), "kernel", "second", 1)
	listing := slotSASS(slotPTX)
	offset := len(strings.Split(strings.TrimSuffix(slotPTX, "\n"), "\n")) - 4
	listingSecond := strings.Replace(slotSASS(slotPTX), "Function : kernel", "Function : second", 1)
	for _, line := range []int{7, 9, 11} {
		listingSecond = strings.Replace(listingSecond, fmt.Sprintf("line %d\n", line), fmt.Sprintf("line %d\n", line+offset), 1)
	}
	listing += listingSecond
	report, err := VerifyInferredSharedConversions([]byte(ir), inferSlots(t, ir), []byte(ptx), []byte(listing))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Slots) != 4 || len(report.Divergences) != 2 || report.Divergences[0].Slot != "@kernel/%b" || report.Divergences[1].Slot != "@second/%b" {
		t.Fatalf("function origins merged: %+v", report)
	}
}

func TestSlotProvenanceDeclinedRewritePreservesDebugBytes(t *testing.T) {
	ir := strings.Replace(slotIR, "align 4, !dbg !10", "align 4   , !dbg !10  ; kept comment", 1)
	ir = strings.Replace(ir, " ret void", " call void @escape(ptr %a)\n call void @escape(ptr %b)\n ret void", 1)
	got, report, err := nvvmx.InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.RewrittenSlots) != 0 || !bytes.Equal(got, []byte(ir)) {
		t.Fatalf("declined rewrite changed metadata bytes:\n%s", got)
	}
}
