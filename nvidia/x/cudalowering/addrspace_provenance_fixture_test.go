package cudalowering

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvvm"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

const (
	twoSlotAddressSpaceSeed = uint32(35)
	twoSlotSlotA            = "@kernel/%slot_a"
	twoSlotSlotB            = "@kernel/%slot_b"
	twoSlotSourceFile       = "/fixture/addrspace_two_slot_ab.ll"
	twoSlotLineA            = 10
	twoSlotColA             = 3
	twoSlotLineB            = 20
	twoSlotColB             = 7
)

// twoSlotAddressSpaceABInput returns a single NVVM 2.0 IR module containing two
// distinguishable shared-memory pointer slots (%slot_a and %slot_b) with
// distinct direct debug locations and distinct numerical contributions to the
// kernel output.
func twoSlotAddressSpaceABInput() []byte {
	return []byte(`target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@shared_a = internal addrspace(3) global i32 undef, align 4
@shared_b = internal addrspace(3) global i32 undef, align 4

define void @kernel(ptr addrspace(1) %out, i32 %input) !dbg !2 {
entry:
  %slot_a = alloca ptr, align 8
  %slot_b = alloca ptr, align 8
  %seed_a = add i32 %input, 3
  %seed_b = shl i32 %input, 1
  store i32 %seed_a, ptr addrspace(3) @shared_a, align 4
  store i32 %seed_b, ptr addrspace(3) @shared_b, align 4
  store ptr addrspacecast (ptr addrspace(3) @shared_a to ptr), ptr %slot_a, align 8
  store ptr addrspacecast (ptr addrspace(3) @shared_b to ptr), ptr %slot_b, align 8
  %pa = load ptr, ptr %slot_a, align 8
  %pb = load ptr, ptr %slot_b, align 8
  %va = load i32, ptr %pa, align 4, !dbg !10
  %vb = load i32, ptr %pb, align 4, !dbg !20
  %weighted_a = mul i32 %va, 5
  %weighted_b = mul i32 %vb, 11
  %sum = add i32 %weighted_a, %weighted_b
  %result = add i32 %sum, 7
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{ptr @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 2}
!2 = distinct !DISubprogram(name: "kernel", scope: !3, file: !3, line: 1)
!3 = !DIFile(filename: "addrspace_two_slot_ab.ll", directory: "/fixture")
!10 = !DILocation(line: 10, column: 3, scope: !2)
!20 = !DILocation(line: 20, column: 7, scope: !2)
`)
}

func twoSlotContributionA(input uint32) uint32 {
	return (input + 3) * 5
}

func twoSlotContributionB(input uint32) uint32 {
	return (input << 1) * 11
}

func twoSlotExpectedOutput(input uint32) uint32 {
	return twoSlotContributionA(input) + twoSlotContributionB(input) + 7
}

func verifyTwoSlotAddressSpaceOutput(input, got uint32) error {
	wantA := twoSlotContributionA(input)
	wantB := twoSlotContributionB(input)
	want := wantA + wantB + 7
	if got == want {
		return nil
	}
	switch got {
	case wantB + 7:
		return fmt.Errorf("output=%d omits %s contribution (%d); want %d", got, twoSlotSlotA, wantA, want)
	case wantA + 7:
		return fmt.Errorf("output=%d omits %s contribution (%d); want %d", got, twoSlotSlotB, wantB, want)
	case (input<<1)*5 + (input+3)*11 + 7:
		return fmt.Errorf("output=%d swaps %s and %s contributions; want %d", got, twoSlotSlotA, twoSlotSlotB, want)
	case (input+3)*5 + (input+3)*11 + 7:
		return fmt.Errorf("output=%d aliases both slots to %s seed; want %d", got, twoSlotSlotA, want)
	case (input<<1)*5 + (input<<1)*11 + 7:
		return fmt.Errorf("output=%d aliases both slots to %s seed; want %d", got, twoSlotSlotB, want)
	default:
		return fmt.Errorf("output=%d, want %d (%s=%d + %s=%d + 7)", got, want, twoSlotSlotA, wantA, twoSlotSlotB, wantB)
	}
}

func sha256Hex(b []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(b))
}

func verifyTwoSlotInferenceOrigins(original, rewritten []byte, report *nvvmx.InferenceReport) (*nvvmx.SlotProvenance, error) {
	if report == nil {
		return nil, fmt.Errorf("nil inference report")
	}
	provenance, err := report.Provenance()
	if err != nil {
		return nil, fmt.Errorf("provenance: %w", err)
	}
	origSHA := sha256Hex(original)
	prepSHA := sha256Hex(rewritten)
	if provenance.OriginalSHA256 != origSHA || provenance.PreparedSHA256 != prepSHA || origSHA == prepSHA {
		return nil, fmt.Errorf("unexpected IR provenance digests: orig=%s prep=%s", provenance.OriginalSHA256, provenance.PreparedSHA256)
	}
	if len(provenance.Slots) != 2 {
		return nil, fmt.Errorf("inferred slots=%d, want 2", len(provenance.Slots))
	}
	wantSlots := []struct {
		slot  string
		value string
		line  int
		col   int
	}{
		{twoSlotSlotA, "%pa", twoSlotLineA, twoSlotColA},
		{twoSlotSlotB, "%pb", twoSlotLineB, twoSlotColB},
	}
	for i, want := range wantSlots {
		info, ok := report.Slots[want.slot]
		if !ok || info.Effective != nvvmx.AddrSpaceShared || info.State != nvvmx.StateShared {
			return nil, fmt.Errorf("slot %s not inferred shared: %+v", want.slot, info)
		}
		origin := provenance.Slots[i]
		if origin.Slot != want.slot || origin.Reason != "" || len(origin.Uses) != 1 {
			return nil, fmt.Errorf("slot %d origin mismatch: %+v", i, origin)
		}
		use := origin.Uses[0]
		wantLoc := nvvmx.SourceLocation{Function: "kernel", File: twoSlotSourceFile, Line: want.line, Column: want.col}
		if use.Value != want.value || use.Operation != "load" || use.IRLine < 1 || use.Location != wantLoc {
			return nil, fmt.Errorf("slot %s terminal use mismatch: %+v, want loc %+v", want.slot, use, wantLoc)
		}
	}
	if provenance.Slots[0].Slot == provenance.Slots[1].Slot || provenance.Slots[0].Uses[0].Location == provenance.Slots[1].Uses[0].Location {
		return nil, fmt.Errorf("two-slot fixture origins are not distinct: %+v", provenance.Slots)
	}
	for _, needle := range []string{
		"%slot_a = alloca ptr addrspace(3), align 8",
		"%slot_b = alloca ptr addrspace(3), align 8",
		"store ptr addrspace(3) @shared_a, ptr %slot_a, align 8",
		"store ptr addrspace(3) @shared_b, ptr %slot_b, align 8",
		"%pa = load ptr addrspace(3), ptr %slot_a, align 8",
		"%pb = load ptr addrspace(3), ptr %slot_b, align 8",
		"%va = load i32, ptr addrspace(3) %pa, align 4, !dbg !10",
		"%vb = load i32, ptr addrspace(3) %pb, align 4, !dbg !20",
	} {
		if !bytes.Contains(rewritten, []byte(needle)) {
			return nil, fmt.Errorf("rewritten IR missing %q", needle)
		}
	}
	return provenance, nil
}

func verifyTwoSlotBeforeReport(report *Report, origSHA, prepSHA, ptxSHA, sassSHA string) error {
	if report == nil {
		return fmt.Errorf("nil before report")
	}
	if len(report.Slots) != 2 || report.Slots[0].Slot != twoSlotSlotA || report.Slots[1].Slot != twoSlotSlotB {
		return fmt.Errorf("before report slots=%+v, want [%s %s]", report.Slots, twoSlotSlotA, twoSlotSlotB)
	}
	slotByID := make(map[string]SlotEvidence, 2)
	for _, s := range report.Slots {
		if s.Reason != "" {
			return fmt.Errorf("before slot %s unavailable: %s", s.Slot, s.Reason)
		}
		if s.OriginalSHA256 != origSHA || s.PreparedSHA256 != prepSHA || s.PTXSHA256 != ptxSHA || s.SASSSHA256 != sassSHA {
			return fmt.Errorf("before slot %s digest mismatch: %+v", s.Slot, s)
		}
		if len(s.Uses) != 1 || (len(s.Uses[0].SharedPCs) == 0 && len(s.Uses[0].ConversionPCs) == 0) || len(s.Uses[0].PTXLines) == 0 {
			return fmt.Errorf("before slot %s lacks terminal-use coverage: %+v", s.Slot, s.Uses)
		}
		slotByID[s.Slot] = s
	}
	if report.Err() == nil || len(report.Divergences) == 0 {
		return fmt.Errorf("before report contains no attributed shared conversions")
	}
	sassConversions := 0
	for _, div := range report.Divergences {
		if div.Kind != SharedConversion {
			continue
		}
		s, ok := slotByID[div.Slot]
		if !ok || div.IRLine != s.Uses[0].IRLine {
			return fmt.Errorf("unattributed or mismatched conversion divergence: %+v", div)
		}
		if s.Verified {
			return fmt.Errorf("slot %s with conversion divergence marked verified", div.Slot)
		}
		if div.SASSCount > 0 {
			if div.SASSAddress == 0 || len(s.Uses[0].ConversionPCs) == 0 {
				return fmt.Errorf("SASS conversion divergence missing PC witness: %+v", div)
			}
			sassConversions += div.SASSCount
		}
	}
	if sassConversions == 0 {
		return fmt.Errorf("before report has no recognized SASS shared conversion: %+v", report.Divergences)
	}
	return nil
}

func verifyTwoSlotAfterReport(report *Report, origSHA, prepSHA, ptxSHA, sassSHA string) error {
	if report == nil {
		return fmt.Errorf("nil after report")
	}
	if report.Err() != nil || len(report.Divergences) != 0 {
		return fmt.Errorf("after report has divergences: %v (%+v)", report.Err(), report.Divergences)
	}
	if len(report.Slots) != 2 || report.Slots[0].Slot != twoSlotSlotA || report.Slots[1].Slot != twoSlotSlotB {
		return fmt.Errorf("after report slots=%+v, want [%s %s]", report.Slots, twoSlotSlotA, twoSlotSlotB)
	}
	for _, s := range report.Slots {
		if !s.Verified || s.Reason != "" {
			return fmt.Errorf("after slot %s not verified: %+v", s.Slot, s)
		}
		if s.OriginalSHA256 != origSHA || s.PreparedSHA256 != prepSHA || s.PTXSHA256 != ptxSHA || s.SASSSHA256 != sassSHA {
			return fmt.Errorf("after slot %s digest mismatch: %+v", s.Slot, s)
		}
		if len(s.Uses) != 1 || len(s.Uses[0].SharedPCs) == 0 || len(s.Uses[0].ConversionPCs) != 0 || len(s.Uses[0].PTXLines) == 0 {
			return fmt.Errorf("after slot %s lacks direct shared witness coverage: %+v", s.Slot, s.Uses)
		}
	}
	return nil
}

// Supplied-artifact host controls for the two-slot fixture. These authored PTX
// and SASS texts exercise the real parsers and per-slot provenance verifier on
// CPU without invoking clang, NVVM, ptxas, nvdisasm, or CUDA device APIs; they
// do not attest compiler metadata acceptance on target hardware.
const twoSlotMixedPTX = `.version 8.0
.target sm_121
.address_size 64
.file 1 "/fixture/addrspace_two_slot_ab.ll"
.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u32 kernel_param_1
)
{
	.loc 1 5 1
	st.shared.u32 [%r0], %r9;
	.loc 1 20 7
	ld.u32 %r8, [%rd2];
	.loc 1 10 3
	ld.shared.u32 %r11, [%r10];
	ret;
}
`

func twoSlotMixedSASS(ptx string) string {
	line := func(op string) int {
		for i, s := range strings.Split(ptx, "\n") {
			if strings.Contains(s, op) {
				return i + 1
			}
		}
		panic("missing PTX opcode in two-slot fixture: " + op)
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

const twoSlotAfterPTX = `.version 8.0
.target sm_121
.address_size 64
.file 1 "/fixture/addrspace_two_slot_ab.ll"
.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u32 kernel_param_1
)
{
	.loc 1 5 1
	st.shared.u32 [%r0], %r9;
	.loc 1 10 3
	ld.shared.u32 %r11, [%r10];
	.loc 1 20 7
	ld.shared.u32 %r8, [%r0];
	ret;
}
`

func twoSlotAfterSASS(ptx string) string {
	lineA, lineB, lineStore := 0, 0, 0
	for i, s := range strings.Split(ptx, "\n") {
		switch {
		case strings.Contains(s, "st.shared.u32"):
			lineStore = i + 1
		case strings.Contains(s, "%r11"):
			lineA = i + 1
		case strings.Contains(s, "%r8"):
			lineB = i + 1
		}
	}
	return fmt.Sprintf(`code for sm_121
Function : kernel
/*0000*/ MOV R0, 0x400;
/*0010*/ MOV R10, 0x500;
//## File ".nv_debug_ptx_txt", line %d
/*0020*/ STS [R0], R9;
//## File ".nv_debug_ptx_txt", line %d
/*0030*/ LDS R11, [R10];
//## File ".nv_debug_ptx_txt", line %d
/*0040*/ LDS R8, [R0];
/*0050*/ EXIT;
`, lineStore, lineA, lineB)
}

func assertNoNativeLibsLoaded(t *testing.T) {
	t.Helper()
	if driver.Loaded() || nvvm.Loaded() {
		t.Fatalf("native library loaded before host control: driver=%v nvvm=%v", driver.Loaded(), nvvm.Loaded())
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvvm.Loaded() {
			t.Errorf("native library loaded during host control: driver=%v nvvm=%v", driver.Loaded(), nvvm.Loaded())
		}
	})
}

func TestTwoSlotProvenanceFixtureOrigins(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	orig := twoSlotAddressSpaceABInput()
	rewritten, report, err := nvvmx.InferAddressSpaces(orig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := verifyTwoSlotInferenceOrigins(orig, rewritten, report); err != nil {
		t.Fatal(err)
	}
}

func TestTwoSlotProvenanceAttribution(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	orig := twoSlotAddressSpaceABInput()
	rewritten, inference, err := nvvmx.InferAddressSpaces(orig)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := verifyTwoSlotInferenceOrigins(orig, rewritten, inference)
	if err != nil {
		t.Fatal(err)
	}
	sassText := twoSlotMixedSASS(twoSlotMixedPTX)
	if _, err := cudaptx.Parse(strings.NewReader(twoSlotMixedPTX)); err != nil {
		t.Fatalf("parse mixed two-slot PTX: %v", err)
	}
	if _, err := cudasass.Parse(strings.NewReader(sassText)); err != nil {
		t.Fatalf("parse mixed two-slot SASS: %v", err)
	}
	report, err := VerifyInferredSharedConversions(orig, inference, []byte(twoSlotMixedPTX), []byte(sassText))
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyTwoSlotBeforeReport(report, provenance.OriginalSHA256, provenance.PreparedSHA256, sha256Hex([]byte(twoSlotMixedPTX)), sha256Hex([]byte(sassText))); err != nil {
		t.Fatal(err)
	}
	if len(report.Divergences) != 1 || report.Divergences[0].Slot != twoSlotSlotB || report.Divergences[0].SASSAddress != 0x60 || report.Divergences[0].IRLine != provenance.Slots[1].Uses[0].IRLine {
		t.Fatalf("expected B-only conversion attribution, got %+v", report.Divergences)
	}
	if !report.Slots[0].Verified || report.Slots[1].Verified {
		t.Fatalf("slot verification mismatch: A=%+v B=%+v", report.Slots[0], report.Slots[1])
	}

	// Swapping the source locations of slot B and slot A moves the conversion
	// finding to slot A while slot B becomes verified, never implicating both.
	swappedPTX := strings.NewReplacer(
		".loc 1 20 7", ".loc 1 10 3",
		".loc 1 10 3", ".loc 1 20 7",
	).Replace(twoSlotMixedPTX)
	swappedSASS := twoSlotMixedSASS(swappedPTX)
	moved, err := VerifyInferredSharedConversions(orig, inference, []byte(swappedPTX), []byte(swappedSASS))
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyTwoSlotBeforeReport(moved, provenance.OriginalSHA256, provenance.PreparedSHA256, sha256Hex([]byte(swappedPTX)), sha256Hex([]byte(swappedSASS))); err != nil {
		t.Fatal(err)
	}
	if len(moved.Divergences) != 1 || moved.Divergences[0].Slot != twoSlotSlotA || moved.Divergences[0].IRLine != provenance.Slots[0].Uses[0].IRLine {
		t.Fatalf("expected conversion attribution to move to %s, got %+v", twoSlotSlotA, moved.Divergences)
	}
	if moved.Slots[0].Verified || !moved.Slots[1].Verified {
		t.Fatalf("moved slot verification mismatch: A=%+v B=%+v", moved.Slots[0], moved.Slots[1])
	}

	// Moving B's conversion location onto A's location without witnessing B
	// must refuse coverage rather than implicating both slots.
	collidingPTX := strings.Replace(twoSlotMixedPTX, ".loc 1 20 7", ".loc 1 10 3", 1)
	collidingReport, err := VerifyInferredSharedConversions(orig, inference, []byte(collidingPTX), []byte(twoSlotMixedSASS(collidingPTX)))
	if err == nil {
		t.Fatalf("expected refusal when B's location collides with A and leaves B unwitnessed: %+v", collidingReport)
	}
	for _, s := range collidingReport.Slots {
		if s.Verified {
			t.Fatalf("colliding location marked slot verified: %+v", s)
		}
	}
}

func TestTwoSlotProvenanceAfterCoverage(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	orig := twoSlotAddressSpaceABInput()
	rewritten, inference, err := nvvmx.InferAddressSpaces(orig)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := verifyTwoSlotInferenceOrigins(orig, rewritten, inference)
	if err != nil {
		t.Fatal(err)
	}
	sassText := twoSlotAfterSASS(twoSlotAfterPTX)
	if _, err := cudaptx.Parse(strings.NewReader(twoSlotAfterPTX)); err != nil {
		t.Fatalf("parse after two-slot PTX: %v", err)
	}
	if _, err := cudasass.Parse(strings.NewReader(sassText)); err != nil {
		t.Fatalf("parse after two-slot SASS: %v", err)
	}
	report, err := VerifyInferredSharedConversions(orig, inference, []byte(twoSlotAfterPTX), []byte(sassText))
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyTwoSlotAfterReport(report, provenance.OriginalSHA256, provenance.PreparedSHA256, sha256Hex([]byte(twoSlotAfterPTX)), sha256Hex([]byte(sassText))); err != nil {
		t.Fatal(err)
	}
}

func TestTwoSlotProvenanceRefusals(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	baseIR := string(twoSlotAddressSpaceABInput())
	baseSASS := twoSlotAfterSASS(twoSlotAfterPTX)

	for _, tc := range []struct {
		name    string
		ir      string
		ptx     string
		sass    string
		wantErr string
	}{
		{
			name:    "omit slot A terminal use in SASS",
			ir:      baseIR,
			ptx:     twoSlotAfterPTX,
			sass:    strings.Replace(baseSASS, "LDS R11, [R10]", "NOP", 1),
			wantErr: "terminal use missing or elided in observed SASS",
		},
		{
			name:    "omit slot B terminal use in SASS",
			ir:      baseIR,
			ptx:     twoSlotAfterPTX,
			sass:    strings.Replace(baseSASS, "LDS R8, [R0]", "NOP", 1),
			wantErr: "terminal use missing or elided in observed SASS",
		},
		{
			name:    "omit slot A debug location in IR",
			ir:      strings.Replace(baseIR, ", !dbg !10", "", 1),
			ptx:     twoSlotAfterPTX,
			sass:    baseSASS,
			wantErr: "terminal use lacks a direct resolved location",
		},
		{
			name:    "omit slot B debug location in IR",
			ir:      strings.Replace(baseIR, ", !dbg !20", "", 1),
			ptx:     twoSlotAfterPTX,
			sass:    baseSASS,
			wantErr: "terminal use lacks a direct resolved location",
		},
		{
			name:    "replace slot A SASS load witness with store",
			ir:      baseIR,
			ptx:     twoSlotAfterPTX,
			sass:    strings.Replace(baseSASS, "LDS R11, [R10]", "STS [R10], R11", 1),
			wantErr: "SASS memory operation conflicts with IR use",
		},
		{
			name:    "replace slot B SASS load witness with store",
			ir:      baseIR,
			ptx:     twoSlotAfterPTX,
			sass:    strings.Replace(baseSASS, "LDS R8, [R0]", "STS [R0], R8", 1),
			wantErr: "SASS memory operation conflicts with IR use",
		},
		{
			name:    "replace slot A PTX load with store",
			ir:      baseIR,
			ptx:     strings.Replace(twoSlotAfterPTX, "ld.shared.u32 %r11, [%r10]", "st.shared.u32 [%r10], %r11", 1),
			sass:    baseSASS,
			wantErr: "memory operation conflicts with IR use",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, inference, err := nvvmx.InferAddressSpaces([]byte(tc.ir))
			if err != nil {
				t.Fatal(err)
			}
			report, err := VerifyInferredSharedConversions([]byte(tc.ir), inference, []byte(tc.ptx), []byte(tc.sass))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("VerifyInferredSharedConversions err = %v, want substring %q (report=%+v)", err, tc.wantErr, report)
			}
			if report == nil {
				t.Fatal("expected non-nil partial report on refusal")
			}
			for _, s := range report.Slots {
				if s.Verified {
					t.Fatalf("refused analysis marked slot verified: %+v", s)
				}
			}
		})
	}
}

func TestTwoSlotProvenanceNumericalOracle(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	seed := twoSlotAddressSpaceSeed
	want := twoSlotExpectedOutput(seed)
	if want != 967 {
		t.Fatalf("twoSlotExpectedOutput(%d) = %d, want 967", seed, want)
	}
	if err := verifyTwoSlotAddressSpaceOutput(seed, want); err != nil {
		t.Fatalf("oracle rejected valid two-slot output: %v", err)
	}
	for _, tc := range []struct {
		name    string
		got     uint32
		wantSub string
	}{
		{"drop slot A contribution", twoSlotContributionB(seed) + 7, twoSlotSlotA},
		{"drop slot B contribution", twoSlotContributionA(seed) + 7, twoSlotSlotB},
		{"swap slot A and B contributions", (seed<<1)*5 + (seed+3)*11 + 7, "swaps"},
		{"alias both slots to slot A", (seed+3)*5 + (seed+3)*11 + 7, twoSlotSlotA},
		{"alias both slots to slot B", (seed<<1)*5 + (seed<<1)*11 + 7, twoSlotSlotB},
		{"legacy single-slot constant", 42, "want 967"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := verifyTwoSlotAddressSpaceOutput(seed, tc.got)
			if err == nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("verifyTwoSlotAddressSpaceOutput(%d, %d) = %v, want error containing %q", seed, tc.got, err, tc.wantSub)
			}
		})
	}
}

// withAddressSpaceNativeLease is the native consumer's sole entry point to
// tools, driver initialization, compilation and device execution.
func withAddressSpaceNativeLease(getenv func(string) string, run func()) error {
	if _, err := cudalock.Inherited(getenv); err != nil {
		return err
	}
	run()
	return nil
}

func TestTwoSlotNativeAdmission(t *testing.T) {
	assertNoNativeLibsLoaded(t)
	dir := filepath.Join(t.TempDir(), "lock")
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	}()
	unavailable := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(unavailable, []byte("not a namespace directory"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		held string
		dir  string
		want bool
	}{
		{"unset", "", dir, false},
		{"presence_only", "1", dir, false},
		{"malformed_pid", "bogus", dir, false},
		{"mismatched_owner", strconv.Itoa(os.Getpid() + 1), dir, false},
		{"unavailable_namespace", strconv.Itoa(os.Getpid()), unavailable, false},
		{"canonical_live_owner", strconv.Itoa(os.Getpid()), dir, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			err := withAddressSpaceNativeLease(func(key string) string {
				switch key {
				case "GPU_LOCK_HELD":
					return tc.held
				case "GPU_LOCKDIR":
					return tc.dir
				}
				return ""
			}, func() { calls++ })
			if tc.want {
				if err != nil || calls != 1 {
					t.Fatalf("admission = %v, callback calls = %d; want one admitted callback", err, calls)
				}
			} else if !errors.Is(err, cudalock.ErrNotInherited) || calls != 0 {
				t.Fatalf("admission = %v, callback calls = %d; want refused without callback", err, calls)
			}
		})
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	calls := 0
	err = withAddressSpaceNativeLease(func(key string) string {
		if key == "GPU_LOCK_HELD" {
			return strconv.Itoa(os.Getpid())
		}
		if key == "GPU_LOCKDIR" {
			return dir
		}
		return ""
	}, func() { calls++ })
	if !errors.Is(err, cudalock.ErrNotInherited) || calls != 0 {
		t.Fatalf("released owner admission = %v, callback calls = %d; want refused without callback", err, calls)
	}
}
