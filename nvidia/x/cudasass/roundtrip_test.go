package cudasass_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// roundTripFixture is nvdisasm -c -hex output for testdata/roundtrip/kernel.cu,
// compiled for sm_121a. See testdata/roundtrip/README.md for provenance.
const roundTripFixture = "testdata/roundtrip/sm121a-roundtrip.sass"

func readRoundTripFixture(t *testing.T) *cudasass.Disassembly {
	t.Helper()
	data, err := os.ReadFile(roundTripFixture)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.Parse(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func roundTripFunction(t *testing.T) *cudasass.Function {
	t.Helper()
	d := readRoundTripFixture(t)
	fn := d.Function("intmix")
	if fn == nil {
		t.Fatalf("no intmix function in %s", roundTripFixture)
	}
	return fn
}

// TestByteIdenticalRoundTripOnCoveredSubset is the step 4 gate. Every
// instruction of a covered form must re-encode from its disassembled text to
// exactly the bytes nvdisasm printed beside it.
func TestByteIdenticalRoundTripOnCoveredSubset(t *testing.T) {
	v := cudasass.RoundTrip(roundTripFunction(t), cudasass.SM121a)
	covered, identical, uncovered := v.Counts()
	if covered == 0 {
		t.Fatalf("no instruction was covered; the gate demonstrated nothing\n%s", v)
	}
	for _, r := range v.Failures() {
		t.Errorf("%#x %s (%s): want %v, got %v, differing bits %v %s",
			r.Address, r.Opcode, r.Form, r.Want, r.Got, r.Difference, r.Err)
	}
	t.Logf("verdict: %s", v)
	t.Logf("%d covered, %d byte-identical, %d outside the subset", covered, identical, uncovered)
}

// TestRoundTripVerdictHasFailedCase requires the verdict to record at least one
// instruction it could not re-encode. A gate whose covered subset happened to
// be the whole program would report success without having been tested.
func TestRoundTripVerdictHasFailedCase(t *testing.T) {
	v := cudasass.RoundTrip(roundTripFunction(t), cudasass.SM121a)
	_, _, uncovered := v.Counts()
	if uncovered == 0 {
		t.Error("every instruction was covered; the verdict records no negative case")
	}
}

// TestFailedVerdictBlocksDependentSteps checks that a verdict with a failing
// instruction blocks steps 5 and 6, and that an empty verdict does too.
func TestFailedVerdictBlocksDependentSteps(t *testing.T) {
	for _, tt := range []struct {
		name string
		v    cudasass.Verdict
		want bool
	}{
		{"empty", cudasass.Verdict{Arch: cudasass.SM121a}, true},
		{"only uncovered", cudasass.Verdict{Arch: cudasass.SM121a, Results: []cudasass.RoundTripResult{{Opcode: "LDC"}}}, true},
		{"one failure", cudasass.Verdict{Arch: cudasass.SM121a, Results: []cudasass.RoundTripResult{
			{Opcode: "S2R", Form: "S2R Rd, SR", Identical: true},
			{Opcode: "IADD", Form: "IADD Rd, Ra, URb", Identical: false},
		}}, true},
		{"all identical", cudasass.Verdict{Arch: cudasass.SM121a, Results: []cudasass.RoundTripResult{
			{Opcode: "S2R", Form: "S2R Rd, SR", Identical: true},
		}}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.Blocks(); got != tt.want {
				t.Errorf("Blocks() = %v, want %v", got, tt.want)
			}
			if tt.v.Feasible() == tt.v.Blocks() {
				t.Error("Feasible and Blocks agree; they must be negations")
			}
		})
	}
}

// TestEncodingNotGeneralizedAcrossSMVersions checks that every entry point
// refuses an architecture it was not calibrated on, rather than decoding
// sm_121a fields out of another architecture's bits.
func TestEncodingNotGeneralizedAcrossSMVersions(t *testing.T) {
	fn := roundTripFunction(t)
	inst := fn.Instructions[0]
	for _, arch := range []cudasass.Arch{"sm_90", "sm_100a", "sm_121", ""} {
		if _, ok := cudasass.DecodeControl(inst.Encoding, arch); ok {
			t.Errorf("DecodeControl accepted %q", arch)
		}
		if _, err := cudasass.ReEncode(inst, arch); err == nil {
			t.Errorf("ReEncode accepted %q", arch)
		}
		if _, err := cudasass.EncodeControl(inst.Encoding, cudasass.ControlCode{Stall: 1}, arch); err == nil {
			t.Errorf("EncodeControl accepted %q", arch)
		}
		if _, err := cudasass.Dependencies(cudasass.BasicBlock{}, arch); err == nil {
			t.Errorf("Dependencies accepted %q", arch)
		}
	}
	// sm_121 is the family and sm_121a is the architecture-specific target.
	// They are not interchangeable and the model is calibrated only on the
	// latter.
	if _, ok := cudasass.DecodeControl(inst.Encoding, "sm_121"); ok {
		t.Error("sm_121 was accepted for an sm_121a model")
	}
}

// TestUndocumentedEncodingRecordedAsUndocumented checks that an instruction
// outside the covered subset is reported as uncovered rather than guessed at.
func TestUndocumentedEncodingRecordedAsUndocumented(t *testing.T) {
	fn := roundTripFunction(t)
	var uncovered *cudasass.Instruction
	for i := range fn.Instructions {
		if _, ok := cudasass.Covered(fn.Instructions[i]); !ok {
			uncovered = &fn.Instructions[i]
			break
		}
	}
	if uncovered == nil {
		t.Skip("every instruction in the fixture is covered")
	}
	if _, err := cudasass.ReEncode(*uncovered, cudasass.SM121a); err == nil {
		t.Errorf("ReEncode(%s) succeeded; an uncovered form must not be encoded", uncovered.Opcode)
	} else if !strings.Contains(err.Error(), "outside the covered subset") {
		t.Errorf("error %q does not say the form is uncovered", err)
	}
	v := cudasass.RoundTrip(fn, cudasass.SM121a)
	for _, r := range v.Results {
		if r.Address == uncovered.Address && r.Form != "" {
			t.Errorf("uncovered instruction %#x was reported under form %q", r.Address, r.Form)
		}
	}
}

// TestControlCodeFieldsDecodedForNamedArch checks the control-code decode
// against values transcribed from the fixture by hand.
func TestControlCodeFieldsDecodedForNamedArch(t *testing.T) {
	// IADD R0, R7.reuse, UR6 ; /* 0x0000000607007c35 */
	//                          /* 0x041fe2000f8e0000 */
	e := cudasass.Encoding{Lo: 0x0000000607007c35, Hi: 0x041fe2000f8e0000}
	c, ok := cudasass.DecodeControl(e, cudasass.SM121a)
	if !ok {
		t.Fatal("DecodeControl reported no control code")
	}
	want := cudasass.ControlCode{Stall: 1, Yield: true, WriteBarrier: 7, ReadBarrier: 7, WaitMask: 0x01, Reuse: 0x1}
	if c != want {
		t.Errorf("control = %+v, want %+v", c, want)
	}
	if !c.Waits(0) {
		t.Error("Waits(0) = false, want true")
	}
	if c.Waits(1) {
		t.Error("Waits(1) = true, want false")
	}
	// Re-applying a decoded control code must not disturb the encoding.
	got, err := cudasass.EncodeControl(e, c, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if got != e {
		t.Errorf("EncodeControl(DecodeControl(e)) = %v, want %v", got, e)
	}
}

// TestStallRangeIsPerFormNotPerArchitecture records the correction that the
// fixture forced: the stall field is four bits wide, but the values the
// hardware accepts differ by instruction. NOP takes all sixteen and ptxas
// emits it with stall 0; IADD Rd, Ra, URb takes only 1-11. A range measured on
// one instruction must not be applied to another.
func TestStallRangeIsPerFormNotPerArchitecture(t *testing.T) {
	nop, ok := cudasass.ObservedStallRange("NOP")
	if !ok {
		t.Fatal("NOP stall range not recorded")
	}
	iadd, ok := cudasass.ObservedStallRange("IADD Rd, Ra, URb")
	if !ok {
		t.Fatal("IADD stall range not recorded")
	}
	if nop == iadd {
		t.Fatal("the two swept forms agree; the per-form distinction is untested")
	}
	if nop.Min != 0 || nop.Max != 15 {
		t.Errorf("NOP stall range = %+v, want {0 15}", nop)
	}
	if iadd.Min != 1 || iadd.Max != 11 {
		t.Errorf("IADD stall range = %+v, want {1 11}", iadd)
	}
	if _, ok := cudasass.ObservedStallRange("S2R Rd, SR"); ok {
		t.Error("S2R reports a stall range that was never swept")
	}
	// A stall value legal for NOP must still encode, even though it is
	// rejected by the hardware on IADD: the check belongs to the form.
	e := cudasass.Encoding{Lo: 0x0000000000007918, Hi: 0x000fc00000000000}
	c, _ := cudasass.DecodeControl(e, cudasass.SM121a)
	if c.Stall != 0 {
		t.Fatalf("fixture NOP stall = %d, want 0", c.Stall)
	}
	if _, err := cudasass.EncodeControl(e, c, cudasass.SM121a); err != nil {
		t.Errorf("re-encoding a real ptxas NOP was rejected: %v", err)
	}
}

// TestMutatedRegisterReEncodes is the half of the step 4 gate that byte
// identity alone does not decide: whether a *changed* instruction can be
// encoded, which is what a reordering or rewriting search would have to do.
//
// The expected encodings below were produced by [cudasass.ReEncode] and then
// disassembled on the target with nvdisasm 13.4, which reported exactly the
// mutated instruction in each case. nvdisasm, not this package, is the
// authority for the bytes; see testdata/roundtrip/README.md for the commands.
func TestMutatedRegisterReEncodes(t *testing.T) {
	fn := roundTripFunction(t)
	find := func(opcode string) cudasass.Instruction {
		t.Helper()
		for _, inst := range fn.Instructions {
			if inst.Opcode == opcode {
				if _, ok := cudasass.Covered(inst); ok {
					return inst
				}
			}
		}
		t.Fatalf("no covered %s in the fixture", opcode)
		return cudasass.Instruction{}
	}
	for _, tt := range []struct {
		opcode   string
		operands string
		want     cudasass.Encoding
		// disassembled is what nvdisasm reported for want on the target.
		disassembled string
	}{
		{"S2R", "R31, SR_TID.X", cudasass.Encoding{Lo: 0x00000000001f7919, Hi: 0x000e220000002100}, "S2R R31, SR_TID.X"},
		{"S2R", "R200, SR_TID.X", cudasass.Encoding{Lo: 0x0000000000c87919, Hi: 0x000e220000002100}, "S2R R200, SR_TID.X"},
		{"IADD", "R31, R7, UR6", cudasass.Encoding{Lo: 0x00000006071f7c35, Hi: 0x041fe2000f8e0000}, "IADD R31, R7.reuse, UR6"},
		{"IADD", "R200, R7, UR6", cudasass.Encoding{Lo: 0x0000000607c87c35, Hi: 0x041fe2000f8e0000}, "IADD R200, R7.reuse, UR6"},
	} {
		t.Run(tt.opcode+" "+tt.operands, func(t *testing.T) {
			inst := find(tt.opcode)
			inst.Operands = tt.operands
			got, err := cudasass.ReEncode(inst, cudasass.SM121a)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("ReEncode = %v, want %v (nvdisasm reported %q for want)", got, tt.want, tt.disassembled)
			}
			// The mutation must actually change the bytes, or the test would
			// pass on an encoder that ignored its input.
			if got == inst.Encoding {
				t.Error("mutated encoding equals the original")
			}
		})
	}
}

// TestUnsweptFormReportsAbsentStallRange makes the absence falsifiable. Every
// covered form that was never swept must report absence rather than a
// plausible 0-15, and the reported range must be the zero value so a caller
// that ignores ok cannot read a permissive range out of it.
func TestUnsweptFormReportsAbsentStallRange(t *testing.T) {
	swept := map[string]bool{"NOP": true, "IADD Rd, Ra, URb": true}
	for _, form := range []string{"NOP", "EXIT", "S2R Rd, SR", "IADD Rd, Ra, URb"} {
		r, ok := cudasass.ObservedStallRange(form)
		if ok != swept[form] {
			t.Errorf("ObservedStallRange(%q) reported %v, want %v", form, ok, swept[form])
			continue
		}
		if !ok && r != (cudasass.StallRange{}) {
			t.Errorf("ObservedStallRange(%q) absent but returned %+v", form, r)
		}
	}
	if _, ok := cudasass.ObservedStallRange("LDG Rd, [Ra]"); ok {
		t.Error("a form outside the covered subset reports a swept stall range")
	}
}

// roundTripCubin is the CUBIN the fixture disassembly was produced from,
// built locally from testdata/roundtrip/kernel.cu. It is stored hex-encoded,
// as the other CUBIN fixtures in this package are. See the README there.
const roundTripCubin = "testdata/roundtrip/sm121a-roundtrip.cubin.hex"

func readRoundTripCubin(t *testing.T) []byte {
	t.Helper()
	text, err := os.ReadFile(roundTripCubin)
	if err != nil {
		t.Fatal(err)
	}
	b, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		t.Fatalf("decode %s: %v", roundTripCubin, err)
	}
	return b
}

// instructionAt returns the index of the fixture instruction at addr.
func instructionAt(t *testing.T, fn *cudasass.Function, addr uint64) int {
	t.Helper()
	for i, inst := range fn.Instructions {
		if inst.Address == addr {
			return i
		}
	}
	t.Fatalf("no instruction at %#x in %s", addr, fn.Name)
	return -1
}

// TestSchedulingPerturbingMutationRejectedAtPatchTime is the step 5 gate on
// the mutation class. The control code is carried out of band from the
// original bytes, which is sound only while the dependency graph is unchanged,
// so a mutation that changes an edge must be refused when the patch is built
// -- not at load, and not by the device.
func TestSchedulingPerturbingMutationRejectedAtPatchTime(t *testing.T) {
	fn := roundTripFunction(t)
	s2r := instructionAt(t, fn, 0x10) // S2R R7, SR_TID.X

	// Scheduling-invariant: same form, same registers, different special
	// register. Nothing in the graph moves.
	invariant := cudasass.Mutation{
		Index:       s2r,
		Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"},
	}
	plan, err := cudasass.CheckMutation(fn, cudasass.SM121a, invariant, nil)
	if err != nil {
		t.Fatalf("scheduling-invariant mutation rejected: %v", err)
	}
	if plan.Before == plan.After {
		t.Fatalf("mutation produced the original encoding %v; the test would pass without mutating anything", plan.Before)
	}
	origControl, _ := cudasass.DecodeControl(plan.Before, cudasass.SM121a)
	newControl, _ := cudasass.DecodeControl(plan.After, cudasass.SM121a)
	if newControl != origControl {
		t.Errorf("control code changed: %v -> %v; it must be carried unchanged", origControl, newControl)
	}

	// Scheduling-perturbing: renaming the destination onto R0 destroys the
	// RAW edges to the instructions that read R7 and creates a WAW with the
	// instruction that writes R0.
	perturbing := cudasass.Mutation{
		Index:       s2r,
		Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R0, SR_TID.X"},
	}
	_, err = cudasass.CheckMutation(fn, cudasass.SM121a, perturbing, nil)
	if err == nil {
		t.Fatal("renaming the destination register was accepted; it changes RAW and WAW edges")
	}
	if !strings.Contains(err.Error(), "perturbs scheduling") {
		t.Errorf("rejection does not name the reason: %v", err)
	}
	t.Logf("rejected: %v", err)
}

// TestRenameOntoLiveRegisterRejectedWithoutEdgeChange is the case the edge
// comparison alone does not catch: a register the block never reads but a
// successor does. The mutation adds no edge and is still wrong, which is why
// liveness is checked separately and why a rename is routed through the model
// rather than assumed safe.
func TestRenameOntoLiveRegisterRejectedWithoutEdgeChange(t *testing.T) {
	fn := roundTripFunction(t)
	nop := instructionAt(t, fn, 0xe0) // the first NOP after the branch

	rep := cudasass.Instruction{Opcode: "S2R", Operands: "R1, SR_TID.X"}
	liveR1 := cudasass.LiveSet{{Class: cudasass.GPR, Num: 1}: true}

	// With R1 live on exit from the block, writing it is a clobber.
	_, err := cudasass.CheckMutation(fn, cudasass.SM121a, cudasass.Mutation{Index: nop, Replacement: rep}, liveR1)
	if err == nil {
		t.Fatal("writing a register that is live on exit was accepted")
	}
	if !strings.Contains(err.Error(), "live after it") {
		t.Errorf("rejection does not name liveness: %v", err)
	}
	if strings.Contains(err.Error(), "perturbs scheduling") {
		t.Errorf("rejected by the edge comparison, so this test is not exercising the liveness check: %v", err)
	}
	t.Logf("rejected: %v", err)

	// With R1 dead, the same mutation is the scheduling-invariant one.
	if _, err := cudasass.CheckMutation(fn, cudasass.SM121a, cudasass.Mutation{Index: nop, Replacement: rep}, cudasass.LiveSet{}); err != nil {
		t.Errorf("with R1 dead the same mutation must be legal, got: %v", err)
	}

	// A missing liveness claim is not an empty one.
	if _, err := cudasass.CheckMutation(fn, cudasass.SM121a, cudasass.Mutation{Index: nop, Replacement: rep}, nil); err == nil {
		t.Error("omitting liveOut for a block that does not end in EXIT was accepted")
	}
}

// TestPatchLandsAtTheDisassembledAddress checks the patch path against the
// real CUBIN the fixture disassembly came from: the bytes at the file offset
// the path computes must be the bytes nvdisasm printed at that address. A
// patch written at the wrong offset produces an image that still loads, so
// this is checked here and not by a load.
func TestPatchLandsAtTheDisassembledAddress(t *testing.T) {
	fn := roundTripFunction(t)
	im, err := cudasass.OpenCubin(readRoundTripCubin(t))
	if err != nil {
		t.Fatal(err)
	}
	idx := instructionAt(t, fn, 0x10)
	plan, err := cudasass.CheckMutation(fn, cudasass.SM121a,
		cudasass.Mutation{Index: idx, Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	patched, err := im.Apply("intmix", []cudasass.MutationPlan{plan})
	if err != nil {
		t.Fatalf("applying a checked plan to the CUBIN it was built from: %v", err)
	}
	if len(patched.Bytes) != len(readRoundTripCubin(t)) {
		t.Errorf("patched image is %d bytes, original is %d; the path must not move anything",
			len(patched.Bytes), len(readRoundTripCubin(t)))
	}
	if len(patched.Patches) != 1 {
		t.Fatalf("applied %d patches, want 1", len(patched.Patches))
	}
	got := patched.Patches[0]
	want := plan.After.Bytes()
	if !bytes.Equal(patched.Bytes[got.FileOffset:got.FileOffset+16], want[:]) {
		t.Errorf("bytes at %#x are not the replacement", got.FileOffset)
	}
	// Everything else is untouched.
	orig := readRoundTripCubin(t)
	for i := range orig {
		if uint64(i) >= got.FileOffset && uint64(i) < got.FileOffset+16 {
			continue
		}
		if orig[i] != patched.Bytes[i] {
			t.Fatalf("byte %#x changed outside the patched instruction", i)
		}
	}
	t.Logf("patched %s at file offset %#x (address %#x)", patched.Section, got.FileOffset, got.Address)
}

// TestPatchAtWrongOffsetRejected is the mutation check on the patch path
// itself: a plan whose target address is shifted by one instruction must be
// refused because the bytes there are not what it was built from. Without
// this check the misplaced patch produces a loadable CUBIN.
func TestPatchAtWrongOffsetRejected(t *testing.T) {
	fn := roundTripFunction(t)
	im, err := cudasass.OpenCubin(readRoundTripCubin(t))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := cudasass.CheckMutation(fn, cudasass.SM121a,
		cudasass.Mutation{Index: instructionAt(t, fn, 0x10), Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		addr uint64
		want error
	}{
		{"one instruction later", plan.Address + 16, cudasass.ErrOriginalMismatch},
		{"one byte later", plan.Address + 1, cudasass.ErrPatchUnaligned},
		{"past the section", plan.Address + 0x10000, cudasass.ErrPatchOutOfRange},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shifted := plan
			shifted.Address = tc.addr
			if _, err := im.Apply("intmix", []cudasass.MutationPlan{shifted}); !errors.Is(err, tc.want) {
				t.Fatalf("Apply at %#x = %v, want %v", tc.addr, err, tc.want)
			}
		})
	}
	// A plan for the wrong kernel's section is the same class of mistake.
	if _, err := im.Apply("addk", []cudasass.MutationPlan{plan}); !errors.Is(err, cudasass.ErrOriginalMismatch) {
		t.Errorf("applying intmix's plan to addk = %v, want %v", err, cudasass.ErrOriginalMismatch)
	}
	if _, err := im.Apply("nosuchkernel", []cudasass.MutationPlan{plan}); !errors.Is(err, cudasass.ErrTextSectionAbsent) {
		t.Errorf("applying to an absent section = %v, want %v", err, cudasass.ErrTextSectionAbsent)
	}
}

// TestPatchedCubinLoadFailsClosed requires the load gate to refuse an image it
// cannot vouch for. On this toolchain every CUBIN carries a second, compressed
// copy of each text section that the patch path cannot update, so a patched
// image built from the fixture is refused; an image with nothing patched is
// refused too, because loading it would qualify nothing.
func TestPatchedCubinLoadFailsClosed(t *testing.T) {
	fn := roundTripFunction(t)
	im, err := cudasass.OpenCubin(readRoundTripCubin(t))
	if err != nil {
		t.Fatal(err)
	}
	mirrors := im.Mirrors("intmix")
	if len(mirrors) == 0 {
		t.Skip("this CUBIN has no mirrored text section; the refusal this test covers cannot arise")
	}
	plan, err := cudasass.CheckMutation(fn, cudasass.SM121a,
		cudasass.Mutation{Index: instructionAt(t, fn, 0x10), Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	patched, err := im.Apply("intmix", []cudasass.MutationPlan{plan})
	if err != nil {
		t.Fatal(err)
	}
	if err := patched.Loadable(); !errors.Is(err, cudasass.ErrMirroredText) {
		t.Fatalf("Loadable = %v, want %v naming %v", err, cudasass.ErrMirroredText, mirrors)
	}
	// The gate refuses even with the lock held: the lock is a precondition,
	// not a substitute for knowing the image is the one that executes.
	if err := cudasass.AuthorizeLoad(patched, heldLockEnv(t)); !errors.Is(err, cudasass.ErrMirroredText) {
		t.Errorf("AuthorizeLoad with the lock held = %v, want %v", err, cudasass.ErrMirroredText)
	}
	if _, err := im.Apply("intmix", nil); !errors.Is(err, cudasass.ErrNothingPatched) {
		t.Errorf("Apply with no plans = %v, want %v", err, cudasass.ErrNothingPatched)
	}
	if err := (cudasass.PatchedImage{}).Loadable(); !errors.Is(err, cudasass.ErrNothingPatched) {
		t.Errorf("an unpatched image is loadable; loading it would qualify nothing")
	}
	t.Logf("refused: %v", patched.Loadable())
}

// TestCandidateLoadRequiresGPULock requires the GPU lock before a candidate
// binary of unknown validity reaches a shared device.
func TestCandidateLoadRequiresGPULock(t *testing.T) {
	// An image with no mirror to refuse, so the lock is the only gate left.
	patched := cudasass.PatchedImage{
		Kernel:  "intmix",
		Section: ".text.intmix",
		Patches: []cudasass.AppliedPatch{{Address: 0x10, FileOffset: 0x910}},
	}
	held := heldLockEnv(t)
	dir := held("GPU_LOCKDIR")
	for _, tc := range []struct {
		name    string
		env     map[string]string
		wantErr bool
	}{
		{"unset", map[string]string{"GPU_LOCKDIR": dir}, true},
		{"empty", map[string]string{"GPU_LOCK_HELD": "", "GPU_LOCKDIR": dir}, true},
		{"zero", map[string]string{"GPU_LOCK_HELD": "0", "GPU_LOCKDIR": dir}, true},
		{"truthy", map[string]string{"GPU_LOCK_HELD": "yes", "GPU_LOCKDIR": dir}, true},
		// gpu-lock env GPU_LOCK_HELD=1 overwrites the holder's pid.
		{"one", map[string]string{"GPU_LOCK_HELD": "1", "GPU_LOCKDIR": dir}, true},
		{"pid the lock does not name", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(os.Getpid() + 1), "GPU_LOCKDIR": dir}, true},
		{"held", map[string]string{"GPU_LOCK_HELD": held("GPU_LOCK_HELD"), "GPU_LOCKDIR": dir}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := cudasass.AuthorizeLoad(patched, func(k string) string { return tc.env[k] })
			if tc.wantErr {
				if !errors.Is(err, cudasass.ErrLockNotHeld) {
					t.Fatalf("AuthorizeLoad = %v, want %v", err, cudasass.ErrLockNotHeld)
				}
				return
			}
			if err != nil {
				t.Fatalf("AuthorizeLoad with the lock held = %v", err)
			}
		})
	}
	if err := cudasass.AuthorizeLoad(patched, nil); !errors.Is(err, cudasass.ErrLockNotHeld) {
		t.Errorf("AuthorizeLoad with no environment = %v, want %v", err, cudasass.ErrLockNotHeld)
	}
}

// heldLockEnv acquires a GPU lock in a temporary directory and returns the
// environment a process running under it would see.
func heldLockEnv(t *testing.T) func(string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "lock")
	h, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Release(); err != nil {
			t.Error(err)
		}
	})
	env := map[string]string{"GPU_LOCK_HELD": strconv.Itoa(os.Getpid()), "GPU_LOCKDIR": dir}
	return func(k string) string { return env[k] }
}
