package cudaptx

import "testing"

// PTX ISA 9.4 binds .loc_intermediate to the immediately following
// instruction, permits at most one per instruction, and keeps it out of the
// primary line table (docs/ptx/11-directives.md:1810, 1816, 1819-1821).
const intermediateSrc = `.version 9.4
.target sm_121
.address_size 64
.file 1 "kernel.cu"
.file 2 "tileIR_source.ir"
.visible .entry k()
{
	.loc 1 72 5
	.loc_intermediate 2 23 6
	mul.lo.s32 %r1, %r2, %r3;
	add.s32 %r4, %r1, 1;
	ret;
}
`

func TestLocIntermediateBindsToOneInstruction(t *testing.T) {
	m, err := ParseString(intermediateSrc)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Kernels) != 1 {
		t.Fatalf("got %d kernels, want 1", len(m.Kernels))
	}
	insts := m.Kernels[0].Instructions
	if len(insts) != 3 {
		t.Fatalf("got %d instructions, want 3", len(insts))
	}
	if got := insts[0].IntermediateLoc.Line; got != 23 {
		t.Errorf("mul.lo.s32 intermediate line = %d, want 23", got)
	}
	for _, i := range []int{1, 2} {
		if got := insts[i].IntermediateLoc; got.FileIndex != 0 {
			t.Errorf("%s carries no .loc_intermediate but reports %v", insts[i].Opcode, got)
		}
	}
}

func TestLocIntermediateStaysOutOfLoc(t *testing.T) {
	m, err := ParseString(intermediateSrc)
	if err != nil {
		t.Fatal(err)
	}
	// The intermediate location names an IR file. Reporting it as Loc hands a
	// consumer a .ir path where it documented a CUDA source position, and the
	// SASS tier has no such file to agree with.
	for _, inst := range m.Kernels[0].Instructions {
		if inst.Loc.FileIndex == 2 {
			t.Errorf("%s: Loc resolved to the intermediate file %q", inst.Opcode, m.Files[2])
		}
	}
	if got := m.Kernels[0].Instructions[0].Loc.Line; got != 72 {
		t.Errorf("mul.lo.s32 Loc line = %d, want 72 from the .loc", got)
	}
}

// intermediateOnlySrc has an instruction whose only source directive is a
// .loc_intermediate, the case where a merge into Loc is observable.
const intermediateOnlySrc = `.version 9.4
.target sm_121
.address_size 64
.file 1 "kernel.cu"
.file 2 "tileIR_source.ir"
.visible .entry k()
{
	.loc_intermediate 2 23 6
	mul.lo.s32 %r1, %r2, %r3;
	ret;
}
`

func TestLocIntermediateDoesNotSupplyLoc(t *testing.T) {
	m, err := ParseString(intermediateOnlySrc)
	if err != nil {
		t.Fatal(err)
	}
	inst := m.Kernels[0].Instructions[0]
	if inst.IntermediateLoc.Line != 23 {
		t.Errorf("IntermediateLoc.Line = %d, want 23", inst.IntermediateLoc.Line)
	}
	// No .loc is in scope. PTX ISA 9.4 keeps .loc_intermediate out of the
	// primary line table, so the instruction has no source position at all and
	// must report none: an IR path here is a wrong answer, not a missing one.
	if inst.Loc.FileIndex != 0 {
		t.Errorf("Loc = %+v, want the zero Location", inst.Loc)
	}
}

func TestLocClearsPendingIntermediate(t *testing.T) {
	src := `.version 9.4
.target sm_121
.file 1 "kernel.cu"
.file 2 "tileIR_source.ir"
.visible .entry k()
{
	.loc_intermediate 2 23 6
	.loc 1 72 5
	mul.lo.s32 %r1, %r2, %r3;
	ret;
}
`
	m, err := ParseString(src)
	if err != nil {
		t.Fatal(err)
	}
	// At most one .loc_intermediate applies to an instruction, and a following
	// .loc starts a new source position; the stale intermediate does not carry
	// across it.
	if got := m.Kernels[0].Instructions[0].IntermediateLoc; got.FileIndex != 0 {
		t.Errorf("IntermediateLoc = %+v after an intervening .loc, want none", got)
	}
}
