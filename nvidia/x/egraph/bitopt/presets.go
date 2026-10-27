package bitopt

import "slices"

// Spec is an alias for [Problem] describing a bit-routing specification.
type Spec = Problem

// Program is an alias for [Solution] describing a synthesized bit-routing program.
type Program = Solution

// Check verifies that prog satisfies spec and passes [Solution.Check].
func Check(spec Spec, prog Program) error {
	if prog.Problem.Words != spec.Words || !slices.Equal(prog.Problem.Outputs, spec.Outputs) || !slices.Equal(prog.Problem.Ops, spec.Ops) {
		prog.Problem = spec
	}
	return prog.Check()
}

// E2M1ToBF16Spec returns the bit-routing specification for unpacking 8*words
// 4-bit E2M1 values (packed across words 32-bit input words) into 4*words
// bf16x2 registers holding unscaled E2M1 subnormals (v * 2^-126, with the
// 3-bit magnitude at bits 6..8 and the sign at bit 15 of each 16-bit half,
// mask 0x81C081C0).
func E2M1ToBF16Spec(words int) Spec {
	if words < 1 {
		words = 1
	}
	return Spec{
		Words:   words,
		Outputs: slices.Repeat([]uint32{0x81C081C0}, 4*words),
	}
}

// INT4ToBF16Spec returns the bit-routing specification for unpacking 8
// 4-bit biased integers from 1 32-bit word into 4 bf16x2 mantissa-field
// registers (bits 0..3 of each 16-bit half, mask 0x000F000F).
func INT4ToBF16Spec() Spec {
	return Spec{
		Words:   1,
		Outputs: []uint32{0x000F000F, 0x000F000F, 0x000F000F, 0x000F000F},
	}
}

var (
	nvfp4Prog1 = Program{
		Problem: E2M1ToBF16Spec(1),
		Moves: []Move{
			{Op: Ident, A: 0, B: 0},
			{Op: Shl, A: 0, B: 0, N: 4},
			{Op: Funnel, A: 0, B: 0, N: 10},
			{Op: Shl, A: 0, B: 0, N: 6},
			{Op: Shr, A: 0, B: 0, N: 2},
		},
		Parts: [][]Part{
			{{Move: 0, Mask: 0x81C081C0}},
			{{Move: 1, Mask: 0x81C081C0}},
			{{Move: 2, Mask: 0x81C081C0}},
			{{Move: 3, Mask: 0x80C080C0}, {Move: 4, Mask: 0x01000100}},
		},
	}

	nvfp4Prog2 = Program{
		Problem: E2M1ToBF16Spec(2),
		Moves: []Move{
			{Op: Shl, A: 0, B: 0, N: 2},
			{Op: Shl, A: 0, B: 0, N: 6},
			{Op: Funnel, A: 0, B: 0, N: 12},
			{Op: Shl, A: 1, B: 1, N: 4},
			{Op: Funnel, A: 1, B: 1, N: 10},
			{Op: Funnel, A: 1, B: 1, N: 14},
			{Op: Perm, A: 0, B: 1, Sel: 0x4163},
			{Op: Ident, A: 0, B: 0},
			{Op: Ident, A: 1, B: 1},
		},
		Parts: [][]Part{
			{{Move: 0, Mask: 0x81C081C0}},
			{{Move: 1, Mask: 0x81C081C0}},
			{{Move: 2, Mask: 0x81C081C0}},
			{{Move: 3, Mask: 0x81C081C0}},
			{{Move: 4, Mask: 0x81C081C0}},
			{{Move: 5, Mask: 0x81C081C0}},
			{{Move: 6, Mask: 0x81C081C0}},
			{{Move: 7, Mask: 0x01800180}, {Move: 8, Mask: 0x80408040}},
		},
	}
)

func init() {
	if err := Check(E2M1ToBF16Spec(1), nvfp4Prog1); err != nil {
		panic(err)
	}
	if err := Check(E2M1ToBF16Spec(2), nvfp4Prog2); err != nil {
		panic(err)
	}
}

// NVFP4Program1Word returns a copy of the verified 9-instruction 1-word
// E2M1-to-BF16 bit-routing program for [E2M1ToBF16Spec](1).
func NVFP4Program1Word() Program {
	return cloneProgram(nvfp4Prog1)
}

// NVFP4Program2Word returns a copy of the verified 16-instruction 2-word
// E2M1-to-BF16 bit-routing program for [E2M1ToBF16Spec](2) (8 instructions per
// word using a cross-word byte permute).
func NVFP4Program2Word() Program {
	return cloneProgram(nvfp4Prog2)
}

func cloneProgram(p Program) Program {
	parts := make([][]Part, len(p.Parts))
	for i, ps := range p.Parts {
		parts[i] = slices.Clone(ps)
	}
	return Program{
		Problem: Spec{
			Words:   p.Problem.Words,
			Outputs: slices.Clone(p.Problem.Outputs),
			Ops:     slices.Clone(p.Problem.Ops),
		},
		Moves: slices.Clone(p.Moves),
		Parts: parts,
	}
}
