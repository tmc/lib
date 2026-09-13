package cudasass_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestNormalizeOperands(t *testing.T) {
	tests := []struct {
		name     string
		opcode   string
		operands string
		want     string
	}{
		{
			name:     "strip memory offset with plus hex",
			opcode:   "LDG.E.CONSTANT.F32",
			operands: "R4, [R2.64 + 0x10]",
			want:     "R4, [R2.64]",
		},
		{
			name:     "strip memory offset with minus hex",
			opcode:   "LDG.E.F32",
			operands: "R4, [R2.64 - 0x20]",
			want:     "R4, [R2.64]",
		},
		{
			name:     "strip memory offset with decimal",
			opcode:   "STS.32",
			operands: "[R1 + 16], R4",
			want:     "[R1], R4",
		},
		{
			name:     "constant bank parameter drift",
			opcode:   "LDC",
			operands: "R0, c[0x0][0x160]",
			want:     "R0, c[0x0]",
		},
		{
			name:     "constant bank parameter drift higher offset",
			opcode:   "LDC",
			operands: "R0, c[0x0][0x168]",
			want:     "R0, c[0x0]",
		},
		{
			name:     "preserve non-drift register offset",
			opcode:   "LDG.E",
			operands: "R4, [R2.64+UR4]",
			want:     "R4, [R2.64+UR4]",
		},
		{
			name:     "preserve LUT truth table mask",
			opcode:   "LOP3.LUT",
			operands: "R6, R4, R5, 0xca",
			want:     "R6, R4, R5, 0xca",
		},
		{
			name:     "normalize branch target with backticks",
			opcode:   "BRA",
			operands: "`0x120`",
			want:     "<target>",
		},
		{
			name:     "normalize branch target hex",
			opcode:   "BRA",
			operands: "0x160",
			want:     "<target>",
		},
		{
			name:     "normalize branch target dot",
			opcode:   "BRA",
			operands: ".",
			want:     "<target>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cudasass.NormalizeOperands(tt.opcode, tt.operands)
			if got != tt.want {
				t.Errorf("NormalizeOperands(%q, %q) = %q, want %q", tt.opcode, tt.operands, got, tt.want)
			}
		})
	}
}

func TestNormalizeLine(t *testing.T) {
	tests := []struct {
		name string
		line string
		want string
	}{
		{
			name: "instruction with hex offset address and comment",
			line: "/* 0010 */   @P0 LDG.E.CONSTANT.F32 R4, [R2.64+0x10] ; // load const",
			want: "@P0 LDG.E.CONSTANT.F32 R4, [R2.64]",
		},
		{
			name: "plain instruction with hex address",
			line: "/*0020*/   IMAD.WIDE R0, R5, R6, R0 ;",
			want: "IMAD.WIDE R0, R5, R6, R0",
		},
		{
			name: "branch instruction target normalization",
			line: "/*0050*/   BRA `0x180` ;",
			want: "BRA <target>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cudasass.NormalizeLine(tt.line)
			if got != tt.want {
				t.Errorf("NormalizeLine(%q) = %q, want %q", tt.line, got, tt.want)
			}
		})
	}
}

func TestDiffFunctions(t *testing.T) {
	tests := []struct {
		name              string
		baseSASS          string
		targetSASS        string
		wantMatched       int
		wantAdded         int
		wantDeleted       int
		wantInstDelta     int
		wantBranchDelta   int
		wantScalarDelta   int
		wantSIMDDelta     int
		wantSpillAdded    int
		wantSpillRemoved  int
		wantSpillDelta    int
		wantSubstitutions []cudasass.OpcodeSubstitution
	}{
		{
			name: "identical kernels",
			baseSASS: `
--- Kernel: test_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`,
			targetSASS: `
--- Kernel: test_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`,
			wantMatched:   3,
			wantAdded:     0,
			wantDeleted:   0,
			wantInstDelta: 0,
		},
		{
			name: "scalar vs SIMD IDP4A",
			baseSASS: `
--- Kernel: gemv ---
/*0100*/ LDG.E.F32 R4, [R2.64] ;
/*0110*/ SHR.B64 R5, R4, 0x8 ;
/*0120*/ IMAD.WIDE R0, R5, R6, R0 ;
/*0130*/ IMAD.WIDE R1, R7, R8, R1 ;
/*0140*/ EXIT ;
`,
			targetSASS: `
--- Kernel: gemv ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ LOP3.LUT R6, R4, R5, 0xca ;
/*0030*/ IDP4A.U8.U8 R0, R6, R7, R0 ;
/*0040*/ IDP4A.S8.S8 R1, R8, R9, R1 ;
/*0050*/ EXIT ;
`,
			wantMatched:     2, // LDG.E.F32 and EXIT match
			wantAdded:       3, // LOP3 + 2 IDP4A
			wantDeleted:     3, // SHR + 2 IMAD
			wantInstDelta:   0,
			wantScalarDelta: -2, // base has SHR, 2 IMAD (3). target has LOP3 (1). 1 - 3 = -2
			wantSIMDDelta:   2,  // 0 -> 2
			wantSubstitutions: []cudasass.OpcodeSubstitution{
				{From: "IMAD", To: "IDP4A", Count: 2, FromCount: 2, ToCount: 2},
				{From: "SHR", To: "LOP3", Count: 1, FromCount: 1, ToCount: 1},
			},
		},
		{
			name: "loop unrolling with memory offset drift",
			baseSASS: `
--- Kernel: loop_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ BRA 0x10 ;
`,
			targetSASS: `
--- Kernel: loop_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ LDG.E.F32 R4, [R2.64+0x4] ;
/*0040*/ FFMA R0, R4, R5, R0 ;
/*0050*/ BRA 0x10 ;
`,
			wantMatched:     3, // First 2 instructions and branch match
			wantAdded:       2, // Unrolled iteration added
			wantDeleted:     0,
			wantInstDelta:   2,
			wantBranchDelta: 0,
			wantScalarDelta: 1, // 1 extra FFMA
		},
		{
			name: "register spill additions",
			baseSASS: `
--- Kernel: spill_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`,
			targetSASS: `
--- Kernel: spill_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ STS.32 [R1+0x10], R4 ;
/*0030*/ LDS.32 R4, [R1+0x10] ;
/*0040*/ FFMA R0, R4, R5, R0 ;
/*0050*/ EXIT ;
`,
			wantMatched:      3, // LDG, FFMA, EXIT match
			wantAdded:        2, // STS + LDS
			wantDeleted:      0,
			wantInstDelta:    2,
			wantSpillAdded:   2,
			wantSpillRemoved: 0,
			wantSpillDelta:   2,
		},
		{
			name: "register spill removals",
			baseSASS: `
--- Kernel: unspill_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ STS.32 [R1+0x10], R4 ;
/*0030*/ LDS.32 R4, [R1+0x10] ;
/*0040*/ FFMA R0, R4, R5, R0 ;
/*0050*/ EXIT ;
`,
			targetSASS: `
--- Kernel: unspill_kernel ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`,
			wantMatched:      3,
			wantAdded:        0,
			wantDeleted:      2, // STS + LDS removed
			wantInstDelta:    -2,
			wantSpillAdded:   0,
			wantSpillRemoved: 2,
			wantSpillDelta:   -2,
		},
		{
			name: "opcode substitution FFMA to HFMA2 and HMMA",
			baseSASS: `
--- Kernel: tensor_kernel ---
/*0010*/ FFMA R0, R4, R5, R0 ;
/*0020*/ FFMA R1, R6, R7, R1 ;
/*0030*/ EXIT ;
`,
			targetSASS: `
--- Kernel: tensor_kernel ---
/*0010*/ HFMA2 R0, R4, R5, R0 ;
/*0020*/ HMMA.16816.F32 R1, R6, R7, R1 ;
/*0030*/ EXIT ;
`,
			wantMatched:     1,
			wantAdded:       2,
			wantDeleted:     2,
			wantScalarDelta: -2,
			wantSIMDDelta:   2,
			wantSubstitutions: []cudasass.OpcodeSubstitution{
				{From: "FFMA", To: "HFMA2", Count: 1, FromCount: 2, ToCount: 1},
				{From: "FFMA", To: "HMMA", Count: 1, FromCount: 1, ToCount: 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseDisasm, err := cudasass.ParseString(tt.baseSASS)
			if err != nil {
				t.Fatalf("ParseString base: %v", err)
			}
			targetDisasm, err := cudasass.ParseString(tt.targetSASS)
			if err != nil {
				t.Fatalf("ParseString target: %v", err)
			}

			if len(baseDisasm.Functions) == 0 || len(targetDisasm.Functions) == 0 {
				t.Fatal("empty parsed functions")
			}

			diff := cudasass.DiffFunctions(*baseDisasm.Functions[0], *targetDisasm.Functions[0])

			if got := diff.MatchedCount(); got != tt.wantMatched {
				t.Errorf("MatchedCount() = %d, want %d", got, tt.wantMatched)
			}
			if got := diff.AddedCount(); got != tt.wantAdded {
				t.Errorf("AddedCount() = %d, want %d", got, tt.wantAdded)
			}
			if got := diff.DeletedCount(); got != tt.wantDeleted {
				t.Errorf("DeletedCount() = %d, want %d", got, tt.wantDeleted)
			}
			if got := diff.Architecture.InstructionDelta; got != tt.wantInstDelta {
				t.Errorf("InstructionDelta = %d, want %d", got, tt.wantInstDelta)
			}
			if tt.wantBranchDelta != 0 && diff.Architecture.BranchDelta != tt.wantBranchDelta {
				t.Errorf("BranchDelta = %d, want %d", diff.Architecture.BranchDelta, tt.wantBranchDelta)
			}
			if tt.wantScalarDelta != 0 && diff.Architecture.ScalarDelta != tt.wantScalarDelta {
				t.Errorf("ScalarDelta = %d, want %d", diff.Architecture.ScalarDelta, tt.wantScalarDelta)
			}
			if tt.wantSIMDDelta != 0 && diff.Architecture.SIMDDelta != tt.wantSIMDDelta {
				t.Errorf("SIMDDelta = %d, want %d", diff.Architecture.SIMDDelta, tt.wantSIMDDelta)
			}
			if diff.Spills.TotalAdded != tt.wantSpillAdded {
				t.Errorf("Spills.TotalAdded = %d, want %d", diff.Spills.TotalAdded, tt.wantSpillAdded)
			}
			if diff.Spills.TotalRemoved != tt.wantSpillRemoved {
				t.Errorf("Spills.TotalRemoved = %d, want %d", diff.Spills.TotalRemoved, tt.wantSpillRemoved)
			}
			if diff.Spills.Delta() != tt.wantSpillDelta {
				t.Errorf("Spills.Delta() = %d, want %d", diff.Spills.Delta(), tt.wantSpillDelta)
			}
			for _, wantSub := range tt.wantSubstitutions {
				found := false
				for _, gotSub := range diff.Architecture.Substitutions {
					if gotSub.From == wantSub.From && gotSub.To == wantSub.To && gotSub.Count == wantSub.Count {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("missing substitution %+v; got %+v", wantSub, diff.Architecture.Substitutions)
				}
			}
		})
	}
}

func TestDiffDisassembly(t *testing.T) {
	baseSASS := `
--- Kernel: k1 ---
/*0010*/ IMAD R0, R1, R2, R0 ;
/*0020*/ EXIT ;

--- Kernel: k2 ---
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`
	targetSASS := `
--- Kernel: k1 ---
/*0010*/ IDP4A.U8.U8 R0, R1, R2, R0 ;
/*0020*/ EXIT ;

--- Kernel: k3 ---
/*0010*/ EXIT ;
`

	base, err := cudasass.ParseString(baseSASS)
	if err != nil {
		t.Fatalf("Parse base: %v", err)
	}
	target, err := cudasass.ParseString(targetSASS)
	if err != nil {
		t.Fatalf("Parse target: %v", err)
	}

	diff, err := cudasass.Diff(base, target)
	if err != nil {
		t.Fatalf("Diff: %v", err)
	}

	if len(diff.Functions) != 3 {
		t.Fatalf("len(Functions) = %d, want 3", len(diff.Functions))
	}

	k1Diff := diff.Function("k1")
	if k1Diff == nil {
		t.Fatal("k1 diff not found")
	}
	if k1Diff.Architecture.SIMDDelta != 1 {
		t.Errorf("k1 SIMDDelta = %d, want 1", k1Diff.Architecture.SIMDDelta)
	}

	k2Diff := diff.Function("k2")
	if k2Diff == nil {
		t.Fatal("k2 diff not found")
	}
	if k2Diff.DeletedCount() != 2 {
		t.Errorf("k2 DeletedCount = %d, want 2", k2Diff.DeletedCount())
	}

	k3Diff := diff.Function("k3")
	if k3Diff == nil {
		t.Fatal("k3 diff not found")
	}
	if k3Diff.AddedCount() != 1 {
		t.Errorf("k3 AddedCount = %d, want 1", k3Diff.AddedCount())
	}

	// Test nil inputs
	if _, err := cudasass.Diff(nil, target); err == nil {
		t.Error("Diff(nil, target) should fail")
	}
	if _, err := cudasass.Diff(base, nil); err == nil {
		t.Error("Diff(base, nil) should fail")
	}
}

func ExampleDiff() {
	baseSASS := `
--- Kernel: gemv ---
/*0010*/ IMAD.WIDE R0, R1, R2, R0 ;
/*0020*/ EXIT ;
`
	targetSASS := `
--- Kernel: gemv ---
/*0010*/ IDP4A.U8.U8 R0, R1, R2, R0 ;
/*0020*/ EXIT ;
`
	base, _ := cudasass.ParseString(baseSASS)
	target, _ := cudasass.ParseString(targetSASS)

	diff, _ := cudasass.Diff(base, target)
	fnDiff := diff.Function("gemv")
	fmt.Printf("SIMD delta: %+d\n", fnDiff.Architecture.SIMDDelta)
	for _, sub := range fnDiff.Architecture.Substitutions {
		fmt.Printf("Substituted: %s -> %s\n", sub.From, sub.To)
	}
	// Output:
	// SIMD delta: +1
	// Substituted: IMAD -> IDP4A
}
