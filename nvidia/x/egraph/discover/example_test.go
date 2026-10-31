package discover_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/discover"
)

func ExampleDiscoverSubByteIdiom() {
	_, gb10Winner, err := discover.DiscoverSubByteIdiom("sm_121a", discover.IdiomE2M1Dequant)
	if err != nil {
		panic(err)
	}
	_, b200Winner, err := discover.DiscoverSubByteIdiom("sm_100a", discover.IdiomE2M1Dequant)
	if err != nil {
		panic(err)
	}
	_, rubinWinner, err := discover.DiscoverSubByteIdiom("sm_107f", discover.IdiomE2M1Dequant)
	if err != nil {
		panic(err)
	}
	fmt.Printf("sm_121a: %s (%d inst, verified=%v, mutant_rejected=%v)\n",
		gb10Winner.Name, gb10Winner.InstructionCount, gb10Winner.VerifiedExhaustive, gb10Winner.MutantRejected)
	fmt.Printf("sm_100a: %s (%d inst, verified=%v)\n",
		b200Winner.Name, b200Winner.InstructionCount, b200Winner.VerifiedExhaustive)
	fmt.Printf("sm_107f: %s (%d inst, verified=%v)\n",
		rubinWinner.Name, rubinWinner.InstructionCount, rubinWinner.VerifiedExhaustive)
	// Output:
	// sm_121a: prmt_lop3_lut_e2m1_dequant (3 inst, verified=true, mutant_rejected=true)
	// sm_100a: native_cvt_f16x2_e2m1x2 (1 inst, verified=true)
	// sm_107f: ptx94_quad_fma_e2m1p4x4 (1 inst, verified=true)
}

func ExampleRunMCTSDiscovery() {
	// (x & 0x00FF00FF) | ((y & 0x00FF00FF) << 8)
	baseline := discover.Program{NumInputs: 2, Insts: []discover.Inst{
		{Op: discover.OpAnd, Src: [3]discover.Operand{discover.R(0), discover.I(0x00FF00FF)}},
		{Op: discover.OpAnd, Src: [3]discover.Operand{discover.R(1), discover.I(0x00FF00FF)}},
		{Op: discover.OpShl, Src: [3]discover.Operand{discover.R(3)}, Imm: 8},
		{Op: discover.OpOr, Src: [3]discover.Operand{discover.R(2), discover.R(4)}},
	}}
	spec := func(in []uint32) uint32 { return in[0]&0x00FF00FF | (in[1]&0x00FF00FF)<<8 }
	idiom, err := discover.RunMCTSDiscovery("byte_plane", spec, baseline, discover.MCTSConfig{Seed: 1})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%d -> %d insts, %d -> %d cycles, mutants rejected %d/%d\n",
		idiom.BaselineInsts, idiom.DiscoveredInsts, idiom.BaselineCycles, idiom.DiscoveredCycles,
		idiom.Mutants.Rejected, idiom.Mutants.Total)
	fmt.Print(idiom.PTX)
	fmt.Println(idiom.RuleRHS)
	// Output:
	// 4 -> 1 insts, 13 -> 4 cycles, mutants rejected 16/16
	// prmt.b32 %t0, %in0, %in1, 0x6240;
	// (prmt_6240 ?a ?b)
}
