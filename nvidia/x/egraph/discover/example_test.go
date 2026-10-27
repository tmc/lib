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
