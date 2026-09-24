package cudacmp_test

import (
	"context"
	"fmt"
	"os"
	"regexp"

	"github.com/tmc/lib/nvidia/x/cudacmp"
)

func ExampleFamilyClassifier_Classify() {
	fc := cudacmp.DefaultFamilyClassifier

	symbols := []string{
		"void <unnamed>::gateup_iq2_swiglu_q8_d2r_kernel<(int)32, (int)32>",
		"mul_mat_q",
		"attention_tokentile",
		"rms_norm_small",
		"repack_weights_aligned_kernel",
	}

	for _, sym := range symbols {
		fmt.Printf("%-30s -> %s\n", fc.Normalize(sym), fc.Classify(sym))
	}

	// Output:
	// gateup_iq2_swiglu_q8_d2r_kernel -> experts
	// mul_mat_q                      -> dense_q8
	// attention_tokentile            -> attention
	// rms_norm_small                 -> norm_prep
	// repack_weights_aligned_kernel  -> model_prep
}

func ExampleRender() {
	reports := []cudacmp.DifferentialReport{
		{
			Phase:              cudacmp.PhasePrefill,
			Family:             cudacmp.FamilyExperts,
			Availability:       cudacmp.AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        12_000_000,
			GoDurationNs:       10_000_000,
			DeltaNs:            -2_000_000,
			Speedup:            1.20,
			NullDriftNs:        50_000,
			ExceedsDrift:       true,
			StatisticallyValid: true,
		},
	}

	_ = cudacmp.Render(os.Stdout, reports)

	// Output:
	// PHASE    FAMILY   C TIME    GO TIME   DELTA     SPEEDUP  DRIFT FLOOR  EXCEEDS DRIFT?
	// prefill  experts  12.00 ms  10.00 ms  -2.00 ms  1.20x    50.00 µs     YES
}

func ExampleCommand() {
	// Digest only the result line, so the throughput line, which differs
	// from run to run, cannot make two runs of one arm disagree.
	arm := cudacmp.Command{
		Path:       "/bin/sh",
		Args:       []string{"-c", `echo "result 42"; echo "tok/s $$"`},
		NoPhaseArg: true,
		Digest:     regexp.MustCompile(`^result `),
	}
	r, err := arm.Runner()(context.Background(), cudacmp.PhaseDecode)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Digest[:12])
	// Output:
	// 258a08a02096
}
