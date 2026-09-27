package cudaschedule_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func Example() {
	const ptx = `.version 8.7
.target sm_90
.address_size 64

.visible .entry race_kernel()
{
    .reg .b32 %r0;
    .reg .b64 %rd0;
    atom.global.add.u32 %r0, [%rd0], 1;
    bar.sync 0;
    ret;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		panic(err)
	}
	fmt.Printf("sites found: %d\n", len(sites))

	rewrite, err := cudaschedule.Perturb(ptx, cudaschedule.Options{
		Seed:      42,
		Intensity: 1.0,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("sites injected: %d\n", rewrite.Report.SitesInjected)

	verdict := cudaschedule.Classify(cudaschedule.RunRecord{
		ExitCode: 0,
		Exited:   true,
		Stdout:   "PASS: verified\n",
	})
	fmt.Printf("verdict: %s\n", verdict)

	// Output:
	// sites found: 2
	// sites injected: 2
	// verdict: Pass
}

func ExampleAnalyze() {
	const ptx = `.version 8.7
.target sm_90
.address_size 64

.visible .entry example()
{
    .reg .b32 %r0;
    .reg .b64 %rd0;
    red.global.add.u32 [%rd0], %r0;
    ret;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		panic(err)
	}
	for _, s := range sites {
		fmt.Printf("%s: %s\n", s.Kind, s.Head)
	}

	// Output:
	// Reduction: red.global.add.u32
}

func ExamplePerturb() {
	const ptx = `.version 8.7
.target sm_90
.address_size 64

.visible .entry example()
{
    .reg .b32 %r0;
    .reg .b64 %rd0;
    fence.sc.gpu;
    ret;
}
`
	rewrite, err := cudaschedule.Perturb(ptx, cudaschedule.Options{
		Seed:      1,
		Intensity: 0.0, // empty draw
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("injected: %d\n", rewrite.Report.SitesInjected)

	// Output:
	// injected: 0
}

func ExampleClassify() {
	rec := cudaschedule.RunRecord{
		ExitCode: 0,
		Exited:   true,
		Stdout:   "MISMATCH at index 0\n",
	}
	v := cudaschedule.Classify(rec)
	fmt.Printf("%s (finding: %v)\n", v, v.IsFinding())

	// Output:
	// Mismatch (finding: true)
}
