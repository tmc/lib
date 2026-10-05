package cudaschedule_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

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

	digest := sha256.Sum256([]byte(ptx))
	verdict := cudaschedule.Classify(cudaschedule.RunRecord{
		ExitCode:          0,
		Exited:            true,
		Stdout:            "PASS: verified\n",
		PTXDigest:         hex.EncodeToString(digest[:]),
		ExpectedPTXDigest: hex.EncodeToString(digest[:]),
		PTXDigestSource:   "runner_receipt", // synthetic recorded outcome
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

func ExampleReplay() {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry example() {
    membar.gl;
    ret;
}
`
	opts := cudaschedule.Options{Seed: 17, Intensity: 2}
	drawn, err := cudaschedule.Perturb(ptx, opts)
	if err != nil {
		panic(err)
	}
	replayed, err := cudaschedule.Replay(ptx, opts, drawn.Report.Decisions)
	if err != nil {
		panic(err)
	}
	fmt.Println(replayed.PTX == drawn.PTX)
	empty, err := cudaschedule.Replay(ptx, opts, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(empty.PTX == ptx)
	// Output:
	// true
	// true
}

func ExampleClassify() {
	// Synthetic recorded evidence illustrates classification, not device proof.
	digest := sha256.Sum256([]byte("synthetic recorded PTX workload"))
	sha := hex.EncodeToString(digest[:])
	rec := cudaschedule.RunRecord{
		ExitCode:          0,
		Exited:            true,
		Stdout:            "MISMATCH at index 0\n",
		PTXDigest:         sha,
		ExpectedPTXDigest: sha,
		PTXDigestSource:   "runner_receipt",
	}
	v := cudaschedule.Classify(rec)
	fmt.Printf("%s (finding: %v)\n", v, v.IsFinding())

	// Output:
	// Mismatch (finding: true)
}

func ExampleCampaignArms() {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry race_kernel() {
    .reg .b32 %r0;
    .reg .b64 %rd0;
    atom.global.add.u32 %r0, [%rd0], 1;
    ret;
}
`
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: ptx, SeedStart: 2, SeedEnd: 3, Intensity: 2, ConfirmRuns: 1,
		Arms: &cudaschedule.CampaignArms{
			PlantedBugPTXSource: ptx + "\n// planted bug\n",
			CleanPTXSource:      ptx + "\n// clean control\n",
		},
		RunVariant: func(_ context.Context, _ uint64, _ bool, input string) (cudaschedule.RunRecord, error) {
			// A synthetic host workload; a device runner reports its loaded PTX.
			digest := sha256.Sum256([]byte(input))
			rec := cudaschedule.RunRecord{Exited: true, PTXDigest: hex.EncodeToString(digest[:])}
			if strings.Contains(input, "planted bug") && strings.Contains(input, "nanosleep.u32") {
				rec.Stdout = "MISMATCH planted bug"
				return rec, nil
			}
			rec.Stdout = "PASS"
			return rec, nil
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(report.Arms.PlantedBug.Outcome)
	fmt.Println(report.Arms.CleanControl.Outcome)

	// Output:
	// detected
	// clean
}
