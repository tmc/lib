package cudasanitizer_test

import (
	"context"
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleGatedRunner() {
	cfg := cudasanitizer.Config{
		Tool:        cudasanitizer.Memcheck,
		MissingTool: cudasanitizer.MissingToolSkip,
		Gating:      cudasanitizer.GatingCoverageIncrease,
	}

	runner, err := cudasanitizer.NewGatedRunner(cfg, 32, 100)
	if err != nil {
		panic(err)
	}

	trace1 := make([]byte, 32)
	trace1[16] = 1

	trial1 := cudasanitizer.GatedTrial{
		Name:  "kernel_variant_a",
		Trace: trace1,
		SanitizeRun: func(ctx context.Context) (*cudasanitizer.Result, error) {
			return &cudasanitizer.Result{Outcome: cudasanitizer.OutcomeClean}, nil
		},
	}

	res1, _ := runner.RunTrial(context.Background(), trial1)
	fmt.Printf("Trial 1 novel: %v, dispatched: %v\n", res1.Novel, res1.SanitizerDispatched)

	// Repeating same trace bypasses compute-sanitizer
	res2, _ := runner.RunTrial(context.Background(), trial1)
	fmt.Printf("Trial 2 novel: %v, dispatched: %v\n", res2.Novel, res2.SanitizerDispatched)

	// Output:
	// Trial 1 novel: true, dispatched: true
	// Trial 2 novel: false, dispatched: false
}

func ExampleSanitizerClassifier() {
	classifier := cudasanitizer.NewSanitizerClassifier()

	m, _ := cudatrial.NewThreadEdgeMap(32, 1)
	m.Record(0, 3)

	identity := cudasanitizer.CrashIdentity{
		Kernel:  "vector_add",
		Image:   sha256.Sum256([]byte("img")),
		Epoch:   "campaign_1",
		Trial:   "trial_0",
		Input:   sha256.Sum256([]byte("inp0")),
		Ordinal: 0,
	}
	failing := identity
	failing.Trial = "trial_1"
	failing.Ordinal = 1
	failing.Input = sha256.Sum256([]byte("inp1"))

	trace := cudasanitizer.CompletedCrashTrace{
		Identity: identity,
		Complete: true,
		MapSize:  32,
		Bitmap:   m.Snapshot(),
	}

	res := &cudasanitizer.Result{
		Config:  cudasanitizer.Config{Tool: cudasanitizer.Memcheck},
		Outcome: cudasanitizer.OutcomeSanitizerErrors,
		Log:     cudasanitizer.ParseLog("========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n"),
	}

	_, isNovel1, _ := classifier.Classify(res, trace, failing)
	fmt.Printf("First crash observation novel: %v\n", isNovel1)

	// Duplicate observation from another trial
	failing2 := failing
	failing2.Trial = "trial_2"
	failing2.Ordinal = 1
	_, isNovel2, _ := classifier.Classify(res, trace, failing2)
	fmt.Printf("Second crash observation novel: %v\n", isNovel2)
	fmt.Printf("Unique crashes recorded: %d\n", classifier.UniqueCount())

	// Output:
	// First crash observation novel: true
	// Second crash observation novel: false
	// Unique crashes recorded: 1
}
