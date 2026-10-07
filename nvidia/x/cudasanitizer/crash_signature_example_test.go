package cudasanitizer_test

import (
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleCrashSignature() {
	m, _ := cudatrial.NewThreadEdgeMap(32, 1)
	m.Record(0, 3) // Caller-supplied host model, not device coverage.
	p := cudasanitizer.CrashIdentity{Kernel: "kernel", Image: sha256.Sum256([]byte("image")), Epoch: "epoch", Trial: "previous", Input: sha256.Sum256([]byte("previous input"))}
	f := p
	f.Trial, f.Input, f.Ordinal = "failure", sha256.Sum256([]byte("failing input")), 1
	r := &cudasanitizer.Result{Config: cudasanitizer.Config{Tool: cudasanitizer.Memcheck}, Outcome: cudasanitizer.OutcomeSanitizerErrors, Log: cudasanitizer.ParseLog("========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n")}
	observation, err := cudasanitizer.CrashSignature(r, cudasanitizer.CompletedCrashTrace{Identity: p, Complete: true, MapSize: 32, Bitmap: m.Snapshot()}, f)
	fmt.Println(observation.Preceding.Trial, observation.Failure.Trial, err)
	// Output: previous failure <nil>
}
