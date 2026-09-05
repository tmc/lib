package cudasanitizer_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

// Running a package's own tests under compute-sanitizer is the common case.
// Main handles both halves of the re-execution: in the parent it launches the
// tool, and in the child it runs the tests.
func ExampleMain() {
	main := func(m *testing.M) {
		os.Exit(cudasanitizer.Main(m, cudasanitizer.Config{
			Tool: cudasanitizer.Memcheck,
			// Leaks count toward the tool's error summary.
			LeakCheck: cudasanitizer.ToggleYes,
			// Without compute-sanitizer installed, run the tests
			// uninstrumented and say so, rather than claiming a pass.
			MissingTool: cudasanitizer.MissingToolSkip,
			// A report of zero errors means nothing unless the kernel
			// under test actually ran.
			RequireOutput: []string{"--- PASS: TestVectorAdd"},
			Timeout:       10 * time.Minute,
			LogDir:        "testdata/sanitizer-logs",
		}))
	}
	_ = main
}

// A test that drives a separate GPU program rather than running itself under
// the tool uses Check, which classifies the run and reports it to t.
func ExampleCheck() {
	test := func(t *testing.T) {
		cudasanitizer.Check(t, cudasanitizer.Config{
			Tool:          cudasanitizer.Racecheck,
			MissingTool:   cudasanitizer.MissingToolFail,
			RequireOutput: []string{"reduction complete"},
			Timeout:       2 * time.Minute,
		}, "./testdata/reduction")
	}
	_ = test
}

// Run returns a Result rather than a bool, because one exit status stands for
// four different situations that call for four different reactions.
func ExampleRun() {
	res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
		Tool:        cudasanitizer.Initcheck,
		MissingTool: cudasanitizer.MissingToolFail,
		Timeout:     time.Minute,
	}, "./vectoradd")
	if err != nil {
		panic(err) // the configuration is unusable
	}
	switch res.Outcome {
	case cudasanitizer.OutcomeClean:
		fmt.Println("no uninitialized reads")
	case cudasanitizer.OutcomeSanitizerErrors:
		fmt.Println(res.Details())
	case cudasanitizer.OutcomeApplicationFailed:
		fmt.Println("the program itself failed:", res.ExitCode)
	case cudasanitizer.OutcomeToolFailure:
		if errors.Is(res.Err, cudasanitizer.ErrToolNotFound) {
			fmt.Println("compute-sanitizer is not installed; nothing was checked")
		} else {
			fmt.Println("no verdict:", res.Err)
		}
	}
}

// The argument vector always carries an explicit nonzero --error-exitcode.
// compute-sanitizer's own default is 0, which makes a run with detected errors
// look like a clean one.
func ExampleConfig_Args() {
	cfg := cudasanitizer.Config{
		Tool:            cudasanitizer.Memcheck,
		MissingTool:     cudasanitizer.MissingToolFail,
		LeakCheck:       cudasanitizer.ToggleYes,
		RequireCUDAInit: cudasanitizer.ToggleYes,
	}
	args, err := cfg.Args("/tmp/report.log", "./vectoradd", "-n", "1024")
	if err != nil {
		panic(err)
	}
	for _, a := range args {
		fmt.Println(a)
	}
	// Output:
	// --tool
	// memcheck
	// --error-exitcode
	// 66
	// --log-file
	// /tmp/report.log
	// --require-cuda-init
	// yes
	// --leak-check
	// full
	// ./vectoradd
	// -n
	// 1024
}

// ParseLog reads a verdict out of the tool's report. Only lines carrying the
// tool's own "=========" prefix are read, so output from the program under test
// cannot forge a summary.
func ExampleParseLog() {
	const report = `========= COMPUTE-SANITIZER
vector add ok
========= Invalid __global__ write of size 4 bytes
=========     at 0x70 in vectorAdd(int *, int *, int *, int)
=========     Address 0x7f8a00000400 is out of bounds
=========
========= ERROR SUMMARY: 1 error
`
	log := cudasanitizer.ParseLog(report)
	fmt.Println("summary found:", log.SummaryFound)
	fmt.Println("errors:", log.Errors)
	fmt.Println("first finding:", log.Findings[0].Headline)
	// Output:
	// summary found: true
	// errors: 1
	// first finding: Invalid __global__ write of size 4 bytes
}

// A report with errors and an exit status of 0 is the case a naive runner gets
// wrong. The verdict comes from the report, so the absence of a summary is not
// silence but a missing verdict.
func ExampleParseLog_noVerdict() {
	truncated := cudasanitizer.ParseLog("========= COMPUTE-SANITIZER\n")
	fmt.Println("summary found:", truncated.SummaryFound, "errors:", truncated.Errors)

	clean := cudasanitizer.ParseLog("========= ERROR SUMMARY: 0 errors\n")
	fmt.Println("summary found:", clean.SummaryFound, "errors:", clean.Errors)
	// Output:
	// summary found: false errors: 0
	// summary found: true errors: 0
}
