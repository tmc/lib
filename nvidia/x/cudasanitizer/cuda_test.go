package cudasanitizer_test

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

// The tests in this file are the ones that establish something about CUDA. They
// need a GPU, a CUDA installation with compute-sanitizer on PATH, and two
// programs supplied by the operator:
//
//	CUDASANITIZER_TEST_CLEAN_PROG   a program whose kernels are correct
//	CUDASANITIZER_TEST_DIRTY_PROG   a program with a known device memory error
//	CUDASANITIZER_TEST_RACE_PROG    a program with a known shared-memory race
//
// Each must print the marker named alongside it below when it has done its
// work, so that a run which starts, initializes CUDA and does nothing is not
// mistaken for a validation.
//
// Nothing here runs on a machine without a GPU, and no test in this package
// substitutes for them. The fixture tests in run_test.go exercise this
// package's own orchestration against a stand-in executable; they say nothing
// about whether compute-sanitizer detects a real error, about the tool's output
// format on any particular CUDA release, or about any GPU.
//
// Run them with:
//
//	LIB_CUDA_HARDWARE_TEST=1 \
//	CUDASANITIZER_TEST_CLEAN_PROG=./testdata/vectoradd \
//	CUDASANITIZER_TEST_DIRTY_PROG=./testdata/vectoradd-oob \
//	CUDASANITIZER_TEST_RACE_PROG=./testdata/reduce-race \
//	go test -run CUDA -v ./nvidia/x/cudasanitizer/

const (
	cleanMarker = "vector add ok"
	dirtyMarker = "vector add done"
	raceMarker  = "reduction complete"
)

func hardwareProg(t *testing.T, env string) string {
	t.Helper()
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") == "" {
		t.Skip("LIB_CUDA_HARDWARE_TEST is empty: these tests need compute-sanitizer and CUDA programs built ahead of time, which no probe can supply")
	}
	if _, err := cudasanitizer.LookTool(""); err != nil {
		t.Fatalf("hardware validation requires compute-sanitizer: %v", err)
	}
	prog := os.Getenv(env)
	if prog == "" {
		t.Fatalf("hardware validation requires %s", env)
	}
	return prog
}

// TestCleanProgramCUDA is the physical clean case: a correct program must come
// back clean, and the run must carry evidence that its kernels actually ran.
func TestCleanProgramCUDA(t *testing.T) {
	prog := hardwareProg(t, "CUDASANITIZER_TEST_CLEAN_PROG")
	res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
		Tool:            cudasanitizer.Memcheck,
		MissingTool:     cudasanitizer.MissingToolFail,
		LeakCheck:       cudasanitizer.ToggleYes,
		RequireCUDAInit: cudasanitizer.ToggleYes,
		RequireOutput:   []string{cleanMarker},
		Timeout:         5 * time.Minute,
		LogDir:          t.TempDir(),
	}, prog)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != cudasanitizer.OutcomeClean {
		t.Fatalf("Outcome = %v, want clean\n%s", res.Outcome, res.Details())
	}
	if !strings.Contains(res.Stdout+res.Stderr, cleanMarker) {
		t.Errorf("the run produced no evidence that the kernels executed:\n%s", res.Details())
	}
	t.Log(res.Summary())
	t.Log(res.Details())
}

// TestKnownBadProgramCUDA is the physical known-bad case. It is the test that
// would catch this package reporting a pass for a program with real errors.
func TestKnownBadProgramCUDA(t *testing.T) {
	prog := hardwareProg(t, "CUDASANITIZER_TEST_DIRTY_PROG")
	res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
		Tool:            cudasanitizer.Memcheck,
		MissingTool:     cudasanitizer.MissingToolFail,
		RequireCUDAInit: cudasanitizer.ToggleYes,
		RequireOutput:   []string{dirtyMarker},
		Timeout:         5 * time.Minute,
		LogDir:          t.TempDir(),
	}, prog)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != cudasanitizer.OutcomeSanitizerErrors {
		t.Fatalf("Outcome = %v, want sanitizer-errors for a program with a known out-of-bounds write\n%s",
			res.Outcome, res.Details())
	}
	if res.Log.Errors == 0 {
		t.Errorf("Log.Errors = 0 with outcome %v; the report and the verdict disagree", res.Outcome)
	}
	if len(res.Log.Findings) == 0 {
		t.Error("no findings were retained; a detected error the reader cannot see is of little use")
	}
	// The trap: on a tool build that ignores --error-exitcode the process
	// exits 0 here. That must not change the verdict.
	t.Logf("exit status %d, %d errors reported", res.ExitCode, res.Log.Errors)
	t.Log(res.Details())
}

// TestKnownRaceCUDA is the racecheck counterpart. It is not Go's race
// detector: -race says nothing about the hazard this looks for.
func TestKnownRaceCUDA(t *testing.T) {
	prog := hardwareProg(t, "CUDASANITIZER_TEST_RACE_PROG")
	res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
		Tool:            cudasanitizer.Racecheck,
		MissingTool:     cudasanitizer.MissingToolFail,
		RacecheckReport: "all",
		RequireCUDAInit: cudasanitizer.ToggleYes,
		RequireOutput:   []string{raceMarker},
		Timeout:         10 * time.Minute,
		LogDir:          t.TempDir(),
	}, prog)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != cudasanitizer.OutcomeSanitizerErrors {
		t.Fatalf("Outcome = %v, want sanitizer-errors for a program with a known shared-memory race\n%s",
			res.Outcome, res.Details())
	}
	t.Log(res.Details())
	if res.Log.Hazards == 0 && res.Log.Errors == 0 {
		t.Errorf("neither hazards nor errors were parsed out of the report:\n%s", res.Log.Text)
	}
}

// TestToolVersionCUDA records which compute-sanitizer produced the reports
// above. A report format is a property of a particular tool build, so a run
// without this line is harder to interpret later.
func TestToolVersionCUDA(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") == "" {
		t.Skip("LIB_CUDA_HARDWARE_TEST is empty: these tests need compute-sanitizer and CUDA programs built ahead of time, which no probe can supply")
	}
	version, err := cudasanitizer.ToolVersion(context.Background(), "")
	if errors.Is(err, cudasanitizer.ErrToolNotFound) {
		t.Fatalf("hardware validation requires compute-sanitizer: %v", err)
	}
	if err != nil {
		t.Fatalf("ToolVersion: %v", err)
	}
	t.Log(version)
}
