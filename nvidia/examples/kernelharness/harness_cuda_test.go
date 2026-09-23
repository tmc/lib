package main

import (
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// smokeConfig keeps the CUDA path quick. It is not a measurement configuration:
// three samples of eight launches establishes that the timer runs, not a
// performance estimate.
func smokeConfig() Config {
	return Config{Trials: 2, Seed: 1, Warmup: 4, Samples: 3, LaunchesPerSample: 8,
		Timeout: 10 * time.Second, Baseline: true}
}

// TestFixtureOutcomes is the discriminating test: each fixture must produce its
// own outcome. A taxonomy where two different failures land on the same value
// has thrown away the part a refinement loop would have learned from, and only
// running all of them together shows that it has not.
//
// testdata/axpby_fault.cu is deliberately absent from this set. It poisons the
// CUDA context, so every later fixture in the same process would report the
// same fault. See TestDeviceFaultFixture.
func TestFixtureOutcomes(t *testing.T) {
	dev := cudatest.Require(t)

	task, err := LookupTask("axpby")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		file string
		want Outcome
	}{
		{"testdata/axpby_correct.cu", OutcomeOK},
		{"testdata/axpby_wrong.cu", OutcomeWrongAnswer},
		{"testdata/axpby_partial.cu", OutcomeWrongAnswer},
		{"testdata/axpby_nocompile.cu", OutcomeCompileError},
		{"testdata/axpby_noentry.cu", OutcomeMissingEntry},
	}
	seen := map[Outcome]string{}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			source, err := os.ReadFile(tc.file)
			if err != nil {
				t.Fatal(err)
			}
			rep, err := Run(dev, task, tc.file, string(source), smokeConfig())
			if err != nil {
				t.Fatal(err)
			}
			if rep.Outcome != tc.want {
				t.Fatalf("outcome %q, want %q (reason: %s)", rep.Outcome, tc.want, rep.Reason)
			}
			if rep.ContextPoisoned {
				t.Fatalf("%s poisoned the context; later fixtures in this process are meaningless", tc.file)
			}
			if rep.Outcome == OutcomeWrongAnswer {
				last := rep.Correctness.Trials[len(rep.Correctness.Trials)-1]
				if last.Comparison.Worst == nil {
					t.Error("a wrong answer reported no worst element")
				}
			}
			if rep.Outcome == OutcomeCompileError && rep.Reason == "" {
				t.Error("a compile error reported no compiler diagnostic")
			}
			seen[rep.Outcome] = tc.file
		})
	}
	// Three distinct outcomes over five fixtures. If the harness collapsed
	// them, this is where it shows.
	if len(seen) < 3 {
		t.Fatalf("five fixtures produced only %d distinct outcomes: %v", len(seen), seen)
	}
}

// A correct candidate must produce a resolved device-time measurement and a
// speedup against the task's own baseline. This is the control on the timer: a
// timer that never resolves would still let every correctness test above pass.
func TestCorrectCandidateIsMeasured(t *testing.T) {
	dev := cudatest.Require(t)
	task, err := LookupTask("axpby")
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile("testdata/axpby_correct.cu")
	if err != nil {
		t.Fatal(err)
	}
	rep, err := Run(dev, task, "testdata/axpby_correct.cu", string(source), smokeConfig())
	if err != nil {
		t.Fatal(err)
	}
	if rep.Outcome != OutcomeOK {
		t.Fatalf("outcome %q: %s", rep.Outcome, rep.Reason)
	}
	if rep.Timing == nil || !rep.Timing.Resolved {
		t.Fatal("the device timer did not resolve the candidate arm")
	}
	if rep.Timing.MedianNanosPerLaunch <= 0 {
		t.Fatalf("median device time per launch is %v", rep.Timing.MedianNanosPerLaunch)
	}
	if len(rep.Timing.SampleNanos) != smokeConfig().Samples {
		t.Fatalf("got %d samples, want %d", len(rep.Timing.SampleNanos), smokeConfig().Samples)
	}
	if rep.Baseline == nil || !rep.Baseline.Resolved {
		t.Fatal("the baseline arm did not resolve")
	}
	if rep.Speedup == nil || rep.Speedup.Ratio <= 0 {
		t.Fatalf("speedup = %+v", rep.Speedup)
	}
	if rep.Compile == nil || rep.Compile.PTXSHA256 == "" {
		t.Fatal("no PTX digest was recorded")
	}
}

// Every built-in task's own baseline kernel must pass that task's own oracle.
// A task whose baseline fails its own reference has a wrong reference, a wrong
// baseline, or a tolerance that cannot be met, and every verdict it issues
// about a candidate would be suspect.
func TestBaselinesPassTheirOwnOracle(t *testing.T) {
	dev := cudatest.Require(t)
	for _, name := range TaskNames() {
		t.Run(name, func(t *testing.T) {
			task := tasks[name]
			cfg := smokeConfig()
			cfg.Baseline = false // the baseline is the candidate here
			rep, err := Run(dev, task, name+"_baseline", task.Baseline, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if rep.Outcome != OutcomeOK {
				t.Fatalf("task %s baseline: outcome %q: %s", name, rep.Outcome, rep.Reason)
			}
		})
	}
}

// TestDeviceFaultFixture runs the faulting fixture. It poisons the CUDA context
// for the whole process, so it is opt-in and must be the only test in its run:
//
//	KERNELHARNESS_FAULT=1 go test ./examples/kernelharness -run TestDeviceFaultFixture
func TestDeviceFaultFixture(t *testing.T) {
	if os.Getenv("KERNELHARNESS_FAULT") != "1" {
		t.Skip("set KERNELHARNESS_FAULT=1 to run the faulting fixture; it poisons the CUDA context for the whole process")
	}
	dev := cudatest.Require(t)
	task, err := LookupTask("axpby")
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile("testdata/axpby_fault.cu")
	if err != nil {
		t.Fatal(err)
	}
	rep, err := Run(dev, task, "testdata/axpby_fault.cu", string(source), smokeConfig())
	if err != nil {
		// Cleanup after a fault returns the same fault; the classified report
		// is what matters, and Run returns it alongside the error.
		t.Logf("cleanup after the fault reported: %v", err)
	}
	if rep == nil {
		t.Fatal("no report")
	}
	if rep.Outcome != OutcomeDeviceFault {
		t.Fatalf("outcome %q, want %q (reason: %s)", rep.Outcome, OutcomeDeviceFault, rep.Reason)
	}
	if !rep.ContextPoisoned {
		t.Error("a device fault did not mark the context poisoned")
	}
}
