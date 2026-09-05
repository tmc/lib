package cudasanitizer

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestIsChild(t *testing.T) {
	tests := []struct {
		args []string
		want bool
	}{
		{[]string{"pkg.test"}, false},
		{[]string{"pkg.test", "-test.v=true"}, false},
		{[]string{"pkg.test", "-" + ChildFlagName}, true},
		{[]string{"pkg.test", "--" + ChildFlagName}, true},
		{[]string{"pkg.test", "-" + ChildFlagName + "=true"}, true},
		{[]string{"pkg.test", "-" + ChildFlagName + "=false"}, false},
		{[]string{"pkg.test", "-" + ChildFlagName, "-test.run=X"}, true},
		// A bare word is an argument to the program under test, not our flag.
		{[]string{"pkg.test", ChildFlagName}, false},
		{[]string{"pkg.test", "-test.run=" + ChildFlagName}, false},
	}
	for _, tt := range tests {
		if got := isChild(tt.args); got != tt.want {
			t.Errorf("isChild(%q) = %v, want %v", tt.args, got, tt.want)
		}
	}
}

// TestMainChildRunsTests checks the half of the recursion guard that stops the
// loop: a process carrying the child flag runs the tests instead of launching
// compute-sanitizer again.
func TestMainChildRunsTests(t *testing.T) {
	t.Setenv(ActiveEnv, "1")
	ran := false
	code := main1(context.Background(),
		[]string{"pkg.test", "-" + ChildFlagName, "-test.run=TestX"},
		Config{Tool: Memcheck, MissingTool: MissingToolFail},
		func() int { ran = true; return 3 },
		io.Discard)
	if !ran {
		t.Error("the child did not run the tests")
	}
	if code != 3 {
		t.Errorf("exit code = %d, want the test runner's 3", code)
	}
}

// TestMainSpoofedChildFlagStillInstruments checks that the child flag alone,
// without the environment variable Main sets on the process it launches, does
// not persuade a parent that it is the child. Believing it would run the tests
// with no instrumentation at all and exit 0, so that a run which validated
// nothing would be indistinguishable from a clean one. `go test ./pkg -args
// -cudasanitizer.child` is enough to produce that argv.
func TestMainSpoofedChildFlagStillInstruments(t *testing.T) {
	t.Setenv(ActiveEnv, "")
	ran := false
	var diag strings.Builder
	code := main1(context.Background(),
		[]string{"pkg.test", "-" + ChildFlagName, "-test.run=TestX"},
		Config{Tool: Memcheck, Path: filepath.Join(t.TempDir(), "absent"), MissingTool: MissingToolFail},
		func() int { ran = true; return 0 },
		&diag)
	if ran {
		t.Error("the tests ran uninstrumented because a stray flag was mistaken for the child marker")
	}
	if code == 0 {
		t.Errorf("exit code = 0, want a failure: nothing was instrumented")
	}
	if !strings.Contains(diag.String(), ActiveEnv) {
		t.Errorf("diagnostics did not mention %s:\n%s", ActiveEnv, diag.String())
	}
}

// TestMainRefusesToRecurse covers the other half: a parent that finds itself
// already running under compute-sanitizer but is not marked as the child stops
// loudly instead of forking forever.
func TestMainRefusesToRecurse(t *testing.T) {
	t.Setenv(ActiveEnv, "1")
	var diag bytes.Buffer
	ran := false
	code := main1(context.Background(),
		[]string{"pkg.test"},
		Config{Tool: Memcheck, MissingTool: MissingToolFail, Path: fixtureTool(t)},
		func() int { ran = true; return 0 },
		&diag)
	if ran {
		t.Error("the tests ran; the parent should have refused before doing anything")
	}
	if code == 0 {
		t.Error("exit code = 0, want nonzero")
	}
	if !strings.Contains(diag.String(), "refusing") {
		t.Errorf("diagnostics = %q, want an explanation of the refusal", diag.String())
	}
}

func TestMainInvalidConfig(t *testing.T) {
	var diag bytes.Buffer
	code := main1(context.Background(), []string{"pkg.test"}, Config{Tool: Memcheck},
		func() int { t.Error("tests ran under an invalid config"); return 0 }, &diag)
	if code == 0 {
		t.Error("exit code = 0, want nonzero for an invalid config")
	}
	if !strings.Contains(diag.String(), "MissingTool") {
		t.Errorf("diagnostics = %q, want the unset policy named", diag.String())
	}
}

// TestMainMissingToolPolicies pins the two ways an absent compute-sanitizer may
// be handled. Neither reports a validated pass.
func TestMainMissingToolPolicies(t *testing.T) {
	absent := filepath.Join(t.TempDir(), "compute-sanitizer-that-is-not-installed")

	t.Run("skip runs the tests uninstrumented and says so", func(t *testing.T) {
		var diag bytes.Buffer
		ran := false
		code := main1(context.Background(), []string{"pkg.test"},
			Config{Tool: Memcheck, MissingTool: MissingToolSkip, Path: absent},
			func() int { ran = true; return 0 }, &diag)
		if !ran {
			t.Error("the tests did not run under MissingToolSkip")
		}
		if code != 0 {
			t.Errorf("exit code = %d, want the test runner's 0", code)
		}
		if !strings.Contains(diag.String(), "WITHOUT sanitizer instrumentation") {
			t.Errorf("diagnostics = %q, want the absence of validation announced", diag.String())
		}
	})

	t.Run("fail does not run the tests", func(t *testing.T) {
		var diag bytes.Buffer
		code := main1(context.Background(), []string{"pkg.test"},
			Config{Tool: Memcheck, MissingTool: MissingToolFail, Path: absent},
			func() int { t.Error("tests ran under MissingToolFail"); return 0 }, &diag)
		if code == 0 {
			t.Error("exit code = 0, want nonzero when required validation is unavailable")
		}
		if !strings.Contains(diag.String(), "compute-sanitizer not found") {
			t.Errorf("diagnostics = %q, want the missing tool named", diag.String())
		}
	})
}

// TestMainPreservesTestArguments runs the parent path end to end against the
// fixture, which records the argument vector it was given without launching
// anything. It checks that the child is marked, that the test binary is the
// program under test, and that the caller's own test flags survive in order.
func TestMainPreservesTestArguments(t *testing.T) {
	argvFile := filepath.Join(t.TempDir(), "argv")
	osArgs := []string{"pkg.test", "-test.run=TestVectorAdd", "-test.v=true", "-test.count=2", "-test.timeout=30m"}

	var diag bytes.Buffer
	cfg := Config{
		Tool:        Memcheck,
		MissingTool: MissingToolFail,
		Path:        fixtureTool(t),
		Env: fixtureEnv(
			fakeArgvFile+"="+argvFile,
			fakeReport+"="+cleanReport,
		),
		Stdout: io.Discard,
		Stderr: io.Discard,
	}
	code := main1(context.Background(), osArgs, cfg,
		func() int { t.Error("the parent ran the tests itself"); return 0 }, &diag)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for a clean run\n%s", code, diag.String())
	}

	data, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("reading recorded argv: %v", err)
	}
	argv := strings.Split(string(data), "\n")

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	i := indexOf(argv, exe)
	if i < 0 {
		t.Fatalf("argv = %q, want the test binary %q as the program under test", argv, exe)
	}
	got := argv[i+1:]
	want := append([]string{"-" + ChildFlagName}, osArgs[1:]...)
	if len(got) != len(want) {
		t.Fatalf("child args = %q, want %q", got, want)
	}
	for j := range want {
		if got[j] != want[j] {
			t.Fatalf("child args = %q, want %q", got, want)
		}
	}
	// The child must be recognizable to itself from exactly these arguments.
	if !isChild(append([]string{exe}, got...)) {
		t.Errorf("isChild(%q) = false; the child would relaunch the sanitizer", got)
	}
}

// TestMainAddsNoTestsRanEvidence checks that a run matching no tests is not
// reported as a clean sanitizer validation.
func TestMainAddsNoTestsRanEvidence(t *testing.T) {
	var diag bytes.Buffer
	cfg := Config{
		Tool:        Memcheck,
		MissingTool: MissingToolFail,
		Path:        fixtureTool(t),
		Env: fixtureEnv(
			fakeReport+"="+cleanReport,
			fakeStdout+"="+noTestsRan,
		),
		Stdout: io.Discard,
		Stderr: io.Discard,
	}
	code := main1(context.Background(), []string{"pkg.test", "-test.run=NoSuchTest"}, cfg,
		func() int { return 0 }, &diag)
	if code == 0 {
		t.Errorf("exit code = 0, want nonzero: zero errors over zero tests is not a validation\n%s", diag.String())
	}
	if !strings.Contains(diag.String(), noTestsRan) {
		t.Errorf("diagnostics = %q, want the missing work explained", diag.String())
	}
}

func TestReportOutcomes(t *testing.T) {
	clean := &Result{Outcome: OutcomeClean, Log: &Log{SummaryFound: true}}
	fake := &fakeTB{TB: t}
	clean.Report(fake)
	if fake.errors > 0 || fake.fatals > 0 || fake.skips > 0 {
		t.Errorf("a clean result reported %+v, want only a log line", fake)
	}

	bad := &Result{Outcome: OutcomeSanitizerErrors, Log: ParseLog(errorReport)}
	fake = &fakeTB{TB: t}
	bad.Report(fake)
	if fake.errors != 1 {
		t.Errorf("sanitizer errors reported %+v, want one Error", fake)
	}
	if !strings.Contains(fake.last, "Invalid __global__ write") {
		t.Errorf("the reported message omits the findings: %s", fake.last)
	}
}

// fakeTB records what Report did without failing the enclosing test. Fatal is
// recorded rather than obeyed, which is safe here because Report calls it last.
type fakeTB struct {
	testing.TB
	errors, fatals, skips int
	last                  string
}

func (f *fakeTB) Helper()                   {}
func (f *fakeTB) Log(args ...any)           { f.last = sprint(args) }
func (f *fakeTB) Error(args ...any)         { f.errors++; f.last = sprint(args) }
func (f *fakeTB) Errorf(s string, a ...any) { f.errors++; f.last = s }
func (f *fakeTB) Fatal(args ...any)         { f.fatals++; f.last = sprint(args) }
func (f *fakeTB) Fatalf(s string, a ...any) { f.fatals++; f.last = s }
func (f *fakeTB) Skip(args ...any)          { f.skips++; f.last = sprint(args) }
func (f *fakeTB) Skipf(s string, a ...any)  { f.skips++; f.last = s }
func (f *fakeTB) SkipNow()                  { f.skips++ }

func sprint(args []any) string {
	var b strings.Builder
	for _, a := range args {
		if s, ok := a.(string); ok {
			b.WriteString(s)
		}
	}
	return b.String()
}

// TestSelfLaunchMarker is the test the self-launch child runs. It prints a
// marker on stdout, which the parent requires as evidence that the child
// really executed rather than merely starting.
func TestSelfLaunchMarker(t *testing.T) {
	if os.Getenv(selfTestEnv) == "" {
		t.Skip("only meaningful inside TestSelfLaunchTerminates")
	}
	fmt.Println(selfTestMark)
}

// TestSelfLaunchTerminates is the end-to-end check on the recursion guard. It
// runs this test binary as if it were a package whose TestMain calls Main,
// with a stand-in tool that really does execute the application it is given.
// The binary therefore launches itself once, and must then stop.
//
// A broken sentinel shows up here as a process that never returns, so the run
// is bounded and a timeout fails the test.
func TestSelfLaunchTerminates(t *testing.T) {
	tool := fixtureTool(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	cmd := exec.Command(exe, "-test.run=TestSelfLaunchMarker", "-test.v=true", "-test.count=1")
	cmd.Env = append(os.Environ(),
		selfTestEnv+"=1",
		selfToolEnv+"="+tool,
		fakeExecApp+"=1",
		fakeReport+"="+cleanReport,
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	cmd.WaitDelay = 5 * time.Second
	group, err := startGroup(cmd)
	if err != nil {
		t.Fatal(err)
	}
	defer group.close()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err = <-done:
	case <-ctx.Done():
		_ = group.kill()
		<-done
		t.Fatalf("the self-launching test binary did not terminate within the deadline; "+
			"the child sentinel failed and it re-executed itself\n%s", out.String())
	}
	if err != nil {
		t.Fatalf("self-launching run failed: %v\n%s", err, out.String())
	}

	got := out.String()
	if n := strings.Count(got, selfTestMark); n != 1 {
		t.Errorf("the marker appeared %d times, want exactly 1; more than one means the binary re-launched itself\n%s",
			n, got)
	}
	if !strings.Contains(got, "compute-sanitizer --tool memcheck: clean") {
		t.Errorf("the parent did not report a clean run:\n%s", got)
	}
	// The child must have been given the sentinel and the active marker.
	if strings.Contains(got, "refusing") {
		t.Errorf("the recursion guard fired, so the child flag did not reach the child:\n%s", got)
	}
}
