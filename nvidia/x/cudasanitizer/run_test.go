package cudasanitizer

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// baseConfig is a Config that passes validation, pointed at the fixture that
// impersonates compute-sanitizer.
func baseConfig(t *testing.T) Config {
	t.Helper()
	return Config{
		Tool:        Memcheck,
		MissingTool: MissingToolFail,
		Path:        fixtureTool(t),
	}
}

func run(t *testing.T, cfg Config, env []string) *Result {
	t.Helper()
	cfg.Env = env
	res, err := Run(context.Background(), cfg, "/nonexistent/app-under-test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return res
}

func TestOutcomeClean(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeStdout+"=vector add ok",
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeClean {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeClean, res.Details())
	}
	if !res.Log.SummaryFound || res.Log.Errors != 0 {
		t.Errorf("Log = %+v, want a summary with 0 errors", res.Log)
	}
	if !strings.Contains(res.Stdout, "vector add ok") {
		t.Errorf("Stdout = %q, want the application output retained", res.Stdout)
	}
}

// TestSanitizerErrorsWithExitStatusZero is the trap this package exists for.
// compute-sanitizer's --error-exitcode defaults to 0, so a tool build or
// wrapper that ignores our flag reports errors and still exits 0. The verdict
// must come from the report, not the status.
func TestSanitizerErrorsWithExitStatusZero(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+errorReport,
		fakeStdout+"=vector add ok",
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeSanitizerErrors {
		t.Fatalf("Outcome = %v, want %v; a clean exit status must not outrank a report of errors\n%s",
			res.Outcome, OutcomeSanitizerErrors, res.Details())
	}
	if res.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0 for this fixture", res.ExitCode)
	}
	if res.Log.Errors != 2 {
		t.Errorf("Log.Errors = %d, want 2", res.Log.Errors)
	}
	if len(res.Log.Findings) != 2 {
		t.Errorf("len(Log.Findings) = %d, want 2", len(res.Log.Findings))
	}
	if !strings.Contains(res.Details(), "Invalid __global__ write") {
		t.Errorf("Details() does not carry the report:\n%s", res.Details())
	}
}

func TestSanitizerErrorsWithErrorExitCode(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+errorReport,
		fakeExit+"="+strconv.Itoa(DefaultErrorExitCode),
	))
	if res.Outcome != OutcomeSanitizerErrors {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeSanitizerErrors, res.Details())
	}
	if res.ExitCodeCollision {
		t.Error("ExitCodeCollision = true, want false: the report agreed with the exit status")
	}
}

func TestApplicationFailed(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeStderr+"=panic: nil map",
		fakeExit+"=2",
	))
	if res.Outcome != OutcomeApplicationFailed {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeApplicationFailed, res.Details())
	}
	if res.ExitCode != 2 {
		t.Errorf("ExitCode = %d, want 2", res.ExitCode)
	}
}

// TestExitCodeCollision covers an application that happens to exit with the
// status reserved for --error-exitcode. The report says zero errors, so this is
// the application's own failure and the ambiguity is recorded rather than
// silently resolved as sanitizer errors.
func TestExitCodeCollision(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeExit+"="+strconv.Itoa(DefaultErrorExitCode),
	))
	if res.Outcome != OutcomeApplicationFailed {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeApplicationFailed, res.Details())
	}
	if !res.ExitCodeCollision {
		t.Error("ExitCodeCollision = false, want true")
	}
	if !strings.Contains(res.Summary(), "reserved for --error-exitcode") {
		t.Errorf("Summary() does not mention the collision: %s", res.Summary())
	}
}

// TestMissingEvidence covers a run that initializes and exits cleanly without
// doing the work the caller cares about. Zero errors over no work is not a
// clean validation.
func TestMissingEvidence(t *testing.T) {
	cfg := baseConfig(t)
	cfg.RequireOutput = []string{"vector add ok"}
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeStdout+"=cuda initialized",
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeApplicationFailed {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeApplicationFailed, res.Details())
	}
	if len(res.MissingEvidence) != 1 || res.MissingEvidence[0] != "vector add ok" {
		t.Errorf("MissingEvidence = %q, want [%q]", res.MissingEvidence, "vector add ok")
	}
}

func TestForbiddenEvidence(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ForbidOutput = []string{noTestsRan}
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeStdout+"="+noTestsRan,
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeApplicationFailed {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeApplicationFailed, res.Details())
	}
	if len(res.ForbiddenEvidence) != 1 {
		t.Errorf("ForbiddenEvidence = %q, want one entry", res.ForbiddenEvidence)
	}
}

// TestNoVerdict covers a tool that never reached a summary line. Nothing can be
// concluded from such a run, so it is a tool failure rather than a pass, even
// though the process exited 0.
func TestNoVerdict(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+truncatedReport,
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeToolFailure {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeToolFailure, res.Details())
	}
	if !errors.Is(res.Err, ErrNoVerdict) {
		t.Errorf("Err = %v, want it to wrap ErrNoVerdict", res.Err)
	}
}

func TestNoLogAtAll(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeNoLog+"=1",
		fakeStderr+"=compute-sanitizer: could not open the application",
		fakeExit+"=1",
	))
	if res.Outcome != OutcomeToolFailure {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeToolFailure, res.Details())
	}
	if !errors.Is(res.Err, ErrNoVerdict) {
		t.Errorf("Err = %v, want it to wrap ErrNoVerdict", res.Err)
	}
}

// TestReportOnStreams covers a tool that ignored --log-file and printed its
// report on stderr instead. The verdict is still read rather than discarded.
func TestReportOnStreams(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(
		fakeNoLog+"=1",
		fakeStderr+"="+errorReport,
		fakeExit+"=0",
	))
	if res.Outcome != OutcomeSanitizerErrors {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeSanitizerErrors, res.Details())
	}
	if res.Log.Errors != 2 {
		t.Errorf("Log.Errors = %d, want 2", res.Log.Errors)
	}
}

func TestToolNotFound(t *testing.T) {
	cfg := baseConfig(t)
	cfg.Path = filepath.Join(t.TempDir(), "compute-sanitizer-that-is-not-installed")
	res, err := Run(context.Background(), cfg, "/nonexistent/app-under-test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != OutcomeToolFailure {
		t.Fatalf("Outcome = %v, want %v", res.Outcome, OutcomeToolFailure)
	}
	if !errors.Is(res.Err, ErrToolNotFound) {
		t.Errorf("Err = %v, want it to wrap ErrToolNotFound", res.Err)
	}
	if res.ExitCode != -1 {
		t.Errorf("ExitCode = %d, want -1 for a run that never started", res.ExitCode)
	}
}

// TestTimeoutKillsProcessGroup checks that a timeout is a tool failure and that
// the whole process group dies. The fixture leaves a grandchild holding the
// inherited output pipes; killing only the direct child would leave Run blocked
// until KillDelay, which is set far longer than the assertion window.
func TestTimeoutKillsProcessGroup(t *testing.T) {
	cfg := baseConfig(t)
	cfg.Timeout = 200 * time.Millisecond
	cfg.KillDelay = 60 * time.Second

	start := time.Now()
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeSpawn+"=1",
		fakeSleep+"=5m",
		fakeExit+"=0",
	))
	elapsed := time.Since(start)

	if res.Outcome != OutcomeToolFailure {
		t.Fatalf("Outcome = %v, want %v; a timeout is never a pass\n%s",
			res.Outcome, OutcomeToolFailure, res.Details())
	}
	if !errors.Is(res.Err, ErrTimeout) {
		t.Errorf("Err = %v, want it to wrap ErrTimeout", res.Err)
	}
	if elapsed > 20*time.Second {
		t.Errorf("Run took %s; the grandchild holding the pipes was not killed with the group", elapsed)
	}
}

func TestContextCancelIsNotAPass(t *testing.T) {
	cfg := baseConfig(t)
	cfg.KillDelay = 10 * time.Second
	cfg.Env = fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeSleep+"=5m",
	)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()
	res, err := Run(ctx, cfg, "/nonexistent/app-under-test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != OutcomeToolFailure {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeToolFailure, res.Details())
	}
	if !errors.Is(res.Err, context.Canceled) {
		t.Errorf("Err = %v, want it to wrap context.Canceled", res.Err)
	}
}

// TestArgvSeenByTool checks what actually reaches compute-sanitizer: an
// explicit nonzero --error-exitcode, the selected tool, a log file, and the
// application with its own arguments after all the options.
func TestArgvSeenByTool(t *testing.T) {
	argvFile := filepath.Join(t.TempDir(), "argv")
	cfg := baseConfig(t)
	cfg.Tool = Racecheck
	cfg.RacecheckReport = "all"
	cfg.RequireCUDAInit = ToggleYes
	cfg.Env = fixtureEnv(
		fakeArgvFile+"="+argvFile,
		fakeReport+"="+cleanReport,
	)
	res, err := Run(context.Background(), cfg, "/nonexistent/app-under-test", "-run", "TestX", "-v")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != OutcomeClean {
		t.Fatalf("Outcome = %v, want %v\n%s", res.Outcome, OutcomeClean, res.Details())
	}
	data, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("reading recorded argv: %v", err)
	}
	argv := strings.Split(string(data), "\n")
	want := []string{"--tool", "racecheck", "--error-exitcode", strconv.Itoa(DefaultErrorExitCode)}
	if len(argv) < len(want) {
		t.Fatalf("argv = %q, too short", argv)
	}
	for i, w := range want {
		if argv[i] != w {
			t.Fatalf("argv[%d] = %q, want %q (argv = %q)", i, argv[i], w, argv)
		}
	}
	tail := argv[len(argv)-4:]
	wantTail := []string{"/nonexistent/app-under-test", "-run", "TestX", "-v"}
	for i, w := range wantTail {
		if tail[i] != w {
			t.Fatalf("argv tail = %q, want %q", tail, wantTail)
		}
	}
	if !contains(argv, "--racecheck-report") || !contains(argv, "--require-cuda-init") {
		t.Errorf("argv = %q, want the configured options present", argv)
	}
	if contains(argv, "0") {
		t.Errorf("argv = %q, contains a bare 0; --error-exitcode must never be 0", argv)
	}
}

func TestLogRetention(t *testing.T) {
	dir := t.TempDir()
	cfg := baseConfig(t)
	cfg.LogDir = dir
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+errorReport,
		fakeExit+"="+strconv.Itoa(DefaultErrorExitCode),
	))
	if res.LogPath == "" {
		t.Fatal("LogPath is empty, want the retained log")
	}
	data, err := os.ReadFile(res.LogPath)
	if err != nil {
		t.Fatalf("reading the retained log: %v", err)
	}
	if !strings.Contains(string(data), "ERROR SUMMARY: 2 errors") {
		t.Errorf("retained log does not hold the report:\n%s", data)
	}
}

func TestLogRemovedWhenNotRetained(t *testing.T) {
	cfg := baseConfig(t)
	res := run(t, cfg, fixtureEnv(fakeReport+"="+cleanReport))
	if res.LogPath != "" {
		t.Errorf("LogPath = %q, want empty when Config.LogDir is unset", res.LogPath)
	}
	if !strings.Contains(res.Log.Text, "ERROR SUMMARY") {
		t.Errorf("Log.Text = %q, want the report retained in memory", res.Log.Text)
	}
}

func TestOutputTruncation(t *testing.T) {
	cfg := baseConfig(t)
	cfg.MaxOutputBytes = 16
	res := run(t, cfg, fixtureEnv(
		fakeReport+"="+cleanReport,
		fakeStdout+"="+strings.Repeat("x", 200),
	))
	if len(res.Stdout) != 16 {
		t.Errorf("len(Stdout) = %d, want 16", len(res.Stdout))
	}
	if !res.StdoutTruncated {
		t.Error("StdoutTruncated = false, want true")
	}
}

func TestRunRejectsInvalidConfig(t *testing.T) {
	res, err := Run(context.Background(), Config{Tool: Memcheck}, "/bin/true")
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v, want ErrInvalidConfig for an unset MissingTool policy", err)
	}
	if res != nil {
		t.Error("Result is non-nil for a rejected config")
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func TestEvidenceBeyondRetainedOutput(t *testing.T) {
	for _, tc := range []struct {
		name            string
		stream          string
		require, forbid []string
		want            Outcome
	}{
		{"required stdout", fakeStdout, []string{"work completed"}, nil, OutcomeClean},
		{"required stderr", fakeStderr, []string{"work completed"}, nil, OutcomeClean},
		{"forbidden stdout", fakeStdout, nil, []string{"work completed"}, OutcomeApplicationFailed},
		{"forbidden stderr", fakeStderr, nil, []string{"work completed"}, OutcomeApplicationFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t)
			cfg.MaxOutputBytes = 16
			cfg.RequireOutput, cfg.ForbidOutput = tc.require, tc.forbid
			res := run(t, cfg, fixtureEnv(fakeReport+"="+cleanReport, tc.stream+"="+strings.Repeat("x", 200)+"work completed"))
			if res.Outcome != tc.want {
				t.Fatalf("outcome = %v, want %v\n%s", res.Outcome, tc.want, res.Details())
			}
			if len(res.Stdout) > 16 || len(res.Stderr) > 16 {
				t.Fatal("output retention limit exceeded")
			}
		})
	}
}

func TestTruncatedFallbackReportIsNotClean(t *testing.T) {
	cfg := baseConfig(t)
	cfg.MaxOutputBytes = 128
	res := run(t, cfg, fixtureEnv(fakeNoLog+"=1", fakeStderr+"="+cleanReport+strings.Repeat("x", 256)+"\n"+errorReport, fakeExit+"=0"))
	if res.Outcome != OutcomeToolFailure || !errors.Is(res.Err, ErrNoVerdict) {
		t.Fatalf("truncated report produced %v (%v)\n%s", res.Outcome, res.Err, res.Details())
	}
}

func TestLookToolExplicitPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "compute-sanitizer")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := LookTool(path)
	if err != nil {
		t.Fatalf("LookTool(%q): %v", path, err)
	}
	if got != path {
		t.Errorf("LookTool(%q) = %q, want the path as given", path, got)
	}
	missing := filepath.Join(dir, "absent")
	if _, err := LookTool(missing); !errors.Is(err, ErrToolNotFound) {
		t.Errorf("LookTool(%q) error = %v, want it to wrap ErrToolNotFound", missing, err)
	}
}

func TestToolVersionLine(t *testing.T) {
	// Outputs measured from the compute-sanitizer builds installed on the
	// GB10 host. The banner line is identical across every release, so a
	// function that returned it would record nothing.
	tests := []struct {
		name string
		out  string
		want string
	}{
		{
			name: "cuda 13.4",
			out:  "NVIDIA (R) Compute Sanitizer\nCopyright (c) 2020-2026 NVIDIA Corporation\nVersion 2026.3.0.0 (build 38637409) (public-release)\n",
			want: "Version 2026.3.0.0 (build 38637409) (public-release)",
		},
		{
			name: "cuda 13.0",
			out:  "NVIDIA (R) Compute Sanitizer\nCopyright (c) 2020-2025 NVIDIA Corporation\nVersion 2025.3.1.0 (build 36400806) (public-release)\n",
			want: "Version 2025.3.1.0 (build 36400806) (public-release)",
		},
		{
			name: "distribution package 2022.4.1",
			out:  "NVIDIA (R) Compute Sanitizer\nCopyright (c) 2020-2023 NVIDIA Corporation\nVersion 2022.4.1\n",
			want: "Version 2022.4.1",
		},
		{
			name: "leading blank lines",
			out:  "\n\n  Version 2026.3.0.0\n",
			want: "Version 2026.3.0.0",
		},
		{
			name: "no version line falls back to the first line",
			out:  "\nsome other build\nand more\n",
			want: "some other build",
		},
		{name: "empty", out: "\n \n", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toolVersionLine(tt.out); got != tt.want {
				t.Errorf("toolVersionLine() = %q, want %q", got, tt.want)
			}
		})
	}
}
