package cudasanitizer

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
)

const (
	// ChildFlagName is the flag [Main] prepends to the child test binary's
	// arguments to mark it as the process running under compute-sanitizer.
	// The flag marks the child, but it does not by itself identify one:
	// arguments do leak, through `go test -args`, wrapper scripts and copied
	// command lines. [Child] requires [ActiveEnv] as well, so that a stray
	// flag cannot make a parent skip instrumentation and report success.
	ChildFlagName = "cudasanitizer.child"

	// ActiveEnv is set to "1" in the child's environment. [Main] refuses to
	// launch compute-sanitizer when it is already set, which stops a parent
	// that somehow lost the child flag from re-executing itself forever.
	// Descendant processes can also read it to tell that they are running
	// under the sanitizer.
	ActiveEnv = "CUDASANITIZER_ACTIVE"

	// noTestsRan is the testing package's warning for a run that matched no
	// tests. [Main] treats it as evidence that the intended work did not
	// execute, because a sanitizer report of zero errors over zero tests is
	// not a validation.
	noTestsRan = "testing: warning: no tests to run"
)

// registerChildFlag defines ChildFlagName so that the child's flag.Parse, which
// runs inside m.Run, accepts it. Registration is idempotent.
func registerChildFlag() {
	if flag.CommandLine.Lookup(ChildFlagName) == nil {
		flag.Bool(ChildFlagName, false, "set by cudasanitizer.Main on the test binary it runs under compute-sanitizer")
	}
}

// Child reports whether this process is the test binary that [Main]
// re-executed under compute-sanitizer.
//
// Both the flag and [ActiveEnv] must be present. The flag alone is not
// trusted: it is an ordinary argument, so `go test -args -cudasanitizer.child`
// or a stale wrapper script puts it in a parent's os.Args, and a parent that
// believed it were the child would run the tests uninstrumented and exit 0
// with nothing to show that no validation happened. Only [Main] sets ActiveEnv,
// and it sets it solely on the process it launches under the tool.
//
// It inspects os.Args directly rather than the parsed flag value, because
// TestMain runs before the testing package parses flags.
func Child() bool { return isChild(os.Args) && os.Getenv(ActiveEnv) == "1" }

func isChild(args []string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			continue
		}
		name, value, hasValue := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-"), "=")
		if name != ChildFlagName {
			continue
		}
		if !hasValue {
			return true
		}
		if b, err := strconv.ParseBool(value); err == nil {
			return b
		}
	}
	return false
}

// Main runs a package's tests under compute-sanitizer. It is meant to be the
// whole body of TestMain:
//
//	func TestMain(m *testing.M) {
//		os.Exit(cudasanitizer.Main(m, cudasanitizer.Config{
//			Tool:          cudasanitizer.Memcheck,
//			MissingTool:   cudasanitizer.MissingToolSkip,
//			RequireOutput: []string{"--- PASS: TestVectorAdd"},
//		}))
//	}
//
// In the child process — the one launched under compute-sanitizer, recognized
// by [ChildFlagName] in its arguments — Main runs m.Run and returns its exit
// code, so the tests themselves behave normally.
//
// In the parent, Main re-executes the current test binary under
// compute-sanitizer and returns 0 only for [OutcomeClean]. The parent's own
// os.Args are passed through unchanged, with [ChildFlagName] prepended, so
// -run, -v, -count, -timeout and -args all reach the tests the caller asked
// for. Note that -test.timeout is enforced inside the child, where every kernel
// launch is instrumented and correspondingly slower; a suite that fits in the
// default timeout uninstrumented may need a larger one here.
//
// When compute-sanitizer is not installed, [Config.MissingTool] decides:
// [MissingToolFail] returns a nonzero code without running anything, and
// [MissingToolSkip] runs the tests uninstrumented after announcing on stderr
// that no sanitizer validation took place. Main never reports success for
// validation that did not happen.
//
// Config.Timeout also bounds the preliminary tool version query.
//
// Main writes its own diagnostics to stderr and passes the child's output
// through to the parent's stdout and stderr as it is produced.
func Main(m *testing.M, cfg Config) int {
	registerChildFlag()
	return main1(context.Background(), os.Args, cfg, m.Run, os.Stderr)
}

// main1 is Main with its arguments, test runner, diagnostics stream and context
// injected, so that both halves can be tested without a real test binary.
func main1(ctx context.Context, osArgs []string, cfg Config, run func() int, diag io.Writer) int {
	logf := func(format string, args ...any) {
		fmt.Fprintf(diag, "cudasanitizer: "+format+"\n", args...)
	}

	flagged, active := isChild(osArgs), os.Getenv(ActiveEnv) == "1"
	if flagged && active {
		return run()
	}
	if flagged {
		// The flag arrived from somewhere other than Main: an -args
		// pass-through, a wrapper script, a copied command line. Treating it
		// as proof of childhood would silently skip instrumentation, so say
		// so and instrument this process as the parent it actually is.
		logf("-%s is set but %s is not, so this process was not launched by Main.", ChildFlagName, ActiveEnv)
		logf("ignoring the flag and running compute-sanitizer here rather than skipping instrumentation.")
	}

	if err := cfg.validate(); err != nil {
		logf("%v", err)
		return 2
	}
	if v := os.Getenv(ActiveEnv); v != "" {
		logf("%s=%s is already set but this process is not marked with -%s;", ActiveEnv, v, ChildFlagName)
		logf("refusing to launch compute-sanitizer again rather than re-executing forever.")
		logf("unset %s if it was inherited from an unrelated run.", ActiveEnv)
		return 2
	}

	if cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()
	}

	toolPath, err := LookTool(cfg.Path)
	if err != nil {
		switch cfg.MissingTool {
		case MissingToolSkip:
			logf("%v", err)
			logf("running the tests WITHOUT sanitizer instrumentation; this run validates nothing about")
			logf("device memory errors, races, uninitialized reads, or synchronization.")
			return run()
		default:
			logf("%v", err)
			logf("Config.MissingTool is %s, so this is a failure rather than a skip.", cfg.MissingTool)
			return 1
		}
	}
	if version, err := ToolVersion(ctx, toolPath); err == nil {
		logf("%s", version)
	}

	exe, err := os.Executable()
	if err != nil {
		logf("locating the test binary: %v", err)
		return 2
	}

	child := cfg
	if child.Env == nil {
		child.Env = os.Environ()
	}
	child.Env = append(child.Env, ActiveEnv+"=1")
	if child.Stdout == nil {
		child.Stdout = os.Stdout
	}
	if child.Stderr == nil {
		child.Stderr = os.Stderr
	}
	child.ForbidOutput = append(append([]string(nil), cfg.ForbidOutput...), noTestsRan)

	args := append([]string{"-" + ChildFlagName}, osArgs[1:]...)
	res, err := Run(ctx, child, exe, args...)
	if err != nil {
		logf("%v", err)
		return 2
	}

	if res.Outcome == OutcomeClean {
		logf("%s", res.Summary())
		return 0
	}
	fmt.Fprintln(diag, "cudasanitizer: "+res.Details())
	return 1
}

// Check runs prog under compute-sanitizer and reports the outcome to t. It is
// [Run] followed by [Result.Report], for tests that drive a separate GPU
// program rather than running themselves under the tool.
//
// A configuration error fails the test immediately, since it is a bug in the
// test rather than a finding about the program.
func Check(t testing.TB, cfg Config, prog string, args ...string) *Result {
	t.Helper()
	res, err := Run(context.Background(), cfg, prog, args...)
	if err != nil {
		t.Fatalf("cudasanitizer.Run: %v", err)
		return nil
	}
	res.Report(t)
	return res
}

// Report records the result on t.
//
// A clean run is logged. Sanitizer errors and application failures are errors,
// reported with the full tool report and the program's output, because a
// finding the reader cannot see is nearly useless. A tool failure is fatal,
// except that a missing compute-sanitizer under [MissingToolSkip] skips the
// test with an explicit notice. Report never leaves a run silently passing.
func (r *Result) Report(t testing.TB) {
	t.Helper()
	switch r.Outcome {
	case OutcomeClean:
		t.Log(r.Summary())
	case OutcomeSanitizerErrors, OutcomeApplicationFailed:
		t.Error(r.Details())
	default:
		if errors.Is(r.Err, ErrToolNotFound) && r.Config.MissingTool == MissingToolSkip {
			t.Skipf("%v; no sanitizer validation was performed", r.Err)
			return
		}
		t.Fatal(r.Details())
	}
}
