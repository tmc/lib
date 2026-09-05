package cudasanitizer

import (
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// The tests in this package do not run compute-sanitizer and do not touch a
// GPU. They run a fixture executable that impersonates compute-sanitizer: it
// reads the --log-file path out of its own arguments, writes a caller-supplied
// report there, prints caller-supplied text, and exits with a caller-supplied
// status.
//
// That is enough to pin this package's behavior — argument construction,
// subprocess and process-group handling, log parsing, and the mapping from
// (exit status, report) to [Outcome] — including the case the package exists
// for, a report full of errors delivered with exit status 0.
//
// It is host orchestration evidence and nothing else. It says nothing about
// CUDA, nothing about the real tool's report format on any given release, and
// nothing about whether a real race would be detected. Those claims need the
// hardware tests in cuda_test.go.

// Environment variables the fixture reads. They are namespaced so that a
// stray value cannot influence anything but the fixture.
const (
	fakeArgvFile     = "CUDASANITIZER_FIXTURE_ARGV_FILE"
	fakeReport       = "CUDASANITIZER_FIXTURE_REPORT"
	fakeNoLog        = "CUDASANITIZER_FIXTURE_NO_LOG"
	fakeStdout       = "CUDASANITIZER_FIXTURE_STDOUT"
	fakeStderr       = "CUDASANITIZER_FIXTURE_STDERR"
	fakeExit         = "CUDASANITIZER_FIXTURE_EXIT"
	fakeSleep        = "CUDASANITIZER_FIXTURE_SLEEP"
	fakeSpawn        = "CUDASANITIZER_FIXTURE_SPAWN_SLEEPER"
	fakeChildPID     = "CUDASANITIZER_FIXTURE_CHILD_PID"
	fakeChildNoPipes = "CUDASANITIZER_FIXTURE_CHILD_NO_PIPES"
	fakeSleeper      = "CUDASANITIZER_FIXTURE_SLEEPER"
	fakeExecApp      = "CUDASANITIZER_FIXTURE_EXEC_APP"
)

// Environment variables that turn this test binary into a package whose
// TestMain runs its own tests under the tool, for the end-to-end self-launch
// test.
const (
	selfTestEnv  = "CUDASANITIZER_SELFTEST"
	selfToolEnv  = "CUDASANITIZER_SELFTEST_TOOL"
	selfTestMark = "SELFTEST-MARKER"
)

const fixtureSource = `// Command fixture impersonates compute-sanitizer for cudasanitizer's tests.
package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func main() {
	if os.Getenv("CUDASANITIZER_FIXTURE_SLEEPER") != "" {
		// A grandchild holding the inherited output pipes, used to check
		// that a timeout kills the whole process group.
		time.Sleep(5 * time.Minute)
		return
	}

	args := os.Args[1:]
	if p := os.Getenv("CUDASANITIZER_FIXTURE_ARGV_FILE"); p != "" {
		if err := os.WriteFile(p, []byte(strings.Join(args, "\n")), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "fixture:", err)
			os.Exit(120)
		}
	}

	logPath := ""
	for i, a := range args {
		if a == "--log-file" && i+1 < len(args) {
			logPath = args[i+1]
		}
	}
	if report := os.Getenv("CUDASANITIZER_FIXTURE_REPORT"); report != "" &&
		logPath != "" && os.Getenv("CUDASANITIZER_FIXTURE_NO_LOG") == "" {
		if err := os.WriteFile(logPath, []byte(report), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "fixture:", err)
			os.Exit(120)
		}
	}
	if s := os.Getenv("CUDASANITIZER_FIXTURE_STDOUT"); s != "" {
		fmt.Fprintln(os.Stdout, s)
	}
	if s := os.Getenv("CUDASANITIZER_FIXTURE_STDERR"); s != "" {
		fmt.Fprintln(os.Stderr, s)
	}
	if os.Getenv("CUDASANITIZER_FIXTURE_EXEC_APP") != "" {
		// Behave like the real tool: run the application that follows the
		// options and pass its exit status back. Every option this package
		// emits is a "--name value" pair, so the application is the first
		// argument that does not follow one.
		i := 0
		for i+1 < len(args) && strings.HasPrefix(args[i], "--") {
			i += 2
		}
		if i >= len(args) {
			fmt.Fprintln(os.Stderr, "fixture: no application in", args)
			os.Exit(121)
		}
		app := exec.Command(args[i], args[i+1:]...)
		app.Stdout = os.Stdout
		app.Stderr = os.Stderr
		app.Stdin = os.Stdin
		err := app.Run()
		var exit *exec.ExitError
		switch {
		case err == nil:
			os.Exit(0)
		case errors.As(err, &exit):
			os.Exit(exit.ExitCode())
		default:
			fmt.Fprintln(os.Stderr, "fixture:", err)
			os.Exit(122)
		}
	}
	if os.Getenv("CUDASANITIZER_FIXTURE_SPAWN_SLEEPER") != "" {
		c := exec.Command(os.Args[0])
		c.Env = append(os.Environ(), "CUDASANITIZER_FIXTURE_SLEEPER=1")
		c.Stdout = os.Stdout
		c.Stderr = os.Stderr
        if os.Getenv("CUDASANITIZER_FIXTURE_CHILD_NO_PIPES") != "" {
            c.Stdout, c.Stderr = nil, nil
        }
		if err := c.Start(); err != nil {
			fmt.Fprintln(os.Stderr, "fixture:", err)
			os.Exit(120)
		}

        if path := os.Getenv("CUDASANITIZER_FIXTURE_CHILD_PID"); path != "" {
            if err := os.WriteFile(path, []byte(strconv.Itoa(c.Process.Pid)), 0o600); err != nil {
                fmt.Fprintln(os.Stderr,"fixture:",err)
                os.Exit(120)
            }
        }
	}
	if d := os.Getenv("CUDASANITIZER_FIXTURE_SLEEP"); d != "" {
		if dur, err := time.ParseDuration(d); err == nil {
			time.Sleep(dur)
		}
	}
	code := 0
	if c := os.Getenv("CUDASANITIZER_FIXTURE_EXIT"); c != "" {
		code, _ = strconv.Atoi(c)
	}
	os.Exit(code)
}
`

var (
	fixtureOnce sync.Once
	fixturePath string
	fixtureErr  error
	fixtureDir  string
)

// fixtureTool builds the fixture executable once per test binary and returns
// its path.
func fixtureTool(t *testing.T) string {
	t.Helper()
	fixtureOnce.Do(func() {
		dir, err := os.MkdirTemp("", "cudasanitizer-fixture-")
		if err != nil {
			fixtureErr = err
			return
		}
		fixtureDir = dir
		write := func(name, body string) {
			if fixtureErr == nil {
				fixtureErr = os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644)
			}
		}
		write("go.mod", "module cudasanitizerfixture\n\ngo 1.24\n")
		write("main.go", fixtureSource)
		if fixtureErr != nil {
			return
		}
		bin := filepath.Join(dir, "fixture")
		cmd := exec.Command("go", "build", "-o", bin, ".")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod", "GOPROXY=off")
		if out, err := cmd.CombinedOutput(); err != nil {
			fixtureErr = &buildError{err: err, out: string(out)}
			return
		}
		fixturePath = bin
	})
	if fixtureErr != nil {
		t.Fatalf("building the compute-sanitizer fixture: %v", fixtureErr)
	}
	return fixturePath
}

type buildError struct {
	err error
	out string
}

func (e *buildError) Error() string { return e.err.Error() + ": " + e.out }

func TestMain(m *testing.M) {
	if os.Getenv(selfTestEnv) != "" {
		// This process stands in for a package that runs its own tests
		// under compute-sanitizer. TestSelfLaunchTerminates starts it.
		os.Exit(Main(m, Config{
			Tool:          Memcheck,
			MissingTool:   MissingToolFail,
			Path:          os.Getenv(selfToolEnv),
			RequireOutput: []string{selfTestMark},
			Timeout:       time.Minute,
		}))
	}
	code := m.Run()
	if fixtureDir != "" {
		os.RemoveAll(fixtureDir)
	}
	os.Exit(code)
}

// fixtureEnv builds an environment for a fixture run from key=value pairs,
// starting from the current environment.
func fixtureEnv(pairs ...string) []string {
	return append(os.Environ(), pairs...)
}

// Reports the fixture writes, in compute-sanitizer's line format.
const (
	cleanReport = `========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
`

	errorReport = `========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x70 in vectorAdd(int *, int *, int *, int)
=========     by thread (255,0,0) in block (3,0,0)
=========     Address 0x7f8a00000400 is out of bounds
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure"
=========     Saved host backtrace up to driver entry point at error
=========
========= ERROR SUMMARY: 2 errors
`

	truncatedReport = `========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 8 bytes
=========     at 0x30 in kernel(void)
`
)
