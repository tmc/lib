package cudasanitizer

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Outcome is what a run turned out to be.
//
// compute-sanitizer reports these four situations through a single exit status,
// which conflates them. They call for four different reactions, so this package
// keeps them apart.
type Outcome int

// The possible outcomes of a run. The zero value is [OutcomeToolFailure], so a
// Result that was never filled in does not read as a pass.
const (
	// OutcomeToolFailure means no verdict exists: compute-sanitizer could
	// not be found or started, the run was cancelled or timed out, or the
	// tool produced no complete report. It is never a pass, and in particular a
	// timeout is this outcome rather than a slow success.
	OutcomeToolFailure Outcome = iota

	// OutcomeClean means the run completed, the tool reported zero errors,
	// and any evidence the caller required was present.
	OutcomeClean

	// OutcomeSanitizerErrors means the tool reported errors. This is the
	// outcome even when the process exited 0, which is what happens when
	// compute-sanitizer's --error-exitcode is left at its default.
	OutcomeSanitizerErrors

	// OutcomeApplicationFailed means the tool completed and reported no
	// errors, but the application exited nonzero or did not produce the
	// output the caller required as evidence that the intended work ran.
	OutcomeApplicationFailed
)

// String returns a short name for the outcome.
func (o Outcome) String() string {
	switch o {
	case OutcomeClean:
		return "clean"
	case OutcomeSanitizerErrors:
		return "sanitizer-errors"
	case OutcomeApplicationFailed:
		return "application-failed"
	case OutcomeToolFailure:
		return "tool-failure"
	}
	return fmt.Sprintf("Outcome(%d)", int(o))
}

// Sentinel errors reported through [Result.Err] for [OutcomeToolFailure].
var (
	// ErrToolNotFound means the compute-sanitizer executable was not found.
	// [Config.MissingTool] decides whether that is a skip or a failure.
	ErrToolNotFound = errors.New("compute-sanitizer not found")

	// ErrTimeout means the run exceeded [Config.Timeout] or the caller's
	// context deadline. The runner kills its process group on Unix, its
	// process job on Windows, and the direct child on other platforms.
	ErrTimeout = errors.New("run timed out")

	// ErrNoVerdict means no complete report was available: the tool emitted
	// no summary, or its fallback output was truncated. A run that ends
	// this way is classified as a tool failure no matter what the exit
	// status was, because no verdict was reached.
	ErrNoVerdict = errors.New("compute-sanitizer produced no complete report")
)

// Result describes one run. [Run] always returns a non-nil Result when the
// Config was valid, including for runs that never started.
type Result struct {
	// Config is the configuration the run used, retained as evidence of
	// what was actually asked for.
	Config Config

	// Outcome classifies the run. Read this rather than ExitCode.
	Outcome Outcome

	// Err carries the reason for [OutcomeToolFailure]: [ErrToolNotFound],
	// [ErrTimeout], [ErrNoVerdict], or an error from starting or waiting on
	// the process. It is nil for the other outcomes.
	Err error

	// Args is the argument vector that was executed, starting with the
	// compute-sanitizer executable.
	Args []string

	// ExitCode is the process exit status, or -1 if the process never ran
	// or was killed by a signal. It is retained for diagnosis; it is not
	// the verdict.
	ExitCode int

	// ErrorExitCode is the --error-exitcode value that was passed. It is
	// never 0.
	ErrorExitCode int

	// ExitCodeCollision reports that the application exited with the same
	// status as ErrorExitCode while the tool's report showed no errors, so
	// the exit status alone would have been read as sanitizer errors.
	ExitCodeCollision bool

	// Log is the parsed tool report. It is never nil.
	Log *Log

	// LogPath is where the tool's report was left on disk, set only when
	// [Config.LogDir] asked for it to be kept.
	LogPath string

	// Stdout and Stderr are the retained copies of the application's
	// output, truncated to [Config.MaxOutputBytes] per stream.
	Stdout, Stderr string

	// StdoutTruncated and StderrTruncated report whether output was
	// dropped from the retained copies.
	StdoutTruncated, StderrTruncated bool

	// MissingEvidence lists [Config.RequireOutput] entries that did not
	// appear in the application's output, and ForbiddenEvidence lists
	// [Config.ForbidOutput] entries that did. Either makes the run
	// [OutcomeApplicationFailed] rather than [OutcomeClean].
	MissingEvidence   []string
	ForbiddenEvidence []string

	// Duration is how long the run took.
	Duration time.Duration
}

// Run executes prog under compute-sanitizer and classifies the result.
//
// The returned error is non-nil only for a Config that cannot be used; in that
// case the Result is nil. Everything that can go wrong once the configuration
// is valid — a missing tool, a failure to start, a timeout, an application
// crash, reported sanitizer errors — is reported through the Result, so callers
// switch on [Result.Outcome] rather than on an error.
//
// Run bounds the subprocess with ctx and, if set, [Config.Timeout], whichever
// expires first. On Unix, the run owns a process group that is killed on
// expiry and cleaned up after the tool exits, including descendants whose
// parent already exited. Windows uses a Job Object to own and terminate
// the process tree. Other platforms currently kill only the direct child.
// Either expiry produces [OutcomeToolFailure] with [ErrTimeout]; a timed-out
// run is never reported as a pass.
func Run(ctx context.Context, cfg Config, prog string, args ...string) (*Result, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if prog == "" {
		return nil, fmt.Errorf("%w: empty program path", ErrInvalidConfig)
	}

	res := &Result{
		Config:        cfg,
		Outcome:       OutcomeToolFailure,
		ExitCode:      -1,
		ErrorExitCode: cfg.errorExitCode(),
		Log:           &Log{},
	}

	toolPath, err := LookTool(cfg.Path)
	if err != nil {
		res.Err = err
		return res, nil
	}

	logPath, keep, err := logFile(cfg.LogDir)
	if err != nil {
		res.Err = fmt.Errorf("creating sanitizer log file: %w", err)
		return res, nil
	}
	if keep {
		res.LogPath = logPath
	} else {
		defer os.Remove(logPath)
	}

	toolArgs, err := cfg.Args(logPath, prog, args...)
	if err != nil {
		return nil, err
	}
	res.Args = append([]string{toolPath}, toolArgs...)

	if cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()
	}

	stdout := &capBuffer{limit: cfg.maxOutputBytes(), tee: cfg.Stdout, evidence: newOutputEvidence(cfg)}
	stderr := &capBuffer{limit: cfg.maxOutputBytes(), tee: cfg.Stderr, evidence: newOutputEvidence(cfg)}

	cmd := exec.Command(toolPath, toolArgs...)
	cmd.Dir = cfg.Dir
	cmd.Env = cfg.Env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = cfg.killDelay()

	start := time.Now()
	runErr := runContext(ctx, cmd)
	res.Duration = time.Since(start)

	res.Stdout, res.StdoutTruncated = stdout.result()
	res.Stderr, res.StderrTruncated = stderr.result()
	logText, _ := os.ReadFile(logPath)
	res.Log = ParseLog(string(logText))
	incompleteFallback := !res.Log.SummaryFound && (res.StdoutTruncated || res.StderrTruncated)
	if !res.Log.SummaryFound && !incompleteFallback {
		// The tool may have failed before writing the log, or written its
		// report to the inherited streams instead. Parse what was captured
		// so a report that did arrive is not thrown away. Truncated output
		// cannot establish a verdict: an early zero summary may precede
		// errors in the omitted part of a later report.
		if alt := ParseLog(res.Stderr + "\n" + res.Stdout); alt.SummaryFound {
			res.Log = alt
		}
	}

	if cmd.ProcessState != nil {
		res.ExitCode = cmd.ProcessState.ExitCode()
	}

	res.checkEvidence(stdout, stderr)
	res.classify(ctx, runErr)
	if incompleteFallback && errors.Is(res.Err, ErrNoVerdict) {
		res.Err = fmt.Errorf("%w: fallback output was truncated", ErrNoVerdict)
	}
	return res, nil
}

type processCleanup interface {
	kill() error
	close() error
}

// runContext bounds cmd with ctx and cleans up its process group when the
// run ends. WaitDelay also bounds pipes kept open after the leader exits.
func runContext(ctx context.Context, cmd *exec.Cmd) error {
	return runContextWithStart(ctx, cmd, func(cmd *exec.Cmd) (processCleanup, error) {
		return startGroup(cmd)
	})
}

// cleanupError distinguishes teardown failure from the application's exit status.
// A joined ExitError must not hide failure to release the process group.
type cleanupError struct{ err error }

func (e *cleanupError) Error() string { return "clean up process group: " + e.err.Error() }
func (e *cleanupError) Unwrap() error { return e.err }

func runContextWithStart(ctx context.Context, cmd *exec.Cmd, start func(*exec.Cmd) (processCleanup, error)) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	group, err := start(cmd)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := group.close(); closeErr != nil {
			err = errors.Join(err, &cleanupError{closeErr})
		}
	}()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		// Kill the whole group: compute-sanitizer runs the application as
		// a separate process, so killing the direct child alone can leave
		// the instrumented program running and holding the pipes open.
		killErr := group.kill()
		select {
		case <-done:
		case <-time.After(cmd.WaitDelay):
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
			}
			<-done
		}
		if killErr != nil {
			return errors.Join(ctx.Err(), &cleanupError{killErr})
		}
		return ctx.Err()
	}
}

// classify applies the outcome rules. The ordering matters: a report of errors
// outranks the exit status, and a run with no report outranks everything,
// because it licenses no conclusion at all.
func (r *Result) classify(ctx context.Context, runErr error) {
	switch {
	case errors.Is(runErr, context.DeadlineExceeded), errors.Is(ctx.Err(), context.DeadlineExceeded):
		r.Outcome = OutcomeToolFailure
		r.Err = fmt.Errorf("%w after %s: %w", ErrTimeout, r.Duration.Round(time.Millisecond), errors.Join(context.DeadlineExceeded, runErr))
		return
	case errors.Is(runErr, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
		r.Outcome = OutcomeToolFailure
		r.Err = fmt.Errorf("run cancelled after %s: %w", r.Duration.Round(time.Millisecond), errors.Join(context.Canceled, runErr))
		return
	}
	var cleanupErr *cleanupError
	if errors.As(runErr, &cleanupErr) {
		r.Outcome = OutcomeToolFailure
		r.Err = runErr
		return
	}
	var exitErr *exec.ExitError
	if runErr != nil && !errors.As(runErr, &exitErr) {
		r.Outcome = OutcomeToolFailure
		r.Err = fmt.Errorf("running %s: %w", DefaultToolName, runErr)
		return
	}

	if !r.Log.SummaryFound {
		r.Outcome = OutcomeToolFailure
		r.Err = fmt.Errorf("%w (exit status %d); the run reached no verdict, so it is not a pass",
			ErrNoVerdict, r.ExitCode)
		return
	}
	if r.Log.Errors > 0 || r.Log.RaceErrors > 0 {
		r.Outcome = OutcomeSanitizerErrors
		return
	}

	switch {
	case r.ExitCode == r.ErrorExitCode:
		// The report says zero errors, so this is the application's own
		// status colliding with the code reserved for the tool.
		r.ExitCodeCollision = true
		r.Outcome = OutcomeApplicationFailed
	case r.ExitCode != 0:
		r.Outcome = OutcomeApplicationFailed
	case len(r.MissingEvidence) > 0 || len(r.ForbiddenEvidence) > 0:
		r.Outcome = OutcomeApplicationFailed
	default:
		r.Outcome = OutcomeClean
	}
}

// Summary returns a one-paragraph account of the run, suitable for a test log.
// It never includes the full tool report; use [Result.Details] for that.
func (r *Result) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "compute-sanitizer --tool %s: %s", r.Config.Tool, r.Outcome)
	if r.Err != nil {
		fmt.Fprintf(&b, " (%v)", r.Err)
	}
	fmt.Fprintf(&b, "; exit status %d, --error-exitcode %d, %s",
		r.ExitCode, r.ErrorExitCode, r.Duration.Round(time.Millisecond))
	if r.Log.SummaryFound {
		fmt.Fprintf(&b, "; report: %d errors", r.Log.Errors)
		if r.Log.RaceErrors > 0 {
			fmt.Fprintf(&b, ", racecheck: %d errors", r.Log.RaceErrors)
		}
		if r.Log.Hazards > 0 {
			fmt.Fprintf(&b, ", %d hazards", r.Log.Hazards)
		}
		if r.Log.LeakedBytes > 0 {
			fmt.Fprintf(&b, ", %d bytes leaked in %d allocations", r.Log.LeakedBytes, r.Log.LeakedAllocations)
		}
	} else {
		b.WriteString("; report: none")
	}
	if r.ExitCodeCollision {
		fmt.Fprintf(&b, "; note: the application exited with %d, the status reserved for --error-exitcode",
			r.ErrorExitCode)
	}
	if len(r.MissingEvidence) > 0 {
		fmt.Fprintf(&b, "; missing required output %q", r.MissingEvidence)
	}
	if len(r.ForbiddenEvidence) > 0 {
		fmt.Fprintf(&b, "; forbidden output present %q", r.ForbiddenEvidence)
	}
	return b.String()
}

// Details returns the summary followed by the retained evidence: the tool's
// report and the application's output. A detected error the caller cannot read
// is close to useless, so failure paths print this rather than the summary
// alone.
func (r *Result) Details() string {
	var b strings.Builder
	b.WriteString(r.Summary())
	b.WriteString("\ncommand: ")
	b.WriteString(strings.Join(r.Args, " "))
	if r.LogPath != "" {
		b.WriteString("\nlog file: ")
		b.WriteString(r.LogPath)
	}
	section := func(name, body string, truncated bool) {
		if strings.TrimSpace(body) == "" {
			return
		}
		b.WriteString("\n--- ")
		b.WriteString(name)
		if truncated {
			b.WriteString(" (truncated)")
		}
		b.WriteString(" ---\n")
		b.WriteString(strings.TrimRight(body, "\n"))
	}
	section("compute-sanitizer report", r.Log.Text, false)
	section("application stdout", r.Stdout, r.StdoutTruncated)
	section("application stderr", r.Stderr, r.StderrTruncated)
	return b.String()
}

// toolkitEnvVars are the environment variables that conventionally name a CUDA
// toolkit root. They are consulted before the default prefixes.
var toolkitEnvVars = []string{"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"}

// toolkitRoots are the install prefixes searched when no environment variable
// names one. /usr/local/cuda is first because a multi-version install points it
// at the version the machine is configured for.
var toolkitRoots = []string{
	"/usr/local/cuda",
	"/usr/local/cuda-13",
	"/usr/local/cuda-12",
	"/opt/cuda",
	"/usr/lib/cuda",
}

// toolkitSubdirs are the directories inside a toolkit root that hold a
// compute-sanitizer executable, in preference order: bin holds a small
// launcher, and the compute-sanitizer directory holds the real binary beside
// the libraries it needs.
var toolkitSubdirs = []string{"bin", "compute-sanitizer"}

// ToolSearchPaths returns the executables [LookTool] tries before PATH, in
// order. The list is empty on GOOS values Compute Sanitizer does not ship for,
// leaving PATH as the only source.
//
// A toolkit install is preferred over PATH because a distribution package can
// leave an older compute-sanitizer earlier on PATH than the toolkit's own. Such
// a binary is not merely out of date: it does not know the current devices, and
// one that shipped against different sanitizer libraries fails to start at all
// rather than reporting that it is too old. Checking the toolkit first picks the
// build that matches the libraries beside it.
//
// A non-empty Config.Path, or a toolkit root in CUDA_HOME, CUDA_PATH or
// CUDA_ROOT, overrides the defaults.
func ToolSearchPaths() []string { return toolSearchPaths(runtime.GOOS, os.Getenv) }

// toolSearchPaths is ToolSearchPaths with the platform and environment
// injected, so the Linux ordering can be tested on any host.
func toolSearchPaths(goos string, getenv func(string) string) []string {
	if goos != "linux" {
		return nil
	}
	var roots []string
	for _, env := range toolkitEnvVars {
		if v := getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	roots = append(roots, toolkitRoots...)

	seen := make(map[string]bool, len(roots)*len(toolkitSubdirs))
	paths := make([]string, 0, len(roots)*len(toolkitSubdirs))
	for _, root := range roots {
		for _, sub := range toolkitSubdirs {
			p := filepath.Join(root, sub, DefaultToolName)
			if seen[p] {
				continue
			}
			seen[p] = true
			paths = append(paths, p)
		}
	}
	return paths
}

// LookTool returns the path to the compute-sanitizer executable. A non-empty
// path is resolved the way a shell would, including a bare name on PATH. An
// empty path tries the toolkit installs [ToolSearchPaths] names and then
// [DefaultToolName] on PATH. The error wraps [ErrToolNotFound].
func LookTool(path string) (string, error) {
	if path != "" {
		found, err := exec.LookPath(path)
		if err != nil {
			return "", fmt.Errorf("%w: %v", ErrToolNotFound, err)
		}
		return found, nil
	}
	for _, candidate := range ToolSearchPaths() {
		if isExecutableFile(candidate) {
			return candidate, nil
		}
	}
	found, err := exec.LookPath(DefaultToolName)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrToolNotFound, err)
	}
	return found, nil
}

// isExecutableFile reports whether path is a regular file with an execute bit
// set. A directory of the same name, or a file that cannot be executed, is not
// a candidate.
func isExecutableFile(path string) bool {
	fi, err := os.Stat(path)
	if err != nil || !fi.Mode().IsRegular() {
		return false
	}
	return fi.Mode().Perm()&0o111 != 0
}

// ToolVersion runs compute-sanitizer --version and returns the line naming the
// build, for recording which one produced a report. An empty path resolves the
// tool the way [LookTool] does.
//
// The first line of --version output is a product banner identical across every
// release, so it identifies nothing; the build is on the line that starts with
// "Version", as in "Version 2026.3.0.0 (build 38637409) (public-release)". When
// no line says that, the first non-empty line is returned instead.
func ToolVersion(ctx context.Context, path string) (string, error) {
	found, err := LookTool(path)
	if err != nil {
		return "", err
	}
	cmd := exec.Command(found, "--version")
	cmd.WaitDelay = time.Second
	output := &capBuffer{limit: 64 << 10}
	cmd.Stdout, cmd.Stderr = output, output
	err = runContext(ctx, cmd)
	if err != nil {
		return "", fmt.Errorf("%s --version: %w", found, err)
	}
	out, truncated := output.result()
	if truncated {
		return "", fmt.Errorf("%s --version: output exceeds 64 KiB", found)
	}
	if v := toolVersionLine(out); v != "" {
		return v, nil
	}
	return "", fmt.Errorf("%s --version: no output", found)
}

// toolVersionLine picks the build identifier out of --version output. It
// returns the line introducing the version, else the first non-empty line, else
// the empty string.
func toolVersionLine(out string) string {
	first := ""
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "Version") {
			return line
		}
		if first == "" {
			first = line
		}
	}
	return first
}

// logFile returns a path for the tool's report. When dir is empty the file is
// created in the system temporary directory and keep is false, meaning the
// caller removes it after reading.
func logFile(dir string) (path string, keep bool, err error) {
	keep = dir != ""
	if keep {
		if err := os.MkdirAll(dir, 0o777); err != nil {
			return "", false, err
		}
	}
	f, err := os.CreateTemp(dir, "compute-sanitizer-*.log")
	if err != nil {
		return "", false, err
	}
	name := f.Name()
	if err := f.Close(); err != nil {
		return "", false, err
	}
	if keep {
		name, err = filepath.Abs(name)
		if err != nil {
			return "", false, err
		}
	}
	return name, keep, nil
}

// capBuffer retains at most limit bytes of what is written to it while passing
// everything through to tee.
type capBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	limit     int
	truncated bool
	tee       io.Writer
	evidence  outputEvidence
}

func (b *capBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	b.evidence.write(p)
	if room := b.limit - b.buf.Len(); room > 0 {
		if len(p) <= room {
			b.buf.Write(p)
		} else {
			b.buf.Write(p[:room])
			b.truncated = true
		}
	} else if len(p) > 0 {
		b.truncated = true
	}
	b.mu.Unlock()
	if b.tee != nil {
		return b.tee.Write(p)
	}
	return len(p), nil
}

func (b *capBuffer) result() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String(), b.truncated
}
