package cudasanitizer

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// DefaultToolName is the name of the compute-sanitizer executable sought when
// [Config.Path] is empty, in the toolkit installs [ToolSearchPaths] names and
// then on PATH.
const DefaultToolName = "compute-sanitizer"

// DefaultErrorExitCode is the value passed as --error-exitcode when
// [Config.ErrorExitCode] is zero.
//
// compute-sanitizer defaults --error-exitcode to 0, which makes a run with
// detected errors indistinguishable from a clean run by exit status alone.
// This package never uses that default. The value is deliberately far from the
// codes a Go test binary produces (0 pass, 1 fail, 2 flag or panic exit) so
// that a collision with the application's own status is unlikely; [Run]
// reports the remaining collision explicitly through
// [Result.ExitCodeCollision].
const DefaultErrorExitCode = 66

// DefaultMaxOutputBytes bounds how much of the application's stdout and stderr
// each [Result] retains when [Config.MaxOutputBytes] is zero.
const DefaultMaxOutputBytes = 1 << 20

// DefaultKillDelay bounds waiting for inherited output pipes to close after
// the process exits, used when [Config.KillDelay] is zero.
const DefaultKillDelay = 5 * time.Second

// Tool selects a compute-sanitizer checking tool. The values are the tool's own
// --tool argument spellings; this package does not invent an abstraction over
// them.
type Tool string

// The checking tools compute-sanitizer provides. Exactly one runs per
// invocation; checking a program with several means several runs.
const (
	// Memcheck detects out-of-bounds and misaligned device memory accesses,
	// invalid API usage, and (with [Config.LeakCheck]) leaked allocations.
	Memcheck Tool = "memcheck"
	// Racecheck detects shared memory data hazards between
	// threads of a kernel.
	Racecheck Tool = "racecheck"
	// Initcheck detects device reads of uninitialized global memory.
	Initcheck Tool = "initcheck"
	// Synccheck detects invalid usage of synchronization primitives.
	Synccheck Tool = "synccheck"
)

func (t Tool) valid() bool {
	switch t {
	case Memcheck, Racecheck, Initcheck, Synccheck:
		return true
	}
	return false
}

// Toggle is a three-state setting for the tool's yes/no options. The zero
// value leaves compute-sanitizer's own default in place and emits no flag.
type Toggle int

// Toggle states.
const (
	ToggleDefault Toggle = iota // emit no flag
	ToggleYes                   // emit yes
	ToggleNo                    // emit no
)

func (t Toggle) arg() (string, bool) {
	switch t {
	case ToggleYes:
		return "yes", true
	case ToggleNo:
		return "no", true
	}
	return "", false
}

// MissingToolPolicy says what a caller wants to happen when compute-sanitizer
// is not installed.
//
// There is deliberately no default. Quietly reporting success for validation
// that never ran is the failure this package exists to prevent, so the choice
// between required validation and optional validation is made explicitly, in
// the caller's source, where a reader can see it.
type MissingToolPolicy int

// Missing-tool policies.
const (
	// MissingToolUnset is the zero value and is not a usable policy. [Run]
	// and [Main] reject a Config that leaves it unset.
	MissingToolUnset MissingToolPolicy = iota

	// MissingToolFail treats an absent compute-sanitizer as a failure:
	// [Result.Report] calls Fatal and [Main] returns a nonzero exit code.
	// Use it where sanitizer validation is a requirement of the run.
	MissingToolFail

	// MissingToolSkip treats an absent compute-sanitizer as a skip:
	// [Result.Report] calls Skip and [Main] runs the tests uninstrumented
	// after announcing on stderr that no sanitizer validation happened.
	// The outcome is still [OutcomeToolFailure]; the policy changes the
	// reaction, not the classification.
	MissingToolSkip
)

func (p MissingToolPolicy) String() string {
	switch p {
	case MissingToolFail:
		return "fail"
	case MissingToolSkip:
		return "skip"
	}
	return "unset"
}

// Config describes one compute-sanitizer invocation.
//
// The zero Config is not usable: [Config.Tool] and [Config.MissingTool] must
// both be set. Every other field has a documented default.
//
// The typed option fields map one-to-one onto compute-sanitizer options and
// keep the tool's own flag names. Options this package does not model go in
// [Config.ExtraArgs].
type Config struct {
	// Tool selects the checking tool (--tool). Required.
	Tool Tool

	// MissingTool says how an absent compute-sanitizer is handled.
	// Required; see [MissingToolPolicy].
	MissingTool MissingToolPolicy

	// Path is the compute-sanitizer executable. Empty means find
	// [DefaultToolName] the way [LookTool] does: the CUDA toolkit installs
	// [ToolSearchPaths] names, then PATH.
	Path string

	// ErrorExitCode is passed as --error-exitcode. Zero selects
	// [DefaultErrorExitCode], so there is no way to configure the tool's
	// own default of 0, which hides detected errors behind a successful
	// application.
	ErrorExitCode int

	// RequireCUDAInit maps to --require-cuda-init, whose tool default is
	// yes. Note that satisfying it proves only that some CUDA API call was
	// reached, not that the intended kernels ran; use RequireOutput for
	// that.
	RequireCUDAInit Toggle

	// LeakCheck maps to --leak-check: ToggleYes emits "full", ToggleNo
	// emits "no". Leaks reported under "full" count toward the tool's
	// error summary. Meaningful for [Memcheck].
	LeakCheck Toggle

	// TrackUnusedMemory maps to --track-unused-memory, which the tool spells
	// as a bare flag: ToggleYes emits it and ToggleNo emits nothing, leaving
	// the tool's default. Meaningful for [Initcheck].
	TrackUnusedMemory Toggle

	// CheckAPIMemoryAccess maps to --check-api-memory-access. Meaningful
	// for [Memcheck].
	CheckAPIMemoryAccess Toggle

	// Kill maps to --kill, which terminates the application on a
	// communication error. The tool spells it as a bare flag: ToggleYes emits
	// it and ToggleNo emits nothing, leaving the tool's default.
	Kill Toggle

	// NVTX maps to --nvtx, enabling NVTX range support in reports.
	NVTX Toggle

	// RacecheckReport maps to --racecheck-report: "analysis", "hazard" or
	// "all". Meaningful for [Racecheck].
	RacecheckReport string

	// TargetProcesses maps to --target-processes: "application-only" or
	// "all". Use "all" when the program under test spawns the processes
	// that actually touch the GPU.
	TargetProcesses string

	// Demangle maps to --demangle: "full", "simple" or "no".
	Demangle string

	// DestroyOnDeviceError maps to --destroy-on-device-error: "context" or
	// "kernel".
	DestroyOnDeviceError string

	// PrintLimit maps to --print-limit, the maximum number of errors
	// printed per kernel launch. Zero emits no flag.
	PrintLimit int

	// LaunchTimeout maps to --launch-timeout, the tool's own wait for the
	// application to launch. It is rounded down to whole seconds. Zero
	// emits no flag. It bounds the launch only; use [Config.Timeout] to
	// bound the whole run.
	LaunchTimeout time.Duration

	// ExtraArgs are passed to compute-sanitizer ahead of the application.
	// They may not repeat an option this Config already manages; [Run]
	// rejects that rather than emitting the option twice.
	ExtraArgs []string

	// Dir is the working directory for the run. Empty means the current
	// process's directory.
	Dir string

	// Env is the child environment. Nil means the current process's
	// environment. Entries are appended to it in [Main].
	Env []string

	// Stdout and Stderr, when non-nil, receive a copy of the application's
	// output as it is produced. A bounded copy is retained on the [Result]
	// regardless.
	Stdout, Stderr io.Writer

	// RequireOutput lists substrings that must appear in the application's
	// combined output for the run to count as evidence that the intended
	// work executed. A run missing any of them is
	// [OutcomeApplicationFailed]. Matching includes output beyond MaxOutputBytes.
	RequireOutput []string

	// ForbidOutput lists substrings whose presence means the run did not do
	// the intended work, such as a harness warning that nothing ran. A run
	// containing any of them is [OutcomeApplicationFailed], including matches
	// beyond MaxOutputBytes.
	ForbidOutput []string

	// Timeout bounds the whole run. Zero means the run is bounded only by
	// the context passed to [Run]. Whichever deadline comes first applies,
	// and either produces [OutcomeToolFailure], never a pass.
	Timeout time.Duration

	// KillDelay bounds waiting for output pipes after the process exits.
	// Cancellation terminates the owned processes immediately; this delay
	// does not postpone that signal. Zero means [DefaultKillDelay].
	KillDelay time.Duration

	// LogDir, when non-empty, is a directory in which the tool's log file
	// is kept after the run and named by [Result.LogPath]. When empty the
	// log is written to a temporary file, read into [Result.Log], and
	// removed.
	LogDir string

	// MaxOutputBytes bounds the retained copy of each of the application's
	// output streams. Zero means [DefaultMaxOutputBytes].
	MaxOutputBytes int
}

// ErrInvalidConfig is returned by [Run] for a Config that cannot be used.
var ErrInvalidConfig = errors.New("invalid config")

// managedOptions are the compute-sanitizer options this package emits itself.
// ExtraArgs may not repeat them.
var managedOptions = []string{
	"--tool",
	"--error-exitcode",
	"--log-file",
	"--require-cuda-init",
	"--leak-check",
	"--track-unused-memory",
	"--check-api-memory-access",
	"--kill",
	"--nvtx",
	"--racecheck-report",
	"--target-processes",
	"--demangle",
	"--destroy-on-device-error",
	"--print-limit",
	"--launch-timeout",
}

func (c Config) validate() error {
	if !c.Tool.valid() {
		return fmt.Errorf("%w: Tool %q is not one of %q, %q, %q, %q",
			ErrInvalidConfig, string(c.Tool), Memcheck, Racecheck, Initcheck, Synccheck)
	}
	switch c.MissingTool {
	case MissingToolFail, MissingToolSkip:
	default:
		return fmt.Errorf("%w: MissingTool must be MissingToolFail or MissingToolSkip; "+
			"there is no default because a missing sanitizer must never read as a pass", ErrInvalidConfig)
	}
	if c.ErrorExitCode < 0 || c.ErrorExitCode > 255 {
		return fmt.Errorf("%w: ErrorExitCode %d is outside 1..255 (0 selects DefaultErrorExitCode)",
			ErrInvalidConfig, c.ErrorExitCode)
	}
	if err := checkEnum("RacecheckReport", c.RacecheckReport, "analysis", "hazard", "all"); err != nil {
		return err
	}
	if err := checkEnum("TargetProcesses", c.TargetProcesses, "application-only", "all"); err != nil {
		return err
	}
	if err := checkEnum("Demangle", c.Demangle, "full", "simple", "no"); err != nil {
		return err
	}
	if err := checkEnum("DestroyOnDeviceError", c.DestroyOnDeviceError, "context", "kernel"); err != nil {
		return err
	}
	if c.PrintLimit < 0 {
		return fmt.Errorf("%w: PrintLimit %d is negative", ErrInvalidConfig, c.PrintLimit)
	}
	for _, arg := range c.ExtraArgs {
		name := arg
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		for _, managed := range managedOptions {
			if name == managed {
				return fmt.Errorf("%w: ExtraArgs sets %s, which this package manages", ErrInvalidConfig, managed)
			}
		}
	}
	return nil
}

func checkEnum(field, value string, allowed ...string) error {
	if value == "" {
		return nil
	}
	for _, a := range allowed {
		if value == a {
			return nil
		}
	}
	return fmt.Errorf("%w: %s %q is not one of %q", ErrInvalidConfig, field, value, allowed)
}

// errorExitCode reports the --error-exitcode value this Config uses. It is
// never 0.
func (c Config) errorExitCode() int {
	if c.ErrorExitCode == 0 {
		return DefaultErrorExitCode
	}
	return c.ErrorExitCode
}

func (c Config) maxOutputBytes() int {
	if c.MaxOutputBytes <= 0 {
		return DefaultMaxOutputBytes
	}
	return c.MaxOutputBytes
}

func (c Config) killDelay() time.Duration {
	if c.KillDelay <= 0 {
		return DefaultKillDelay
	}
	return c.KillDelay
}

// flags returns the options this Config emits, in order, with the argument each
// one carries. It is the single account of what the command line contains, so
// [Config.Args] and [Config.CheckOptions] cannot describe different runs.
//
// A flag with no value is one the tool spells without an argument. Passing one
// anyway is rejected outright ("extra yes detected on the input command line")
// rather than ignored, which is why arity is part of what CheckOptions checks.
func (c Config) flags(logFile string) []cudatool.Flag {
	var flags []cudatool.Flag
	add := func(name, value string) {
		flags = append(flags, cudatool.Flag{Name: name, Value: value})
	}
	// toggle emits an option the tool spells with a yes-or-no argument.
	toggle := func(name string, t Toggle, yes, no string) {
		v, ok := t.arg()
		if !ok {
			return
		}
		if v == "yes" {
			add(name, yes)
		} else {
			add(name, no)
		}
	}
	// bare emits an option the tool spells without an argument, so ToggleNo
	// selects the tool's own default by emitting nothing.
	bare := func(name string, t Toggle) {
		if t == ToggleYes {
			flags = append(flags, cudatool.Flag{Name: name})
		}
	}
	add("--tool", string(c.Tool))
	add("--error-exitcode", strconv.Itoa(c.errorExitCode()))
	if logFile != "" {
		add("--log-file", logFile)
	}
	toggle("--require-cuda-init", c.RequireCUDAInit, "yes", "no")
	toggle("--leak-check", c.LeakCheck, "full", "no")
	bare("--track-unused-memory", c.TrackUnusedMemory)
	toggle("--check-api-memory-access", c.CheckAPIMemoryAccess, "yes", "no")
	bare("--kill", c.Kill)
	toggle("--nvtx", c.NVTX, "yes", "no")
	if c.RacecheckReport != "" {
		add("--racecheck-report", c.RacecheckReport)
	}
	if c.TargetProcesses != "" {
		add("--target-processes", c.TargetProcesses)
	}
	if c.Demangle != "" {
		add("--demangle", c.Demangle)
	}
	if c.DestroyOnDeviceError != "" {
		add("--destroy-on-device-error", c.DestroyOnDeviceError)
	}
	if c.PrintLimit > 0 {
		add("--print-limit", strconv.Itoa(c.PrintLimit))
	}
	if secs := int(c.LaunchTimeout / time.Second); secs > 0 {
		add("--launch-timeout", strconv.Itoa(secs))
	}
	return flags
}

// Args returns the full compute-sanitizer argument vector for running prog with
// appArgs, writing the tool's report to logFile. The returned slice starts with
// the compute-sanitizer options, then prog, then appArgs, matching the tool's
// "compute-sanitizer [options] app [app args]" form. It does not include the
// executable itself.
//
// Args reports an error for a Config that [Run] would reject. It runs nothing:
// whether the installed compute-sanitizer accepts these options is
// [Config.CheckOptions]'s question, because answering it means executing the
// tool.
func (c Config) Args(logFile, prog string, appArgs ...string) ([]string, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if prog == "" {
		return nil, fmt.Errorf("%w: empty program path", ErrInvalidConfig)
	}
	args := cudatool.Args(c.flags(logFile)...)
	args = append(args, c.ExtraArgs...)
	args = append(args, prog)
	args = append(args, appArgs...)
	return args, nil
}

// CheckOptions reports whether the compute-sanitizer this Config resolves to
// accepts the options the Config would emit, by name and by arity.
//
// It is separate from [Config.Args] because it runs the tool: resolving the
// executable and asking it for its version is a subprocess, and Args is a pure
// function of the Config that a test can call without a toolkit installed. [Run]
// calls it before starting a run, so a mismatch is reported as a Go error naming
// the tool and its version rather than as a rejected command line with no
// report. Callers that build a command line themselves, from [Config.Args], can
// call it directly.
//
// The names in [Config.ExtraArgs] are checked too, but not their arity: how a
// caller spells an option this package does not model is the caller's business,
// while whether the installed tool has that option at all is knowable here.
//
// A Config whose options are all accepted returns nil, and so does one whose
// tool cannot be resolved or whose capabilities cannot be determined: an
// executable that reports no version and whose --help cannot be read supports a
// claim about nothing. [Run] reports a missing tool through
// [Result.Outcome] and [Config.MissingTool] instead.
func (c Config) CheckOptions(ctx context.Context) error {
	if err := c.validate(); err != nil {
		return err
	}
	caps, err := cudatool.Capabilities(ctx, DefaultToolName, c.Path)
	if err != nil {
		return nil
	}
	// The log path's value does not matter to the check, only that --log-file
	// is an option that takes one.
	return errors.Join(caps.Check(c.flags("compute-sanitizer.log")...), caps.CheckNames(c.ExtraArgs...))
}
