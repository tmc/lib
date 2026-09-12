// Package cudasanitizer runs a program under the NVIDIA compute-sanitizer
// command-line tool and classifies what happened.
//
// # What this package is not
//
// It is not a binding for the CUDA sanitizer public API (the callback and
// patching interface in sanitizer_*.h). Nothing here loads libsanitizer or
// subscribes to device events; compute-sanitizer is driven as a subprocess.
//
// It is also not Go's race detector. The -race flag instruments Go code on the
// CPU. [Racecheck] instruments CUDA shared memory accesses on the
// GPU. Enabling one says nothing about the other, and a package can need both.
//
// # The exit-code trap
//
// compute-sanitizer's --error-exitcode option defaults to 0. With that default
// the tool exits 0 after reporting memory errors, races, or leaks, as long as
// the application under test exited 0. A runner that consults only the exit
// status therefore reports a clean pass for a program full of memory errors.
//
// This package always passes an explicit nonzero --error-exitcode
// ([DefaultErrorExitCode] unless [Config.ErrorExitCode] is set) and never
// relies on the tool's default. It also parses the tool's report, so a run is
// classified from the report rather than the exit status alone.
//
// A single exit status conflates four situations that call for four different
// reactions. [Run] separates them as [Outcome] values:
//
//   - [OutcomeClean]: the run completed and the tool reported nothing.
//   - [OutcomeSanitizerErrors]: the tool reported errors.
//   - [OutcomeApplicationFailed]: the application failed for its own reasons,
//     or did not produce the evidence the caller required.
//   - [OutcomeToolFailure]: the tool did not run, did not finish, or produced
//     no verdict. This includes a timeout and a missing compute-sanitizer
//     binary. It is never a pass.
//
// # Proving the work ran
//
// --require-cuda-init defaults to yes, so compute-sanitizer reports an error if
// the application never reaches a CUDA API call. That is a weak guarantee: a
// program that initializes CUDA and launches nothing satisfies it and reports
// zero errors. Zero errors over zero kernels is not a clean validation.
//
// Callers therefore declare what the run must show. [Config.RequireOutput]
// lists substrings that must appear in the application's own output and
// [Config.ForbidOutput] lists substrings that must not. A run missing required
// evidence is [OutcomeApplicationFailed], not [OutcomeClean], and the
// unsatisfied entries are retained on the [Result].
//
// # Missing tooling
//
// [Config.MissingTool] has no default. A zero [MissingToolPolicy] is a
// configuration error, because the one behavior that must never happen by
// accident is a green run on a machine where compute-sanitizer was never
// installed. Callers choose [MissingToolFail] (validation is required) or
// [MissingToolSkip] (validation is optional, and its absence is announced).
//
// # Running a test binary under the tool
//
// [Main] re-executes the current test binary under compute-sanitizer:
//
//	func TestMain(m *testing.M) {
//		os.Exit(cudasanitizer.Main(m, cudasanitizer.Config{
//			Tool:          cudasanitizer.Memcheck,
//			MissingTool:   cudasanitizer.MissingToolSkip,
//			RequireOutput: []string{"vector add ok"},
//		}))
//	}
//
// The obvious implementation of that pattern re-executes the test binary,
// re-enters TestMain, and forks forever. [Main] marks the child with an
// explicit -cudasanitizer.child flag prepended to the child's arguments, and
// additionally sets a CUDASANITIZER_ACTIVE environment variable so that a
// parent that somehow finds itself already running under the tool refuses to
// relaunch instead of recursing. See [Main] for the details.
//
// [Main] passes the parent's os.Args through to the child unchanged, so -run,
// -v, -count, -timeout and -args reach the tests that the caller asked for.
// Note that -test.timeout is measured in the child, where every kernel launch
// is instrumented; a suite that fits comfortably in the default timeout without
// the sanitizer may not fit under it.
//
// # What the tests in this package establish
//
// The tests here drive fixture executables: small Go programs, built into a
// temporary directory, that impersonate compute-sanitizer by writing a chosen
// report to the --log-file path and exiting with a chosen status. They pin the
// classification rules, including the case of a report full of errors delivered
// with exit status 0.
//
// That is host orchestration evidence and nothing more. It establishes that
// this package's argument construction, subprocess handling, log parsing, and
// outcome classification behave as documented. It establishes nothing about
// CUDA, nothing about the real tool's output format on any particular release,
// and nothing about detecting a real race on a real GPU. The tests that would
// establish those things are in cuda_test.go and require a GPU, a CUDA
// installation, and the real binary.
package cudasanitizer
