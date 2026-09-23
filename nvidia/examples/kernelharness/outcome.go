package main

import (
	"context"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
)

// Outcome names what happened to a candidate. A refinement loop reads this
// field: the distinctions here are the ones that change what it should do next.
type Outcome string

const (
	// OutcomeOK means the candidate compiled, loaded, ran, and matched the
	// reference on every trial.
	OutcomeOK Outcome = "ok"
	// OutcomeCompileError means NVRTC rejected the source. The compiler log is
	// the evidence and is reported verbatim.
	OutcomeCompileError Outcome = "compile_error"
	// OutcomeLoadError means the driver rejected the compiled image: a PTX JIT
	// failure, an unsupported PTX version, or no binary for this device. The
	// source compiled; the artifact is unusable here.
	OutcomeLoadError Outcome = "load_error"
	// OutcomeMissingEntry means the module loaded but does not define the entry
	// symbol the task requires. Almost always a missing extern "C" or a typo,
	// which is a different edit from a wrong answer.
	OutcomeMissingEntry Outcome = "missing_entry"
	// OutcomeLaunchRejected means the driver refused the launch before running
	// it: too many registers for the block size, an invalid configuration. No
	// device work happened, so the context is still usable.
	OutcomeLaunchRejected Outcome = "launch_rejected"
	// OutcomeDeviceFault means the kernel ran and faulted: an illegal address,
	// a misaligned access, a device-side assert. The CUDA context is poisoned;
	// see Report.ContextPoisoned.
	OutcomeDeviceFault Outcome = "device_fault"
	// OutcomeTimeout means the kernel did not complete within the deadline.
	// Nothing here kills a running kernel, so the context is poisoned too.
	OutcomeTimeout Outcome = "timeout"
	// OutcomeWrongAnswer means the kernel ran to completion and its output is
	// outside tolerance. The report carries the worst element.
	OutcomeWrongAnswer Outcome = "wrong_answer"
	// OutcomeNoDevice means no CUDA device was available. It is not a verdict
	// on the candidate and must never be read as one.
	OutcomeNoDevice Outcome = "no_device"
	// OutcomeHarnessError means the harness itself failed, or hit a driver
	// error it does not classify. It is not a verdict on the candidate either.
	OutcomeHarnessError Outcome = "harness_error"
)

// poisoning lists the outcomes after which the CUDA context is unusable. The
// driver documents these codes as leaving the process in an inconsistent state:
// every subsequent CUDA call returns the same error. A harness that kept going
// would report the first fault again for every later candidate, so the CLI
// stops and a loop must re-exec it per candidate.
func (o Outcome) poisonsContext() bool {
	return o == OutcomeDeviceFault || o == OutcomeTimeout
}

// classify maps an error from a CUDA call to an outcome. The second result
// reports whether the error was recognized: an unrecognized driver code becomes
// OutcomeHarnessError, never OutcomeOK, so a code added to a future driver
// cannot be absorbed as a pass.
func classify(err error) (Outcome, bool) {
	if err == nil {
		return OutcomeOK, true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return OutcomeTimeout, true
	}
	var res driver.Result
	if !errors.As(err, &res) {
		return OutcomeHarnessError, false
	}
	switch res {
	case driver.Success:
		// A Success wrapped in an error is a harness bug, not a pass.
		return OutcomeHarnessError, false

	case driver.ErrNoDevice, driver.ErrInvalidDevice, driver.ErrDeviceUnavailable,
		driver.ErrNotInitialized, driver.ErrStubLibrary:
		return OutcomeNoDevice, true

	case driver.ErrInvalidPtx, driver.ErrUnsupportedPtxVersion, driver.ErrNoBinaryForGpu,
		driver.ErrInvalidImage, driver.ErrInvalidSource, driver.ErrJitCompilerNotFound,
		driver.ErrJitCompilationDisabled, driver.ErrUnsupportedDevsideSync:
		return OutcomeLoadError, true

	case driver.ErrNotFound:
		return OutcomeMissingEntry, true

	case driver.ErrLaunchOutOfResources, driver.ErrInvalidValue, driver.ErrInvalidHandle,
		driver.ErrCooperativeLaunchTooLarge, driver.ErrInvalidClusterSize,
		driver.ErrLaunchIncompatibleTexturing, driver.ErrFunctionNotLoaded:
		return OutcomeLaunchRejected, true

	case driver.ErrLaunchTimeout:
		return OutcomeTimeout, true

	case driver.ErrIllegalAddress, driver.ErrMisalignedAddress, driver.ErrIllegalInstruction,
		driver.ErrInvalidAddressSpace, driver.ErrInvalidPc, driver.ErrHardwareStackError,
		driver.ErrLaunchFailed, driver.ErrAssert, driver.ErrEccUncorrectable,
		driver.ErrNvlinkUncorrectable, driver.ErrContained, driver.ErrTensorMemoryLeak,
		driver.ErrExternalDevice:
		return OutcomeDeviceFault, true
	}
	return OutcomeHarnessError, false
}

// exitCode maps an outcome to a process exit status. A caller that reads only
// the status still cannot confuse "no device" with "the candidate is correct".
func (o Outcome) exitCode() int {
	switch o {
	case OutcomeOK:
		return 0
	case OutcomeNoDevice:
		return 3
	case OutcomeHarnessError:
		return 4
	}
	return 1
}
