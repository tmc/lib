package main

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestClassify(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
		want Outcome
	}{
		{"nil is ok", nil, OutcomeOK},
		{"deadline", fmt.Errorf("sync: %w", context.DeadlineExceeded), OutcomeTimeout},
		{"no device", driver.ErrNoDevice, OutcomeNoDevice},
		{"invalid ptx", driver.ErrInvalidPtx, OutcomeLoadError},
		{"unsupported ptx version", driver.ErrUnsupportedPtxVersion, OutcomeLoadError},
		{"no binary for gpu", driver.ErrNoBinaryForGpu, OutcomeLoadError},
		{"missing symbol", driver.ErrNotFound, OutcomeMissingEntry},
		{"out of resources", driver.ErrLaunchOutOfResources, OutcomeLaunchRejected},
		{"illegal address", driver.ErrIllegalAddress, OutcomeDeviceFault},
		{"misaligned", driver.ErrMisalignedAddress, OutcomeDeviceFault},
		{"device assert", driver.ErrAssert, OutcomeDeviceFault},
		{"launch failed", driver.ErrLaunchFailed, OutcomeDeviceFault},
		{"launch timeout", driver.ErrLaunchTimeout, OutcomeTimeout},
		{"wrapped", fmt.Errorf("launch: %w", driver.ErrIllegalAddress), OutcomeDeviceFault},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, recognized := classify(tt.err)
			if got != tt.want {
				t.Errorf("classify(%v) = %q, want %q", tt.err, got, tt.want)
			}
			if !recognized {
				t.Errorf("classify(%v) reported the error as unrecognized", tt.err)
			}
		})
	}
}

// An error the harness does not know must never be absorbed as a pass. This is
// the control: without it, adding a driver code to the switch could silently
// turn a failure into "ok".
func TestClassifyRefusesToGuess(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
	}{
		{"plain error", errors.New("something else")},
		{"unmapped driver code", driver.ErrKeyRotation},
		{"success wrapped as an error", driver.Success},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, recognized := classify(tt.err)
			if got != OutcomeHarnessError {
				t.Errorf("classify(%v) = %q, want %q", tt.err, got, OutcomeHarnessError)
			}
			if recognized {
				t.Error("an unmapped error was reported as recognized")
			}
		})
	}
}

// The outcomes must not collapse: a loop reads this field to decide what to
// change, so two different failures reaching the same exit code would still
// have to be distinguishable in the report.
func TestOutcomeExitCodesSeparateTheThreeKinds(t *testing.T) {
	if OutcomeOK.exitCode() != 0 {
		t.Error("ok must exit 0")
	}
	if OutcomeNoDevice.exitCode() == OutcomeOK.exitCode() {
		t.Error("no_device shares an exit code with ok: an absent GPU would read as a passing kernel")
	}
	for _, o := range []Outcome{OutcomeCompileError, OutcomeLoadError, OutcomeMissingEntry,
		OutcomeLaunchRejected, OutcomeDeviceFault, OutcomeTimeout, OutcomeWrongAnswer} {
		if o.exitCode() != 1 {
			t.Errorf("%s exits %d, want 1", o, o.exitCode())
		}
	}
	if OutcomeHarnessError.exitCode() == 1 {
		t.Error("harness_error shares an exit code with a failed candidate")
	}
}

func TestPoisoningOutcomes(t *testing.T) {
	poison := map[Outcome]bool{OutcomeDeviceFault: true, OutcomeTimeout: true}
	for _, o := range []Outcome{OutcomeOK, OutcomeCompileError, OutcomeLoadError,
		OutcomeMissingEntry, OutcomeLaunchRejected, OutcomeDeviceFault, OutcomeTimeout,
		OutcomeWrongAnswer, OutcomeNoDevice, OutcomeHarnessError} {
		if got := o.poisonsContext(); got != poison[o] {
			t.Errorf("%s.poisonsContext() = %v, want %v", o, got, poison[o])
		}
	}
}
