package cudaregress

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

var (
	// ErrIncidentNotAdmitted indicates an incident lacks valid evidence admission.
	ErrIncidentNotAdmitted = fmt.Errorf("cudaregress: incident not admitted by evidence contract: %w", cudaerrors.ErrUnsupported)

	// ErrArtifactMissing indicates a required source, binary, input, or reference artifact is not accessible.
	// This constitutes a reconstruction blocker.
	ErrArtifactMissing = errors.New("cudaregress: required artifact is missing on disk (reconstruction blocker)")

	// ErrArtifactCorrupted indicates an artifact's content hash does not match its declared digest.
	// This constitutes a reconstruction blocker.
	ErrArtifactCorrupted = errors.New("cudaregress: artifact content digest mismatch (reconstruction blocker)")

	// ErrEnvironmentDrift indicates execution environment differs from the declared plan.
	ErrEnvironmentDrift = errors.New("cudaregress: environment drift detected")

	// ErrUnsupportedTimingChannel indicates an unmeasured or unsupported timing channel was requested.
	ErrUnsupportedTimingChannel = fmt.Errorf("cudaregress: requested timing channel is unsupported: %w", cudaerrors.ErrUnsupported)

	// ErrCorrectnessOracleFailed indicates candidate outputs failed the correctness oracle.
	ErrCorrectnessOracleFailed = errors.New("cudaregress: output correctness oracle evaluation failed")

	// ErrUnsafeCommandRefused indicates an arbitrary shell command or unverified executable was refused.
	ErrUnsafeCommandRefused = errors.New("cudaregress: execution of arbitrary shell command refused by policy")

	// ErrExecutionHalted indicates replay execution stopped due to failure or cancellation.
	ErrExecutionHalted = errors.New("cudaregress: replay execution halted")

	// ErrNoiseFloorExceeded indicates the null control baseline-vs-baseline drift exceeds allowable noise.
	ErrNoiseFloorExceeded = errors.New("cudaregress: null control noise floor exceeds threshold")
)
