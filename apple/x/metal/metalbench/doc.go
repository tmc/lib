//go:build darwin

// Package metalbench provides experimental wall-clock timing and empirical
// autotuning harnesses for Apple Silicon Metal.
//
// Time includes command-buffer creation, submission, and synchronized waiting;
// it is not a Metal GPU-counter or hardware-event measurement. The package has
// no production consumer. Results require a qualified idle host and a separate
// correctness oracle before they support a kernel-selection decision.
package metalbench
