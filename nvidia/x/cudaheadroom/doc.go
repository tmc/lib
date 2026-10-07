// Package cudaheadroom checks shared-memory plans against a loaded CUDA kernel.
// Device user limits and compiled static storage are counted once. Driver
// reservation is an optional diagnostic, not a deduction from the user limit.
//
// Callers must serialize kernel configuration, plan mutation, launch and module
// lifetime. A successful check is a snapshot, not an atomic launch reservation
// or authentication of the kernel image. Only the default shared-memory mode is
// supported; unavailable required attributes return errors.
package cudaheadroom
