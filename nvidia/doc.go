// Package nvidia groups Go bindings for NVIDIA libraries in its subpackages.
//
// Import the binding package for the library you need, such as driver or nvml.
// Higher-level helpers live under x. Native operations require the corresponding
// runtime libraries and, where applicable, a supported device and driver.
// Platform availability, resource ownership and error behavior are documented
// by each binding package.
//
// # Compatibility
//
// All binding packages share one module version. Before v1, a new CUDA toolkit
// minor starts a new Go minor release; patch releases preserve the binding API.
// An incompatible update to a separately versioned library waits for the next
// minor release. Generation inputs are recorded in upstream.json and do not
// determine which installed runtimes are supported.
//
// Packages under x/ are outside the compatibility promise. Their APIs may
// change or be removed in patch releases.
//
// # Supported runtimes
//
// The supported runtime scope is CUDA 12 and 13 on linux/amd64 and CUDA 13 on
// linux/arm64. Darwin supports host checks, such as layout and loader tests,
// and has no CUDA runtime support. Support for a runtime major does not mean
// every library, operation, GPU or driver combination has been qualified;
// consult the package documentation and the workload's validation results.
// A library's ABI major is independent of the CUDA toolkit major.
package nvidia
