// Package stage builds and pins CUDA host-glue libraries with the entry ABI
// int stage(struct stage_ctx *). Context carries named device addresses,
// a stream and caller-retained CUDA function handles. Every registered native
// entry receives a pointer; no aggregate is passed by value.
//
// Builder uses cc or nvcc in a caller-owned retained directory. Load compares
// the library's compiler-evaluated context layout with the actual Go layout.
// Library pins protect code lifetime only: the caller must also retain device
// allocations and function generation pins until stream completion. A zero
// host return establishes submission, not device completion. Failed unload is
// retryable. Compiler source, output and logs remain for qualification receipts.
//
// Stage libraries are C host code, not Go c-shared libraries with a second Go
// runtime. Go plugins also cannot be unloaded and require a matching toolchain.
// Loading C libraries runs arbitrary host constructors and is not isolation.
// Linux and Darwin are supported; other platforms refuse native loading.
package stage
