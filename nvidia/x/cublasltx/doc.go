// Package cublasltx provides an owning cuBLASLt matrix multiplication engine.
//
// An Engine serializes operations because they share a handle and workspace.
// Call Close after all submitted stream work has completed. A closed Engine
// cannot be reused.
package cublasltx
