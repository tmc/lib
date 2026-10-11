// Cudahotswap watches a local CUDA source file and reports checked pushes.
//
// The runner retains a vector-add baseline and a replay fixture with FP32
// inputs, stable weights, and a contiguous FP32 output. Candidates use:
//
//	extern "C" __global__ void replay(float *x, float *y, const float *w, unsigned n)
//
// Each output must equal x[i]+w[i]. Use -entry to change the entry name.
// Correctness uses the actual replay checker; failed candidates are reported
// as JSON and watching continues. Device faults stop the process.
//
// Usage:
//
//	cudahotswap -watch kernel.cu [-poll] [-I include] [-device 0]
//
// For an egraph-emitted kernel, provide its expression sidecar, explicit named
// FP32 arrays, and entry. Inputs follow the sidecar's lexical parameter order,
// then the output pointer; this mode has no n argument:
//
//	cudahotswap -watch chain.cu -entry chain -sidecar chain.egraph.json -inputs inputs.json
//
// The checker evaluates the frozen expression and poisons output before each
// real launch. It permits absolute error 1e-5 plus relative error 1e-5 for FP32
// arithmetic and device transcendental rounding. Edits never change the oracle;
// source attribution is invalidated when edited source differs. Input JSON is
// bounded to 64 names and 65536 total elements. Empty or nonfinite arrays refuse.
// Sidecar mode accepts only the emitted pointer signature;
// preprocessor changes and compiler flags are unsupported.
//
// Native runs need reviewed source and packet acceptance and an authorized
// GPU environment. The runner acquires cudalock for setup, each Service
// request, and final teardown, releasing it while waiting for source changes.
package main
