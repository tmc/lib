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
// Native runs need reviewed source and packet acceptance and an authorized
// GPU environment. The runner acquires cudalock for setup, each Service
// request, and final teardown, releasing it while waiting for source changes.
package main
