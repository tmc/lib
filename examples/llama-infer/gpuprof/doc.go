// Package gpuprof provides structured workload estimation and benchmark output
// formatting for neural network inference benchmarks.
//
// Workload estimates calculate parameter memory traffic (model-weight-GB/s) and
// dense arithmetic intensity (dense-model-TFLOPs/s) from declared model dimensions
// and precision. Hardware counter observations (e.g. DRAM byte transactions and
// hardware execution cycles) require counter-backed measurement sessions and are
// marked unmeasured when absent.
package gpuprof
