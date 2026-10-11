// Package cudadiag provides evidence-admitted diagnostic mechanisms for CUDA
// workloads.
//
// It isolates root execution bottlenecks—specifically distinguishing host
// submission delays from competing device bottlenecks—using deterministic trace
// contracts and target OS scheduler evidence.
//
// Elapsed wall time, summed kernel time, interval-union busy time, host API
// duration, runnable/scheduled intervals, and wait time are kept strictly
// separate.
//
// Every diagnostic claim requires prerequisites admitted via package
// [github.com/tmc/lib/nvidia/x/cudaprof/tracecontract] and identities tracked via
// package [github.com/tmc/lib/nvidia/x/cudaworkload]. Unmatched returns, missing
// scheduler events, clock uncertainty, and trace loss result in withheld
// findings rather than speculative claims. Overlapping power or clock changes
// are treated as non-causal associations unless validated through an explicit
// [InterventionPlan].
package cudadiag
