// Package cudadiagdemo exercises GPU diagnosis and regression APIs with a
// synthetic concurrent Go service fixture. It records application pre-entry and
// simulated API intervals in a process-local monotonic epoch, carries them in
// the actual recorder snapshot, and decodes that snapshot through the existing
// workload joiner and diagnoser. Device intervals are explicitly synthetic;
// scheduler states are never inferred from an injection setting or timer wait.
//
// Results separate service, pre-entry, API and device timing. Missing captured
// evidence withholds dependent findings. The existing regression runner executes
// a portable service intervention with an independent output oracle. Noisy
// comparisons can remain unknown. Results do not qualify physical GPU capture,
// scheduler cause, collector-off/on cost, or hardware support. Run the portable
// example through x/cmd/gpudiagdemo; native execution is a separate opt-in path.
package cudadiagdemo
