// Package cudagreen plans and creates CUDA green contexts that partition a
// GPU's streaming multiprocessors among workloads.
//
// A green context (CUDA 12.4 and later) owns a set of SMs; kernels launched
// on its streams run only on those SMs. Partitioning keeps a long-running
// kernel, such as a persistent prefill megakernel, from delaying a
// latency-sensitive one, such as a decode step, until its CTAs drain.
//
// [ComputePlan] carves a device into one [Partition] per [Spec], sized by
// target fraction and minimum, rounded to the split [Granularity] and to the
// role's cluster size. [VerifyPlan] checks a plan's invariants: disjoint
// masks, mask population equal to the assigned count, alignment and
// minimums. Both are pure Go and run anywhere.
//
// [Open] creates the green contexts for a plan through the driver bindings
// in [github.com/tmc/lib/nvidia/driver]: it splits the device's SM resource
// into granule-sized groups with cuDevSmResourceSplitByCount, combines each
// partition's groups into one descriptor with cuDevResourceGenerateDesc,
// and creates a green context and a non-blocking stream for each. The
// driver may make fewer groups than the SM count allows (4 groups of 8 on
// the 48-SM GB10); [DeviceSplit] asks it how many, and a plan computed with
// that limit fits. The
// driver does not report which physical SMs a group holds, so a
// [Partition]'s [Mask] numbers SMs logically, in plan order; it proves the
// plan is disjoint, not that particular physical SMs were chosen.
//
// The native tests, gated on CUDA_GREEN_NATIVE=1 and an inherited GPU
// lock, create the green contexts on device 0 and log the launch latency of
// a small kernel beside a long one with and without them. The latencies are
// logged, not asserted.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch
// releases.
package cudagreen
