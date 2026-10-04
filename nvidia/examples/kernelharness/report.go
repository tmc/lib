package main

// unmeasured is the standing list of quantities this harness does not report.
// It is copied into every report.
func unmeasured() []string {
	return []string{
		"occupancy: not queried",
		"achieved memory throughput: not measured; BytesPerLaunch is the task's required traffic, not what the kernel moved",
		"device cache state across samples: uncontrolled, see timing.input_policy",
		"SM clock and power state during measurement: not sampled",
		"CUBIN or SASS artifact digest: not recorded; compile.ptx_sha256 binds only the PTX to the source",
		"register and shared-memory use per thread: not queried",
		"numerical behaviour outside the generated input range: not exercised",
		"concurrent device contention during measurement: not detected",
	}
}
