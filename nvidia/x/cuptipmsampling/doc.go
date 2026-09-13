// Package cuptipmsampling provides CUPTI Performance Monitor (PM) sampling for NVIDIA GPUs.
//
// PM sampling periodically samples hardware performance counters across the whole
// GPU device into a hardware buffer at fixed time or cycle intervals.
//
// Samples are device-wide aggregate measurements over distinct time spans,
// not per-kernel or per-thread measurements. They must not be attributed to a
// specific kernel launch without explicit timestamp or interval correlation.
//
// CounterAvailability queries the hardware counter availability image for a device.
// NewSession configures and enables PM sampling on a GPU device, managing the
// lifecycle of periodic hardware counter sampling, decoding hardware buffers into
// counter data images, and querying per-sample timestamp bounds.
//
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
package cuptipmsampling
