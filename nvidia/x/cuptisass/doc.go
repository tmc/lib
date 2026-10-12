// Package cuptisass provides CUPTI SASS metrics data collection for NVIDIA GPUs.
//
// SASS metrics collection dynamically instruments kernel instructions at runtime
// to collect hardware metrics such as instruction execution counts, memory
// traffic, and branch behaviors at per-instruction PC granularity.
//
// SupportedMetrics queries the available SASS metrics for a target GPU chip.
// MetricProperties inspects details for a specific metric name.
// NewSession establishes an instrumented collection session on a CUDA context,
// enabling instruction patching across kernel launches and flushing decoded
// records per instruction.
//
// Range, PM sampling, and SASS metric sessions share a process-local device
// lease. This library policy avoids competing measurements and instrumentation;
// it is not a claim that all native profiler pairs are incompatible. Activity
// tracing is not excluded. Callers still coordinate PC sampling, raw CUPTI
// bindings, other libraries, and other processes when measurement integrity
// requires isolation. Native errors remain authoritative for external conflicts.
// Close must succeed before dropping a session or destroying its context. Native
// cleanup failures retain the lease and resources for an explicit retry.
//
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
// A later release of the same CUDA major uses the newest of those layouts at
// or below it; a new major or an older release is refused.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// [Channel] is the host end of NVBit's device-to-host record channel. It is
// pure Go and runs on any platform: it drains batches from a doorbell and a
// buffer the device writes, and [Channel.Receive] stops at the sentinel record
// NVBit's record_reg_vals flush kernel pushes. No device channel has been
// qualified by a native run yet; [CheckDeviceChannel] reports that as pending.
//
// No producer exists for a Channel. Outside its tests, nothing in this module
// pushes records into one: NVBit's producer is device code it injects into the
// instrumented kernel, and this module neither links nor injects NVBit. The
// protocol is modelled from NVBit's core/utils/channel.hpp and
// record_reg_vals' tool_func/flush_channel.cu, and every test drives a
// Channel with input the test synthesizes. The device kernels in the Linux
// native test are a test-authored copy of NVBit's ChannelDev that has not
// run; a passing run would qualify transport through that copy, not any
// instrumentation producer. No test can detect the missing producer:
// [CheckDeviceChannel] stands for an unqualified device, not for an absent
// producer.
//
// A Channel is unrelated to [NewSession]. SASS metrics are collected by
// CUPTI's own instrumentation and are read through [Session], never through a
// Channel, and no record a Channel has received measures a kernel.
package cuptisass
