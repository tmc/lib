// Devicequalify refuses to record a measurement from a GPU whose health
// was never established.
//
// Hostqualify checks that the CPU is quiet. It reads the load average and
// nothing else, so it passes on a machine whose GPU has stopped working.
// Two 2048-token CUDA runs timed out with the load qualifier reporting
// 0.36 and 1.06; afterwards nvidia-smi reported ERR/N/A. Whether the
// device was already degraded during those runs is not established by any
// retained receipt, because none recorded device state at run time. That
// gap is what this closes.
//
// Run it before and after a measurement, retaining both:
//
//	devicequalify -o before.json
//	devicequalify -o after.json
//
// The gate is a functional probe, not a topology assertion: the CUDA
// driver must initialize, report at least one device, and answer for that
// device's name, compute capability and memory. nvidia-smi can list a GPU
// that cuInit then refuses with NO_DEVICE, so listing is not health.
//
// Descriptive fields are recorded without being gated, because on the one
// host in question they do not discriminate. A post-reboot baseline whose
// CUDA probe completes cleanly still reports PCIe 2.5 GT/s x1 against a
// x16 maximum, and still reports driver model Unknown under
// /proc/driver/nvidia. A predicate over either field would refuse a
// working device, so neither is a default gate and the driver model is not
// read at all. Gate on the link only with an explicitly supplied
// expectation, when a healthy baseline for that machine is in hand:
//
//	devicequalify -expect-link-width 16 -expect-link-gen 5
//
// It exits non-zero when the device does not qualify, so a measurement
// script can gate on it, and writes its observations as JSON so a receipt
// can bind what was seen rather than asserting it in prose.
package main
