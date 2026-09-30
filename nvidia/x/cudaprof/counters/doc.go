// Package counters selects qualified observed launches and compares explicit
// hardware-counter receipts. It does not collect counters or infer a hardware
// bottleneck from elapsed time. Collection belongs in a separate replay run
// using cuptirange, with native metric definitions and replay qualification.
package counters
