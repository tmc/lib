// Package counters selects qualified launches, collects native counters through
// explicit application replay, and compares compatible counter receipts.
// Collect uses a separate bounded user replay with native symbol/geometry and
// application output witnesses. Counter values do not establish a bottleneck.
package counters
