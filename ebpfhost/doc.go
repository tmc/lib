// Package ebpfhost collects Linux tracefs events from isolated trace instances.
// It uses kernel uprobes and scheduler tracepoints, not an eBPF program. Opening
// a collector requires tracefs access; loss is unknown unless independently
// established from kernel statistics. The supported source platforms are linux/amd64 and linux/arm64.
package ebpfhost
