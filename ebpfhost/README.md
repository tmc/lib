# ebpfhost

Linux/amd64 tracefs capture and ELF function resolution. This backend installs
uprobes, return probes and optional scheduler tracepoints in an owned trace
instance. It does not load an eBPF program; `bpfload` is a separate facility.

Tracefs must be mounted and writable. Capture requires appropriate kernel
privileges. The instance uses the kernel `mono` clock. Raw line timestamps do
not establish sensor timing, cross-process clock mappings, precise workload
joins or loss-free acquisition. Per-CPU buffers are bounded by BufferKB; kernel
overwrite and other losses remain unknown to this raw backend.

PID filtering covers existing threads. With scheduler capture, payload filters
are fixed to that initial thread snapshot; threads created later are not
covered. Child following is disabled so forked processes cannot acquire the parent identity.
Use a fixed-thread workload or treat thread coverage as incomplete.

Close disables instance events, unregisters only owned probe names and removes
the instance. All partial setup failures attempt the same cleanup and return
cleanup errors. Read cancellation polls at 10 ms while idle and limits each raw
line to 1 MiB. No live kernel or GPU qualification is claimed by CPU tests.
