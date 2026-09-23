// Package cudabpf is the NVIDIA-specific half of a driver-level BPF policy
// bridge: the hook semantics, chunk placement, eviction ranking and
// time-slice-group attributes of an open-source GPU kernel driver whose UVM
// and scheduler have been patched to export typed struct_ops tables.
//
// The generic Linux plumbing -- bpf(2), BTF parsing, map and program objects,
// struct_ops registration and link lifetime -- is deliberately not here. It is
// reached through [Loader], [BTF] and [Kfuncs], which a consumer in another
// module satisfies. Nothing in this package imports a BPF library, opens a
// file descriptor or issues a syscall, so the whole policy layer is testable
// on a host that cannot load BPF at all.
//
// # Availability
//
// None of the tables this package describes are mainline Linux. They are
// registered by an out-of-tree kernel module against a patched open GPU
// driver, so on an ordinary target every hook is absent, and absent is the
// expected result rather than an error. [Discover] reports what a target
// actually provides; absence is carried as [HookStatus].Available false with a
// reason, never as a zero-valued success.
//
// # What a host test proves
//
// Nothing in this package qualifies kernel behavior. Host tests cover
// encoding, type checking, attach-request validation, decision logic and
// fallback. A claim that a policy changed residency, eviction order or the
// share of the GPU a tenant received requires the kernel, driver and GPU
// identity recorded by [Capabilities], a contended workload, and an oracle
// outside the policy path.
package cudabpf
