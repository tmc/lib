// Package cudabpf validates NVIDIA experimental driver-policy capabilities,
// owns compiled-object attachments through a consumer-supplied loader, and
// provides host models of placement and scheduling decisions.
//
// It has no Linux loader dependency. The consumer implements Loader and BTF
// using actual target metadata. BTF type names alone do not prove registered
// struct_ops callbacks or callable kfuncs: discovery requires registration,
// callback prototype, context layout, and kfunc admission evidence.
//
// # Experimental ABI
//
// The inventory describes gpu_ext v0.0.1013 (a23fbf8b). These are extension
// source interfaces, not mainline Linux or NVIDIA driver interfaces. Every
// hook is expected to be absent on stock drivers. Discover records absence;
// Attach refuses unsupported or mismatched hooks before loading an object.
// No live driver with these hooks has been qualified for this package.
//
// # Host models and kernel policy
//
// Go policies and Kfuncs model decisions in host tests. They are not compiled
// into BPF, called by the kernel, or userspace wrappers for kernel kfuncs.
// Attach loads caller-supplied compiled bytes; it does not connect those
// programs to Go callbacks. Host-model DisableHook does not modify a kernel
// attachment. Actual policy is disabled by Detach or Close. Active per-hook
// control, process filtering, bytecode authoring, driver-default restoration,
// and asynchronous kernel prefetch need an admitted extension implementation.
//
// The driver retains final authority over any submitted decision. A host
// policy result proves neither residency change nor GPU scheduling behavior.
// Native claims require exact kernel, driver and GPU identity, extension BTF,
// independent correctness controls, and raw load/attach/workload receipts.
package cudabpf
