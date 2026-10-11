// Package cudatracehost collects Linux CUDA host API observations through BPF
// user probes or tracefs. It does not measure GPU completion or execute device
// BPF programs.
//
// Collection supports Linux amd64 SysV and arm64 AAPCS64 functions in resolved library files.
// Select exactly one API layer to avoid counting runtime-to-driver nesting as
// independent operations. The BPF backend records allocation output values on
// successful returns, kernel task births and exec identities, with explicit
// unavailable flags. Tracefs cannot recover saved allocation output arguments
// or verify thread birth identities. Both backends preserve raw observations.
//
// CUDA context lifetimes, device duration and pre-attachment history remain
// unavailable. Scheduler capture covers the attachment thread snapshot; BPF
// probes cover only that snapshot; new threads remain unobserved. Perf uprobe
// inheritance is refused because it can prevent unprivileged thread creation. Ring loss
// counters and unknown tracefs loss are separate from API success.
// Optional user stacks are bounded frame-pointer observations, not complete
// language stacks. Async allocation observations describe queued host calls and
// do not establish allocation lifetime or GPU completion.
//
// Native arm64 privileged attachment remains unqualified.
//
// Historical pre-inheritance-fix CUDA safety failed. On driver 580.178.04, pre-initialization
// attachment returned CUDA_ERROR_OPERATING_SYSTEM from cuInit while the matched
// collector-off workload passed. The corrected snapshot-only amd64 collector
// passed a bounded L4 control. That receipt does not establish native arm64
// attachment, complete thread coverage or production overhead bounds.
package cudatracehost
