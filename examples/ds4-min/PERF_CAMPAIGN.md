# DS4 prefill parity campaign

Objective: reduce the Go CUDA prefill time for the canonical 2,048-token,
two-token workload on DGX GB10 while preserving the frozen output contract.

Pinned workload:

- model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- context: 4096; prefill: 2048; generated tokens: 2
- canonical prefill SHA-256: `3c190c73110cec7aed9f22916a1c84216eb7338071040846cc1d0bd992456aa5`
- continuation tokens: `15342`, `201`

Each stage must retain the existing arithmetic contract and pass focused tests
plus all 16 prefix cases and 48 snapshots of `TestCUDAPrefillGather`. A stage
needs three alternating clean timings, a baseline and candidate cudaprof/pprof
capture, and a recorded binary/source hash. Summed profiler kernel time is
diagnostic; clean wall time decides whether a change improves performance.

GPU work is serialized under `gpu-lock`. A candidate that changes output,
fails a negative control, or lacks same-workload before/after evidence is not
accepted. Directly copying C accumulation order is prohibited unless the
existing Q8 reduction-order diagnostic proves bit identity.

Stages, in priority order:

1. Retile the existing expert-centric Q2 down gather.
2. Improve already-batched mixed-attention kernels.
3. Prototype exact-order dense Q8 Tensor Core work.
4. Profile and extend the qualified HC batching path.
5. Retile or fuse IQ2 gate/up work without changing rounding boundaries.
