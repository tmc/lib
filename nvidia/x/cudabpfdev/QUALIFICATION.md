# Native qualification

`TestNativeInjectionFixture` checks on the host that full-warp (32 threads)
and partial-warp (17 threads) stub candidates are produced and parseable by
`cudaptx`. It does not prove native assembly or GPU behavior.

`TestNativeInjection` includes the same stub attach cases alongside the
original/candidate output comparison. It derives each expected map and miss
count from `Evaluate` for one active warp, then scales by the launched warp
count. The test is gated by `CUDABPFDEV_NATIVE=1` and acquires a `cudalock`
lease. Exact frozen candidate 577e21dd4 passed on GB10 sm_121a on 2026-10-10
with CUDA13.0.88 and driver580.173.02. Original/candidate numerical comparison,
partial/full warp counter and miss controls, and stub injection passed under
canonical gpu-lock. See the coordinator receipt
`/Users/tmc/tmp/lib-bpf/20261010-device/GB10-RECEIPT.md`.
This receipt does not qualify H100 or performance: Ollama was present.

After device ordinal, target, and workload authorization are granted, run only
this command from the `nvidia/` module directory:

```sh
GOWORK=off CUDABPFDEV_NATIVE=1 CUDABPFDEV_DEVICE=<authorized-ordinal> CUDABPFDEV_TARGET=<sm_100a-or-sm_121a> /home/tmc/ml-explore/bin/gpu-lock go test -count=1 ./x/cudabpfdev -run '^TestNativeInjection$'
```

## Compiler gates

`TestPTXAssembler`, `TestInjectionAssembler` and
`TestStubInjectionAssembler` are compile-only gates. The last compiles both
original and rewritten stub kernels for sm_100a and sm_121a, logging image
digests and compiler commands. The original stub definition precedes its call;
no unresolved forward-call prototype is emitted. These tests skip when ptxas
is absent. PTX 9.0 requires CUDA 13; CUDA 12.9 is not a qualification backend.
Compiler acceptance for these fixtures passed with CUDA13.0.88 on GB10;
physical numerical/counter equivalence passed for sm_121a at 577e21dd4.
The extended native harness additionally checks redzones around input/output
buffers, detects a deliberate physical redzone corruption, and runs two private
modules/maps concurrently on independent threads/streams. Those additional
controls require their own exact candidate receipt. Performance/overhead and
H100 remain open. No sm_89 native admission was added for the available L4 host.

Nonzero shared staging remains explicitly unsupported. Device bytecode is the
package's bounded v1 dialect, not arbitrary Linux eBPF or general bpftime
bytecode. Stub forward declarations remain unsupported; only a single defined
empty function is admitted. Duplicate bodies, predicates and hidden body
storage/directives are rejected.
