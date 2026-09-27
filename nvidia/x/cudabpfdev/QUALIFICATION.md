# Native qualification

`TestNativeInjectionFixture` checks on the host that full-warp (32 threads)
and partial-warp (17 threads) stub candidates are produced and parseable by
`cudaptx`. It does not prove native assembly or GPU behavior.

`TestNativeInjection` includes the same stub attach cases alongside the
original/candidate output comparison. It derives each expected map and miss
count from `Evaluate` for one active warp, then scales by the launched warp
count. The test is gated by `CUDABPFDEV_NATIVE=1` and acquires a `cudalock`
lease. It has not been run; native assembly and GPU behavior remain
unqualified.

After device ordinal, target, and workload authorization are granted, run only
this command from the `nvidia/` module directory:

```sh
GOWORK=off CUDABPFDEV_NATIVE=1 CUDABPFDEV_DEVICE=<authorized-ordinal> CUDABPFDEV_TARGET=<sm_100a-or-sm_121a> /home/tmc/ml-explore/bin/gpu-lock go test -count=1 ./x/cudabpfdev -run '^TestNativeInjection$'
```
