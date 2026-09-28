# Source and qualification

This package implements the policy layout and inference equations in PufferLib
5.0. The local reference checkout used for qualification is PufferAI/PufferLib
commit `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.

`breakout/breakout.cu` is byte-identical to that checkout's
`ocean/breakout/breakout.cu`. `breakout/breakout_opt.cu` is the local optimized
implementation. The PufferLib license is retained in `LICENSE.pufferlib`.
The small headers in `breakout/shim` replace host-only facilities for NVRTC.

The CPU oracle in `testdata/golden.c` includes PufferLib's `src/puffercpu.c`.
Rebuilding it with `-O2 -ffp-contract=off` reproduces `testdata/golden.txt` exactly
on the qualification host. Tests compare both the Go host implementation and
CUDA implementation with those values.

On the GB10 qualification host, the synthetic network cases, batch tests and
trained Breakout checkpoint test passed. Both Breakout implementations passed
the recorded 256-environment, 2000-step trajectory and generated layout checks.
These checks cover the supplied cases; they do not establish equivalence for
every environment configuration or a performance advantage.

The generated bindings target the `next` CUDA APIs. Regenerate them with a
compatible libclang and its CUDA resource headers using the two cudabind
`go:generate` directives in `breakout/breakout.go`. The separate copy directive
requires `PUFFERLIB` and should only run when intentionally refreshing the
upstream source.
