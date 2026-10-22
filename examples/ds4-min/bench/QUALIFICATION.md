# C-vs-Go qualification protocol

This is the preregistered comparison protocol for the warm prefill benchmark.

- Run exactly five `C, Go, Go, C` blocks under one GPU lease. Do not extend the
  run after a failed gate; replace the run only if a pre-registered void rule
  fires.
- Require every capture receipt to be `MEASURED`, lossless, and digest-gated.
- Record GPU clocks, temperature, and power immediately before and after each
  arm. Both C and Go perform the same registered warm prefill before opening
  the measured range; only the steady-state prefill window is qualified. The
  C runner must honor `DS4_C_WARM_PREFILL=1`; the harness fails closed if its
  runner does not report that warm pass.
- For each block, pair the first C with the first Go and the second C with the
  second Go. Report every signed gap, the median, and the minimum/maximum.
- Require every pair to have the same sign.
- Compute the within-arm noise floors from the ten C-vs-C and Go-vs-Go
  differences: five `C2-C1` and five `G2-G1` values. Compute each
  implementation's median absolute deviation, then use the larger. Apply a
  fixed nonzero MAD
  floor of 1 ms to avoid treating a quantized zero as noiseless. Qualify the
  median gap only when it exceeds three times the larger floored MAD.
- Retain the old two-percent pair-spread number as a diagnostic, but it is not
  the qualification gate.
- A block is void only for a pre-registered infrastructure failure: missing or
  non-MEASURED receipt, loss, digest failure, clock/temperature telemetry
  failure, or an explicitly recorded GPU reset. A void block is replaced once;
  otherwise the exact five-block result is reported as failed.
- Emit both whole-phase and `--warmup-until=hc_expand|hc_cached_expand` gap
  reports. Every capture must contain the boundary kernel; if it is absent or
  renamed, the run is invalid rather than silently falling back to a fixed cut.
  The boundary-defined window is the registered steady-state window; the
  difference is the prologue report.

This protocol is performance-only. Numerical parity still requires the existing
cross-implementation endpoint and boundary evidence.
