# JIT vector addition

This composition uses `x/cuda` for context ownership, `x/nvrtcx` for cached
runtime compilation, `x/cudamem` for typed device buffers, and `x/cudalaunch`
for kernel argument packing and dispatch.

    GOWORK=off go run .
