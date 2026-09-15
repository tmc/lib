# metalcount

Counts Metal dispatches, command buffers, commits, waits and encoders in an
unmodified process. Injected with `DYLD_INSERT_LIBRARIES`, it swizzles the
concrete driver classes behind the `MTL*` protocols and writes a JSON tally.

The counts are deterministic integers rather than timings, so they do not need a
quiet host: contention changes when work runs, not how much of it is submitted.
That makes submission structure measurable on a machine that cannot pass a
throughput qualification.

One instrument counts both the C oracle and the Go port, so the two arms are
compared by the same definition instead of by two hand-written counters.

    clang -dynamiclib -fobjc-arc -O2 -o libmetalcount.dylib metalcount.m \
        -framework Metal -framework Foundation

    METALCOUNT_OUT=counts.json \
    DYLD_INSERT_LIBRARIES=$PWD/libmetalcount.dylib ./program

`hooks_installed` reports which of the eight hooks bound; a zero there means a
selector was absent and its count is not evidence.

## Calibration

`control.m` performs a known 7 dispatches, 3 command buffers, 3 commits, 2 waits,
3 compute encoders and 1 blit encoder. Build and run it under the library before
trusting a measurement; it must report exactly those numbers, and editing its
dispatch counts must move the tally.

## Measuring per-token cost

Setup and prefill are fixed costs that a single run cannot separate from
generation. Run two token counts and take the difference, then confirm linearity
with a third point that the two-point line predicts. Go binaries exit through a
raw syscall and never run C `atexit` handlers, which is why the tally is
rewritten on every commit and wait rather than only at exit.
