# What a Metal phase profile would have to do to be comparable

The CUDA lane has a reviewed phase-attribution instrument
(`cuda-phase-profile-candidate-v3`): nine named phases over a warmed
`EvalBatch` of eight tokens, timed with CUDA events recorded into the default
stream, gated on exact `Float32bits` identity against an uninstrumented arm.
The obvious next step is the Metal counterpart, so the two backends can be
ranked against each other. This note records what stands in the way, so the
counterpart is not built on a timing source that cannot be compared.

## Host markers are not the analogue of a CUDA event

A CUDA event is recorded *into the stream*. The interval between two events is
device time, whatever the host was doing meanwhile.

Metal encodes dispatches into a command encoder and the GPU runs them
asynchronously; the host returns from an encode call long before the work runs.
A `time.Now()` marker between two encode calls therefore measures *encoding*
cost, not execution. Two such numbers can be subtracted, and the result will
look exactly like a phase duration, but it is not one, and it is not the same
quantity the CUDA profile reports. Ranking Metal phases by host markers against
CUDA phases timed by stream events would be a cross-backend comparison of two
different measurements — the failure mode this project keeps rediscovering.

The one existing Metal instrument of this shape, `phaseTimer` in
`metal_profile_darwin_test.go`, is honest about this: it attributes the
*sweep's* host wall clock — gguf read, model map, view checks, shader
compilation — which is genuinely host-side work, and it tags its records
`kind: "host"`.

## What the comparable source would be

`metalprof.SampleBuffer` (`apple/x/metal/metalprof/counters.go`) wraps
`MTLCounterSampleBuffer` over the `MTLCommonCounterSetTimestamp` counter set
and resolves hardware GPU timestamps. The package states the rule this note is
about: *callers must not substitute host time for GPU counter values*, and it
returns `ErrTimestampCountersUnavailable` rather than degrading to host time.

That is the right source. Two things are in the way.

**The phases are inside one encoder.** `metalBatch` creates a single
`MTLComputeCommandEncoder` per command buffer (`metal_batch_darwin.go:480-491`)
and reuses it across every dispatch until `Wait` ends encoding and commits — one
command buffer per layer, which is what the 134 → 91 command-buffer count
measured. So `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` give layer
granularity, not phase granularity. Phase boundaries need
`sampleCountersInBuffer:atSampleIndex:withBarrier:` between dispatches within
the encoder. The binding exists on the generated encoder protocol
(`apple/metal/compute_command_encoder_protocol.gen.go`); `x/metal` does not
wrap it, so wrapping it is prerequisite work.

**The barrier argument is a real choice, not a detail.** With `barrier: true`
the timestamp is bounded to the dispatch boundary but serializes work that
Metal would otherwise overlap, so the profile perturbs the thing it measures
more than the CUDA events do. With `barrier: false` the sample is cheap but not
precisely attributable to a boundary. Whichever is chosen must be stated in the
result's scope, and the two backends' perturbation is then not equal — which
bounds how finely their rankings may be compared even when both are honest.

## The gate to copy

Independent of timing, the CUDA candidate's structure is worth reproducing
exactly: run the production path uninstrumented, retain the full-logit bits,
`Reset`, run instrumented, and refuse to publish unless the two agree bit for
bit — with phase counts validated against a closed-form expectation so an
instrument that silently stopped marking cannot pass. Identity is the right
gate here precisely because the markers are meant to be inert; it is the
inverse of the compressor-defer case, where the load-bearing evidence was a
count that moved.

## Scope this does not have

Both instruments profile eight warmed tokens at positions 0..7. Attention cost
grows with position and the FFN phases do not, so neither ranking transfers to
the position range where the 2048-token runs stopped (CUDA 1254, Metal 1579).
Neither of those two runs produced a rate: the CUDA one carries an
unestablished device-health scope, and the Metal one failed its after-run host
qualification at load 5.33 against a registered limit of 4.00, because the
agent holding the slot was compiling on the same machine.
