# What a Metal phase profile would have to do to be comparable

The CUDA lane has a reviewed phase-attribution instrument
(`cuda-phase-profile-candidate-v3`): nine named phases over a warmed
`EvalBatch` of eight tokens, timed with CUDA events recorded into the default
stream, gated on exact `Float32bits` identity against an uninstrumented arm.
The obvious next step is the Metal counterpart, so the two backends can be
ranked against each other. This note records what stands in the way, so the
counterpart is not built on a timing source that cannot be compared.

## Host markers are not the analogue of a CUDA event

A CUDA event is recorded *into the stream*, and the interval between two
events is device-timestamp elapsed time. That is not execution-only time: it
includes stream idle, so if the host is slow to submit, the gap is inside the
number. What it does give is a device clock reading taken at a point in the
stream's own order.

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

That is the right source, but what this device permits is narrower than the
API suggests, and the limit is physical rather than a matter of bindings.

**Command-buffer times are two coarse regions per layer, not nine phases.**
`metalBatch` creates a single `MTLComputeCommandEncoder` per command buffer
(`metal_batch_darwin.go:480-491`) and reuses it across every dispatch until
`Wait` ends encoding and commits. The layer path builds *two* batches, at
`metal_runtime_darwin.go:584` and `:730`, waited at `:665` and `:844`, with the
closing work and the FFN sharing the second; 2*43 + 5 = 91 is the retained
command-buffer count. So `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` give two
real GPU-time regions per layer — worth having, and cheap — but not the
granularity the CUDA instrument reports.

**Phase granularity inside one encoder is unavailable on this device.** The
binding is not the obstacle: `metalBatch.enc` is the raw
`github.com/tmc/apple/metal.MTLComputeCommandEncoder`
(`metal_batch_darwin.go:20,480`), whose generated protocol declares
`SampleCountersInBufferAtSampleIndexWithBarrier`
(`apple/metal/compute_command_encoder_protocol.gen.go:146`), and
`metalprof.SampleBuffer.Raw()` hands back the matching
`MTLCounterSampleBuffer`. The device is. `TestMetalCounterCalibration` asks
this M4 Max directly and it answers no:

	supports sampling at dispatch boundary: false
	supports sampling at blit boundary:     false
	supports sampling at stage boundary:    true
	supports sampling at draw boundary:     false

Only `MTLCounterSamplingPointAtStageBoundary` is permitted, which attaches two
sample indices to a whole compute pass through
`MTLComputePassDescriptor.sampleBufferAttachments`. It times an encoder, not a
dispatch. So on this hardware, phase granularity means *one encoder per phase*
— a change to how the layer path encodes, not an added marker inside it. That
is a change to the structure of the encoding, where a CUDA event is an addition
to a stream; how the two compare in magnitude is not measured here.

The same run establishes two other physical numbers. A sample buffer accepts at
most 4,096 timestamps (the driver names the range 8..32768 bytes at 8 bytes a
sample), which bounds any single-buffer scheme. And a GPU tick is a nanosecond:
correlating the two clocks with `sampleTimestamps:gpuTimestamp:` across a
100 ms host interval gives 0.999851 ns per tick, so resolved timestamps need no
scaling.

**Counter sampling buys nothing over command-buffer times at pass
granularity.** In the same run, the stage-boundary pass span resolved to
1,401,541 ns while `GPUEndTime - GPUStartTime` for that command buffer reported
1.40175 ms — the same quantity to within 0.015%. `MTLCommandBuffer`'s GPU times
are already the cheap form of the only granularity this device offers, so the
counter machinery is worth adding only if a pass ever holds more than one phase
worth timing, which by the paragraph above it cannot.

**The barrier argument is moot here.** With dispatch-boundary sampling
refused, the `withBarrier:` choice — bounding a timestamp to a dispatch at the
cost of serializing overlap — never arises on this device. It would matter on
hardware that permits per-dispatch sampling, and is recorded only so the
question is not reopened. Both instruments perturb scheduling; nothing measured
here establishes which perturbs more.

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
qualification at load 5.33 against a registered limit of 4.00. The agent
holding the slot was compiling on the same machine during the window, which
makes it the plausible cause; the load reading establishes contention, not
which process caused it.

## What the calibration does not establish

`TestMetalCounterCalibration` is model-free and reads device capability, buffer
capacity and the tick unit; those are properties of the hardware and do not
depend on how busy the host is. The one timing number it reports, the 1.4 ms
pass span, was taken at host load 8.73 against a registered limit of 4.00. It is
used here only as a self-consistency check between two clocks measured over the
same interval, which contention does not invalidate, and it is not a rate.

A separate defect surfaced while establishing this: `metalprof`'s own
`TestSampleBufferHardwareTimestamps` fails on this M4 Max, because it asserts
support for `MTLCounterSamplingPointAtBlitBoundary` before using it. Run with
`METALPROF_HARDWARE_TEST=1` it reports "device does not support counter sampling
at blit boundaries". The test encodes an assumption this device does not meet;
that is for the package owner, not this note.
