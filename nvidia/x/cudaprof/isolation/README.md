# Sampled GPU isolation

Isolation is independent of collector completeness, shape, correctness and
stationarity. Schema v1 has three states: `clear` means **sampled clear**,
`contaminated` retains an overlapping outside-process observation, and
`unchecked` means usable bindings or observation coverage are unavailable.
Contamination takes precedence over observer errors and unknown completion.
Sampling may miss brief activity. Resident memory does not quantify timing bias.

`cudaprof capture --observe-isolation ...` adapts the established GPU-window
observer policy without acquiring a lock. Run the entire capture under the
host's GPU lock wrapper. The observer reads the lock directory from
`GPU_LOCKDIR`, defaulting to the one `x/cudalock` uses; it verifies the
live local owner and ancestry, reads physical UUIDs and compute-process residency using bounded
`nvidia-smi` subprocesses, and retains monotonic query boundaries. It records
process generations as Linux boot ID plus `/proc/PID/stat` starttime. Unknown
ancestry, containers without supported attribution, query errors and vanished
processes remain unresolved. It never excludes all same-user processes.

The capture interval is independently retained as a process admission/termination
window, or a qualified profiler phase when available. Broader observer findings
remain in the receipt. Only a supported observation wholly inside the capture
window establishes capture contamination; a query straddling a boundary leaves
that overlap uncertain. Independent runs do not share an execution timeline.

On hosts with multiple physical GPUs the native adapter retains every sampled
device but cannot bind CUDA target ordinals to physical UUIDs; capture isolation
remains unchecked until that mapping is independently supported. The typed
validator supports multiple individually bound devices.

Current generic capture admission cannot prove final relevant GPU completion
across followed or orphaned contexts. A clean native observer therefore remains
`unchecked`, with the completion limitation explicit. The typed validator can
establish sampled clear when independently bound completion, physical UUID,
process generation, clock and per-device coverage are all available. Current
`bench-gpu-window` text claims omit these bindings and cannot establish v1 clear.
No automatic conversion treats a text `clear` or `GPU_LOCK_HELD` as proof.

`--isolation-receipt FILE` retains bounded parent v1 evidence and validates it
against independently retained capture bindings. It is mutually exclusive with
`--observe-isolation`. `--require-isolation` requires sampled clear after writing
the artifact; default capture exit behavior is unchanged. Campaigns reject
contaminated captures through their existing whole-block exclusions, retaining
all original files. Unchecked is not silently promoted to qualified isolation.

Receipt bytes are retained as `gpu-isolation-v1.json`, independently hashed from
the bytes parsed. Capture status includes the disposition and separate binding.
Generic pack/extract preserves the attachment. Reader validation rejects duplicate
keys, unknown fields, unsupported versions, trailing JSON and files over 1 MiB.
Malformed bounded receipts are retained with unchecked status. The attachment
hash is recorded outside the hashed receipt, avoiding a circular dependency.
A final capture artifact hash belongs in the separate archive manifest.

A parent receipt arriving later can be retained explicitly:

```sh
cudaprof isolation bind -receipt observer.json -o bound.pprof capture.pprof
cudaprof trace check -require isolation bound.pprof
```

`isolation bind` publishes a new artifact atomically without replacement. It
retains every original receipt, original status bytes, previous binding manifests,
and the original capture SHA-256 in bounded provenance. A hash-addressed parent
receipt and `gpu-isolation-binding-v1.json` bind new evidence to independently
retained capture identity. Repeated binding preserves contamination and all prior
source hashes; a clean parent cannot erase an earlier outside observation. Input
and output paths, symlinks and hard-link collisions are refused. Malformed receipts
and contradictory hashes produce no new output. Legacy captures without an
independent binding remain unchecked. Successful publication is diagnostic, not a
performance qualification. It never claims late evidence was present at initial
admission or silently rewrites the original capture.
Local receipt hashes detect binding changes; they do not attest observer honesty.

CPU tests cover clear/contaminated/unchecked, contamination plus errors, process
reuse, clock mismatch, boundary uncertainty, missing GPU coverage, legacy/malformed
receipts, strict JSON, immutable attachment round trips, and ancestry sampling.
Controlled clean, arranged-residency and short-process GPU experiments must run
inside one coordinated `gpu-lock` hold. Native observer overhead has not been
measured; those live controls are distinct from the CPU fixture evidence.
