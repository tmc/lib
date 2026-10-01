# Export detail and evidence

`cudaprof trace export` and `cudaprof trace compare` use the legacy exporters.
All detail modes retain the raw timeline once. `-detail=raw` is the default.
`compact` adds phase and token aggregates without materialization or delivery
tracks; `token` adds observed token scopes and delivery; `operation` also adds
layer and operation groups. Summary hierarchy tracks omit activity leaves that
are already present on the raw tracks. Missing metadata remains unknown.

`-summary` selects `operation`. Combining it with a different explicit detail
is an error regardless of flag order. Summary marker and domain options require
summary detail, except that submission links may select their API domain in raw
mode. Use either `-operation-domain` or its legacy spelling `-summary-domain`.
Marker inference remains separately labeled and does not establish delivery.

Exports use a sibling temporary file and rename after successful completion.
The output must differ from every input, including hard links. `-output-bytes=N`
limits trace or archive bytes; its default is 536870912. Archive trace members
also have a 536870912-byte working buffer limit. An exceeded limit refuses the
export without replacing an existing output. A conservative preflight bounds
all opted-in projections at 100000 tracks
combined, including auxiliary evidence and both arms of an archive. Refusal
occurs before trace bytes are written; no records are silently omitted. Summary
references default to a bound of 2000000. `-max-tracks=N` and
`-max-summary-references=N` explicitly raise or lower these positive limits. The byte limit bounds all output. The decoded
record budget also caps the total rendered raw rows across journals.
`-memory-bytes=N` sets a scoped Go soft memory budget (default 805306368),
respecting a lower caller limit. This is a garbage-collector target, not an RSS
limit or a promise that every capture fits. Export releases the aggregate
pprof samples before decoding the timeline because journals supply the evidence.
Compact is opt-in; no performance
claim or change to the default follows from these limits.

SQL-visible `cudaprof.track.v1` instants identify raw and summary tracks through
versioned arguments, independently of display names. Raw records retain journal,
sequence and capture hash. Summary host union and per-device GPU union are
separate aggregations. Capture-level interval tracks are canonical; descendant
tracks are detail views, so summing all summary slices double-counts work.

Comparison archives bind exact member bytes in `.cudaprof-bindings.json`,
including the manifest, alignment receipt and both traces. This is an integrity
index, not an authenticity signature. Both arms retain independently qualified
collector, workload and stationarity receipts and one fixed alignment offset.
Memory, telemetry, lifecycle and submission evidence use that same offset.

Isolation is an independent qualification instant with the bound observer
receipt. Sampled-clear observations do not establish unsampled absence or a
timing effect. Observer residency intervals are not drawn on the CUPTI clock
without an explicit clock relationship; missing observer evidence stays unchecked.

## Recorded resource validation

An existing Go Gemma fixed-32 capture (SHA256
`7825263db4470fabd2c482c359efde1265bd3e2d0aa9bc935e910ce8046ff957`)
was exported without collecting another workload. The preregistered gate was
peak RSS at most 1 GiB, output at most 512 MiB, retained raw counts and zero
supported importer errors. Initial raw and compact trials failed the RSS gate
at 1306368 and 1416352 KiB. Those failed measurements were retained.

CPU/allocation profiles showed aggregate profile decoding and per-record
serialization allocations. Dropping unused aggregate samples alone still
failed the RSS gate at 1178260 KiB. The scoped 768 MiB Go memory target then
recorded raw peak RSS 837176 KiB and final compact peak RSS 836808 KiB, including the combined track preflight.
Raw output was 163621092 bytes on 318 imported tracks. Final compact output was
171567953 bytes on 320 imported tracks. Official Trace Processor v58.2 retained
270163 raw rows and reported zero error statistics for each projection.
These are fixture-specific resource observations; allocation volume remained
about 5.1 GB per raw export, and no general speedup or full-model resource claim
follows. Compact remains opt-in. Isolation for this old capture is unchecked.

Artifact publication uses retained producer capability as provenance. It never
initializes a CUDA context to query the current host device; absence of retained
properties remains absence, including on hosts with a visible GPU.

Export currently decodes journals into the raw rendering projection and, when
semantic detail is requested, a second typed timeline projection. Native trace
analysis uses its own qualified session; export has not yet unified those two
decoders. Both projections retain the same source identity and apply decode
gates; this transitional implementation makes no single-decode resource claim.
