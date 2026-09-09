# A standalone Metal profiling example for Go

Status: in-progress / slices A, B, C implemented and qualified on hardware. Slices D (native ISA / exact instruction PC attribution) blocked on platform runtime capabilities; Slice E deferred.

## Outcome

Build `apple/examples/metal-trace`: a small, copyable Go program that runs and checks real Metal compute work, records Go execution events and measured GPU intervals, and produces one portable `capture.pprof`. A recipient can inspect Go and Metal Shading Language (MSL) source, available compiler intermediate representation and GPU assembly, counters, and the Go trace without the original source tree or an Apple GPU. Native capture/replay still requires a qualified Mac.

Match the NVIDIA example's strength through numerical checks, recorded code identity, honest measurement semantics, rich source navigation, reusable public packages, and independently reproducible offline viewing. Do not promise identical APIs or hardware features. MSL is the source language, AIR is a compiler intermediate representation, a metallib is a library container, and device ISA is a distinct artifact. AIR is not proof of the final instructions executed by a pipeline.

This plan is separate from plans 001–003 and must not expand the NVIDIA agent's current assignment. Reuse their transport lessons, but do not make Metal depend on the NVIDIA module or block either implementation on a shared backend framework. Estimate the deep profiling work after the capability and code-extraction investigation in step 1.

## Current source and gaps

Paths are relative to the lib repository unless otherwise stated. The adjacent baseline file fingerprints inspected source, including files in the sibling gputrace checkout; inspect changes before implementation rather than overwriting them.

| Source | Existing behavior | Development consequence |
| --- | --- | --- |
| `apple/x/metal/metal.go` | Device, queue, runtime source compilation, pipeline and buffer helpers | Keep actual compilation, allocation, encoding, submission and validation visible in the example |
| `apple/x/metal/metalprof/counters.go` | Timestamp-set discovery and counter-buffer resolution | Generalize through a runtime capability inventory; audit units, error sentinels, returned lengths, supported sampling points and Objective-C ownership |
| `apple/x/metal/metalprof/capture.go` | Programmatic capture and external MTLReplayer invocation | Reuse capture ownership; add bounded replay execution, output validation and recorded tool provenance; process exit alone is not successful collection |
| `apple/x/metal/metalprof/proto.go` | Fixed dispatch-count/time schema and synthetic addresses | Add measured schemas and source-only locations; do not present synthetic addresses as native PCs |
| `apple/x/metal/metalprof/flightrecorder.go` | Caller-supplied durations; record-call wall timestamps; hard-coded `MetalShaders.metal` in snapshots | Preserve existing documented behavior while providing an explicit interval/source API; do not use its current timeline as a GPU clock alignment mechanism |
| `apple/x/metal/metalprof/*_darwin_test.go` | Opt-in timestamp and capture/replay tests | Extend these witnesses instead of treating skipped tests as execution evidence |
| Sibling `gputrace/api_shader.go`, `api_counter.go`, `api_timing.go`, `api_export.go` | Public entry points for shader, counter, timing and export analysis | Inventory usable public contracts and platform dependencies; reuse qualified decoders rather than copying internal packages |
| Sibling `gputrace/docs/STREAMDATA_FORMAT.md`, `docs/research/GTShaderProfiler_BINDING_GAPS.md` | Format evidence and remaining profiler/binary/source-mapping gaps | Research inputs, not a guarantee that every Mac or trace contains ISA, PC data or source costs |
| `apple/go.mod` | Local replacement for `github.com/tmc/apple` | A clean external-client build must explicitly account for this dependency; no claim of a published reproducible module until versions actually resolve |

## Package boundaries

| Location | Responsibility |
| --- | --- |
| `apple/examples/metal-trace` | Flags, explicit workload, CPU reference, warmup, measurements, trace tasks, result summary and capture assembly |
| Existing `apple/x/metal/metalprof` | Metal measurement and capture lifecycle; caller observations; typed metric descriptors and pprof projection |
| New `apple/x/metal/metalcapture` | Pure Go portable recording, integrity/size limits, attached source and code coordinates; no Metal initialization, Xcode invocation or temporary-directory policy |
| New `apple/cmd/metalprof` | Offline CLI and upstream pprof viewer adapter, extraction, source/assembly navigation, optional native enrichment command |
| Sibling `github.com/tmc/gputrace` | Existing trace/replay/shader decoders through public APIs or a versioned command export; any missing public boundary is a separately scoped change in that repository |

Start with one new public package, not a family of speculative wrappers. Keep pure recording and viewing files buildable without Darwin bindings. The example may be Darwin-only; the viewer must have tested non-Darwin builds and at least one actual non-Darwin offline run. If importing gputrace pulls native dependencies into the viewer, enrich on Darwin into portable data before copying the capture; do not require private frameworks at viewing time.

Design producer and reader examples before freezing the API. Prefer concrete values, `Read(io.Reader)` and `Write(io.Writer, ...)`; stream ownership remains with the caller. Separate live Metal resource owners from serialized data. Specify zero-value behavior, close/drain ownership and errors. Use `doc.go`, runnable external-package examples with `// Output:`, and generator-first changes for missing framework bindings. Follow the [Go doc comment guide](https://go.dev/doc/comment).

## Measurement contract

Every observation records its collector, native metric identifier, unit, numeric representation, aggregation rule, scope, availability, device/pipeline identity and live or replay run ID. Enumerate runtime counters and supported sampling points; do not impose the current two-field schema as a ceiling. Record every discovered feature's disposition: collected, unsupported, unavailable, disabled, failed, or not yet implemented, with a reason.

| Data | Required semantics |
| --- | --- |
| GPU timestamps | Resolve completed, valid same-device samples; retain original values and clock conversion provenance; reject missing/sentinel/truncated data |
| Command-buffer GPU duration | Valid only after completion with checked command status; call it a command-buffer interval, even if the buffer contains one dispatch |
| Dispatch interval | Only claim the scope bounded by supported counter sampling points; include barriers/encoder overhead in the stated scope when applicable |
| Host encode, submit and wait spans | Host clock observations; never substitute for GPU execution duration |
| Replay counters and shader costs | Separate run and collection mode, with replay identity, concurrency settings and perturbation metadata; never merge into live counts or timestamps |
| Ratios, rates, gauges, histograms and compiler statistics | Preserve typed native data and denominator/window semantics; do not sum percentages or treat static instruction counts as executed counts |
| PC/instruction data | Require actual addresses and a verified matching code image; absent data remains unavailable, not a zero-filled sample type |

Use integer additive values in pprof only when the projection is defensible and range-safe. Retain floating-point values, uint64 overflow cases, nonadditive observations and raw status in the capture. Multiple homogeneous profiles may live in one capture; expose collector/schema selection explicitly. The default outer profile is the baseline live GPU interval profile. Label counts as dispatches, not hardware PC samples.

Summed busy intervals may exceed elapsed wall time when work overlaps. Preserve queue and interval identities and show union time separately if computed. Go trace tasks/regions describe host execution; attach run/dispatch IDs to join to GPU records. Do not draw GPU work at host completion time or claim CPU/GPU alignment without recorded paired-clock calibration and its error bound. A separate saved GPU timeline can carry device intervals when the Go trace cannot represent them faithfully.

Apple documents counter sampling and clock conversion in [GPU counters and counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers). Its [GPU performance guide](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance) describes shader profiler timing obtained through tracing under overlap; treat that as a separate measurement mode. Device support must be checked against runtime results and the [Metal feature tables](https://developer.apple.com/metal/capabilities/), not inferred solely from the OS version.

## Step 1: Inventory and prove the difficult paths first

Record repository revisions and dirty source hashes; Go, macOS build, architecture, device registry identity, GPU family, Xcode/SDK/compiler and selected developer directory. Inventory the installed SDK, generated bindings, runtime selectors/counter sets and gputrace public APIs. Record each feature's declaration, wrapper, collector, decoder, export type, witness and gap. Include timestamp/statistic counters, command-buffer timing, captures, replay counters, shader profiling, source maps, AIR, native ISA, resource/memory observations, clock correlation, concurrent queues and Metal 4 surfaces where present. Do not call this all-Metal API coverage: it is profiling coverage for the example and reachable measurement families.

Run two bounded exploratory witnesses before committing to an API: (1) validated dispatch plus real timing; (2) capture/replay of that exact pipeline plus source/code extraction. Retain raw output and failures under `~/tmp`. Verify source maps distinguish two kernels and two specializations of one kernel. Determine whether assembly is available from the captured pipeline, compiler tooling, a binary archive, or private tooling; record which route actually works and how identity is established.

Native ISA/source attribution is a first-class milestone, not silently removed from the plan when unavailable. If the selected host cannot produce it, keep baseline work moving but mark full qualification blocked on a named host/tool capability. A tool that disassembles a separate recompile does not satisfy exact-pipeline attribution. Keep any private/Xcode-specific collector opt-in and version-qualified behind the native enrichment boundary. Public API baseline must not require it.

Deliver `004-metal-capabilities.json` with the ledger, supported host matrix and executable verification commands after this investigation. This planning turn does not invent runtime results.

## Step 2: Build the baseline workload and ownership flow

Use two explicit compute stages: vector addition, then a configurable dependent FMA chain. Start with 1,048,576 float32 elements, 512 FMA rounds, one warmup and twelve measured passes, matching the scale of the CUDA example while making no speed-comparison claim. Bound allocation arithmetic and dispatch dimensions; let the user reduce work. Verify every output from both stages, including warmup, against a CPU reference with an explicit float32/FMA policy and finite-error tolerance. Report measured verified outputs as `2 * elements * passes`; warmup is checked but excluded.

Offer `-timing=auto|counter|command-buffer`: auto records the chosen supported mode and reason; explicit counter mode fails when unavailable. Use one stage per command buffer for the initial fallback and name the measured scope honestly. Add an explicit multi-queue overlap workload later, retaining dependency and buffer ownership checks. Make `-capture` and replay/enrichment explicit so the smallest run needs only public Metal execution/timing.

Compile/load through one identifiable path, retain MSL and compilation inputs/options, and hash the library supplied to pipeline creation when loading a metallib. Store function constants, entry point and pipeline identity. If runtime source compilation cannot expose its exact native binary, record that boundary. Keep debug/source compilation options explicit because they may change generated code and timing.

Acquire device, library, pipeline, buffers, counter buffers, encoders and command buffers in visible order. Check command completion/errors before reading results or counters. Submitted work must finish before referenced resources are released; cancellation of a Go wait does not cancel GPU execution. Resolve ownership conventions of generated Objective-C wrappers before adding releases or leaking retained objects. Stop capture and Go tracing before packaging final output. Write the final profile atomically only after validation succeeds; retain failure diagnostics separately.

## Step 3: Implement the portable recording and viewer

Specify a Metal-owned versioned format with a standard gzip pprof outer profile and a namespaced embedded payload, following the already demonstrated NVIDIA single-file approach. It is a project convention carried by pprof comments, not a promise that every pprof transformation preserves attachments. Validate a pinned upstream parse/write round trip; stripping comments must still leave a usable ordinary profile with no false claim that source assets remain.

Attach selected profiles, MSL/Go source, Go trace, typed measurements, capabilities, code index and available AIR/ISA. File roles and metadata must be caller supplied, with no example filenames or workload fields in the public package. A compact capture is the normal sharing artifact. Raw `.gputrace` and replay trees can be much larger: support explicit full inclusion with documented limits and report omitted raw assets. Compact captures must be sufficient for all displayed results but must not claim replayability without the required original assets.

Specify compressed/uncompressed limits, duplicate/path/case-collision rejection, digest checks, unknown-version behavior and bounded decoding. Do not execute captured content. Extraction stays in a fresh caller-selected directory; command-owned temporary files use `~/tmp` and are cleaned on success and failure. Pure reads do not create files. Archive trees must not escape through traversal, links or absolute paths.

Add a pprof object adapter for saved MSL/AIR/ISA coordinates. Keep code image identity, native instruction offsets and logical text coordinates distinct. Source-only frames have no fabricated native address. Whole-dispatch intervals may use documented navigation anchors, but must never become instruction hotness. Avoid duplicating sample values when showing multiple code representations. Store enough provenance to verify a source/assembly line belongs to the captured pipeline specialization.

Provide `metalprof -top`, source/disassembly listing, local HTTP views, profile selection and extraction; finalize syntax through CLI examples. Unavailable assembly or metrics get an explicit explanation. Empty or zero-total profiles must never render `NaN%`. Annotate source using measured attribution only; compile statistics and heuristic estimates, if retained, are separate named data views.

## Step 4: Add deep collectors without narrowing the inventory

Implement runtime-discovered public counter sets and supported sample locations, then native replay enrichment using existing gputrace capabilities. Preserve native counter IDs, units and validity. Resolve counter accumulation/delta semantics and sample windows before exporting. Incompatible modes get separate recorded runs; `-collect=all` reports the full ledger and returns a partial/non-success status when requested reachable collectors remain unimplemented or fail. Unsupported hardware is a visible coverage gap, not a fabricated zero or a reason to hide an entire metric family.

Add shader/source cost and ISA/PC attribution only with the evidence established in step 1. Preserve raw attribution and confidence; do not convert a compiler cost estimate, dispatch duration or pipeline-level sample into measured line-level execution. Validate against an Xcode view of the same retained recording and selected mode where available. Report disagreements rather than normalizing them away.

Do not replay just one dependent stage or queue if it changes the workload. Capture required resources and dependencies, revalidate replay outputs where the tool exposes them, and distinguish replay fidelity checks that cannot be performed. Replay uses an explicit wall-time bound and validates fresh expected output files, not stale files from a previous invocation.

## Step 5: Optional multiple devices and library workloads

Keep single-device Apple Silicon as the primary acceptance target. Enumerate actual Metal devices before offering `-devices`; unified memory is not multiple GPUs. If two suitable physical devices exist, add separately selected independent per-device work, then an explicitly host-staged reduction/gather with all-device numerical checks and transfer/host/device timing separation. Do not describe this as NCCL or assume peer-to-peer transport. Reject unsupported device selections before allocation and preserve per-device clocks; never subtract timestamps from different GPUs.

An optional MPS matrix workload can exercise library-owned kernels later. It must have its own CPU reference and source-only library-call frames unless genuine library code attribution is available. This addition does not replace the owned MSL kernels needed to prove AIR/ISA/source navigation. Neither optional workload blocks single-device baseline delivery; multi-device and library qualification remain separate checklists.

## Acceptance and delivery slices

| Slice | Required evidence before calling it complete | Status |
| --- | --- | --- |
| A: source/capability investigation | Versioned ledger; real timing witness; explicit result for matching pipeline source/AIR/ISA and shader attribution routes | **COMPLETED** (004-metal-capabilities.json generated; verified on Apple M4 Max) |
| B: runnable public example | Small, tail and default runs validate all results; warmup excluded; GPU intervals independently reconcile to profile totals; resource errors handled | **COMPLETED** (`apple/examples/metal-trace` verified on hardware, 257 & 1,048,576 elements) |
| C: single-file/offline package | Capture survives copying/renaming without original files; stock pprof totals match; local source, assembly where collected, and extracted Go trace work; external-module build succeeds | **COMPLETED** (`metalcapture`, `metalprof`, external-client build, isolated directory tests passed) |
| D: full profiling | Every reachable measurement family has a collector/export or explicit unresolved gap; fresh native replay/attribution evidence; raw and projected data reconcile | **DEEP COLLECTOR SLICE COMPLETED / HARDWARE ISA UNRESOLVED** (Replay enrichment, streamData dispatch timing, compiler metrics, and typed observations qualified with separate live/replay profiles; hardware ISA disassembly & runtime PC updates documented as unresolved due to Apple toolchain/target mismatch) |
| E: optional workloads | Actual second-device or MPS execution with numerical, ownership and attribution checks; unsupported/skipped remains unqualified | **DEFERRED** (single-device focus qualified; multi-device deferred) |

Software tests use small, clearly synthetic fixtures: codec round trips, writer failures, corrupted digests, truncation, oversized payloads, unsafe extraction, duplicate assets, invalid schema/unit conversion, integer overflow, missing versus zero measurements, specialization collisions and source/ISA coordinate mismatches. Ensure absent PC data cannot acquire instruction attribution and live/replay profiles cannot merge silently. Test the CLI with `rsc.io/script` txtar cases per repository instructions. Add uncached focused tests, race tests for shared recorder state and vet; do not introduce mock-GPU infrastructure just to test control flow.

Build a copied example in `module example.com/metal-client` with `GOWORK=off` and explicit local dependency replacements only where needed. Exercise all exported example APIs from this external client. Separately test a clean resolved dependency build before claiming install-from-anywhere support. Compile the pure viewer for Darwin and Linux; actually run it on Linux against the single copied file. Keep native framework dependencies out of that path.

Hardware cases: one element, 257 elements with small rounds/passes, and the proposed default. Controls: deliberately wrong finite output and NaN must fail before success output; missing counter capability must fail explicit counter mode; truncated replay output must fail enrichment; mismatched code hash must fail attribution. Retain counts, maximum error, intervals, capabilities, source/tool hashes and process exit status. Set bounds before running; do not enlarge them after failure to declare success.

For final viewing, copy only `capture.pprof` into an otherwise empty directory. Check stock pprof totals, custom source/AIR/ISA views and collector selection; remove access to original source/tool paths. Exercise real browser interactions with AXMCP + CDP in Brave, including a missing metric and a zero-total profile. Extract the Go trace and open it with the recorded compatible Go toolchain. Retain screenshots and machine-readable assertions. Cross-builds and screenshots do not substitute for native collection evidence.

Update slice status and the index with exact passed, failed, skipped and blocked gates. Do not mark the full plan complete after only timing and source work if native ISA or reachable collectors remain unresolved. No commits, publishing, driver changes or alteration of the running NVIDIA task are authorized by this plan. Implementation should use an owned checkout incorporating necessary current source, with artifacts under `~/tmp` and binaries kept out of Git.
