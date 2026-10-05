# Versioning the nvidia module against NVIDIA releases

Status: proposal, rev 3 (2026-10-04). Companion to ../VERSIONING_PLAN.md,
which covers the repo-wide per-directory module split. This document
covers the one module that tracks a moving upstream: how we pin NVIDIA
inputs, regenerate, decide a version, tag, and describe what changed,
and which installed runtimes the result is allowed to bind.

Rev 3 replaces the release trains of rev 2. Rev 2 cut one Go minor per
CUDA toolkit minor and generated from one header. Since 2026-09-29 the
generator reads a floor header, a ceiling header and the pinned
releases between them, and produces one Go API that spans them. A new
CUDA minor therefore adds to the API instead of replacing it, and the
module stays on one tag line, `nvidia/v0.1.x`. "Release line", "Naming
versioned C functions" and "Mixing our versioning with NVIDIA's" below
are new or rewritten; the rulings they rest on are dated where they
are stated. The rest is rev 2, edited where it assumed trains.

Rev 2 separated the *generation source* from the *supported runtime*,
and that separation stands. Its platform scope was ratified by tmc on
2026-09-06; the floors in "Release line" were decided on 2026-09-30 and
2026-10-04.

## The unit of versioning

`github.com/tmc/lib/nvidia` is one module and one tag line
(`nvidia/vX.Y.Z`). driver, cudnn, nccl, and the rest are packages inside
it, not modules, so a cuDNN-only regeneration still cuts a module-wide
tag. That is acceptable and we keep it: consumers get one version to
pin, and the packages share a generator and a load model. Upstream
versions are tracked as *data* (a manifest), never as Go module versions.

Two consequences follow:

- The Go version says how the Go API moved. The manifest says which
  NVIDIA release each package was generated from. They are independent.
- `x/` (hand-written; 457 of 755 nvidia commits at `fd350702` touch only
  files under `nvidia/x/`) lives in the same tag line, while "Release
  line" below exempts it from the compatibility promise. Those two
  facts do not sit together: a consumer reading `nvidia/v0.1.3` cannot
  tell from the version whether a break reaches them, because the tag
  covers a surface the tag promises nothing about. The 38 `x/` entries in
  `../api/nvidia.log` are the first time that had to be argued out of a
  bump by hand, and the first draft of that argument got it backwards.

  The earlier trigger for splitting -- "if its churn starts forcing
  tags that regenerate nothing" -- is retired because it cannot fire.
  One nvidia tag has ever been cut, so no tag has been forced by
  anything, and the predicate stays false however large the churn
  gets.

  Decided 2026-09-27 (tmc): x/ stays in this module. This reverses the
  earlier decision to split it. The exemption therefore has to be
  stated where a consumer reads it: each importable x/ package's doc
  comment says the package is outside the compatibility promise, and
  that is a precondition of the next tag. Today it is carried by this
  file alone. It is not carried by `x/DESIGN.md`, which is excluded by
  `.git/info/exclude`, is in no clone, and does not discuss API
  stability in any case; nor by the x/ package docs, 0 of the 49
  importable ones stating it. See "Release line".

## Problems in the current pipeline

1. `cuda-headers/fetch.sh` already pins: it reads `versions.lock` and a
   normal fetch never asks the CDN for the newest release;
   `-update-lock` rewrites the lock without fetching. What is missing
   is a richer named manifest and a per-directory record of which
   archive produced which directory. The checked-out tree is still a
   mix (cudart 13.3.29 beside cupti 13.3.75 beside cccl 13.2.86).
2. `nvidia/upstream.json` still declares itself `partial-provenance`,
   but the coverage gap it was written about is closed. Measured at
   `fd350702`: all 35 packages carry a recorded generation argv,
   `coverage.missing_argv` is empty, and none carries
   `generation-unrecorded`. 31 are `generation-reproduced`; 4 are
   `generation-partially-reproduced` -- cupti, driver, nvjitlink and
   nvml, whose recorded recipes do not reproduce the committed bytes.

   What remains is non-uniform stamping in the generated code. 27
   packages emit a `provenance.gen.go` carrying the full header list and
   an input SHA-256. Eight do not, and still stamp only the header
   basename ("from cufft.h"): cuBLAS, cuBLASLt, cuSOLVER, cuTENSOR,
   driver, ncclep, ncclm2n and vpi. Five of those eight are the packages
   this item used to name as the argv gaps, so the same generators are
   behind both; fixing their emitter closes this item.
3. `codegen/docs/*.json` carries `"version": ""`; the scraped docs are
   unpinned and undated.
4. `make -C nvidia apidiff` compares the current API with the newest
   `nvidia/v*` tag on linux/amd64 and darwin/arm64. It rejects incompatible
   changes unless `ALLOW_INCOMPATIBLE=1` is explicit. The existing
   `version-check` target instead compares against the reviewed API report;
   both targets remain available. Neither is a hardware qualification gate.

## Pinned inputs

### Header manifest (in cuda-headers)

Replace "latest" with a committed `versions.json` that `fetch.sh`
reads. One entry per redist package:

```json
{
  "platform": "linux-x86_64",
  "packages": {
    "cuda_cudart": {"version": "13.3.29", "source": "cuda/redist"},
    "cuda_cupti":  {"version": "13.3.75", "source": "cuda/redist"},
    "libcudnn":    {"version": "9.20.0.48", "source": "cudnn/redist"},
    "nccl":        {"version": "2.29.7", "cuda": "cuda12.9"},
    "fabricmanager": {"version": "610.43.02"}
  }
}
```

`fetch.sh` gains two modes: `fetch` (download exactly the manifest) and
`update [pkg...]` (resolve latest, rewrite the manifest, then fetch).
Updating and fetching are separate commits so a header bump is a
reviewable diff of both the manifest and the headers. Every extracted
package directory gets a `.version` file containing its archive name,
and `fetch.sh` refuses to run when the git tree is dirty.

NVIDIA publishes a `redistrib_<ver>.json` per CUDA release on the same
CDN; `update` should prefer it to directory scraping so a "CUDA 13.3"
bump moves every toolkit package to the versions NVIDIA shipped
together, rather than to per-package latest.

cuda-headers is tagged after each coherent bump: `cuda-13.3`,
`cudnn-9.20`, `nccl-2.29.7`. The tag is what lib's manifest cites.

### Docs manifest (in codegen)

`fetch-nvidia-docs` writes `docs/manifest.json`: source URL, docs
version string when the site exposes one (cuDNN does), fetch date, and
sha256 of each JSON. The per-file `"version"` field is filled from the
same source. Docs are enrichment only, so a docs refresh never by
itself justifies a tag; it rides the next regeneration.

### Provenance in generated output

gen-nvidia already has a `gen.Header` type with a SourceHash. Emit it in
each package's `doc.gen.go` and write one machine-readable file at the
module root, `nvidia/upstream.json`:

```json
{
  "generator": "github.com/tmc/codegen/cmd/gen-nvidia@<commit>",
  "headers": "github.com/tmc/cuda-headers@cuda-13.3",
  "packages": {
    "driver": {"header": "cuda.h", "package": "cuda_cudart", "version": "13.3.29", "sha256": "…"},
    "cudnn":  {"header": "cudnn.h", "package": "libcudnn", "version": "9.20.0.48", "sha256": "…"},
    "nccl":   {"header": "nccl.h", "package": "nccl", "version": "2.29.7", "sha256": "…"}
  }
}
```

This file is the bridge between the Go tag and the NVIDIA release. It is
also the input to the changelog (below). Expose it in Go as
`nvidia.Upstream()` only if a consumer asks; a data file is enough.

## Regeneration

One entry point in lib: `nvidia/generate.go` with a `//go:generate`
directive per package invoking that package's *recorded* generation
argv with `-headers-dir` pointing at a cuda-headers checkout at the
manifest's tag. Not every package comes from `cmd/gen-nvidia`: NVML is
generated by `cmd/nvml-survey` and cuSPARSE by `cmd/cusparse-survey`.
Using the wrong tool invents phantom API changes. A Makefile target
`make -C nvidia regen` wraps it and then:

1. `gofmt -l` and `go vet ./...` in the module.
2. `go test ./...` on the host, with `GOWORK=off`. Darwin runs the
   layout and loader tests: these are host checks and Darwin evidence.
   Target-platform ABI and runtime checks are separate and are not
   implied by them.
3. `go test ./...` on a CUDA host for the runtime tests. Report skips
   explicitly; an exit-zero run whose GPU tests skipped does not
   establish the target cell.
4. Rewrites `upstream.json`.

A regeneration commit touches only `*.gen.go`, `upstream.json`, and the
go:generate file, with subject `nvidia: regenerate from CUDA 13.3`. Hand
fixes in the generator go in codegen first and are re-run, never
patched into lib.

## Supported runtimes

The generation source and the supported runtime are two different
things, and this document previously conflated them. `upstream.json`
says which NVIDIA headers a package was generated from. It says nothing
about which installed libraries that package is allowed to bind. This
section states the second contract; everything above it states the
first.

### Ratified scope

Generation reads the floor and ceiling releases that "Release line"
describes. Supported runtime is wider on one
architecture and narrower on the other:

| Platform | Supported runtime majors | Why |
|---|---|---|
| linux/amd64 | CUDA 12 and CUDA 13 | Measured: one unchanged executable served both major runtime roots for the variants listed below. |
| linux/arm64 | CUDA 13 only | Policy limited to CUDA 13. The measured GB10 path using NVRTC 12.6.85 failed while compiling for `sm_121`, before its numerical oracle (receipt: BE6AE5E7). Other CUDA 12 paths and arm64 hardware were not qualified. |
| darwin | no CUDA runtime | Host-only tests (layout, loader). Unchanged. |

Ratified by tmc, 2026-09-06. Driver floors are NVIDIA's: 525+ for CUDA
12.x, 580+ for CUDA 13.x. Library ABI major is not toolkit major
(cuFFT `.so.11`/`.so.12`, cuBLAS `.so.12`/`.so.13`); never infer one
from the other.

### Two contracts per workload, not one

The contract is stated per *cell*, not per executable name: executable
and source identity, mode and flags, platform, GPU, driver profile,
and the exact product versions of each runtime root. "CUDA 12" alone
is not an input identity. One variant can owe successful execution on
both amd64 roots and explicit unsupported behavior on an arm64 root;
those are separate cells, not contradictory contracts.

Assign each selected cell one required outcome; record qualification
status separately:

- **successful execution** — the variant runs and produces correct
  results against that runtime root.
- **explicit refusal** — the variant refuses before executing an
  incompatible operation, and does not merely happen to avoid the
  divergent path.

The required behavior determines the qualification work. Explicit
refusal requires a verified rejection point and a distinguishable
error. Successful execution requires compatible contracts for the
operations and fields the variant actually exercises; measured
divergence on that path must be resolved and qualified. Neither
requirement chooses an adapter design by itself: a layout difference
outside the exercised boundary needs no adapter, and an exercised one
does not uniquely select a design. Neither outcome is established
without a receipt.

A ratification row is *executable + mode/flags + input/GPU architecture
+ required output oracle + required native calls and record fields +
unavailable-feature behavior*. Package lists are an inventory index,
not acceptance criteria: import graphs cannot decide this, because a
package can import a type whose layout diverged without ever reaching
it.

### What has been measured

Three amd64 cells passed with one unchanged binary each, verified by
identical executable hash, positive run exit 0, deliberately-wrong
oracle exit 1, and `/proc/self/maps` captured at load and at workload
completion. Driver 580.173.02 throughout; two L4s on
`g2-spot-dual-l4`.

| Variant | Older runtime | Newer runtime | Result |
|---|---|---|---|
| deterministic two-layer inference | NVRTC 12.9.41 | NVRTC 13.0.88 | prompt A, three greedy tokens `[203,203,203]`, exact agreement with the CPU oracle, cold caches. Token equivalence on a deterministic fixture, not full logits. |
| single-rank GEMM plus all-reduce | cuBLAS 12 + NCCL 2.31.2 cuda12.9 | cuBLAS 13.1.1.3 + NCCL 2.28.9 cuda13.0 | exact `[19,43,22,50]` |
| two-rank two-GPU all-reduce | NCCL 2.31.2 cuda12.9 | NCCL 2.28.9 cuda13.0 | `GetUniqueId`/`NewCommInitRank`, one process with two device ranks, two distinct GPU UUIDs, exact sums over 4096 elements per rank |

Separate Darwin arm64 header probes found matching declarations and
signatures for 45 exact driver symbols in a conservative source
boundary for the frozen inference executables. That boundary does not
pass the changed caller-backed NVIDIA records identified in the
census. The sampled cuBLAS GEMM contract matched. NCCL `UniqueId`
measured sizeof 128, alignof 1, internal offset 0 and width 128, using
the same NCCL 2.29.7 header with two toolkit include environments;
this does not compare CUDA-flavor archives. `cuEventElapsedTime_v2` is
declared with matching signatures in both header sets; only the
unsuffixed macro selection differs. Those probes used driver
12.9.79/13.2.51 and cuBLAS 12.9.1.4/13.3.0.5 header profiles, which
are distinct from the runtime table's product versions, and not the
currently dirty 13.3 header tree. The conservative closure includes
optional helpers; it is not a dynamic mode-specific call trace. These
results do not establish Linux ABI or runtime compatibility.
See 392735FA's header-closure MEMO.md and its receipt manifest.

Stated limits, which are part of the result:

- The two NCCL archives differ in product version as well as CUDA
  flavor. This does not isolate CUDA flavor at constant NCCL version.
- One driver profile passed. That is not "CUDA 12 hosts work"; no
  older-driver host was tested.
- On arm64 the CUDA 12 run failed in the compiler, before reaching the
  oracle, so it is not a successful negative numerical control.
- CUPTI activity collection was not run against a live runtime. Its
  version gate is source evidence.
- Not covered at all: real-checkpoint L4 inference, full-logit
  equivalence, multiprocess NCCL rendezvous, distributed GEMM, arm64
  multi-GPU, cuDNN, cuTENSOR, nvCOMP, NVML/GPM product matrices.

These are numerical passes on named cells. They are separate from the
full qualification gate, which is on HOLD for the reasons in "Not yet
built" below; a passing oracle does not advance it.

### Fail-closed today

Two independent gates exist. Both now read record sizes, but they
admit different things: the Go package requires an exact size match for
most kinds, the agent requires only that the runtime record be at least
its compiled prefix.

`nvidia/x/cuptiactivity` gates on record layout. `recordLayouts`
admits any CUPTI 13.x version and then queries
`cuptiActivityGetStructSize` for every kind it decodes, rejecting a
kind whose native size differs from the Go layout; kernel records are
checked against the measured Kernel10-Kernel13 sizes instead. A
different major version is rejected before any size query, callback
registration or activity enable. `recordBytes` bounds each record to
the size that gate admitted, not to the whole activity buffer.

`nvidia/internal/cudainject/native/cupti_agent.c` gates on record size.
It rejects a cross-major pair, CUDA 12 `Kernel9` against CUDA 13
`Kernel10`, and within CUDA 13 it feature-detects
`cuptiActivityGetStructSize` through `dlsym` and requires the runtime
kernel and memcpy records to be at least as large as the structs it was
compiled against. When the symbol is absent, which is every runtime
before 13.2, the size check is skipped and the major check carries the
policy alone.

That replaced a strict `cuptiGetVersion() == CUPTI_API_VERSION`
equality, which was our policy rather than a CUPTI requirement and was
stricter than the layouts require. Measured 2026-09-13 on dgx: an agent
built against each of the 13.0, 13.1, 13.2, 13.3 and 13.4 header sets,
run against each of the five runtimes, all 25 cells collected kernel and
memcpy records with no drops, including a four-minor skew. The reason is
that the compiled struct is a stable prefix and runtime records grow
append-only, so `sizeof(CUpti_ActivityKernel10)` stays 216 and
`sizeof(CUpti_ActivityMemcpy6)` stays 96 across the band while the
runtime records reach 248 and 112.

The agent's `kernel_layout` and `memcpy_layout` fields are layout
*names*, and they stayed `Kernel10` and `Memcpy6` across all five pairs
while the sizes changed underneath. `memcpy_layout` is a fixed string
literal besides. Both are retained as provenance and neither is a gate;
the size is what the policy reads.

Native version responses and rejection behavior remain unqualified.
Loading and the version query may already have native initialization
effects. Successful collection on both majors remains unmet and
requires investigation of older-record support and native delivery.

### Not yet built

- There is no generic runtime ABI-profile admission layer. Explicit
  paths and bare sonames can already select a library outside the
  generation inputs. Widening sonames to manufacture coverage remains
  prohibited; qualification tests the routes as they are.
- The `TestLoadConflict` diagnostic defect (explicit adoption after a
  lazy load kept `Implicit=true`) is fixed in the emitter and
  regenerated for `nccl` and `cufile`; it persists in every generated
  package not yet regenerated against that fix.
- The cudainject size gate is unmeasured outside CUDA 13. The CUDA 12
  `Kernel9` arm is rejected by the major check rather than qualified,
  and PC sampling carries its own record structs and is not covered by
  the kernel and memcpy size checks at all.
- `nvidia/x/cuptiactivity` needs a prefix read. It pins whole 13.2
  runtime records, which is why 13.3 and 13.4 runtimes fail its size
  check, and that failure is unrelated to the agent's version policy.
  The fix is to decode the compiled prefix and admit a larger runtime
  record, not to pin CUPTI to 13.2.
- The lifecycle gate is not established. The existing examples use
  unchecked deferred cleanup and do not demonstrate leak-free repeated
  in-process reuse.
- The executed examples are not evidence of anything native. Measured
  2026-09-14 on dgx (linux/arm64): `go test ./... -run '^Example'`
  produced the same 196 passing examples, with no skips and no
  failures, against every toolkit tried. Examples carrying an `Output:`
  directive print Go values and error strings and do not reach a CUDA
  library, so a green example matrix says nothing about runtime
  compatibility.

  Label matrix cells by what the loader mapped, not by the
  `LD_LIBRARY_PATH` that was set. Every installed toolkit is in this
  host's `ld.so.cache`, and a soname absent from the named directory
  falls back to the cache silently. A cell pointed at a nonexistent
  directory mapped CUDA 13.4's cuBLAS, cuFFT and CUPTI together with
  `/lib/aarch64-linux-gnu/libnvvm.so.4`: not an absence control and not
  a coherent toolkit. A cell pointed at CUDA 12.6 took only
  `libcurand.so.10` from 12.6 and fell back to 13.4 for the four
  `.so.13`/`.so.12` sonames 12.6 does not ship, so cross-major cells are
  structurally invalid rather than merely unqualified. Within CUDA 13
  the pin holds: 13.0 through 13.4 mapped five distinct library sets
  reporting `cuptiGetVersion` 130001, 130101, 130202, 130301 and 130400.
  `Path()` cannot be used for this, because it returns the soname handed
  to `dlopen` and was byte-identical in all five cells; read
  `/proc/self/maps` or the library's own version call.

- The version-sensitive population is the hardware tier behind the
  `*_physical` tags, and it does discriminate. Measured 2026-09-14 on
  dgx with `cublas`, `cublaslt`, `cufft`, `curand`, `nvjitlink`,
  `nvfatbin`, `nvvm`, `cupti` and `nvml` `_physical` tags, `go test
  ./... -count=1 -p 1`, one cell per toolkit:

  | cell | cuptiGetVersion | PASS | SKIP | FAIL |
  |---|---|---|---|---|
  | 13.0 | 130001 | 1508 | 134 | 8 |
  | 13.1 | 130101 | 1508 | 133 | 9 |
  | 13.2 | 130202 | 1508 | 133 | 9 |
  | 13.3 | 130301 | 1508 | 133 | 9 |
  | 13.4 | 130400 | 1508 | 133 | 9 |

  Exactly one result moves across the band. `nvvm.TestCompileAndRun`
  passes on 13.0 and fails on 13.1 through 13.4 with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at module load, after compiling
  1048 bytes of NVVM IR into 830 bytes of PTX for `sm_121`. The host
  driver is 580.173.02; this is a driver ceiling on the PTX that newer
  libNVVM emits, not a binding defect. The other eight failures are
  version-invariant and unrelated to toolkit selection.

Claims remain limited to the exact qualified variant cells and their
stated exclusions. Fixing loader diagnostics, adding an admission
mechanism and passing lifecycle tests would not by themselves qualify
every package or operation in the module. An admission layer is a
design candidate, not an architecture earned by the header histogram,
and its future existence is not approval to widen sonames.

## Release line

The first tag is `nvidia/v0.1.0`. Every later release stays on the
`v0.1.x` line. The version encodes no NVIDIA number (see "Mixing"
below); `upstream.json` records which releases each package spans.

Each generated package names a floor and a ceiling. The ceiling is the
newest pinned release; the floor is the oldest release the package
promises to load against:

| Package kind | Floor | Decided |
|---|---|---|
| `driver`, `nvml` | CUDA 12.2 (driver r535) | 2026-09-30 |
| other libraries shipped in the CUDA toolkit | CUDA 12.2, or the first toolkit release that shipped the library | 2026-09-30 |
| libraries versioned separately (cuDNN, NCCL, DCGM, NVSHMEM, cuTENSOR, cuDSS, the cuQuantum libraries, nvCOMP, gdrcopy, …) | the version generated for `v0.1.0` | 2026-10-04 |

History and floor are separate. The generator also reads pinned headers
of releases older than the floor, so the API changelog and the
`since=` labels say when each declaration appeared. For separately versioned
libraries that history is backfilled from past releases (decided
2026-10-04; scheduled for v0.1.1, so v0.1.0 labels them since the floor). It changes labels only: plain names keep their meaning at
the floor, and the loader never admits a release below the floor, so a
`since=` older than the floor is informational.

The loader tries soname majors newest first, admits a library only when
its reported version is at or above the floor, and gates on the
installed driver. A symbol that an admitted older library lacks returns
`*SymbolError` at call time; `HasFoo` reports it in advance.

Moving a ceiling is additive by construction, and the generator
enforces it rather than a reviewer:

- `CheckAPIRemoval` refuses a regeneration that removes an exported
  name or changes its signature.
- `CheckFrozenNames` refuses one that changes the C symbol a Go name
  means, even when the signature is unchanged. Every generated package
  commits `frozennames.txt` before `v0.1.0` (decided 2026-10-03).
- `-allow-api-removal` overrides both. Using it is a release decision
  and the diff of `frozennames.txt` records it.

So a regeneration onto a newer CUDA, cuDNN or NCCL release is a patch
tag on this line, and so are generator, documentation and x/ fixes. A
change that needs `-allow-api-removal` is not a patch. How such a change
is numbered (a `v0.2.0`, with or without a bridge, see "Fix directives")
is not yet decided; until it is, it waits.

Lowering a floor would change what plain names mean (see the next
section), so it is treated the same way as a removal.

x/ is exempt from the compatibility promise. It is hand-written and
changes at will in patch releases the way golang.org/x does. x/ never
blocks a regeneration. The exemption is stated in every importable x/
package's doc comment before `v0.1.0` (see "The unit of versioning").
A green x/ means every test passes, or skips with a stated reason, on
all three qualification hosts: L4 at the floor driver, L4 at the
ceiling, and GB10 (decided 2026-10-04).

Promotion to `v1.0.0` happens when the naming rules below have held
through several ceiling moves with no `-allow-api-removal`. After v1 a
break is a `/v2` module path, which is the outcome to avoid.

### Set aside: encoding the CUDA release in the version

Two alternatives were considered on 2026-09-30 and set aside "for now":
a minor that carries the CUDA number (`v0.13.400` for CUDA 13.4), and
backfilled tags, one per CUDA minor from the floor up. Either would
mean regenerating each historical release as its own tag. The path stays
open: the first public tag fixes the numbering, and until then nothing
is published (`proxy.golang.org` has no versions of this module).

## Naming versioned C functions

NVIDIA versions C functions in two ways, and the generator treats them
differently. The rule applies to every generated package (decided
2026-10-03 and 2026-10-04).

**Macro redirects** (`#define nvmlInit nvmlInit_v2`, `cuCtxCreate` →
`cuCtxCreate_v4`): one C name, one Go name. When every version has the
same Go signature, the plain name binds at load time to the newest
symbol the installed library exports, newest first, at no per-call cost.

**Explicit siblings** (`nvmlDeviceGetAccountingStats` next to
`nvmlDeviceGetAccountingStats_v2`, usually with different structs):

- The plain Go name binds what the plain C name means at the floor: the
  `#define` target there, else the declaration there. A family absent
  from the floor is evaluated at the earliest release that declares it.
- Every version also gets an explicit name: `FooV1` for the unsuffixed
  symbol, `FooV2` for `_v2`, and so on. The suffix is spelled `V2`, not
  `_v2`.
- Every version has a `HasFooVn` predicate. There is no generated or
  hand-written fallback that calls V2 and converts to V1 shapes; the
  fields do not map mechanically. A caller that supports old and new
  drivers branches:

      if nvml.HasDeviceGetVgpuSchedulerStateV2() {
          err = dev.VgpuSchedulerStateV2(&s2)
      } else {
          err = dev.VgpuSchedulerState(&s)
      }

- A plain name never moves once `frozennames.txt` records it. When
  NVIDIA adds `_v3`, the generator adds `FooV3` and `HasFooV3`; the
  plain name and `FooV2` are unchanged.

A family whose floor ABI the ceiling header no longer declares
(`cuCtxCreate`, the edge-data graph functions) gets a generated shim
that keeps the newest Go signature and binds the old symbol at load.
Zero values for the new-only arguments route to the old symbol; any
other value returns an unsupported error.

### Deprecation docs

Decided 2026-10-04:

- A deprecation is real when NVIDIA marks it `DEPRECATED(x.y)` or says
  the declaration "is deprecated" or "has been deprecated". It gets a Go
  `Deprecated:` paragraph, last in the doc comment, saying since when
  and linking the replacement:

      // Deprecated: Deprecated in CUDA 13.0. Use [Device.VgpuSchedulerStateV2] instead.

  Restating the deprecation without either fact is not allowed. When
  NVIDIA gives neither, the paragraph is `Deprecated: Deprecated by NVIDIA.`
- An announced deprecation ("will be deprecated in a future release")
  is not one. It gets a plain paragraph, spelled exactly `Note:` so that
  go/doc does not collect it as a `NOTE(x):` marker:

      // Note: NVIDIA plans to deprecate this function in a future release.
      // Use [Device.VgpuSchedulerStateV2] instead.

  When a later header makes the deprecation real, the generator turns
  the note into `Deprecated:`.

Under this rule the plain nvml vGPU scheduler methods carry `Deprecated:`
(NVIDIA deprecated them in 13.0) and point at their V2 forms, which
need a 13.2 or later driver. A name is never marked deprecated only
because a newer version exists.

### Deciding the bump, mechanically

`make -C nvidia version-check` computes the next version from three
inputs and prints the tag commands. The name is deliberately not
`release-check`: that target exists at the repository root, means
something else, and is owned by `../RELEASE.md` -- it gates license,
module paths, replacements, placeholder versions, declared platforms and
tagged test packages, and says of itself that it "does not prove
dependency availability or ABI compatibility". `version-check` is the
API-and-version gate for this module alone. Neither subsumes the other,
and a release runs both. Every `version-check` named below is the
nvidia-local gate; where a repo-wide CI job is meant, the root
`release-check` runs alongside it.

`version-check` is specified by what it must read:

- It fails when the working tree is dirty, so a verdict always names a
  revision. This tree is shared; a gate that reads uncommitted work
  reports on a state no tag will ever have.
- It runs apidiff against the retained export data of the last tag and
  keeps both halves. The incompatible set decides the bump; the
  compatible set feeds the changelog. `../api/nvidia.log` was taken with
  `-incompatible` alone, which is why it cannot show the 1139 compatible
  entries this release also carries.
- It writes the new export data (`apidiff -m -w`) as a release artifact
  beside the tag, so the next run's baseline does not depend on a scratch
  directory surviving.
- It diffs the bound-symbol table, which apidiff cannot see.
- It is advisory on `x/`: those entries print under their own heading and
  are excluded from the bump decision, per the exemption above.

`nvidia/Makefile` exists now and implements the first clause and the
incompatible half of the second; the rest is still done by hand, so each
verdict in this document should be read with the revision it was taken
at.

The three inputs:

| Signal | Result |
|---|---|
| apidiff compatible, `frozennames.txt` lines only added | patch bump on `v0.1.x` |
| apidiff incompatible, or a `frozennames.txt` line removed or changed | fail: needs `-allow-api-removal` and a release decision |

apidiff is run as `apidiff -m -incompatible` against export data of the
last tag written with `apidiff -m -w` from the module cache. The
bound-symbol table (`cuMemAlloc` → `cuMemAlloc_v2`) is diffed alongside:
a moved binding of a frozen name is refused, and any other move is a
changelog line even though apidiff is silent on it.

A rename with a stable value is the case only apidiff sees. CUPTI
respelled two `CUpti_ActivityAttribute` enumerators between 13.2 and
13.3 and kept their values (20:
`..._ENABLE_MULTI_SUBSCRIBER_GRAPH_LEVEL_TRACE` became
`..._ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE`; 23: `..._CIG_MODE` became
`..._ENABLE_CIG_MODE`; measured 2026-09-14 on dgx). A constant-drift
check passes before and after. Under the release line the old Go name
cannot be dropped, since `CheckAPIRemoval` refuses it, so libgen keeps
it as a deprecated alias of the new one.

Nothing is published yet. `git ls-remote --tags origin` lists no
`nvidia/` tag, and `proxy.golang.org` has no versions of this module.
Until the first push to
a public remote, the API can still change without cost to anyone.

Rules that keep regenerations additive:

- Deprecated declarations stay bound and marked `Deprecated:`
  (`-deprecated=true` is the default). A symbol leaves only when the
  header drops it.
- Symbols missing from an older installed library return
  `*SymbolError` at call time. That covers *absence* only. It does not
  cover a symbol that is present with a divergent record layout,
  signature or constant value, which is the failure mode "Supported
  runtimes" exists to bound. The minimum tested driver is stated in the
  release notes, not enforced in code, and a release note states the
  supported runtime majors per platform rather than implying that any
  older library works.

## cudart and driver

`driver` binds cuda.h to libcuda. There is no runtime-API package yet;
when one is added it is `nvidia/cudart`, binding cuda_runtime_api.h to
libcudart, a sibling of `driver` and never merged into it:

- **Two libraries, two owners.** libcuda ships with the display driver
  and may be years older than the toolkit; libcudart ships with the
  toolkit and may be absent entirely. The load model is one library per
  package, with one `Load`, `Loaded`, `Path`, `Close`.
- **Two error types and a colliding vocabulary.** `cudaError_t` versus
  `CUresult`, `Malloc` beside `MemAlloc`, about 400 runtime functions
  that mirror driver ones under other names.
- **Interop is by alias.** `cudaStream_t` and `CUstream` are the same
  object, so cudart declares `type Stream = driver.Stream` and friends
  and values cross without conversion, the same cross-package type
  resolution cuBLAS and cuDNN already use.

For versioning they move together. Both headers come from the
`cuda_cudart` redist package, so `upstream.json` pins them to one
version, and moving the toolkit ceiling regenerates both. The difference shows at
runtime, not in tags: missing driver symbols return `*SymbolError`,
which alone does not establish compatibility with an older installed
libcuda. `cudart` is not currently bound. Any future cudart support
must state its runtime-major and workload contract separately from its
generation source; loading `libcudart.so.13` because the ceiling is 13 is
a possible single-major policy, not a consequence of versioning or of
the simultaneous-support goal.

Priority: after pinning and provenance. From pure Go, nvrtc plus the
driver's module loading already covers what the runtime's implicit
context gives nvcc-built programs; cudart earns its keep for interop
with libraries that hand out `cudaStream_t` and for conveniences like
device properties.

## Fix directives between versions

go-iroh v0.1.0 shipped its rename as a two-step upgrade: a bridge tag
(v0.0.2, cut on a `release/v0.0` branch after v0.1.0's API was final)
carried both the old and the new names, with every old name marked
`Deprecated:` and `//go:fix inline` and forwarding to the new one. The
release notes told consumers to `go get` the bridge, run
`go fix -inline ./...`, then `go get` the real release. The next release
deleted the shims. The same shape works here, and generated code makes
it cheaper because the generator knows the rename map.

### What can be inlined

`go fix -inline` rewrites call sites of a function whose body is a
single return of a call, a type alias, or a constant. That covers the
three kinds of churn this module produces:

| Churn | Shim | Source of the mapping |
|---|---|---|
| generator naming rule change (receiver moves, `New` prefix, enum prefix stripping) | `//go:fix inline` func or alias | the generator's rename map, run under old and new rules |
| NVIDIA versioned symbol (`cuFoo_v3` appears) | none: the plain name stays put and `FooV3` is added (see "Naming versioned C functions") | `frozennames.txt` diff |
| NVIDIA renames or retypes a declaration | inline shim when the new signature accepts the old arguments; otherwise `Deprecated:` only | apidiff between the two header generations |

Only the first kind is fully mechanical. The third needs a human to
write the shim body when arguments change meaning; the generator emits
the skeleton with a `// TODO(bridge)` and the release check fails
until it is filled or removed.

### How the generator produces them

`gen-nvidia -bridge <old export data>` takes the apidiff export data of
the last tag (`apidiff -m -w`), generates under current rules, and for
every exported identifier present in the old data and absent in the
new, emits into `<pkg>/legacy.gen.go`:

```go
// MemAllocPitch is the pre-v0.3 name for [DevicePtr.MemAllocPitch].
//
// Deprecated: use [DevicePtr.MemAllocPitch].
//
//go:fix inline
func MemAllocPitch(w, h, n uint) (DevicePtr, uint, error) {
	return DevicePtr(0).MemAllocPitch(w, h, n)
}
```

`legacy.gen.go` is a normal generated file with a build tag of its own
(`//go:build !nolegacy`) so a consumer who has already migrated can
prove they no longer depend on it. The file exists only on bridge tags.

### Release shape

No break is planned on the release line (see "Release line"). If one is
ever approved, it would produce two tags, in this order:

1. `nvidia/v0.2.0` on main: new API, no shims.
2. `nvidia/v0.1.N+1` on branch `release/v0.1`: the last v0.1 tag plus
   the new API plus `legacy.gen.go`. It must contain the new API, since
   the shims call it. This is the bridge.

The release notes would carry the Before/After table (generated from
the same rename map) and the upgrade recipe, verbatim from go-iroh:

```sh
go get github.com/tmc/lib/nvidia@v0.1.N+1
go fix -diff -inline ./...
go fix -inline ./...
go get github.com/tmc/lib/nvidia@v0.2.0
go mod tidy
go build ./... && go test -count=1 ./...
```

`-inline` is required; plain `go fix` also runs unrelated modernizers.

## Mixing our versioning with NVIDIA's

Keep the Go version ours and expose NVIDIA's everywhere else. The
module version cannot carry an NVIDIA number without breaking
something:

- Major = CUDA major (`nvidia/v13.3.0`) forces a `/v13` module path and
  a path change on every CUDA major. cuDNN and NCCL do not share the
  number, so it is also wrong for most of the module.
- Pre-release suffixes (`v0.1.0-cuda13.3`) sort below the release and
  are skipped by `@latest`. Build metadata (`+cuda13.3`) is not
  recognized in tags at all.
- A CUDA-named branch (`@cuda-13.3`) no longer names anything: one
  revision spans every release from the floor to the ceiling. A Go
  `@branch` query also resolves to the branch head, not to the newest
  tag on it, so it would not give a reproducible pin.

What the module exposes instead:

- **The manifest.** `upstream.json` records, per package, the floor,
  the ceiling and every pinned header in between. The release notes
  title is generated from it:
  `nvidia v0.1.0 (CUDA 12.2–13.4, cuDNN 9.x, NCCL 2.x)`.
- **Header constants.** A package exports its ceiling header's version
  macro as `HeaderVersion`. cublasmp, gdrcopy, nvcomp and nvjpeg2k do
  today; `driver` (`CUDA_VERSION`), `cudnn` and `nccl` should. Callers
  compare it with the runtime `GetVersion` result, which is what NVIDIA
  intends the macros for.
- **Per-call requirements.** A name whose symbol postdates the floor
  says "Requires CUDA X.Y or later" in its doc, and has a `Has`
  predicate.

## Publishing

Order, because the proxy is immutable and dependents pin by tag:

1. cuda-headers: manifest bump commit, tag `cuda-13.4`, push.
2. tmc/ffi, then libgen: any runtime or generator changes, commit, push., commit, push. Record the commit in
   `upstream.json`.
3. lib: regeneration commit, `version-check`, tag `nvidia/v0.X.Y`, push
   the commit and the tag together.
4. lib: bump `gonccl` (and `mgccl` transitively) to require the new
   nvidia tag, `go mod tidy`, tag them, push.
5. `go work sync` at the lib root only after step 3's tag is fetchable.

A CI job on lib runs `make check`, per-module build and test, and
`version-check` against the last tag on every push to main, so the
API-diff verdict is visible before anyone tags.

## Changelog

Generate it; do not hand-write it. `nvidia/CHANGELOG.md` is regenerated
from three inputs between two tags, oldest section first:

- `upstream.json` at each tag: per-package NVIDIA version deltas
  ("driver: CUDA 13.2.51 → 13.3.29").
- `apidiff` between the tags, split by package: added, removed, and
  changed exported identifiers. Incompatible changes are listed first.
- The bound-symbol table (name → soname symbol) at each tag, for
  versioned-symbol moves.

Commit subjects are appended as an "Other" section filtered to the
`nvidia/` path, for hand-written x/ work that apidiff cannot describe.

A small Go program in codegen, `cmd/nvidia-changelog`, takes two tags
(or one tag and the working tree), does the three diffs, and prints
Markdown. `version-check` runs it and inserts the result under an
`## Unreleased` heading; tagging renames the heading. This is the
optional piece; everything above stands without it.

## The maintainer's loop

Everything a ceiling move needs, in order, with one command per step:

```sh
# 1. cuda-headers: pin the new release beside the old ones and commit.
make -C ../cuda-headers update CUDA=13.5     # adds the release, fetches
git -C ../cuda-headers commit -am 'cuda 13.5' && git -C ../cuda-headers tag cuda-13.5

# 2. lib/nvidia: regenerate with the new ceiling, test, and check.
make -C nvidia regen HEADERS=../cuda-headers   # go generate + gofmt + vet + host tests
make -C nvidia test-gpu                        # floor and ceiling hosts
make -C nvidia version-check                   # apidiff + frozen names + changelog draft

# 3. Commit and tag what version-check printed.
git commit -am 'nvidia: raise the ceiling to CUDA 13.5'
git tag nvidia/v0.1.N && git push origin main nvidia/v0.1.N

# 4. Dependents.
make -C gonccl bump NVIDIA=v0.1.N && git tag gonccl/v0.x.y
```

A fix release is steps 2 and 3 with no header change. CI runs `regen`
in diff mode and `version-check` on every push to main, so the verdict
is known before tagging and a stale regeneration cannot land unnoticed.

Time budget per ceiling move, once the tooling exists: under an hour.

## First release: nvidia/v0.1.0

Rev 2 planned this release as `v0.2.0`; it was renamed `v0.1.0` on
2026-09-29 when the `rel/v0.2.0` branch was retired. No `nvidia/` tag
exists on any remote. The release is tagged from a signed, restamped
candidate commit, and only when all of these hold for that exact commit:

1. Every generated package reproduces its committed bytes from the argv
   recorded in `upstream.json`, and the provenance includes every header
   the parse opened, transitively (CUPTI recorded only its root headers
   until this was fixed).
2. Every generated package commits `frozennames.txt`, under the naming
   rule above.
3. Deprecation docs follow the rule above: no empty restatements, no
   `Deprecated:` on an announced deprecation.
4. apidiff against published main `1e04c035f` is reviewed per package
   and each incompatible entry is labeled: an NVIDIA header change, a
   naming-rule change, or a fix. Nothing is left unexplained.
5. The physical tests pass on L4 at the r535 floor and on an r610+
   ceiling host, and x/ is green on all three hosts as defined above.
6. Every importable x/ package states the exemption in its doc comment.
7. Each package commits `api.gen.txt`, and `CUDA-API.md`, the per-release
   API changelog, is generated from them with its checks passing, with
   toolkit history from the 12.2 floor (decided 2026-10-04). The
   backfill for separately versioned libraries, complete `since=`
   labels for them, and data-driven NVRTC option checks wait for v0.1.1:
   landing them later changes neither `frozennames.txt` nor apidiff,
   which is the test for deferring anything. Names present at the floor
   and removed before the ceiling stay bound, with a `Deprecated:` paragraph saying when NVIDIA
   deprecated and removed them; their `Has` predicate reports false where
   the symbol is absent.

When every gate holds, the release report gives the exact tmc/ffi,
libgen and lib hashes and the tag, and work stops there. Nothing is tagged or pushed without
tmc's go (decided 2026-10-04).

Earlier decisions that still hold: host buffers are never `uintptr`;
every commit builds the whole repository; separately versioned
libraries refresh in their own commits; the compatibility policy moves
into the `nvidia` package doc, and this file stays a working note.

API decisions from the review of the first candidate (decided
2026-10-05), all made before the tag because each changes exported
names:

- Generated enum types get Go names, not C typedef spellings
  (`CublasoperationT`); a package that reuses another package's enum
  aliases it (`type Operation = cublas.Operation`).
- Device memory is `driver.DevicePtr` in every package.
- Getters return their value: `cublaslt.GetProperty` returns
  `(int32, error)`.
- Every generated wrapper folds the library status into `error`;
  nvdec's `(Result, error)` results go.
- A type with one handle-only release function gets `Close() error`
  and no `Destroy`, including when the C function takes a pointer to
  the handle. No `Deprecated: use Close` shims.
- The package-level library teardown is `UnloadLibrary()`, not
  `Close()`.
- Methods drop `Get` consistently: `driver.Array.Descriptor3D()`.
- Generated names keep the C casing of initialisms (`FieldId`,
  `GpuId`, `NewDeviceByPciBusId`). A rename touches about 1,210 names
  and every caller, so it was declined; the rule is stated here so it
  is a choice rather than an accident.

## What to do first

1. Bring the candidate up to date with main and apply the review
   decisions above in the generators, fix transitive header capture
   in `internal/nvidiaprovenance`, then regenerate every package
   with its recorded recipe, restamp, and run the reproduce gate.
2. Fix the plain names that bind `_v2` where the floor declares v1
   (`nvml` memory info, fan speed, retired pages, power limit and CUDA
   driver version; `driver` `MemAdvise`, `MemPrefetchAsync` and
   `DeviceGetUuid`).
3. Run the gates above on the signed candidate, including the hardware
   gates.
4. Tag `nvidia/v0.1.0` and publish in the order under "Publishing"
   (tmc/ffi before lib), on tmc's go.
