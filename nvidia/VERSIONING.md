# Versioning the nvidia module against NVIDIA releases

Status: proposal, rev 2 (2026-09-06). Companion to ../VERSIONING_PLAN.md,
which covers the repo-wide per-directory module split. This document
covers the one module that tracks a moving upstream: how we pin NVIDIA
inputs, regenerate, decide a version, tag, and describe what changed,
and which installed runtimes the result is allowed to bind.

Rev 2 separates two things rev 1 ran together: the *generation source*
(one coherent CUDA release per train) and the *supported runtime* (what
an installed host may present). "Supported runtimes" below is the only
ratified part of this document; the tag mechanics remain a proposal.
Its platform scope was ratified by tmc on 2026-09-06 and its evidence
comes from the 2026-09-06 CUDA 12/13 measurement campaign. The workload
matrix is deferred and is an output of this document, not an input.

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
  trains" below exempts it from the compatibility promise. Those two
  facts do not sit together: a consumer reading `nvidia/v0.2.0` cannot
  tell from the version whether a break reaches them, because the tag
  covers a surface the tag promises nothing about. The 38 `x/` entries in
  `../api/nvidia.log` are the first time that had to be argued out of a
  bump by hand, and the first draft of that argument got it backwards.

  The earlier trigger for splitting -- "if its churn starts forcing tags
  that regenerate nothing" -- is retired because it cannot fire. One
  nvidia tag has ever been cut, so no tag has been forced by anything,
  and the predicate stays false however large the churn gets. Split
  `nvidia/x` into its own module at the next train, so that "changes at
  will" becomes something a version can say. A module-path change is
  breaking, but it is breaking for nobody until the first public tag, so
  the split is cheapest now and never cheaper later. Until then the
  exemption is carried by this file alone. It is not carried by
  `x/DESIGN.md`, which is excluded by `.git/info/exclude`, is in no
  clone, and does not discuss API stability in any case; nor by the x/
  package docs, 45 of the 49 importable ones saying nothing about it.
  See "Release trains".

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
4. There is no pre-tag API gate. `apidiff` is installed but not wired in.

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
3. `go test ./...` on the dgx box for the runtime tests. See the
   gpu-test-targets note: dgx is the CUDA machine. Report skips
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

Generation source stays single: one coherent CUDA release per train, as
"Release trains" describes. Supported runtime is wider on one
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

## Release trains: how the Go version aligns with NVIDIA's

The Go version cannot encode NVIDIA's numbers (see "Mixing" below), but
it can *track* them. The rule:

**One Go minor per CUDA toolkit minor. Everything else is a patch.**

| Go tag | Train | What it contains |
|---|---|---|
| `nvidia/v0.1.0` | CUDA 13.3 | regeneration of every toolkit package from the 13.3 redist set, plus the latest cuDNN, NCCL, cuTENSOR, cuFile, Fabric Manager at train time |
| `nvidia/v0.1.1` … | CUDA 13.3 | generator fixes, x/ fixes, docs refresh, additive refresh of a non-toolkit library (new cuDNN minor) |
| `nvidia/v0.2.0` | CUDA 13.4 | next regeneration |

Consequences that make this manageable for one maintainer:

- **A train is the only event that can break API.** Between trains,
  regeneration from a newer non-toolkit library is allowed only when
  apidiff reports it additive; otherwise it waits for the next train.
  A consumer on v0.2.x never needs `go fix`.
- **Bridges exist per train, at most.** A breaking train `v0.2.0` gets
  a `v0.1.<last+1>` bridge on the `cuda-13.3` branch. Most trains will
  be additive (NVIDIA rarely removes declarations inside a major) and
  need none.
- **Trains are quarterly at most**, following NVIDIA's toolkit minor
  cadence. Patches are whenever.
- **The mapping is a table, not a formula.** `TRAINS.md` (generated
  from `upstream.json` at each tag) lists Go minor ↔ NVIDIA versions,
  and a `cuda-13.3` branch follows each train so
  `go get ...@cuda-13.3` works (see "Mixing" below).
- **CUDA major bumps** (13 → 14) are still just a train. The last
  CUDA 13 train's branch (`cuda-13.4`) simply stops fast-forwarding and
  becomes the patch-only line. That is a generation-source and Go-API
  choice about which branch receives regenerations; it does not by
  itself retire runtime support for the older major, which changes
  only through an explicit change to the contract in "Supported
  runtimes".
- **x/ is exempt from the compatibility promise.** It is hand-written
  and changes at will in patch releases the way golang.org/x does. x/
  never causes a bridge and never blocks a train. The exemption is
  declared here and nowhere else, which is a defect to close before the
  tag rather than a citation to make. Two earlier authorities for it do
  not hold: `x/DESIGN.md` is a package map, ownership rules and
  validation rules, and says nothing about API stability, so "by
  charter" cited a document that does not contain the charter; and of
  the 49 importable x/ packages, 45 say nothing about stability in any
  .go file, while the four that use the word "experimental"
  (`cudagraph`, `cudalayer`, `cudamlx`, `mxfp4`) apply it to
  implementation maturity, not to compatibility. The denominator is
  every directory under `x/` holding .go files, at any depth, less
  `cmd/`, `testdata/` and `internal/`; internal packages are excluded
  because no consumer can import them, so no promise about them is
  owed. Both counts survive a wider search than "experimental" --
  `changes at will`, `semver`, `patch release`, `breaking change` and
  `unstable` add no genuine hit in either the packages or
  `x/DESIGN.md`. **Prerequisite for the tag:** every x/ package doc
  states the exemption, so that a consumer reading the package learns
  it without reading this file.

Promotion to `v1.0.0` happens when a train regenerates with no
incompatible diff for two consecutive trains and the generator's
naming rules are declared frozen. After v1 the rule tightens: a
breaking train is a `/v2` module, which is the outcome to avoid, so
v1 waits until the generator is stable.

### Deferred: encoding the CUDA release in the minor

Because a train is one Go minor, the minor could carry the CUDA number
directly: `v0.13.300` for the CUDA 13.3 train, `v0.13.301` for a fix on
it, `v0.13.400` for CUDA 13.4. Then `@v0.13` means "newest CUDA 13
bindings" with no branch involved. Plain `v0.13.3` does not work
because it leaves no patch slot.

Decision (2026-09-01): not adopted yet. Trains stay on small minors
(`v0.1`, `v0.2`, …) until the generator settles. The switch is one-way,
since versions only increase, so it is made deliberately at a train
boundary and never retracted. Until then the `cuda-13.x` branch is how
consumers ask by CUDA name.

The one-way constraint binds only after publication. Versions only
increase for a consumer who has resolved one, and none has --
`proxy.golang.org` 404s on `github.com/tmc/lib/nvidia/@v/list`, as
recorded under the CUPTI rename below. Before the first public tag the
numbering can be rewritten outright, so if the CUDA minor is wanted this
is the moment it is free.

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
| `upstream.json` toolkit version changed minor | new train: minor bump |
| apidiff incompatible on a non-train change | fail: wait for a train |
| apidiff incompatible on a train | minor bump plus bridge required |
| anything else | patch bump |

apidiff is run as `apidiff -m -incompatible` against export data of the
last tag written with `apidiff -m -w` from the module cache. The
bound-symbol table (`cuMemAlloc` → `cuMemAlloc_v2`) is diffed alongside:
a moved binding is a train-only change and a changelog line even though
apidiff is silent on it.

The 13.3 train carries one known rename, and a value-based check cannot
see it. CUPTI respelled two `CUpti_ActivityAttribute` enumerators between
13.2 and 13.3 while leaving their values alone. Measured 2026-09-14 by
reading `cupti_activity.h` from the installed 13.2, 13.3 and 13.4
toolkits on dgx:

| value | 13.2 | 13.3 and 13.4 |
|---|---|---|
| 20 | `..._ENABLE_MULTI_SUBSCRIBER_GRAPH_LEVEL_TRACE` | `..._ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE` |
| 23 | `..._CIG_MODE` | `..._ENABLE_CIG_MODE` |

`cupti/enums.gen.go` is frozen at the 13.2 spelling, so regenerating onto
the 13.3.1 profile renames two exported Go identifiers,
`ActivityAttrEnableMultiSubscriberGraphLevelTrace` and
`ActivityAttrCigMode`, and changes what `String()` returns for both.
Nothing in this repository names either identifier, so no build anywhere
fails. Both are in the tagged surface: `nvidia/v0.1.0` is the only nvidia
tag and its `cupti/enums.gen.go` carries both names, as declarations and
as `String()` literals. Because the values stay 20 and 23, a
constant-drift check passes before and after — apidiff, which compares
names, is the only check that reports it.

Nothing is published, which decides how much this costs. `git ls-remote
--tags origin` returns nothing, and `proxy.golang.org` answers 404 for
both `github.com/tmc/lib/nvidia/@v/list` and `v0.1.0.info`. The tag does
exist on the private `dgx` remote, at the same commit, alongside 15 other
module tags, so it is not local-only and rewriting it means force-pushing
there — but no consumer can have fetched it through the module proxy. So
this is a break against a tag, not against a released API, and the
bridge-tag apparatus in "Release trains" is not owed to anyone for this
transition. That stays true only until the first push to a public
remote.

Follow the header: the bindings mirror the C API, and holding a
superseded spelling would misreport what 13.3 and later expose. Ship the
rename as a named breaking change in the v0.2.0 changelog rather than
aliasing it.

These two are changelog entries, not a bump trigger, and the passage
above should not be read as though they decided anything. The v0.2.0
surface is already incompatible with `nvidia/v0.1.0` in 195 recorded
ways, and 157 of those are outside `x/`. Counted in `../api/nvidia.log`
on 2026-09-14: 93 `nvml` entries, 54 `driver` entries, all six
`cufft.Handle.Exec*` methods retyped from typed pointers to `uintptr`,
two in `cusparse`, and one each in `nvrtc` and `nccl`; plus a
struct-shape change on `nvml.RemappedRowsInfoV2`, which is also the
single `!` entry.

The other 38 are under `x/`: 15 whole-package removals and 23 symbols.
By the exemption stated above they neither add to this decision nor
subtract from it, so quote the 157 and list the 38 separately. An
earlier draft of this paragraph led with the 15 `x/` package removals,
which argued against the exemption two sections up. The bump is decided
on the 157 and written down in `../RELEASE.md`. The renames add to the
list; they do not create it.

The risk to manage is procedural, and it is narrow. `../api/nvidia.log`
has no `ActivityAttr` entry, but apidiff emits only differences, so that
records these identifiers as *unchanged* on 2026-09-13 rather than absent
from the baseline. They were unchanged because this tree still carries
the 13.2 spelling: the rename does not exist in our code yet. It appears
the moment someone regenerates against a 13.3 or later toolkit, and that
run is the only one where apidiff can see it. Since apidiff is run by
hand rather than by a gate, a regeneration committed without re-running
it ships the break invisibly.

That is the one thing here worth building: the post-regeneration apidiff
run belongs in the regeneration step itself, not in a person's memory.
Zero `ActivityAttr` matches in both `../api/nvidia.log` and
`../api/nvidia-linux.log` confirm no other artifact we keep can ever
show this class of change.

Rules that keep trains additive:

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

For versioning they are the same train. Both headers come from the
`cuda_cudart` redist package, so `upstream.json` pins them to one
version, and a toolkit minor regenerates both. The difference shows at
runtime, not in tags: missing driver symbols return `*SymbolError`,
which alone does not establish compatibility with an older installed
libcuda. `cudart` is not currently bound. Any future cudart support
must state its runtime-major and workload contract separately from its
generation train; loading `libcudart.so.13` because the train is 13 is
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
| NVIDIA versioned symbol (`cuFoo` → `cuFoo_v3`, same Go name) | usually none, but not automatically: a moved binding can change signature or record layout, or drop an export a supported older library still needs | bound-symbol table diff, plus a native contract and runtime-coverage review |
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

A breaking regeneration produces three tags, in this order:

1. `nvidia/v0.2.0` on main: new API, no shims.
2. `nvidia/v0.1.1` on branch `cuda-13.3`: v0.1.0 plus the
   new API plus `legacy.gen.go`. It must contain the new API, since the
   shims call it. This is the bridge.
3. Nothing further on the branch unless a bug fix needs backporting.

Release notes for v0.2.0 carry the Before/After table (generated from
the same rename map) and the upgrade recipe, verbatim from go-iroh:

```sh
go get github.com/tmc/lib/nvidia@v0.1.1
go fix -diff -inline ./...
go fix -inline ./...
go get github.com/tmc/lib/nvidia@v0.2.0
go mod tidy
go build ./... && go test -count=1 ./...
```

`-inline` is required; plain `go fix` also runs unrelated modernizers.

The bridge exists only when apidiff reports incompatible changes. A
regeneration that only adds symbols tags once.

## Mixing our versioning with NVIDIA's

The short answer: keep the Go version ours, expose NVIDIA's everywhere
else. The module version cannot carry an NVIDIA number without breaking
something:

- Major = CUDA major (`nvidia/v13.3.0`) forces a `/v13` module path
  suffix and a path change on every CUDA major. cuDNN and NCCL do not
  share the number, so it is also wrong for most of the module.
- Minor = CUDA minor (`v0.133.x`) reads as nothing and collides the
  first time a cuDNN-only regeneration needs a bump.
- Pre-release suffixes (`v0.2.0-cuda13.3`) sort below the release and
  are skipped by `@latest`. Build metadata (`+cuda13.3`) is not
  recognized in tags at all.

What does work, and what the plan adopts:

- **CUDA-named release branches.** `go get` accepts any git ref, and
  when the ref's commit carries a semver tag for the module the
  toolchain records that tag, not a pseudo-version. Verified against
  go-iroh: `go list -m github.com/tmc/go-iroh@release/v0.0` resolves to
  `v0.0.2`. Only semver refs get the `nvidia/` directory prefix, so the
  ref name is bare: branch `cuda-13.3`, and

      go get github.com/tmc/lib/nvidia@cuda-13.3

  resolves to the newest `nvidia/v0.1.x` tag on that branch and writes
  that semver into the consumer's go.mod. The user asks by CUDA name
  and gets a reproducible pin. The branch is created at the train and
  fast-forwarded to each patch tag while the train is current; once
  main has started the next train it moves only for backports. It is
  the maintenance branch and the marker in one ref, so there is nothing
  extra to keep in sync. A bare tag would work the same way but cannot
  follow patches.

  Because a train is one Go minor, the same line is also reachable by
  version prefix with no branch at all: `@v0.1` selects the newest
  `v0.1.x` tag (verified: `go-iroh@v0.0` resolves to `v0.0.2`).
  `@v0.2.x` is not a query and fails. `TRAINS.md` lists both spellings
  per train.
- **Constants.** Each package exports what its header defines:
  `driver.HeaderVersion = 13030` (from `CUDA_VERSION`),
  `cudnn.HeaderVersion` from `CUDNN_MAJOR/MINOR/PATCHLEVEL`,
  `nccl.HeaderVersion` from `NCCL_VERSION_CODE`. Consumers compare
  against the runtime `GetVersion` result to decide feature use, which
  is what NVIDIA intends the macros for.
- **Release title.** `nvidia v0.1.0 (CUDA 13.3, cuDNN 9.20, NCCL 2.29.7)`,
  generated from `upstream.json`.
- **Maintenance lines per CUDA major.** When the generation source moves
  from CUDA 13 to 14, the last CUDA 13 minor becomes a branch
  (its `cuda-13.x` branch) that accepts patch tags only. That is the one
  place NVIDIA's cadence shapes our tag graph: parallel lines exist per
  CUDA major, expressed as semver patch series, never as majors.

## Publishing

Order, because the proxy is immutable and dependents pin by tag:

1. cuda-headers: manifest bump commit, tag `cuda-13.3`, push.
   (Same name as lib's branch on purpose: one CUDA name, two repos.)
2. codegen: any generator changes, commit, push. Record the commit in
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

Everything a train needs, in order, with one command per step:

```sh
# 1. cuda-headers: move the manifest to the new redist set and commit.
make -C ../cuda-headers update CUDA=13.4     # rewrites versions.json, fetches
git -C ../cuda-headers commit -am 'cuda 13.4' && git -C ../cuda-headers tag cuda-13.4

# 2. lib/nvidia: regenerate, test, and check.
make -C nvidia regen HEADERS=../cuda-headers   # go generate + gofmt + vet + host tests
make -C nvidia test-gpu                        # runs go test on dgx
make -C nvidia version-check                   # apidiff + symbol diff + changelog draft

# 3. Commit and tag what version-check printed.
git commit -am 'nvidia: regenerate from CUDA 13.4'
git tag nvidia/v0.2.0 && git branch cuda-13.4 && git push origin main cuda-13.4 --tags

# 4. Only if version-check said "bridge required":
git checkout cuda-13.3                         # the previous train's branch
make -C nvidia bridge FROM=nvidia/v0.1.3       # emits legacy.gen.go against v0.2.0
git commit -am 'nvidia: v0.1.4 migration bridge' && git tag nvidia/v0.1.4 && git push --tags

# 5. Dependents.
make -C gonccl bump NVIDIA=v0.2.0 && git tag gonccl/v0.x.y
```

A patch release is steps 2, 3 with no header change. CI runs `regen`
in diff mode and `version-check` on every push to main, so the verdict
is known before tagging and a stale regeneration cannot land unnoticed.

Time budget per train, once the tooling exists: under an hour when
additive, an afternoon when a bridge is needed. That is the bar the
tooling is measured against.

## What to do first

1. cuda-headers: add `versions.json` and `.version` files, make
   `fetch.sh` honor them, tag the current coherent state once the 13.2 /
   13.3 mix is resolved to one release.
2. gen-nvidia: emit `upstream.json` and the per-package Source line.
3. lib/nvidia: add `generate.go` and the two Makefile targets.
4. Regenerate onto CUDA 13.3 (consolidating the 13.2/13.3 mix that
   `v0.1.0` was built from), run `version-check` against `nvidia/v0.1.0`,
   tag `nvidia/v0.1.1` if additive, branch `cuda-13.3`, bump gonccl.
   `v0.1` is the CUDA 13.3 train; `v0.2.0` is reserved for CUDA 13.4 or
   for an incompatible diff, whichever comes first.
5. Mark every x/ package experimental in its doc comment.
6. gen-nvidia `-bridge`, `HeaderVersion` constants, `TRAINS.md`.
7. cmd/nvidia-changelog, when time allows.

Remaining policy and qualification work; broader claims require the
relevant cells to pass.

8. Done for two packages (lib `8071942`, codegen `6de8d1a`): explicit
   adoption after a lazy load now clears `Implicit`, fixed in the
   emitter with `nccl` and `cufile` regenerated from pinned inputs and
   native lifecycle regressions added. Every other generated package
   acquires the fix on its own next regeneration and is not refreshed
   by that commit.
9. Decide, per executable variant, "works on both majors" or "fails
   cleanly on the one it does not support", and record the row. That
   list is the deferred workload matrix.
10. Establish the lifecycle gate on the examples: checked cleanup and
    leak-free repeated in-process reuse.
11. Only then consider a runtime ABI-profile admission layer. Do not
    widen sonames before it exists.
