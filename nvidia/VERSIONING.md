# Versioning the nvidia module against NVIDIA releases

Status: proposal (2026-09-01). Companion to ../VERSIONING_PLAN.md, which
covers the repo-wide per-directory module split. This document covers the
one module that tracks a moving upstream: how we pin NVIDIA inputs,
regenerate, decide a version, tag, and describe what changed.

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
- `x/` (hand-written, 62 of 78 nvidia commits to date) lives in the same
  tag line. If its churn starts forcing tags that regenerate nothing,
  split `nvidia/x` into its own module then; not before.

## Problems in the current pipeline

1. `cuda-headers/fetch.sh` resolves "latest" per package at fetch time.
   The checked-out tree is already a mix: cudart 13.3.29 beside cupti
   13.3.75 beside cccl 13.2.86. Nothing records which archive produced
   which directory, so a regeneration is not reproducible.
2. Generated files stamp only the header basename ("from cuda.h"). The
   lib tree carries no upstream version and no input hash, so a
   consumer cannot tell whether `driver` reflects CUDA 13.2 or 13.3.
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
directive per package invoking gen-nvidia with `-headers-dir` pointing
at a cuda-headers checkout at the manifest's tag. A Makefile target
`make -C nvidia regen` wraps it and then:

1. `gofmt -l` and `go vet ./...` in the module.
2. `go test ./...` on the host (darwin runs the layout and loader tests;
   this is the cross-platform guarantee).
3. `go test ./...` on the dgx box for the runtime tests. See the
   gpu-test-targets note: dgx is the CUDA machine.
4. Rewrites `upstream.json`.

A regeneration commit touches only `*.gen.go`, `upstream.json`, and the
go:generate file, with subject `nvidia: regenerate from CUDA 13.3`. Hand
fixes in the generator go in codegen first and are re-run, never
patched into lib.

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
  becomes the patch-only line for anyone still on 13.
- **x/ is exempt from the compatibility promise.** It is hand-written,
  experimental by charter (x/DESIGN.md), and changes at will in patch
  releases the way golang.org/x does. Each x/ package doc states this.
  x/ never causes a bridge and never blocks a train.

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

### Deciding the bump, mechanically

`make -C nvidia release-check` computes the next version from three
inputs and prints the tag commands:

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

Rules that keep trains additive:

- Deprecated declarations stay bound and marked `Deprecated:`
  (`-deprecated=true` is the default). A symbol leaves only when the
  header drops it.
- Symbols missing from an older installed library return
  `*SymbolError` at call time, so a newer train does not break users
  on older drivers. The minimum tested driver is stated in the release
  notes, not enforced in code.

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
runtime, not in tags: `driver` tolerates an old installed libcuda via
`SymbolError`, while `cudart` should be loaded with the soname matching
its train (`libcudart.so.13`) since the runtime ABI is toolkit-major
scoped.

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
| NVIDIA versioned symbol (`cuFoo` → `cuFoo_v3`, same Go name) | none needed; bound symbol moves | bound-symbol table diff |
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
3. lib: regeneration commit, `release-check`, tag `nvidia/v0.X.Y`, push
   the commit and the tag together.
4. lib: bump `gonccl` (and `mgccl` transitively) to require the new
   nvidia tag, `go mod tidy`, tag them, push.
5. `go work sync` at the lib root only after step 3's tag is fetchable.

A CI job on lib runs `make check`, per-module build and test, and
`release-check` against the last tag on every push to main, so the
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
Markdown. `release-check` runs it and inserts the result under an
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
make -C nvidia release-check                   # apidiff + symbol diff + changelog draft

# 3. Commit and tag what release-check printed.
git commit -am 'nvidia: regenerate from CUDA 13.4'
git tag nvidia/v0.2.0 && git branch cuda-13.4 && git push origin main cuda-13.4 --tags

# 4. Only if release-check said "bridge required":
git checkout cuda-13.3                         # the previous train's branch
make -C nvidia bridge FROM=nvidia/v0.1.3       # emits legacy.gen.go against v0.2.0
git commit -am 'nvidia: v0.1.4 migration bridge' && git tag nvidia/v0.1.4 && git push --tags

# 5. Dependents.
make -C gonccl bump NVIDIA=v0.2.0 && git tag gonccl/v0.x.y
```

A patch release is steps 2, 3 with no header change. CI runs `regen`
in diff mode and `release-check` on every push to main, so the verdict
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
   `v0.1.0` was built from), run `release-check` against `nvidia/v0.1.0`,
   tag `nvidia/v0.1.1` if additive, branch `cuda-13.3`, bump gonccl.
   `v0.1` is the CUDA 13.3 train; `v0.2.0` is reserved for CUDA 13.4 or
   for an incompatible diff, whichever comes first.
5. Mark every x/ package experimental in its doc comment.
6. gen-nvidia `-bridge`, `HeaderVersion` constants, `TRAINS.md`.
7. cmd/nvidia-changelog, when time allows.
