# Release readiness

Status: blocked on license selection and dependency publication; not ready. Scope: all 14 library modules in `go.work`. `examples` is
built and tested for development but is not tagged. There is no root module.

The audit began at `75377ee0` on September 12, 2026, with Go 1.27.0 on
Darwin/arm64 (Apple M4 Max). Subsequent working-tree fixes are not committed.
Receipts below are local files under `~/tmp/lib-release/`.

## Open requirements

| Requirement | Current state | Next action |
|---|---|---|
| Repository license | User selection pending | Add the selected license and review third-party redistribution notices |
| Public external dependencies | 22 of 23 declared external versions download; pinned `tmc/ffi` returns 404 with an upstream authentication failure | Make the exact dependency available before releasing its consumers |
| Internal dependency versions | `tpu`, `x/xla`, and `x/gomlx` still have replacements and placeholder versions | Prepare releases in dependency order, then regenerate sums with Go |
| NVIDIA parallel test reliability | One burst run lost a record; an earlier missing-library child timed out | Retain both failures and resolve the concurrency-sensitive qualification boundary |
| Final archives | Draft archive checks exist; license and final revision are not fixed | Recheck immutable final artifacts after license/version preparation |
| API/version review | Darwin export-data comparison completed for the four modules with local tags; candidates below | Confirm final versions and recheck platform-specific APIs on the final revision |
| Native platform coverage | Build targets are declared in `scripts/platforms.txt` and gated by `make platform-check`; runtime qualification remains Darwin/arm64 plus named CUDA hosts | Record named runtime tests, library versions, and skips for each claimed platform |

Origin HEAD was `d26d4c225ffc` (May 19); it predates the `openxla`,
`x/stablehlo`, and `x/xla` modules. Go version queries at that revision failed
with missing `go.mod`. `git ls-remote --tags origin` returned no tags. Recheck
before publication; do not silently move existing local tags.

## Verified changes

- `apple` pins public `tmc/apple v0.6.18`; `hfsync` pins public `go-iroh v0.2.0`.
  Both local replacements were removed. Go generated the sums. Both external
  archives downloaded from the public proxy with private-module bypass disabled.
- The dynamic Apple harness now binds shader dimensions and dispatches one
  GEMV threadgroup per output row. Nonzero CPU-reference tests failed before
  repair and pass afterward for widths 256/257, 19 output rows, and batches
  1/7/8/17. The original remainder loop was independently checked and retained.
- Memory-pressure and decoding harnesses now dispatch complete GEMV row groups
  and check nonzero outputs. Memory-pressure output checks failed before repair.
- The NVIDIA missing-library child now emits verbose output and has a nine-second
  test timeout for stack dumps before its unchanged ten-second parent deadline.
- `make release-check` checks license presence, module paths, replacements,
  placeholder versions, empty workspaces, and external workspace members.
  This Python 3/Go metadata gate does not prove dependency availability or ABI
  compatibility. `make release-selftest` exercises seven acceptance/rejection cases.
- `nvidia/x/cupticheckpoint` called `purego.Dlopen` and `purego.Dlsym`, which
  exist only on unix, so the `nvidia` module did not compile for Windows. The
  two calls now go through the same `openDynamicLibrary`/`lookupLibrarySymbol`
  seam `nvidia/cupti` already used. It was the module's only Windows break,
  and no darwin or linux build could see it.
- `make platform-check` cross-builds every workspace module for the targets it
  claims in `scripts/platforms.txt`, and `release-check` now requires it. The
  claim list must name exactly the modules in `go.work`, so a new module cannot
  join the workspace without declaring its platforms.

## Evidence and limits

| Check | Result | Receipt |
|---|---|---|
| Initial full workspace build/test | All 15 modules passed at the initial revision | `make.log` |
| Workspace gate controls | Nine passed | `make gate-selftest` output |
| Release metadata controls | Seven passed; real metadata gate remains failing | `make release-selftest` output |
| Standalone source tests | Nine leaf modules passed uncached with workspace off; existing dependency cache | `standalone.json` |
| Public dependency downloads | 22 succeeded; only `tmc/ffi` failed | `public-deps.eg7nm3ou/download.json` |
| Draft archive consumers | Ten modules passed via local proxy; NVIDIA package tests serialized; campaign began with empty shared cache | `proxy-check.fcq6a4id/results.json` |
| Apple dispatch regression | Failed before repair; corrected package passed | `dynamic-before.log`, `metalbench-final.log` |
| Other GEMV callers | Output checks failed before repair; package and benchmark smoke passed afterward | `memory-before.log`, `metalbench-callers.log`, `decode-dispatch-smoke.log` |
| Full Apple suite after fixes | Passed uncached, workspace off | `apple-final-source.log` |
| NVIDIA parallel suite | Failed: 1 of 640,000 burst records dropped for queue capacity | `nvidia-diagnostic-run.log` |
| NVIDIA missing-library failure | Child exceeded ten-second parent deadline; focused repeats and later broad run passed | `proxy-check.fcq6a4id/nvidia.log`, `cusparse-subprocess-repeat.log` |
| Burst and deliberate-loss controls | Five repetitions each passed, with full retention in normal runs and reconciled drops in zero-wait controls | `burst-focused.log` |
| Full NVIDIA suite, serialized packages | Passed with unchanged test set, queue, deadlines, and assertions | `nvidia-serial.log` |
| Declared platform builds | 66 of 66 passed at `1584e1c6`; the gate was checked against a false claim and a missing module and rejected both | `make platform-check` output |

The draft archive passes cover `libgit2`, `lldb`, `lm`, `rocm`, `treesitter`,
`x/hftokenizer`, `x/stablehlo`, `hfsync`, `nvidia`, and the pre-repair Apple archive.
These are unpublished drafts, not final licensed artifacts. Subpackage-only
modules require fetching their packages and test dependencies before read-only
tests (`go get -t <module>/...@<version>`).

The old Apple 0.71x result and earlier passing performance results used a
broken dispatch and cannot qualify performance. A one-iteration benchmark
smoke proves execution only. NVIDIA's passing serialized run does not explain
the parallel failure or guarantee lossless admission under arbitrary host load;
its admission wait is bounded at 100 ms. No limits or assertions were relaxed.

A separate draft dependency wave passed source and installed-consumer tests
for `openxla`, `x/stablehlo`, `tpu`, `x/xla`, and `x/gomlx` at local draft
version `v0.1.1`. This run seeded the local proxy with the exact cached
`tmc/ffi` archive. It verifies packaging and dependency ordering, not public
accessibility, TPU execution, or final release versions. Its checksums remain
only in the fixture. Receipt: `proxy-check.fcq6a4id/xla-results.json`.

The final metadata audit still rejects the absent license and internal
replacements/placeholders (`final-metadata.log`). A fresh public-proxy request
for the pinned `ffi` version still returns HTTP 404 (`ffi-public-final.txt`).
License selection has remained pending since the user asked about existing
project licenses. Final licensed archives and publishable dependency manifests
cannot be finalized from the currently available inputs. No commits or tags
were created during this release-readiness pass, and no publication occurred.

## Supported platforms

`scripts/platforms.txt` is the declaration; the table below restates it. Every
entry builds at `1584e1c6` under `make platform-check`, 66 builds in 39 seconds.

| Modules | darwin | linux | windows |
|---|---|---|---|
| `apple`, `examples` | amd64, arm64 | - | - |
| `hfsync`, `lm`, `nvidia`, `x/hftokenizer`, `x/stablehlo` | amd64, arm64 | amd64, arm64 | amd64, arm64 |
| `libgit2`, `lldb`, `openxla`, `rocm`, `tpu`, `treesitter`, `x/gomlx`, `x/xla` | amd64, arm64 | amd64, arm64 | - |

The darwin-only modules bind Apple frameworks. The eight modules in the last
row ship no windows source at all and have never claimed it; that is an
undeclared platform set now declared, not a defect left unfixed.

These are build targets. Compiling for a target is not a tested runtime on it:
native execution evidence in this document is Darwin/arm64 plus the named CUDA
hosts, and every other row is source-level only. The builds use the workspace,
so local dependency resolution still applies.

An earlier pass recorded 26 linux builds with `CGO_ENABLED=0`
(`linux-builds/results.json`); `platform-check` supersedes it and covers
windows, which nothing had built before.

## Proposed versions

These are candidates, not tags or publication instructions. Existing local tags
remain unchanged. All current modules require Go 1.27.0; the four existing
module tags required Go 1.26.2.

| Modules | Candidate | Basis |
|---|---|---|
| `nvidia` | `v0.2.0` | 196 incompatible report entries relative to local `nvidia/v0.1.0` |
| `lldb` | `v0.2.0` | Adds the public `Require` function |
| `rocm`, `treesitter` | `v0.1.1` | Dependency/toolchain updates; no public API differences in this comparison |
| `apple`, `hfsync`, `libgit2`, `lm`, `openxla`, `tpu`, `x/gomlx`, `x/hftokenizer`, `x/stablehlo`, `x/xla` | `v0.1.0` | No existing local module release tags |

`apidiff` from `golang.org/x/exp@v0.0.0-20260824195058-e88cd73687aa`
compared export data from each exact local tag with the current module on
Darwin/arm64 on 2026-09-13, with `GOWORK=off`. The four text reports are
committed under `api/`; export snapshots remain in `~/tmp/lib-release-current/api/`
and are intentionally not committed. Internal packages were excluded. This is
public Go API evidence for that build context, not native ABI or behavioral
compatibility.

NVIDIA migration review must account for removed packages and changed return
contracts, including `nvrtc.Program.GetTileIR`, driver resource and tensor-map
APIs, and pointer accessors in `x/moonep` and `x/nvlshm` that now return errors.
Perfetto field/clock constants also changed. The report lists 15 removed
packages, including `x/cudablock`, `x/cudafuse`, `x/cudagrammar`, and `x/cudasrv`.
The complete incompatibility list is in `api/nvidia.log`; do not describe this
candidate as a compatible patch update.

## Final candidate workflow

1. Resolve the license and external dependency availability.
2. Prepare leaf modules, then `tpu` and `x/xla` after `openxla`/`x/stablehlo`,
   then `x/gomlx`. Remove release-module replacements and pin final versions.
   Use Go commands for sums; never copy draft-proxy sums into final manifests.
3. Review exported API changes against prior module tags. Record each module's
   version, commit, dependency versions, and validation receipts. No root tag.
4. Run `make`, `make gate-selftest`, `make release-selftest`, and
   `make release-check`, which now includes `make platform-check`. Preserve
   failures and native-test skips.
5. Test immutable module archives from independent consumer modules with
   `GOWORK=off`, no local replacements, and an initially empty module cache.
   Before publication, a local proxy can prove packaging and ordering; label
   this evidence as local rather than public installation.
6. Run named native smoke/error-path tests for supported platforms, retaining
   host, revision, toolkit/library versions, and skips. Recheck the actual
   final artifact after any source, license, or dependency changes.

Publication is a separate authorized action. Push only reviewed subdirectory
tags, then verify public consumer installation and advertised examples.

If a published version fails, stop dependent releases, keep tags intact, and
recommend the last verified version. Prepare a corrective patch and, where
needed, retract the defective version in a subsequent module release. Reapply
consumer and native runtime gates before recommending the replacement. There
are no database migrations in this workflow.
