# Release readiness

Status: blocked on license selection and dependency publication; not ready. Scope: all 14 library modules in `go.work`. `examples` is
built and tested for development but is not tagged. There is no root module.

The audit began at `75377ee0` on September 12, 2026, with Go 1.27.0 on
Darwin/arm64 (Apple M4 Max). The last revision it built and tested was
`fa28c9c4`; any commit after that one changes only this document. The repairs it found are
committed; see the "Verified changes" section for what each one was. No tags
were created and no publication occurred. Receipts below are local files under
`~/tmp/lib-release/` and `~/tmp/lib-release-current/`.

## Open requirements

| Requirement | Current state | Next action |
|---|---|---|
| Repository license | User selection pending | Add the selected license and review third-party redistribution notices |
| Public external dependencies | 22 of 23 declared external versions download; pinned `tmc/ffi` returns 404 with an upstream authentication failure | Make the exact dependency available before releasing its consumers |
| Internal dependency versions | `tpu`, `x/xla`, and `x/gomlx` still have replacements and placeholder versions | Prepare releases in dependency order, then regenerate sums with Go |
| NVIDIA parallel test reliability | The burst boundary is measured and attributed: admission waits on journal write latency against a 100 ms deadline, 16-40 ms quiet and 99 ms under host I/O. The missing-library child timeout is unchanged | Decide whether admission keeps the wall-clock deadline or waits on drain progress, then requalify |
| Final archives | Draft archive checks exist; license and final revision are not fixed | Recheck immutable final artifacts after license/version preparation |
| API/version review | Export data compared for the four tagged modules on Darwin, and for `nvidia` on Linux as well; the Windows surface has no baseline because `nvidia/v0.1.0` does not compile for Windows | Confirm final versions and recheck on the final revision |
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
- Four hardware qualification tests behind `nvml_physical` and `cuda` had
  stopped compiling: `nvidia/nvml/physical_test.go` called `Device.TotalMem`
  with its old single-return signature at two sites, and three `example_cuda_test.go`
  files referenced APIs the packages no longer export. The `nvml` calls were
  repaired; the three example files were deleted rather than rewritten against
  a guessed API. No default gate could see any of it: nothing a workstation
  runs selects those tags.
- `make testtag-check` compiles the test package behind every non-GOOS/GOARCH
  build tag in the tracked test files, and `release-check` now requires it.
  The tag list is discovered, not declared, so a new tag is covered the day it
  appears; a tag whose GOOS the owning module does not claim in
  `scripts/platforms.txt` is skipped with that reason rather than failed.
- `cudaprof diff` ranked stall shifts by absolute delta alone over a slice built
  by ranging a map, so tied reasons permuted per run and the top-five truncation
  could drop a different reason each time: two diffs of the same pair of
  profiles disagreed. Ties now break on the reason name.

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
| Burst admission boundary | Eight quiet runs retained 640,000 of 640,000; longest admission wait equals longest batch write to the millisecond in every run | `go test ./internal/cudainject/nativecontrol -run TestBurst -count=8 -v` |
| Burst admission under host I/O | Three runs with 4.8 GB of concurrent writes; both maxima rose together to 99 ms against the 100 ms deadline and still retained everything | same test with a concurrent `dd` load |
| NVIDIA Linux API comparison | 195 incompatible entries plus one type-mismatch warning, identical to the Darwin set in both directions | `api/nvidia-linux.log` |
| NVIDIA Windows API comparison | Not possible: `nvidia/v0.1.0` does not compile for Windows, so the tag has no export data | `api/nvidia-darwin-vs-windows.log` compares current surfaces instead |
| Tagged test packages | 22 built, 1 skipped for an unclaimed GOOS; the gate was checked against an injected signature error in `nvml/physical_test.go` and rejected it | `make testtag-check` output |
| Burst admission failure observed directly | A full `make test` run lost 87 of 640,000 with `longest_wait=113ms longest_write=431ms`: the predicted mechanism, measured on a real failure rather than inferred | `make-test-head.log` |
| Full workspace test at the final revision | All 15 modules, 144 packages passed at `fa28c9c4`; `check`, `gate-selftest`, `release-selftest`, `platform-check` and `testtag-check` all clean | `make-test-f3a541fe.log` and the gate logs beside it |
| Release metadata at the final revision | Fails on the missing license and on `tpu`, `x/xla`, `x/gomlx` replacements and placeholders; nothing else | `release-check-fa28c9c4.log` |
| Pinned `tmc/ffi` availability | Still 404; `git ls-remote` fails authentication, so the repository is private rather than unpublished | `proxy.golang.org` `.info` and `.zip`, rechecked September 13, 2026 |

The draft archive passes cover `libgit2`, `lldb`, `lm`, `rocm`, `treesitter`,
`x/hftokenizer`, `x/stablehlo`, `hfsync`, `nvidia`, and the pre-repair Apple archive.
These are unpublished drafts, not final licensed artifacts. Subpackage-only
modules require fetching their packages and test dependencies before read-only
tests (`go get -t <module>/...@<version>`).

The old Apple 0.71x result and earlier passing performance results used a
broken dispatch and cannot qualify performance. A one-iteration benchmark
smoke proves execution only. No limits or assertions were relaxed.

The NVIDIA burst failure is now explained rather than merely retained. The
producer's admission path waits for the drain to publish a freed ring slot,
and the drain publishes only after its batch write to the journal returns, so
the 100 ms admission deadline is a bound on journal write latency and on
nothing else. The finalized record now carries the longest single admission
wait and the longest single batch write; over 2,500 batches they are equal to
the millisecond in every run, which is what identifies the cause. On a quiet
Darwin/arm64 host the pair peaks at 16-40 ms; driving 4.8 GB of concurrent
host writes moves both to 99 ms without losing a record. Crossing the deadline
is the whole mechanism: the loss is bounded, attributed to
`queue_capacity_dropped`, and reported with the stall that produced it.

A full `make test` run under heavy local load then crossed it and said so:
`burst lost 87 of 640000 offered records (capacity=87 contention=0 oversize=0
longest_wait=113ms longest_write=431ms)`. A 431 ms batch write held admission
past the 100 ms deadline, and the count of lost records equals the capacity
counter exactly. That is the mechanism observed on a failure, not inferred
from passing runs.

Lossless burst admission is therefore a property of the host, not of the
producer alone, and `TestBurst` asserts it strictly. Two policies remain open:
keep the wall-clock deadline and accept attributed loss on a loaded host, or
let admission wait on drain progress instead of elapsed time, which removes
this loss but gives up the per-record latency bound the current design chose
on purpose. The decision belongs with the owner; no assertion was relaxed to
avoid it.

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
cannot be finalized from the currently available inputs. The repairs this pass
found are committed on `main`; no tags were created and no publication occurred.
`make release-check` at `fa28c9c4` fails on exactly those two blockers -- the
missing license and the internal replacements and placeholder versions -- and
on nothing else.

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

`examples/llama-infer` carries 22 test files gated on `linux` inside a module
that cannot build for linux: it depends on `apple`, which is darwin-only
(`apple/x/metal/metalprof/enrich.go:82: undefined: ReplayTraceContext` under
`GOOS=linux`). Those files are unreachable from any platform the module claims,
so `testtag-check` skips the `tpu` tag there with that reason rather than
reporting a failure the module never promised to avoid. `examples` is not
tagged for release, so this is recorded and not fixed here.

An earlier pass recorded 26 linux builds with `CGO_ENABLED=0`
(`linux-builds/results.json`); `platform-check` supersedes it and covers
windows, which nothing had built before.

## Proposed versions

These are candidates, not tags or publication instructions. Existing local tags
remain unchanged. All current modules require Go 1.27.0; the four existing
module tags required Go 1.26.2.

| Modules | Candidate | Basis |
|---|---|---|
| `nvidia` | `v0.2.0` | 195 incompatible report entries plus one type-mismatch warning, relative to local `nvidia/v0.1.0` |
| `lldb` | `v0.2.0` | Adds the public `Require` function |
| `rocm`, `treesitter` | `v0.1.1` | Dependency/toolchain updates; no public API differences in this comparison |
| `apple`, `hfsync`, `libgit2`, `lm`, `openxla`, `tpu`, `x/gomlx`, `x/hftokenizer`, `x/stablehlo`, `x/xla` | `v0.1.0` | No existing local module release tags |

`apidiff` from `golang.org/x/exp@v0.0.0-20260824195058-e88cd73687aa`
compared export data from each exact local tag with the current module on
Darwin/arm64 on 2026-09-13, with `GOWORK=off`. The six text reports are
committed under `api/`; export snapshots remain in `~/tmp/lib-release-current/api/`
and are intentionally not committed. Internal packages were excluded. This is
public Go API evidence for that build context, not native ABI or behavioral
compatibility.

The `nvidia` comparison was repeated with `GOOS=linux`: 195 incompatible
entries and the same single type-mismatch warning, with no entry on either
side that the other lacks (`api/nvidia-linux.log`). The two removal sets are
identical as sets -- sorted, the diff is empty -- though apidiff emits them in
a different order per platform, so a plain `diff` of the two files does report
differences. The warning lines are identical unsorted. The Darwin-only
comparison was therefore not under-reporting for Linux.

Both counts describe one report. `api/nvidia.log` and `api/nvidia-linux.log`
each hold 195 lines opening with `-` and one opening with `!`; quoting 196 for
one platform and 195 for the other counted the warning in one place and not the
other. The warning is a single entry, on `nvml.RemappedRowsInfoV2`, reporting
two different messages for the same object.

Windows has no baseline to compare against. `nvidia/v0.1.0` does not build for
`GOOS=windows`: `x/nvlshm` calls `unix.Close` with no Windows guard, so
`apidiff` cannot write export data for the tag at all. Whatever the tag
promised on Windows, nothing there was ever compiled or compared. The current
tree does build, so `api/nvidia-darwin-vs-windows.log` compares the two
current surfaces instead:

- `x/moonep` and `x/nvlshm` do not exist on Windows.
- `nvdec.SymbolError.Err`, `nvdec.(*SymbolError).Unwrap`, `nvdec.BindFailure`,
  and `nvdec.BindFailures` exist only on unix. The Windows layer comes from an
  older generator lineage that predates them.
- Thirteen `nvdec.DecodeCreateInfo` and `nvdec.SourceDataPacket` fields are
  `uint32` on Windows and `uint64` elsewhere. The generator models C
  `unsigned long`, which is 32 bits under LLP64; `layout_windows_test.go` and
  `windows_abi_test.go` hold that layout. Portable consumer code cannot assume
  either width.

On `windows/arm64` the `nvdec` bindings are absent entirely: `init_windows.gen.go`
and its siblings are `windows && amd64`, so the package compiles down to its
doc, loader, and provenance files and exports no API. It is the only package in
the module that does this, and a build gate cannot see it -- an empty package
compiles. A consumer gets an undefined-symbol error at its own call site.

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
   `make release-check`, which now includes `make platform-check` and
   `make testtag-check`. Preserve failures and native-test skips.
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
