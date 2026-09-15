# Make the PTX/SASS trace example a public API example

Status: completed. Priority: P1. Effort: 2–4 days plus access to a CUDA host. Risk: medium, because source coordinates and saved captures cross package boundaries. Planned on 2026-09-09 at `06c7378`. Verified on NVIDIA GB10 (sm_121, CUDA 13.0, Driver 580.173.02) and local offline darwin/arm64.

## Recommendation

Scope extension: [plan 003](003-cupti-feature-coverage.md) covers the user's additional request for all CUPTI features and all reachable/representable sample types. Its inventory and schema design must inform this plan's capture API before that API is finalized. It supersedes the deferral of new profiling backends below; the original event-only invariants apply to the baseline mode.

Make `examples/ptx-sass-trace` an ordinary client of public packages. Give the shared capture format a focused public home, `github.com/tmc/lib/nvidia/x/cudacapture`, after removing assumptions about this particular workload. Keep CUDA compilation, launch, timing, and validation visible in the example. Keep pprof's viewer integration in `cmd/cudaprof`.

The current import of `internal/profilebundle` is legal: the example is inside the subtree allowed to import it. A user who copies these sources into an unrelated module cannot use that import. This follows Go's [internal package rule](https://go.dev/doc/go1.4#internalpackages). A public example should demonstrate an API that its reader can use from their own module.

This is a design recommendation informed by Go's guidance, not a claim of Go team endorsement. [Organizing a Go module](https://go.dev/doc/modules/layout) supports retaining implementation details in internal packages and separating commands from reusable packages. [Package names](https://go.dev/blog/package-names) recommends judging a package boundary from its callers. Here, both the producer and viewer need captured files and code coordinates; neither needs the other's execution policy.

## Evidence and priorities

Paths below are relative to the repository root, `/Volumes/tmc/go/src/github.com/tmc/lib`.

| Priority | Finding and effect | Evidence | Effort / change risk / confidence |
| --- | --- | --- | --- |
| 1 | Every Go file in the example imports the internal package; copying the example outside the NVIDIA import subtree fails. | `nvidia/examples/ptx-sass-trace/main.go:14`, `capture.go:16`, `workload.go:12` | M / medium / high |
| 1 | The shared bundle owns an exact list of example Go filenames and workload fields. Renaming an example source file changes its readers and packer. | `nvidia/internal/profilebundle/bundle.go:25–40`, `embedded.go:155–170` | M / medium / high |
| 2 | Workload execution mutates the serialization manifest and receives the entire parsed code index. | `nvidia/examples/ptx-sass-trace/workload.go:19`, `:100–105` | S / medium / high |
| 2 | The reader parses CUBIN, PTX, SASS, and CUDA source in one function; it assumes `kernels.cu` and a simple `void name(` declaration. PTX is parsed again by the example. | `nvidia/internal/profilebundle/bundle.go:117–207`; `nvidia/examples/ptx-sass-trace/capture.go:108–123` | M / medium / high |
| 2 | The example depends on a private logical image base that is separately hard-coded in the profile builder. | `nvidia/examples/ptx-sass-trace/workload.go:138`; `nvidia/internal/profilebundle/bundle.go:21–23`; `nvidia/x/cudaprof/proto.go:173–184` | S / medium / high |
| 3 | `capture` combines trace ownership, device setup, source copying, compilation, disassembly, profile writing, and workload execution; the manifest literal is one long line. | `nvidia/examples/ptx-sass-trace/capture.go:28–150` | S / low / high |
| 3 | The example has no tests. Existing bundle tests cover corruption and round trips, but real-code validation is opt-in. | No `*_test.go` in the example; `nvidia/internal/profilebundle/bundle_test.go:49–53`, `embedded_test.go:51–129` | M / low / high |

Two excerpts identify the coupling an executor must remove:

```go
// nvidia/examples/ptx-sass-trace/capture.go:28
func capture(dir string, cfg config) (result *profilebundle.Manifest, err error)

// nvidia/examples/ptx-sass-trace/workload.go:19
func runWorkload(ctx context.Context, dev *cuda.Device, mod *cuda.Module,
    prof *cudaprof.Session, code *profilebundle.Code, cfg config,
    m *profilebundle.Manifest) (err error)
```

Retain the useful existing style: `main` delegates to an error-returning `run`; owners register cleanup immediately; measured intervals use `time.Duration`; `reference` is a plain CPU function. The short error pattern in `nvidia/x/cuda/device.go:25–40` and table cases in `nvidia/internal/profilebundle/bundle_test.go:11–47` are suitable models.

## Scope and baseline

Implementation may change only:

- `nvidia/examples/ptx-sass-trace/`, including new local tests and focused files.
- `nvidia/x/cudacapture/` (new), including package documentation and tests.
- `nvidia/internal/profilebundle/`, to migrate its responsibilities and remove dead code.
- `nvidia/cmd/cudaprof/`, for the new capture API, retained-format support, and tests.
- `nvidia/x/cudaprof/proto.go`, `doc.go`, and associated tests, only to name the existing default mapping start and verify its use.
- `nvidia/README.md` and `nvidia/x/DESIGN.md`, for the package map and example instructions.
- `nvidia/go.mod` and tool-managed `go.sum`, only if CLI tests require adding `rsc.io/script`.
- This plan's status row after implementation verification.

Generated bindings, other examples, `go.work`, other modules, and unrelated profile features are outside scope. Do not stage binaries. Do not write `go.sum` by hand. Work in an owned checkout containing the current untracked sources; a worktree made from HEAD alone will omit this example. Do not reset, stash, or overwrite another session's work. Commit only if requested, using the user's `git-auto-commit-message` and git-notes procedure; this plan does not request a commit or publication.

Before implementation, from the repository root:

```sh
git status --short
git diff --stat 06c7378..HEAD -- nvidia/examples/ptx-sass-trace nvidia/internal/profilebundle nvidia/cmd/cudaprof nvidia/x/cudaprof
```

This plan once carried a `.baseline.sha256` of the inspected example, bundle, viewer and relevant public source, to be checked with `shasum -a 256 -c`. It was removed on 2026-09-14: 14 of its 22 rows no longer matched, nothing in the repository ran the check, and a hash file that is mostly stale reads like a gate without being one. Use the `git diff --stat` above to see what has moved instead. The viewer's object tests changed during review; this plan uses the updated SASS and PTX tests, not the earlier test shape.

Observed on this Mac, with Go `go1.27.0 darwin/arm64` and NVIDIA module directive `go 1.26.2`:

```sh
cd /Volumes/tmc/go/src/github.com/tmc/lib/nvidia
env -u CUDA_PROFILE_BUNDLE GOWORK=off TMPDIR="$HOME/tmp" \
  go test -mod=readonly -count=1 ./examples/ptx-sass-trace \
  ./internal/profilebundle ./cmd/cudaprof ./x/cudaprof ./x/cudaptx ./x/cudasass
```

Result: exit 0; five tested packages passed; the example reported `[no test files]`. A separate verbose run confirmed that `TestCapturedCode` skips without `CUDA_PROFILE_BUNDLE`; installed-disassembler tests also skipped without their real CUBIN inputs. Focused `go vet` produced no diagnostics, and `sh -n examples/ptx-sass-trace/package.sh` passed. These are local software checks, not evidence of CUDA execution or working GPU timing. The repository workspace uses machine-specific paths, so all implementation checks below use `GOWORK=off`.

## Target boundaries

Optional follow-up: [plan 002](002-optional-nccl-capture.md) adds multi-GPU
NCCL AllReduce and AllGather. It depends on this public API cleanup and does
not change this plan's default single-GPU acceptance criteria. Preserve
arbitrary attached Go source, source-only profile frames, opaque metadata,
and rank/device sample labels so that NCCL can use the format without adding
collective-specific schema fields or claiming application PTX/SASS belongs to
NCCL. The format's code roles describe the recorded application kernels;
additional source-only library calls need no invented code image.

| Location | Owns |
| --- | --- |
| Example `main.go` | Flags, validation, output-directory creation, summary and top-level errors |
| Example `capture.go` | Trace lifetime, compile/disassemble/load sequence, capture assembly and final writes |
| Example `workload.go` | Inputs, warmup, launches, event timing, checking outputs, recording samples |
| Example `output.go` (only if needed) | Example metadata and its encoding, embedded source collection, capture output |
| `x/cudacapture` | Portable recorded files, integrity checks, profile transport, code/source coordinate index |
| `x/cudaprof` | Caller-supplied observations, samples and their mapping convention |
| `cmd/cudaprof` | Upstream pprof adapter, command flags, temporary extraction and display policy |

Use one new public package. Separate its format and code-index implementation into files, rather than creating a package for each helper. It must not depend on `github.com/google/pprof/driver`, NVRTC, or CUDA execution. Reusing `github.com/google/pprof/profile`, `x/cudaptx`, and `x/cudasass` is appropriate. There is no need for an interface around the CUDA device or a configurable pipeline framework.

## Step 1: Characterize the behavior that must survive

Add focused tests before changing ownership. Extend the existing bundle round-trip and viewer coordinate tests, using small, explicitly synthetic fixtures for software checks. Construct any minimal ELF fixture in Go test code; do not commit executable or capture binaries. Keep a retained real capture outside the repository for the later hardware check.

Required behavior:

- The measured profile contains only `gpu_time` in nanoseconds. Every measured kernel launch contributes one sample with its kernel/phase/pass labels; warmup contributes none.
- Validation checks both stages, including warmup, using the existing exact vector-add comparison and FMA reference/tolerance. `verified_results` counts only measured outputs: `2 * elements * passes`.
- The CUBIN supplied to the disassembler is also selected by `LoadOwnedOutput(..., nvrtcx.OutputCUBIN)` for execution. PTX remains an inspection artifact from that compilation.
- SASS entry and PTX arithmetic lines remain navigation anchors. Source-only frames do not acquire invented PCs, and source mapping does not alter sample values.
- A standalone `capture.pprof` still works after removing the original capture directory and after an upstream pprof parse/write cycle. Source, assembly, trace, and checksums survive.

Extract only the small pure functions needed for meaningful tests: configuration validation, result checking, and profile frame construction. Cover invalid ranges, one element, a block-tail size such as 257, wrong finite output, NaN/Inf, and the measured/warmup distinction. Do not introduce a mock GPU implementation.

**Verify:** run the baseline command above with `-count=1`; expect exit 0 and actual example tests. Preserve `TestSingleFile`, `TestEmbeddedCorruption`, `TestObjectCoordinates`, `TestPTXCoordinates`, and `TestSourceOnlyFrames` or direct successors with the same assertions.

## Step 2: Design the public capture API around both callers

First write client examples for a producer and a reader in `x/cudacapture/example_test.go`, then implement the smallest API they need. The following is the recommended shape; names may be tightened when compiling those clients, but the boundary and behavior are required:

```go
type Capture struct {
    Profile  *profile.Profile
    Files    map[string][]byte
    Code     CodeFiles
    Metadata json.RawMessage
}

type CodeFiles struct {
    Source, PTX, CUBIN, SASS string
}

func Read(r io.Reader) (*Capture, error)
func Write(w io.Writer, c *Capture) error
func ReadDir(dir string) (*Capture, error)
func WriteDir(dir string, c *Capture) error
func LoadCode(c *Capture) (*Code, error)
```

`Capture` is in-memory data, not an open device or temporary directory. Readers/writers do not close caller-owned streams. `Read` owns the decoded bytes; `Write` borrows its argument for the call, does not mutate it, and returns errors for a missing/invalid profile or file set. Directory helpers operate in the caller's chosen directory and do not silently delete it. Keep types concrete; no option DSL, registry, global mutable `Files`, or generic metadata interface.

Keep the useful `Code`, `Kernel`, and `Instruction` information required by the two callers, with the lookup method they use. Make `LoadCode` take filenames from `CodeFiles` instead of assuming `kernels.cu`. Document the supported CUDA declaration and SASS formats honestly; this work does not require a general C++ parser. Parse PTX once per load. Keep the choice of `add.f32` and `fma.rn.f32` anchors in the example, using the parsed PTX or a narrow lookup on the code index.

Move `elements`, `passes`, `fma_rounds`, interval arrays, verified count, and max error to an unexported example metadata struct. Device/tool identity and the timestamp can live there too. The format layer validates that metadata is valid JSON and preserves it; it does not interpret a compute workload. The viewer can render a generic file/profile view plus metadata. Preserve the familiar summary for this example through a small command-local metadata decoder.

Define the file set from supplied files, with explicit source/PTX/CUBIN/SASS roles. For this bounded format, accept flat portable filenames only: reject empty/dot names, separators, absolute/volume paths, collisions with `gpu.pprof`, `capture.pprof`, or `manifest.json`, and names that collide on supported case-insensitive filesystems. Reject missing required roles, invalid code, and checksum mismatches. Do not weaken the current rejection of unexpected capture versions and multiple embedded payloads. Document size limits before allowing arbitrary attachments; preserve a real baseline capture within those limits.

Changing the fixed v1 manifest into this format warrants a new wire version, not an undocumented v1 reinterpretation. Write v2; retain a small, tested v1 decoder for saved recordings, mapping the old fields and fixed filenames into the new in-memory representation. Keep the old wire schema private. This is saved-data compatibility, not a public alias for the old internal API. For v2, encode roles, file digests, and opaque metadata in `manifest.json` and the embedded pprof payload; exclude the enclosing profile from its own digest exactly as today. Readers validate before materializing files. A format test must verify that directory and embedded forms represent the same recording. A v1 viewer will reject v2 captures, so ship the updated producer and viewer together and include the updated viewer in optional share bundles.

Move transport/code validation tests to their new owner. Retire unused old APIs after both callers switch; do not maintain two independent codecs. Viewer-specific temporary extraction belongs in the command, under `~/tmp`, with cleanup on error and on completion.

**Verify:** `GOWORK=off TMPDIR="$HOME/tmp" go test -mod=readonly -count=1 ./x/cudacapture ./internal/profilebundle ./cmd/cudaprof` from `nvidia/` while the old package still exists; omit its path after removal. Expect all tests to pass. Tests must include a non-example source filename, extra Go source, metadata round trip, old-format read, corrupt/missing file, unsafe filename, failing writer, and unchanged samples. Public API examples need runnable `// Output:` assertions, following the [Go doc comment guide](https://go.dev/doc/comment).

## Step 3: Make the example read in execution order

Replace direct `profilebundle` imports and parameters with the public capture API and small local values. Have `runWorkload` return a local result containing measured intervals, verified count, and maximum error. It should not know JSON field names or mutate a manifest owned by its caller.

Resolve the two kernel functions, launch dimensions, and source anchors before the pass loop. A short local table is sufficient for their names, phases, launch arguments, and frames. Keep the CUDA calls explicit. Keep `runKernel` responsible for start event → launch → end event → synchronize → elapsed time. Return a `time.Duration`; attach the measured sample in the caller after successful timing. Build the stable CUDA/PTX/SASS frames once, and append the Go caller frame at the actual launch-recording site so splitting files cannot silently move the displayed attribution to a helper.

Replace the duplicated logical base with a documented `cudaprof.DefaultMappingStart` constant, used by `NewBuilder` and by the example's `DefaultMappingStart + kernel.Offset` calculation. The viewer should continue reading mapping start/limit/offset from the actual profile. This names an existing convention; it must not change serialized addresses or imply they are device PCs.

Break `capture` only at useful boundaries, for example `compileCode` and `writeCapture`; keep orchestration legible in one place. Use multiline literals. Give ambiguous errors operation context, such as `compile kernels`, `open trace`, or `write capture`, preserving wrapped errors. Prefer `errors.New` for constant errors. Do not wrap every pass-through error or create a cleanup abstraction solely to shorten `defer` statements.

Preserve ownership: buffers/events/stream finish before module/device close, `trace.Stop` precedes closing its file, and successful profile finalization happens before capture packing. Keep `errors.Join` where both an operation and cleanup can fail. Do not casually add signal cancellation: the current ownership guide says cancellation stops waiting, not already submitted GPU work (`nvidia/x/DESIGN.md`, Ownership rules). Any future cancellation path must drain submitted work before freeing its resources.

Use one example-owned embedded file inventory. Collect those files for the capture without mirroring their names in the public package or viewer. Update embedded source and its display line mapping together when files move.

**Verify:** run `GOWORK=off TMPDIR="$HOME/tmp" go test -mod=readonly -count=1 ./examples/ptx-sass-trace ./x/cudacapture ./x/cudaprof ./cmd/cudaprof`; expect exit 0. `rg -n 'internal/|profilebundle' examples/ptx-sass-trace -g '*.go'` must find no matches. Test that the chosen profile mapping contains the emitted SASS address and that source-only frames retain no mapping.

## Step 4: Keep the CLI and documentation small

Retain current workload flags and defaults. Validate each range separately so errors identify the offending flag. Reject unexpected positional arguments before creating output or attempting CUDA. Preserve refusal to overwrite an existing `-out` directory and the `~/tmp` default. Print the output directory only after a complete successful capture; remove a partial final `capture.pprof` on write/close failure while retaining diagnostic intermediate files.

Add CLI integration tests using `rsc.io/script` txtar files per repository instructions. Cover help, bad flags/ranges, an existing output directory, no accidental output on validation errors, and viewer pack/extract behavior. Use subprocess tests for those real command behaviors; do not add callback injection to the production CLI just to simulate CUDA.

Keep the README's primary path to three operations: capture, inspect with `cudaprof`, and open the Go trace. Give prerequisites explicitly and use the producer's recorded Go version for trace instructions, replacing the fixed `go1.26.7` example. Explain whole-launch timing and logical coordinates once near the viewing instructions. Move optional archive/cross-build details into a packaging section or separate document. Preserve `package.sh` and its offline rebuild feature; update its source comparison and metadata-version lookup to use the new recording representation. It must no longer require the exact original set of Go filenames.

Add brief `doc.go` package documentation and runnable public API examples. Match the existing `x/cudaprof/example_test.go` external-package style. Use the Go doc comment guide linked above. Avoid adding dependencies except the requested CLI test harness, and never hand-edit a checksum file.

**Verify:** `GOWORK=off TMPDIR="$HOME/tmp" go test -mod=readonly -count=1 ./examples/ptx-sass-trace ./x/cudacapture ./cmd/cudaprof` and `sh -n examples/ptx-sass-trace/package.sh` both exit 0. Run the actual optional packager on the final real capture in step 5; shell syntax checking alone does not qualify its output.

## Step 5: Prove external use and preserve the recording

Add `examples/ptx-sass-trace/check-external.sh`. It creates an unrelated module under `~/tmp`, copies the example's Go/embedded source assets, requires `github.com/tmc/lib/nvidia`, and adds a local replacement pointing at the executor's owned NVIDIA checkout. Use `module example.com/ptx-sass-client`, not a path under `github.com/tmc/lib/nvidia`. Set `GOWORK=off`; let `go mod tidy` generate sums only in that disposable client. Build to a path inside its temporary root, then run the binary with `-h` without a GPU. This verifies actual copied source, not merely an empty external package importing one new symbol.

Also run a negative control there: temporarily add a Go file importing `github.com/tmc/lib/nvidia/internal/profilebundle` if that package remains, or create an owned internal-only fixture in the temporary dependency copy if it has been removed. The unrelated client must fail with `use of internal package ... not allowed`. Remove the mutation and rebuild successfully. The negative control establishes that workspace settings or a misleading module name have not bypassed the intended boundary.

Final software checks, from `nvidia/`:

```sh
GOWORK=off TMPDIR="$HOME/tmp" go test -mod=readonly -count=1 \
  ./examples/ptx-sass-trace ./x/cudacapture ./cmd/cudaprof \
  ./x/cudaprof ./x/cudaptx ./x/cudasass
GOWORK=off TMPDIR="$HOME/tmp" go vet -mod=readonly \
  ./examples/ptx-sass-trace ./x/cudacapture ./cmd/cudaprof ./x/cudaprof
sh examples/ptx-sass-trace/check-external.sh
```

All must exit 0. Check formatting only on changed Go files. The external script must report its failed internal-import control as an expected result, then its successful clean build.

On an available CUDA host, record source hashes, Go version, driver/device identity, NVRTC, and nvdisasm identity. Run the example with `-elements=257 -rounds=8 -passes=2`, then with the existing defaults. Choose fresh `-out` directories under `~/tmp` and a verified nvdisasm path. The small run must verify 1,028 measured outputs and record two intervals for each kernel; the default run must verify 25,165,824 and record twelve intervals for each. Both must check warmup without counting it in measured totals. Compare profile values with the saved interval arrays, not with prior timing magnitudes.

Add an opt-in `TestCapture` in the example, gated by `CUDA_PROFILE_CAPTURE` pointing at a retained recording. It must fail on invalid files, check the counts/labels/units above for the recording's parameters, verify checksums and source coordinates, and compare summed pprof values with per-kernel recorded event totals. Synthetic tests must not claim that the recorded CUBIN executed; the fresh command and source audit establish the producer path. Keep the real-recording code validation test under its new package owner.

Copy only the new `capture.pprof` to a machine without CUDA and verify:

```sh
GOWORK=off go run -mod=readonly ./cmd/cudaprof -top "$recording"
GOWORK=off go run -mod=readonly ./cmd/cudaprof -list=compute "$recording"
GOWORK=off go run -mod=readonly ./cmd/cudaprof -disasm=compute "$recording"
go tool pprof -top "$recording"
```

`recording` must name the copied standalone profile. Expect matching kernel totals, CUDA/PTX/SASS/Go source navigation, and actual saved SASS. Exercise the HTTP Source and Disassemble views, extract files, and open the trace with the recorded producer toolchain. Re-run the optional `package.sh` on the new directory and verify its `./view`, `./trace`, and vendored offline rebuild instructions. Keep archives and binaries under `~/tmp` and out of staging. Report cross-build success separately from actually running each supported platform.

## Done criteria and maintenance

- [x] The copied example builds in an unrelated module, its help runs without CUDA, and the internal-import negative control fails as intended.
- [x] The example has no direct internal imports, bundle schema types in workload signatures, or private address constants.
- [x] Both producer and viewer use the public capture API. It accepts a different source filename and additional Go files without editing a global inventory.
- [x] Focused tests, examples, vet, and CLI checks pass; corruption, writer-error, old-format, and source-coordinate controls are exercised.
- [x] Fresh small/default CUDA runs retain full validation and measurement semantics; offline viewing, extraction, trace viewing, and optional packaging work.
- [x] v1 recordings remain readable; v2 documents its file rules, limits, metadata, ownership, and version behavior.
- [x] The final diff stays within scope and the plan index records the outcome accurately.

If no CUDA host or retained recording is available during implementation, the public API and software checks can proceed, but the hardware/recording criteria remain unverified. Do not substitute a skipped test for those results.

Stop and reconcile if the baseline hashes differ, a format change breaks an existing recording, ownership requires modifying an out-of-scope package, or the proposed API needs more than one public package to serve these two callers. If a verification fails twice after a reasonable repair, report the exact failure before expanding scope. Do not modify concurrent work to make a baseline pass.

Future review should focus on three contracts: the example remains copyable, files and source coordinates remain tied to the measured binary, and recording metadata never claims instruction sampling or counters that were not collected. New profiling backends, general C++ parsing, performance optimization, and a monorepo-wide style cleanup are deferred.
