# lib

Purego bindings and tooling for GPU compute: NVIDIA CUDA, Apple Metal, and the
OpenXLA/PJRT runtimes that sit on top of them. No cgo, no vendor SDK headers at
build time.

Experimental and not-yet-ready modules live in
[github.com/tmc/libexp](https://github.com/tmc/libexp). Everything in this
repository is listed in `go.work` and is built and tested by `make`.

## Repository layout

There is no root module. Each directory below is an independent Go module with
its own release tags (`<dir>/vX.Y.Z`).

| Module | Import path | Description |
|---|---|---|
| `nvidia` | `github.com/tmc/lib/nvidia` | Purego bindings for the CUDA driver, runtime, math libraries, NCCL, CUPTI, and Compute Sanitizer |
| `apple` | `github.com/tmc/lib/apple` | Purego bindings for Metal, plus Metal capture and profiling |
| `openxla` | `github.com/tmc/lib/openxla` | Purego bindings for the PJRT C API |
| `x/xla` | `github.com/tmc/lib/x/xla` | XLA client, compilation, and execution over `openxla` |
| `x/stablehlo` | `github.com/tmc/lib/x/stablehlo` | StableHLO module construction and serialization |
| `x/gomlx` | `github.com/tmc/lib/x/gomlx` | GoMLX backend over `x/xla` |
| `x/hftokenizer` | `github.com/tmc/lib/x/hftokenizer` | HuggingFace tokenizer.json loading and encoding |
| `tpu` | `github.com/tmc/lib/tpu` | PJRT TPU plugin discovery and loading |
| `lm` | `github.com/tmc/lib/lm` | Language model loading and sampling helpers |
| `hfsync` | `github.com/tmc/lib/hfsync` | HuggingFace repository mirroring |
| `libgit2` | `github.com/tmc/lib/libgit2` | Purego libgit2 bindings |
| `lldb` | `github.com/tmc/lib/lldb` | Purego LLDB C API bindings |
| `treesitter` | `github.com/tmc/lib/treesitter` | Purego tree-sitter parsing bindings |
| `examples` | `github.com/tmc/lib/examples` | Runnable programs that exercise the modules above |

The `x/` prefix marks a module whose API is still moving. It will change without
a major version bump.

## Usage

`go get` the specific module. `go get github.com/tmc/lib` fails by design,
because there is no root module.

```bash
go get github.com/tmc/lib/nvidia@latest
```

## Development

`go.work` covers every module in the repository, which is the invariant that
keeps a module from quietly rotting: if it is here, `make` builds and tests it.
`make check` fails if a module on disk is missing from the workspace.

```bash
make          # build and test every module
make check    # verify go.work covers every module on disk
```

Every module also builds on its own, without the workspace:

```bash
cd nvidia && GOWORK=off go build ./... && GOWORK=off go test ./...
```
