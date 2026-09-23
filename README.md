# lib

Purego bindings and tooling for GPU compute: NVIDIA CUDA, Apple Metal, and the
OpenXLA/PJRT runtimes that sit on top of them. No cgo, no vendor SDK headers at
build time.

## Repository layout

There is no root module. Each directory below is an independent Go module with
its own release tags (`<dir>/vX.Y.Z`).

| Module | Import path | Description |
|---|---|---|
| `nvidia` | `github.com/tmc/lib/nvidia` | Purego bindings for the CUDA driver, runtime, math libraries, NCCL, CUPTI, and Compute Sanitizer |
| `rocm` | `github.com/tmc/lib/rocm` | Purego HIP runtime bindings |
| `apple` | `github.com/tmc/lib/apple` | Purego bindings for Metal, plus Metal capture and profiling |
| `openxla` | `github.com/tmc/lib/openxla` | Purego bindings for the PJRT C API |
| `hf/hftokenizer` | `github.com/tmc/lib/hf/hftokenizer` | HuggingFace tokenizer.json loading and encoding |
| `tpu` | `github.com/tmc/lib/tpu` | PJRT TPU plugin discovery and loading |
| `lm` | `github.com/tmc/lib/lm` | Language model loading and sampling helpers |
| `hf/hfsync` | `github.com/tmc/lib/hf/hfsync` | HuggingFace repository mirroring |
| `lldb` | `github.com/tmc/lib/lldb` | Purego LLDB C API bindings |
| `treesitter` | `github.com/tmc/lib/treesitter` | Purego tree-sitter parsing bindings |
| `gguf` | `github.com/tmc/lib/gguf` | GGUF model container reading |
| `examples` | `github.com/tmc/lib/examples` | Programs that span modules; not released |

Packages under a module's `x/` directory, such as `nvidia/x`, have an API that
is still moving. They will change without a major version bump.
`hf/hftokenizer` carries the same notice in its package documentation.

## Usage

`go get` the specific module. `go get github.com/tmc/lib` fails by design,
because there is no root module.

```bash
go get github.com/tmc/lib/nvidia@latest
```

## Development

`go.work` is not checked in. Create one that uses every module:

```bash
go work init
go work use -r .
```

Build and test a module from its directory:

```bash
cd nvidia && go build ./... && go test ./...
```

To check a module without workspace dependency selection:

```bash
cd nvidia && GOWORK=off go build ./... && GOWORK=off go test ./...
```

Some modules still use relative `replace` directives and unpublished dependency
versions. Building them in this checkout does not prove that consumers can
install them.
