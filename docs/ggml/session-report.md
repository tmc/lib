# Session Report: ggml

Status: artifact-only package plan rendered.

## Semantic Sources

- `llama.h` (reference): higher-level model/session API shape and tokenizer boundary
- `llama.cpp` (implementation): semantic reference for model serving loops; not an ABI input
- `docs/ggml-backends.md` (docs): backend and hardware gate semantics
- ggml.h is the only ABI parse input for the first target.
- llama.h and llama.cpp are semantic references only; they must not be parsed for ABI symbols.
- CPU tensor metadata and graph construction are cheap default gates.
- CUDA, Metal, Vulkan, and other provider execution are manual-gated adapters.

## Default Gates

- Validate package plan before rendering.
- Run generated cheap tests only by default.
- Treat hardware tests as manual gates.

## Completion Rule

Do not call this package complete until artifacts, NotebookLM review, local non-hardware gates, and manual gate notes agree.
