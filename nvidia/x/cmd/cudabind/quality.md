# Cudabind quality review

Cudabind parses a CUDA translation unit with libclang, emits typed Go methods,
and embeds the translation unit and headers for NVRTC compilation. Its current
shape already gives a useful safety boundary: a C pointer parameter becomes a
`cudamem.Slice[T]`, scalar arguments retain their fixed-width Go type, and an
unrepresentable C type returns an `unsupported type` error. Generated launch
methods call the typed `cuda.Kernel` directly; they do not erase those types
through a reflective argument API.

The next improvements should preserve that boundary:

* Keep type conversion fail-closed. In particular, do not silently map unknown
  integer widths, address spaces, vector types, or callable types to `uintptr`
  or bytes. Add an explicit mapping and an ABI check before accepting each new
  class.
* Extend generated provenance beyond the current SHA-256 of embedded files.
  The hash currently detects changed source and includes, while flags such as
  `-std`, `-D`, `-arch`, parser version, and the NVRTC/toolkit version also
  affect the resulting module. Record the generation options and compiler
  identities, and make stale generated output diagnosable.
* Treat C layout as an ABI contract. The generator already uses clang sizes
  and offsets, inserts padding, emits compile-time Go assertions, and generates
  an NVRTC layout test. Expand coverage for alignment, nested records, arrays,
  packed records, unions, and bit-fields before mapping those forms to named Go
  fields. Keep opaque byte arrays for layouts that cannot be represented
  faithfully.
* The current pointer mapping preserves element type but not `const`, pointer
  direction, or a relationship between a pointer and a length parameter. A
  `Slice[T]` carries a length, but generated launch methods do not yet check
  that the length satisfies an adjacent kernel parameter's contract. Add
  const/mutable distinctions only if they can be expressed in a small Go API;
  make length checks explicit metadata rather than guessing from parameter
  names.
* Preserve source locations in diagnostics. Clang locations are already
  available while binding declarations; include the declaration location in
  unsupported-type, overload, and name-collision errors so fixes point back to
  the CUDA source.
* Generated modules own loaded module state and expose `Close`. Document that
  callers must keep device slices alive until asynchronous work finishes. If
  wrappers later add asynchronous copies or deferred cleanup,
  encode those lifetimes in the API rather than relying on generated comments.
* Test ahead-of-time generation and JIT compilation against the same fixture.
  They should agree on source/include hashes, options, lowered C++ names,
  signatures, and ABI layout. The existing generated test checks device
  compiler layout and constants; generator tests should additionally compile
  representative calls and reject wrong scalar and pointer element types at Go
  compile time.

The compile-only generated-call test belongs alongside the generator tests and
uses an overlay over `internal/testvec`; its invalid variants prove the Go
compiler rejects `float32` for a C `int` argument and `Slice[uint32]` for a
`float*` parameter without requiring a GPU.

On the current AArch64 host, `TestGenerate` passes with libclang 19.1.1 and its
matching resource headers. Clang 18 device-only parsing fails while reading the
host glibc `bits/math-vector.h`: that header declares SVE vector names which
the active CUDA device target does not define. Do not inject those names or
switch the parser to host-only mode; either can hide a real target/layout
mismatch. Keep both outcomes visible until a compatible parser/toolkit pair is
available, and include the failing header location in the diagnostic.
