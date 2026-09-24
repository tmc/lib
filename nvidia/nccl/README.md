# NCCL bindings

The six host calls that take `Team` by value use the pinned, ejected registrar
in `internal/ffi`:

- `Comm.TeamRankToWorld` and `Comm.TeamRankToLsa`
- `Comm.GinBarrierCreateRequirement`
- `LsaBarrierCreateRequirement` and `CftBarrierCreateRequirement`
- `Window.CftDeviceLeInfo`

The rank-conversion methods return `(int32, error)`. All six report unavailable
symbols or unsupported signatures through `SymbolError`; inspect `BindFailures`
for registration errors. The local registrar currently supports Linux and Darwin
ARM64. It never retries these calls through purego's struct marshaler.

Other entry points retain purego registration, including communicator creation
with its 128-byte `UniqueId`. Return-only `Team` accessors also retain their
existing binding.

From the `nvidia` module, `go generate` ejects the pinned FFI source before
regenerating the bindings. `FFI_SRC` can name the clean source checkout required
by `scripts/ffi.sh`. Rebuild `nvgen` from the matching libgen checkout after
updating its recipes (`go install ./cmd/nvgen` in libgen). To check the ejected copy independently:

```sh
sh scripts/ffi.sh -verify
```

`TestTeamFFI` compiles a small C library and tests these generated call signatures
in a subprocess. It checks every Team field, following scalar/pointer arguments,
negative input, and a separate stock-purego `UniqueId` call. It does not execute
NCCL GPU operations.
