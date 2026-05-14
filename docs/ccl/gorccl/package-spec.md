# Package Spec: gorccl

Package gorccl provides Go-native RCCL collective operations.

- Import path: `github.com/tmc/lib/gorccl`
- ABI package: `internal/abi`
- ABI library base name: `rccl`
- Public API note: The public package exposes Group, Buffer, and context-driven collectives; low-level ABI bindings stay below internal/abi or provider adapters.

## Parse Inputs

- Headers: `rccl.h`

## Semantic Sources

- `tmc/lib/rocm/rccl` (reference): current low-level generated ABI surface
- `gojaccl/docs/go-jaccl-spec.md` (reference): Go-native collective API quality reference
- `/tmp/collab-F61A-ccl-bindings-to-257A.md` (reference): P0 CCL binding target roadmap
- ROCm device buffers are caller-owned and described by Buffer metadata.
- Blocking collectives require context.Context and must validate Buffer metadata before provider calls.
- Real provider adapters remain build-tagged and manual-gated; fake tests run by default.

## Rendezvous

- Store interface: `Store`
- Fake store: in-memory key/value rendezvous

## Capability Metadata

- Provider metadata: generated `ProviderInfo` fields on `Capabilities`.
- Health/readiness: generated `HealthSnapshot` fields on `Capabilities`.

## Resources

- `Group` wraps `backend` and is owned.

## Operations

- `Barrier` calls backend `barrier`; blocking, context policy `required`.
- `AllReduce` wraps `ncclAllReduce` calls backend `allReduce`; blocking, context policy `required`.
- `AllGather` wraps `ncclAllGather` calls backend `allGather`; blocking, context policy `required`.
  - `dst.Count == src.Count * uint(g.Size())`
- `Broadcast` wraps `ncclBroadcast` calls backend `broadcast`; blocking, context policy `required`.
- `AllToAll` wraps `ncclAlltoAll` calls backend `allToAll`; blocking, context policy `required`.
- `ReduceScatter` wraps `ncclReduceScatter` calls backend `reduceScatter`; blocking, context policy `required`.
  - `src.Count == dst.Count * uint(g.Size())`
- `Send` wraps `ncclSend` calls backend `send`; blocking, context policy `required`.
- `Recv` wraps `ncclRecv` calls backend `recv`; blocking, context policy `required`.

## Adapters

- `rccl` build tag `gorccl_hardware`; kind `rccl`; package `github.com/tmc/lib/rocm/rccl`; runtime `rocm`; device kind `DeviceROCM`; manual gate `hardware`.

## Unsafe Policy

No public unsafe exposure is allowed.

## Element Sets

No element sets declared.
