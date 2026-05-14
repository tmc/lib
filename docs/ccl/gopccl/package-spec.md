# Package Spec: gopccl

Package gopccl provides Go-native PCCL collective operations.

- Import path: `github.com/tmc/lib/gopccl`
- ABI package: `internal/abi`
- ABI library base name: `pccl`
- Public API note: gopccl is a local PCCL-capable collective island below ccl/mgccl; provider, rendezvous, and transport details are internal boundaries.

## Parse Inputs

- Headers: `pccl.h`

## Semantic Sources

- `tmc/lib/pccl` (reference): current low-level generated PCCL ABI surface
- `PrimeIntellect-ai/pccl` (reference): PCCL source API and behavior reference
- `gojaccl/docs/go-jaccl-spec.md` (reference): Go-native collective API quality reference
- `/tmp/collab-F61A-gopccl-interface-to-257A.md` (reference): gopccl public interface guidance
- `/tmp/collab-F61A-ccl-bindings-to-257A.md` (reference): P0 CCL binding target roadmap
- gopccl uses non-generic Buffer metadata and explicit provider boundaries.
- Rendezvous and transport state machines are semantic package inputs, not ABI parse inputs.
- Real PCCL, NCCL, or RCCL providers remain build-tagged and manual-gated.

## Rendezvous

- Store interface: `Store`
- Fake store: in-memory key/value rendezvous

## Capability Metadata

- Provider metadata: generated `ProviderInfo` fields on `Capabilities`.
- Health/readiness: generated `HealthSnapshot` fields on `Capabilities`.

## Fake Transport

- State machine: `deterministic-drop-retry`
- Drop every: `2`
- Retry limit: `1`

## Resources

- `Group` wraps `backend` and is owned.

## Operations

- `Barrier` calls backend `barrier`; blocking, context policy `required`.
- `AllReduce` wraps `pcclAllReduce` calls backend `allReduce`; blocking, context policy `required`.
- `AllGather` wraps `pcclAllGather` calls backend `allGather`; blocking, context policy `required`.
  - `dst.Count == src.Count * uint(g.Size())`
- `Broadcast` wraps `pcclBroadcast` calls backend `broadcast`; blocking, context policy `required`.
- `AllToAll` wraps `pcclAllToAll` calls backend `allToAll`; blocking, context policy `required`.
- `ReduceScatter` wraps `pcclReduceScatter` calls backend `reduceScatter`; blocking, context policy `required`.
  - `src.Count == dst.Count * uint(g.Size())`
- `Send` wraps `pcclSend` calls backend `send`; blocking, context policy `required`.
- `Recv` wraps `pcclRecv` calls backend `recv`; blocking, context policy `required`.
- `PeersPending`: PeersPending reports whether the communicator has pending peers to accept. calls backend `peersPending`; blocking, context policy `pre-post`.
- `UpdateTopology`: UpdateTopology applies pending peer and topology changes. calls backend `updateTopology`; blocking, context policy `pre-post`.
- `OptimizeTopology`: OptimizeTopology measures links and optimizes the collective order. calls backend `optimizeTopology`; blocking, context policy `pre-post`.
- `SynchronizeState`: SynchronizeState reconciles named state tensors across the group. calls backend `synchronizeState`; blocking, context policy `pre-post`.

## Adapters

- `pccl` build tag `gopccl_pccl`; kind `pccl`; package `github.com/tmc/lib/pccl`; runtime `pccl`; device kind `DeviceCUDA`; manual gate `hardware`.
- `nccl` build tag `gopccl_nccl`; kind `nccl`; package `github.com/tmc/lib/nvidia/nccl`; runtime `cuda`; device kind `DeviceCUDA`; manual gate `hardware`.
- `rccl` build tag `gopccl_rccl`; kind `rccl`; package `github.com/tmc/lib/rocm/rccl`; runtime `rocm`; device kind `DeviceROCM`; manual gate `hardware`.

## Unsafe Policy

No public unsafe exposure is allowed.

## Element Sets

No element sets declared.
