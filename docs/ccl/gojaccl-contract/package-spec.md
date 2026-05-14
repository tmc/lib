# Package Spec: gojaccl

Package gojaccl provides Go-native JACCL collective operations.

- Import path: `github.com/tmc/gojaccl`
- ABI package: `internal/abi`
- ABI library base name: `jaccl`
- Public API note: This is a contract-alignment plan for the existing gojaccl package; it is not a replacement for the JACCL RDMA implementation.

## Parse Inputs

- Headers: `jaccl.h`

## Semantic Sources

- `gojaccl/docs/go-jaccl-spec.md` (reference): accepted Go JACCL public API
- `gojaccl/docs/go-doc-output.txt` (reference): current public go doc output
- `lib/jaccl/ring.cpp` (c++): ring collective semantics reference
- `lib/jaccl/rdma.cpp` (implementation): manual RDMA gate semantics reference
- `/tmp/collab-F61A-ccl-bindings-to-257A.md` (reference): P0 CCL binding target roadmap
- JACCL C++ and RDMA sources shape contract, capability, readiness, and health signals only.
- Do not feed C++ implementation sources into ABI parsing.
- Do not claim hardware proof without JACCL_TEST_RDMA and RTR gates.

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
- `AllReduce` wraps `jacclAllReduce` calls backend `allReduce`; blocking, context policy `required`.
- `AllGather` wraps `jacclAllGather` calls backend `allGather`; blocking, context policy `required`.
  - `dst.Count == src.Count * uint(g.Size())`
- `Broadcast` wraps `jacclBroadcast` calls backend `broadcast`; blocking, context policy `required`.
- `AllToAll` wraps `jacclAllToAll` calls backend `allToAll`; blocking, context policy `required`.
- `ReduceScatter` wraps `jacclReduceScatter` calls backend `reduceScatter`; blocking, context policy `required`.
  - `src.Count == dst.Count * uint(g.Size())`
- `Send` wraps `jacclSend` calls backend `send`; blocking, context policy `required`.
- `Recv` wraps `jacclRecv` calls backend `recv`; blocking, context policy `required`.

## Adapters

- `jaccl` build tag `gojaccl_rdma`; manual gate `rdma`.

## Unsafe Policy

No public unsafe exposure is allowed.

## Element Sets

No element sets declared.
