# Topology-aware distributed inference design

This document defines prerequisites and safety boundaries for adapting an
inference plan to a changing cluster. It does not claim that distributed plan
switching, KV migration, or mixed Metal/CUDA execution is implemented.

Status: research and implementation proposal. Referenced components are
proposed unless their paths exist in the current tree.

## Prerequisites

Distributed adaptation begins only after the single-node runtime has:

- an authoritative compiled plan;
- explicit tensor dtype, layout, placement, and alias contracts;
- isolated sequence and KV state;
- true batched prefill and decode APIs;
- retained correctness and performance evidence; and
- stable plan and buffer lifetimes.

Without those contracts, a topology transition cannot determine what state must
move or whether the new execution is semantically equivalent.

## Semantic classes

Every transition belongs to one of two classes.

### Exact transition

An exact transition preserves the model, weights, numerical policy, request
state, and decoding algorithm. It may redistribute computation and tensor
shards, but it does not drop experts, change precision, discard KV entries, or
change token acceptance.

Tensor parallel and pipeline parallel executions may compute the same model,
but they are not a single local e-graph equality. Each is a distributed plan
with explicit weight placement, collective order, stage boundaries, activation
formats, sequence ownership, and failure behavior. Equivalence is established
by the plan semantics and differential tests.

### Degraded transition

A degraded transition changes accuracy, model capacity, context, or decoding
behavior. Examples include dropping MoE experts, reducing weight precision,
truncating KV state, or accepting an unverified speculative token.

These actions are never equality rewrites. They require an opt-in service
policy, quality evaluation, a visible response annotation, and a rollback path.
If no such policy is enabled, insufficient resources cause admission failure or
request failure.

Training synchronization methods such as DiLoCo do not define an exact
inference transition and are outside this design.

## Distributed plan

A plan records:

- membership epoch and node identities;
- device capabilities and supported dtypes;
- model and tokenizer revisions;
- weight and KV shard maps;
- operation placement and stage order;
- collective operations, participants, algorithms, and timeouts;
- activation and transport layouts;
- request and sequence ownership;
- checkpoint and recovery boundaries; and
- exact or degraded semantic class.

The plan is immutable within an epoch. Nodes reject work carrying an unknown
epoch, incompatible plan hash, or model hash.

## Topology observation

Topology input includes observed links and configured device capabilities.
Discovery alone does not establish usable bandwidth, latency, or a direct
route. Calibration probes are bounded, authenticated, and separated from user
traffic.

Each value carries provenance:

- `observed` for a bounded probe or runtime transfer;
- `configured` for an operator-supplied device limit;
- `modeled` for a cost-model result; or
- `unavailable` when no trustworthy source exists.

A cost model proposes candidates. It does not authorize a transition.

## Transition protocol

An exact transition uses a prepare-and-activate protocol:

1. Freeze the target membership epoch and build a candidate plan.
2. Validate model hashes, capabilities, memory capacity, layouts, and collective
   reachability on every participant.
3. Load or transfer immutable weight shards and verify their hashes.
4. Quiesce affected sequences at a declared token boundary.
5. Transfer required KV pages and sequence metadata with checksums.
6. Run a bounded reference or replay check on the destination plan.
7. Activate the complete plan at one epoch boundary.
8. Retain the old plan until all pre-transition steps have drained.

This protocol has barriers at correctness boundaries. “Barrierless” or
“uninterrupted” failover is not a goal unless a later implementation proves an
equivalent protocol with the same state guarantees.

## KV ownership and migration

KV pages have one authoritative owner per epoch or an explicit replicated
ownership protocol. Page identity includes sequence, layer, position range,
dtype, layout, model revision, and epoch.

Migration is idempotent and resumable. The receiver validates metadata and
checksums before publication. A sequence cannot execute on the new plan until
all required pages are present. Cancellation during migration releases both
source and destination leases according to a tested state machine.

Recomputing KV from retained prompt and generated tokens may be a valid recovery
path when latency and determinism permit it. It must be explicit and measured.

## Failure behavior

- A node loss before activation leaves the old epoch authoritative.
- A node loss after activation follows the new plan's recovery policy; it does
  not silently mix epochs.
- A collective timeout fails the affected step and marks its execution state
  uncertain until the backend proves otherwise.
- Partial weight or KV transfer is never published.
- If no exact replacement plan fits, the request waits, fails, or enters an
  explicitly enabled degraded mode.

Every failure includes the plan, epoch, sequence, and operation identity needed
to diagnose it.

## Heterogeneous clusters

Metal and CUDA nodes may participate only through a common tensor and transport
contract. Kernel binaries and calibration results remain backend-specific.
Placement accounts for transfer cost and synchronization, not only configured
TFLOPs or bandwidth.

Mixed-backend floating-point order may require a tolerance rather than bitwise
equality. The accepted tolerance is declared before profiling and validated on
the complete distributed plan.

## Security and admission

Peers authenticate before receiving weights, KV state, prompts, or generated
tokens. Authorization is scoped by model and service. Plans and transfers have
size limits, deadlines, and replay protection. Telemetry from a peer is treated
as untrusted input until bounded by local observations or operator policy.

## Proposed package boundaries

Names below describe concerns, not committed APIs:

```text
internal/distplan   immutable distributed plans and validation
internal/epoch      membership and activation state machine
internal/placement  candidate placement and cost modeling
internal/kvtransfer resumable KV page transfer
internal/calibrate  bounded link and collective probes
```

Transport and collective packages should expose small context-aware operations.
The planner depends on capability interfaces, not on backend-specific global
state.

## Acceptance sequence

1. Two-node static pipeline plan with fixed membership and synthetic small
   tensors, compared with a single-node reference.
2. Tensor-parallel plan with collective ordering and injected timeout tests.
3. Exact transition between two preloaded plans at a request boundary.
4. KV migration for one paused sequence with corruption, cancellation, and
   retry controls.
5. Node-loss tests before and after activation.
6. Real-model, real-device validation with retained route, correctness, and
   performance evidence.
7. Only then, evaluate adaptive placement and mixed-backend plans.

Expert dropping, dynamic quantization, and context truncation remain separate
degraded-service proposals. They are not acceptance criteria for exact
distributed inference.
