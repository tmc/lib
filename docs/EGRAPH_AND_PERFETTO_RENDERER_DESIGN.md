# E-graph and Perfetto renderer design

This document defines a truthful inspection UI for graph selection and GPU
traces in `examples/llama-infer`.

Status: partial implementation. The browser surfaces exist, but the current
graph endpoint is not an executed-plan view and the server does not register the
Perfetto trace endpoint requested by the dashboard.

## Current endpoint status

| Surface | Current behavior | Required label |
| --- | --- | --- |
| `/api/graph/plan` | Builds and returns a small RMSNorm/GEMV demonstration graph | `illustrative`, not `active plan` |
| Telemetry `execution_plan` field | Returns a deterministic configuration descriptor | `descriptive` until it drives dispatch |
| Graph partition labels | May mark operations as GPU or ANE candidates | `suggested target`; no ANE executor is wired |
| FlightRecorder timeline | Reports recorded forward/prefill regions | `recorded region`; not necessarily one event per GPU kernel |
| `/debug/pprof/gpu` | Existing GPU profile fallback used by the dashboard | Preserve its actual format and provenance |
| `/debug/perfetto.pftrace` | Requested by `dashboard.js` but not registered by the server | `unavailable` until implemented and tested |

The first implementation task is to make the UI display these states rather
than implying that a demonstration graph or placement suggestion controlled
execution.

## Data model

### Graph view

A graph response contains:

- `kind`: `illustrative`, `candidate`, `selected`, or `executed`;
- model, plan, phase, and shape-bucket identities when applicable;
- nodes with semantic operation, backend operation, dtype, shape, layout, and
  suggested or actual target;
- edges with tensor identity and byte size when known;
- rewrite provenance and candidate rejection reason;
- selected-plan hash; and
- source revision and dirty state.

`executed` is admitted only when the backend consumed that plan identity and a
trace or test records the same identity. `selected` means the compiler selected
it; it does not prove execution.

### Trace view

A trace event contains:

- backend, queue or stream, and device identity;
- plan, phase, bucket, sequence batch, and operation identity;
- start and duration with clock source;
- grid and threadgroup or block geometry when observed;
- synchronization and host-transfer markers; and
- labels whose provenance is known.

Modeled bytes or FLOPs are separate fields tagged `modeled`. Do not emit them as
hardware-counter observations.

## Graph pipeline

```text
semantic model graph
        |
        +--> candidate rewrites and rejection records
        |
        v
selected compiled plan
        |
        v
backend dispatch consumes plan identity
        |
        v
trace records executed plan and operation identities
```

The renderer should let the user compare these stages, but must not invent a
stage that the runtime does not expose. Until authoritative extraction exists,
the current demonstration graph stays in an explicitly named example endpoint
or response kind.

The graph package owns graph serialization and validation. The HTTP server owns
transport, size limits, and request authorization. Browser code owns layout and
interaction; it does not reinterpret a suggested target as observed execution.

## Perfetto integration

Perfetto's supported embedded flow uses the UI iframe messaging API documented
in the [Perfetto embedding API reference](https://perfetto.dev/docs/visualization/embedding-api-reference):

1. create an iframe for the official Perfetto UI;
2. wait for its `PING` message;
3. verify the message came from the iframe window and an allowed origin; and
4. send a message containing `{perfetto: {buffer, title}}`, transferring the
   trace buffer when supported.

The server first needs a real `/debug/perfetto.pftrace` handler. The handler
must return a valid trace with an explicit content type, bound its size and
generation time, propagate serialization errors, and use the same
authentication policy as other diagnostic endpoints.

The UI should retain a download action as a fallback. It must report whether the
embedded viewer rejected the message, the endpoint was unavailable, or the
trace was empty. Falling back to a different profile format is not a successful
Perfetto load.

### Browser security

- Validate `event.source` against the created iframe window.
- Restrict accepted origins; do not use an unchecked wildcard receiver.
- Apply a Content Security Policy and the cross-origin opener policy required
  by the chosen viewer flow.
- Treat graph labels and trace strings as untrusted text.
- Bound graph node count, trace bytes, and render work.
- Do not expose prompts, tokens, tensor contents, filesystem paths, or peer
  credentials in traces by default.

## UI behavior

The graph explorer provides:

- filtering by semantic operation, backend operation, phase, and state;
- side-by-side candidate and selected-plan views;
- a reason for every rejected candidate;
- explicit `suggested` versus `executed` placement badges; and
- a link from a trace event to its plan operation when both identities match.

The timeline distinguishes host, Metal queue, CUDA stream, transfer, and
synchronization events. If only a whole-forward region exists, render one
region; do not synthesize a per-kernel waterfall.

## API failure semantics

- Unsupported graph or trace data returns a structured `unavailable` response
  or an appropriate HTTP error.
- Serialization errors return errors; they do not produce an empty success.
- Unknown plan identities are rejected.
- A partial trace is labeled partial with its coverage interval.
- A zero-event trace is not accepted as evidence of GPU activity.

## Implementation sequence

1. Rename or label the existing demonstration graph response as illustrative.
2. Define shared graph and trace schemas with provenance and size validation.
3. Thread an authoritative plan identity from compilation through backend
   dispatch and FlightRecorder events.
4. Register and test `/debug/perfetto.pftrace`, including invalid, empty,
   oversized, and unauthorized cases.
5. Implement the guarded Perfetto iframe handshake and retain download fallback.
6. Add candidate, selected, and executed graph views only as the runtime exposes
   those stages.
7. Add browser tests for status labeling, cross-origin rejection, missing data,
   and trace-to-plan linking.

The endpoint and provenance work precedes graph animation or richer rendering.
A precise small view is more useful than a detailed visualization of inferred
state.
