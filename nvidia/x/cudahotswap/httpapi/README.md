# Hot-swap HTTP transports

The application creates `httpapi.New(service, resources)` and serves it on a
listener from `httpapi.Listen`. Unix sockets default to
`unix:/var/run/cudahotswap.sock`; the application owns directory permissions.
TCP uses plain HTTP. Keep it on loopback and use an SSH forward.

All service methods appear in `httpapi.Endpoints()`. Each endpoint takes POST
JSON except GET `/v1/events?since=SEQ`. Events use SSE by default or JSON lines
with `Accept: application/x-ndjson`. The client example in `stream_test.go`
round-trips against an `httptest` server.

Code/configuration mutations pass `label` through to the service and preserve
the actual generation, config revision and State.Label. For operations without a
service label argument, `request_label` is transport request metadata; it does
not change service state. Watch events retain their producer's label and stamp.

Named resources come from the embedding application: checkers, buffers,
streams and graph bindings. A client cannot supply a checker implementation or
native pointer. The application's `Arguments` binding fixes each kernel's
pointer-versus-scalar argument kinds; unbound launches fail closed. PTX integer
storage alone is insufficient to infer that intent. Scalars use
`{ "type": "u32", "value": 32 }`; a buffer uses
`{ "type": "buffer", "name": "x" }`. Fixture buffers additionally declare
their argument index and byte extent. The application must exclude concurrent
buffer writes while replay records or uses them.

Pin and launch responses retain opaque `handle` values even on partial errors.
Release pins with `release_pin`, drain launches with `wait`, and release fixtures
with `close_fixture`, supplying `handle` and `label`. Failed cleanup retains the
owner for retry. Close cancels streams and drains launches before releasing
pins and closing the service; the application frees native resources only after
Close succeeds. The handler limits retained handles to 128 and streams to 32.

Failed checks return HTTP 200 with the classified report. Malformed requests
return 400; typed service poisoning returns 503. Other service failures remain
structured HTTP 200 failures. Cleanup endpoints remain available after poisoning.

## CLI and SSH

Build the CUDA-host CLI from `nvidia/`:

```sh
GOWORK=off go build -o ./cudahotswap ./x/cmd/cudahotswap
./cudahotswap demo -addr tcp:127.0.0.1:8713
```

The demo owns SAXPY buffers and runs a CPU-reference Checker. It acquires
GPU-lock leases for setup, readback and teardown; Service owns per-request launch
leases. Do not place an outer GPU lock around this long-lived server, because it
would block the service's own requests.
Its buffers remain private to the serialized loop and checker. General
applications supply named launch/fixture resources through `httpapi.Resources`.

On the Mac, in a separate terminal:

```sh
ssh -N -L 127.0.0.1:8713:127.0.0.1:8713 dgx
cudahotswap status -addr tcp:127.0.0.1:8713 -name saxpy
cudahotswap push -addr tcp:127.0.0.1:8713 -name saxpy -entry saxpy -source saxpy.cu -label human
cudahotswap tail -addr tcp:127.0.0.1:8713 -since 0
cudahotswap -addr tcp:127.0.0.1:8713 -watch saxpy.cu -name saxpy -entry saxpy
```

Local watch polls every 100 ms and pushes changed bytes. The service's own Watch
endpoint uses the lane 61 directory watcher and debounce semantics.

## MCP

The adapter lives in a separate module, `x/cmd/cudahotswapmcp`. Its SDK dependency
does not enter `nvidia/go.mod`. From that directory:

```sh
GOWORK=off go build -o ./cudahotswapmcp .
./cudahotswapmcp -addr tcp:127.0.0.1:8713
```

The process speaks MCP stdio. Its generated JSON schemas are returned by MCP
`tools/list`; the following table identifies every tool's arguments. A failed
report sets MCP `isError` and preserves the classified reason.

| Tool | Arguments | HTTP endpoint |
| --- | --- | --- |
| push_kernel | name, label, source: code, entry, optional filename/flags | push |
| revert_kernel | name, label | revert |
| list_kernels | none | list |
| kernel_status | name, label | status |
| set_check | name, label, optional checker/fixture/policy/keep_on_failure | set_check |
| set_launch | name, label, geometry | set_geometry |
| record_fixture | name, label, stream, geometry, args, buffers, outputs, optional policy | record_fixture |
| select_version | name, label, version | select |
| wait_for_report | name, since, timeout_seconds (1..120) | events |

Geometry uses `Grid` and `Block` objects, each with `X`, `Y`, `Z`, and `SharedMem`.
A null geometry removes a configured override through set_launch. Fixtures
require geometry, named stream/buffers and outputs. Checker names and fixture
handles must already exist on the server. A retention gap is an explicit error
from wait_for_report, never a fabricated success.

## Native consumer

The adapter's `TestNativeMacMCPWrongThenCorrect` is explicitly gated by
`CUDAHOTSWAP_NATIVE=1`. It runs on the Mac, through the operator-owned forward,
starts the actual MCP binary over stdio, pushes a wrong SAXPY and then the correct
one, and retains both MCP reports and submitted source digests. It requires
absolute `CUDAHOTSWAP_MCP_BINARY` and `CUDAHOTSWAP_ARTIFACTS`, and
`CUDAHOTSWAP_ADDR=tcp:127.0.0.1:8713`. It has not been physically qualified by
the implementation's host gates.
# Host stages

Applications may implement the optional `StageService` extension and supply
`Resources.Stages`. Each entry fixes a retained builder, immutable `StageSpec`,
ordered buffer registry names, stream registry name, CUDA mode and compiler
flags. The client cannot select a compiler executable, context schema or pointer.
The application retains these owners and generation directories until
`Handler.Close` succeeds.

The five routes are `register_stage`, `push_stage`, `revert_stage`, `run_stage`
and `record_stage_fixture`. Registration and push take `name`, `label` and
`stage_code`; compiler flags and mode come from the application binding. Revert
takes only the named stage and label. Stage calls refuse kernel geometry.

Run takes the declared stream name and ordered buffer-only `args`. Its response
has a completion `handle` and no invented state stamp. A handle returned with a
failure still owns work. Call `wait` with that handle and a label, retrying after
cancellation or cleanup failure. Fixture calls require the same ordered arguments,
exact complete buffer extents and explicit outputs/policy; their replay arguments
are nil placeholders for the service-owned scratch buffers. Host-only stages
cannot record native replay fixtures.

The host lifecycle test compiles and loads actual CPU-only C glue. Synthetic
native-dispatch tests cover serialization and refusal, not GPU execution or
native resource qualification.
