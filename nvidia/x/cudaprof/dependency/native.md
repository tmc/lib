# Native dependency evidence

`CUDAPROF_DEPENDENCIES=1` enables optional driver callback receipts. Each receipt
retains callback entry/exit, a process-local observation ordinal, API result,
correlation/context, stream/event handles and timestamp domain. The producer
reports `dependency_coverage=driver_partial`. Runtime-only submissions, default
or blocking streams, unknown submission variants and incomplete graph snapshots
remain partial. Callback collection changes overhead and requires its own
measurement; CPU analysis tests do not measure collector overhead.

`Observe` builds stream ordering only from observed nonblocking stream lifetimes
and nonoverlapping successful callback pairs. Event edges require an observed
live event and a uniquely ordered successful record/wait. Successful stream,
event and context synchronization creates a zero-duration completion node.
Graph edges require explicitly supplied kernel-node dependencies, queried CUPTI
node IDs, exact executable lifecycle versions and observed clone relationships.
Child graph source versions are retained; child DAG expansion remains partial
when distinct copied node identities are unavailable. Graph membership,
submission correlation and timestamp proximity do not supply graph edges.

The controlled workload is `cmd/cudaprof/testdata/dependency.c` with
`dependency_kernel.cu`. It creates two nonblocking streams, a two-node graph,
a parallel kernel, an event wait, a combine kernel, an asynchronous host copy
and a successful stream synchronization. Its independent invariant is four
kernel records plus one memcpy record and host output `[11,22,33,66]`. It emits
one delivered marker and one token-limit marker on the submitting thread.
Missing NVTX entry points fail the workload.

Build without running the GPU:

```
nvcc -ptx -arch=compute_80 testdata/dependency_kernel.cu -o ~/tmp/dependency_kernel.ptx
cc -Wall -Wextra -Werror -I/usr/local/cuda/include testdata/dependency.c \
  -L/usr/local/cuda/lib64/stubs -lcuda -ldl -o ~/tmp/dependency
```

Capture this native program with driver API records, markers, graph lifecycle
and dependency collection enabled. The existing Go/Python subprocess wrappers
can launch it with the PTX argument; instrumentation belongs to the native
submitting thread and does not compare Go and Python inference engines.

```
cudaprof dependency -token 0 -expected-activities 5 \
  -html ~/tmp/dependency.html -perfetto ~/tmp/dependency.pftrace capture.pprof
```

`-expected-activities` is an independently supplied workload invariant, never
the observed capture count. Qualification requires a complete collector receipt,
one scoped token range, one scoped delivered instant, all submitting API calls
inside that range and one successful synchronization in its materialization
range. Every work branch must reach that completion. This qualifies only the
bounded controlled request; arbitrary application captures retain a partial
view. CPU tests and a native build do not constitute live GPU validation.
