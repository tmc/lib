Go stack collection is disabled by default. Capture's `--stacks` option
sets `CUDAPROF_GO_STACKS=1` for programs importing cudaprof. Each process
writes its own bounded journal. Applications must reserve CUPTI Custom2
for Go stacks.

`Do` records the stack at scope entry. A scope may cover several launches.
Use `Submit` immediately around a synchronous CUDA submission to record
its call site instead:

```go
err := cudaprof.Submit(func() error {
    return launchKernel()
})
```

The external ID stays on the submitting OS thread until the callback returns.
Work dispatched to another thread does not inherit it. Graph replay identifies
the replay submission; it does not recreate individual node creation stacks.
Neither path reads stacks from asynchronous CUPTI callbacks.

Call `FinishStackRecording` after all work completes to disable new recording,
wait for entered scopes, and write final producer coverage. Without a final
receipt, a reader cannot claim complete stack collection. For a manually
supplied journal, append `WriteStackCoverage` to the same journal writer before closing it.
Failures and truncation remain visible in the records and receipt.

`StartCaptureHostProfile` explicitly starts a separate runtime execution trace
and CPU profile in the capture sidecar directory. Call its returned `Stop`
before process exit. `StartHostProfile` accepts caller-owned writers instead.
Both helpers refuse an existing runtime trace or CPU profiler and preserve
its owner. Direct calls to runtime start/stop APIs must be coordinated by the
application because the runtime exposes no ownership tokens.

The CPU pprof contains aggregate samples, without individual timestamps.
The execution trace retains timestamped scheduler events and selected stacks;
it is not a timestamped copy of CPU pprof. The sidecar receipt records wall-clock
bounds and reports its GPU clock alignment as unavailable. Retaining these
files alone does not justify positioning CPU samples on a GPU timeline.

Export decodes retained Go 1.27.1 traces using a matching local Go tool.
On Linux, it verifies the producer's native session, process generation, and
unified monotonic clock before drawing CPU samples on separate host tracks.
Other versions or missing clock evidence remain unavailable. The host-stack
query retains timestamp quantization, clock snapshots, and observed read skew.
