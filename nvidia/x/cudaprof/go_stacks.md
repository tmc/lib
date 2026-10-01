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
