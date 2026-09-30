# Controlled native evidence

`cudaprof probe` runs a deterministic one-thread CUDA write with versioned
application metadata. Arrange exclusive GPU use outside the command. The
command does not acquire a GPU lock itself.

```
cudaprof probe
cudaprof probe -managed
cudaprof probe -metrics sm__cycles_elapsed.avg
cudaprof probe -um
```

The ordinary controls reset output to zero, launch, synchronize, and check for
seven. Wall times include reset, launch and synchronization. They are diagnostic
controls, not independent performance estimates. The counter run repeats the
same idempotent operation, checks each replay output, retains native metric
definitions, driver/CUPTI/build versions, replay mode and collection-quality receipts, and evaluates only a successful
counter image. Requested metrics come from the native device inventory; a
failure is retained as JSON and also returns a failing exit status. Collection
refusal is not a zero counter result. Neither this kernel nor its marker
qualifies an application shape such as a Gemma4 GEMV.

`-um` runs in its own process, separately from range collection. It attempts
native configuration of HtoD bytes, DtoH bytes, CPU page faults and GPU page
faults individually, disabling each successful configuration. Receipts retain
native result codes when available; a missing library symbol has no native
result code. Configuration success does not establish that activity collection
works. This probe does not register an activity callback or decode UM records.

Device attributes report allocation/access capabilities. `-managed` changes the
allocation API only. Neither attributes, successful allocation, successful
configuration nor correctness establish physical residency or observed
migration. [NVIDIA's current CUPTI release notes](https://docs.nvidia.com/cupti/release-notes/release-notes.html)
state that unified-memory profiling is unsupported on ARM. The receipt records
that documented limitation independently of the native API probe result.

Counter overhead is explicitly `unmeasured`. Before/after ordinary controls do
not measure the profiled run duration and cannot estimate profiler overhead.
