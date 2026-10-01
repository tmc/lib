# Python stacks

`cudaprof capture --stacks` enables the standard-library-only
`cudaprofstacks.py` adapter. Capture supplies the embedded adapter on
`PYTHONPATH`, preserving existing entries. Instrument synchronous launch sites:

```python
import cudaprofstacks

with cudaprofstacks.scope("forward"):
    forward()
with cudaprofstacks.submission("launch"):
    launch()
```

`scope` captures its caller at region entry. It may cover several launches;
`submission` belongs immediately around one synchronous launch. Neither
propagates attribution to worker threads. A wrapped graph replay identifies
its submitter, not node construction sites. Frames are captured
in Python before submission, never in a CUPTI callback. The adapter reserves
CUPTI CUSTOM2; applications already using that namespace must disable it.
Names document call sites; attribution uses frames and external IDs.

Each region pushes and pops on the same native thread, including exceptional
exits. Failed pushes, failed pops, mismatched popped IDs, or missing native
session/process-generation handshakes mark attribution unavailable. Forked
children reset their state and require their own native handshake. There are
no PyTorch hooks, interpreter tracing hooks, or automatic imports.

Journals intern stacks and retain at most 64 frames per stack and 8 MiB per
process. Frame strings are limited to 4096 characters. A terminal coverage
receipt reports bounded collection; abrupt process termination may leave it
absent. Collection requires both `CUDAPROF_PYTHON_STACKS=1` and
`CUDAPROF_STACK_DIR`, set by capture.

An optional sampler records statistical wall snapshots:

```python
stop = cudaprofstacks.start_sampler(interval=0.01)
try:
    work()
finally:
    stop()
```

Snapshots include blocked Python threads and are not CPU execution samples.
They use `sys._current_frames`, retain native thread IDs when known, and write
separate bounded `python-wall-*.jsonl` sidecars. They are marked
`clock_unavailable` until an exporter verifies the native clock capability;
they never provide submission causality. Sampling acquires the GIL and can
perturb the workload. Timing overhead is not yet measured.

Run CPU-only adapter tests with
`python3 x/cmd/cudaprof/cudaprofstacks_test.py` from the module directory.
