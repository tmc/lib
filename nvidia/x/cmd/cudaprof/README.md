# cudaprof

## Capturing only part of a run

By default `cudaprof capture` collects from the first CUDA call to exit.
To collect only a region, bracket it in the target with the CUDA profiler
API and pass `--capture-range=cudaProfilerApi`:

```sh
cudaprof capture --capture-range=cudaProfilerApi -o region.pprof -- ./app
```

Collection opens when `cudaProfilerStart` (or the driver's `cuProfilerStart`)
returns and closes when `cudaProfilerStop` (or `cuProfilerStop`) is entered,
so neither call is inside the range. Start and stop may repeat; each pair
is recorded as a range edge in the journal.

`--finalize=profiler-stop` is a separate request: it publishes the capture
at the first stop instead of at exit. Use it when the target leaves without
running libc's exit path (a Go program, or one that calls `_exit`). It ends
the capture at the first stop, so do not combine it with a target that
stops and starts again.

### From C or C++

```c
#include <cuda_profiler_api.h>

warmup();
cudaProfilerStart();
step();
cudaDeviceSynchronize();
cudaProfilerStop();
```

Link with `-lcudart`. The synchronize keeps the region's kernels inside it:
the range closes at the stop call, not when the queued work finishes.

### From Python with ctypes

Python programs usually load CUDA through a framework, so call the driver
entry points from `libcuda`, which there is only ever one copy of in a
process. Call them on a thread that already has a current CUDA context,
which is any thread that has done CUDA work:

```python
import ctypes
import torch

cuda = ctypes.CDLL("libcuda.so.1")

warmup()
torch.cuda.synchronize()
if cuda.cuProfilerStart() != 0:
    raise RuntimeError("cuProfilerStart failed")
step()
torch.cuda.synchronize()
if cuda.cuProfilerStop() != 0:
    raise RuntimeError("cuProfilerStop failed")
```

The agent sees the call even when it returns an error, so check the result:
a nonzero return means the range edge was recorded but the driver refused
the call, usually because no context was current.

`capture-status.json` holds a `phase_receipt` for the range. One start/stop
pair is measured there directly. A run that never completed a pair reports
the reason `edge_pair_invalid`, which is how a target that never called
start differs from one that ran no kernels. Several pairs report
`multiple_phases`, with one entry per pair.
