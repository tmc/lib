# Application ranges

Use existing NVTX push/pop ranges with `cudaprof:v1 ` followed by JSON.
The required `phase` is `prefill` or `decode`. Optional `token` and `layer`
are zero-based indices; `operation` is an application label; `lookahead` and
`materialization` are booleans. Optional `dtype` is an application-defined
nonempty string, and `cache_length` is a nonnegative integer. Omission means unknown, including booleans.
Repeat context on nested ranges: missing fields do not inherit values.

The Go package example uses `x/nvtx`. Python applications can use PyTorch's
existing NVTX surface:

```python
import json
from contextlib import contextmanager
import torch

@contextmanager
def operation(enabled, **metadata):
    if not enabled:
        yield
        return
    # Keep push and pop on this host thread.
    message = "cudaprof:v1 " + json.dumps(metadata, separators=(",", ":"))
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

with operation(enabled=True, phase="decode", token=3, layer=0,
               operation="gemv", lookahead=False):
    pass  # Submit application work here.
```

The disabled path does not encode metadata or invoke NVTX. Python requires a
CUDA-enabled PyTorch build when enabled; the Go NVTX implementation is a no-op
without a provider. Neither example measures instrumentation overhead.

Ranges use the collector's NVTX timestamps. They identify host scope, not the
GPU completion boundary. Overlap on different threads is valid. Same-thread
crossing ranges are invalid; nested conflicting labels are ambiguous and must
not be selected as authoritative kernel context. Parsing validates messages
only. Pairing, crossing-range detection, and context conflict receipts belong
to consumers that have timestamps and process/thread identity. An absent range,
an unpaired half, and a malformed message are distinct missing-evidence states.
Keep raw names, intervals, and marker counts alongside parsed metadata.

These observed labels do not replace inferred `arg_reduce` boundaries or prove
workload equivalence. Cross-thread submission and asynchronous work require
correlation evidence beyond range containment.
