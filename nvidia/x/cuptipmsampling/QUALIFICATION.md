# PM workload qualification

`TestPhysicalWorkload` creates a PM-specific host configuration, requires one
pass, runs 20 launches and checks every output against an integer CPU reference.
It requires completed samples, no hardware overflow, a drained hardware buffer,
ordered positive timestamp spans, and finite positive cycle measurements over
intervals overlapping the workload's CUPTI timestamp window.

PM intervals describe the whole device. The window encloses host submission
through copy completion, not just device execution. These measurements are not
per-kernel counters and do not distinguish concurrent kernels. The test does not
infer coverage from a fixed sample count: sampling boundaries can vary. An
external exclusive GPU reservation is part of its measurement assumptions.

The test uses the native PM host configuration API; a Range Profiler plan is not
a valid substitute. The fixture intentionally keeps this low-level configuration
local rather than introducing an unreviewed public scheduling API.

Run from the `nvidia` module inside an exclusive GPU reservation. For the
shared development host, for example:

```sh
mkdir -p "$HOME/tmp/cupti-qualification"
GPU_LANE=cupti-qualification "$HOME/ml-explore/bin/gpu-slot" -- \
  env GOWORK=off TMPDIR="$HOME/tmp" \
  CUPTI_QUALIFICATION_DIR="$HOME/tmp/cupti-qualification" \
  go test -p=1 -tags=cupti_physical -count=1 -v \
  -run '^TestPhysicalWorkload$' ./x/cuptipmsampling
```

The build tag requests actual hardware qualification. This test fails if CUDA,
CUPTI, the requested metric or required tools are unavailable; unsupported is not
a clean measurement. Without the tag it is excluded from normal CPU test runs.
The optional artifact directory must already exist. Each run creates its own
subdirectory containing source, PTX, exact loaded CUBIN, and compiler flags; the
test logs source/CUBIN SHA256 identities. Keep these binary artifacts outside Git.
