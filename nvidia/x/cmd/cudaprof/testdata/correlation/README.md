# Physical stream and graph correlation

From the `nvidia` module, run under an exclusive GPU hold, where
`$GPU_LOCK` is the host's exclusive-GPU wrapper:

```sh
GPU_LANE=correlation $GPU_LOCK -- env \
  GOWORK=off TMPDIR="$HOME/tmp" CUDAPROF_CORRELATION_TEST=1 \
  CUDAPROF_CORRELATION_OUTPUT="$HOME/tmp/correlation" \
  go test ./x/cmd/cudaprof -run '^TestPhysicalCorrelation$' -count=1 -v
```

The test builds the native CUPTI agent from this checkout and runs this Go
fixture in a fresh process. CUDA, CUPTI, NVRTC, a C compiler and a GPU are
required. An explicitly enabled test fails if any prerequisite or evidence is
missing. `CUDAPROF_CORRELATION_AGENT` optionally selects an already-built agent,
for example to check a backport. Without the opt-in the test skips.

The fixture compiles two different `update` kernels with the same symbol and
source filename. It loads native CUBINs, records their SHA-256 identities and
retains the exact source bytes separately, keyed by image hash. `prepare`
copies 32 integers to the device and adds one on a stream. An event connects
that stream to `advance`, which replays a one-node graph three times on another
stream, adding two each time. It synchronizes, copies back, and checks that
every result is seven. NVTX ranges are pinned to the submission thread.

The independent work contract requires four kernels, two 128-byte copies,
one graph node, three replay correlations and two kernel streams. The test
uses the existing activity-to-profile projection to check that every kernel
and copy belongs to its own NVTX operation. It verifies the loaded image hash
for every execution, the source hash, event ordering, host API evidence,
synchronization records, and separate kernel/memory time dimensions. Host
waits must not become device-time samples.

Negative controls remove a real journal frame, remove a parsed activity, remove a loaded image, remove a
source artifact, and substitute the other same-named kernel's source. The
qualification refuses partial or unresolved evidence. Both the capture reader and
the existing gaps report reject the damaged journal; missing work is not
reported as idle. This work-count oracle
belongs to the fixture: a producer drop count of zero cannot establish those
expected counts for an arbitrary application. The test does not add an
automatic application-invariant flag to the CLI or an automatic NVRTC source
collector. The retained source-to-image relation comes from this fixture's
compile manifest, not from matching kernel names or paths.

When an output directory is supplied, a new `run-*` subdirectory retains the ordinary capture bundle,
contract JSON and exact sources for inspection. Detailed test output should be
saved separately. No performance conclusion follows from this tiny fixture.

The regression fixes explicit driver kernel-node attribution through graph
instantiation/cloning. Captured graphs, generic graph-node creation and
runtime-created kernel nodes remain unresolved when their function identity
was not observed. Graph parameter/update callbacks conservatively invalidate
all retained graph-module identities: delayed activity must not be attributed
to a newer function. The table is bounded to 4096 nodes; exhaustion remains
unresolved. Updating graphs is tested with the native callback harness, not a
physical graph-update workload. This is not source-line or PC-sample
qualification through graphs.
