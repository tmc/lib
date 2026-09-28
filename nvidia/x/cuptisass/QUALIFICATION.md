# SASS workload qualification

`TestPhysicalWorkload` executes and checks a 256-thread integer kernel, requires
nonempty instruction records and zero dropped records, and validates every
function name/index and instruction PC against the CUBIN it loaded. It then
executes two identical launches and requires every per-PC metric count to double.
This is execution evidence; the older discovery/empty-session tests are not.

The experiment uses one controlled module. Its native CRC is retained as a
lookup key, not equated with a cryptographic digest or the separate 64-bit PC
sampling CRC. Counts aggregate launches in each flush window. It does not prove
general cross-process, same-name multi-module or graph replay attribution.

The native configuration calls need a current CUDA context even though they take
a device ordinal. Session setup and cleanup now establish the supplied context
on a locked thread and restore the caller's context. Callers need not pin their
own goroutine. CPU tests cover restoration errors without losing cleanup state.

Run from the `nvidia` module inside an exclusive GPU reservation. For the
shared development host, for example:

```sh
mkdir -p "$HOME/tmp/cupti-qualification"
GPU_LANE=cupti-qualification "$HOME/ml-explore/bin/gpu-slot" -- \
  env GOWORK=off TMPDIR="$HOME/tmp" \
  CUPTI_QUALIFICATION_DIR="$HOME/tmp/cupti-qualification" \
  go test -p=1 -tags=cupti_physical -count=1 -v \
  -run '^TestPhysicalWorkload$' ./x/cuptisass
```

The build tag requests actual hardware qualification. This test fails if CUDA,
CUPTI, the requested metric or required tools are unavailable; unsupported is not
a clean measurement. Without the tag it is excluded from normal CPU test runs.
The optional artifact directory must already exist. Each run creates its own
subdirectory containing source, PTX, exact loaded CUBIN, and compiler flags; the
test logs source/CUBIN SHA256 identities. Keep these binary artifacts outside Git.
