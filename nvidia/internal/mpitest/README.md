# Two-rank physical tests

Run these opt-in tests on Linux with cgo, two distinct visible CUDA GPUs, a
compatible cuQuantum SDK, and **CUDA-aware Open MPI**. Both ranks must see GPU
indices 0 and 1; each rank selects its rank-numbered GPU. Run one test job at a
time. A successful MPI version query does not establish CUDA awareness: first
verify an MPI transfer between device buffers on these GPUs with a numerical
oracle. Ordinary CPU-only MPI can build these tests but cannot qualify them.

From the `nvidia` module directory, set paths to existing installations. The
examples assume Open MPI uses `include` and `lib` under its installation prefix;
adjust both consistently if the installation uses a different layout.

```sh
export MPI_PREFIX=/absolute/path/to/cuda-aware-openmpi
export CUQUANTUM_ROOT=/absolute/path/to/cuquantum-sdk
export CUDA_ROOT=/usr/local/cuda
export CUTENSOR_ROOT=/absolute/path/to/cutensor-sdk
export TEST_OUT="$HOME/tmp/nvidia-mpi-tests"
mkdir -p "$TEST_OUT"
export PATH="$MPI_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$CUQUANTUM_ROOT/lib:$CUTENSOR_ROOT/lib:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GOWORK=off CGO_ENABLED=1
export CGO_CFLAGS="-I$MPI_PREFIX/include"
export CGO_LDFLAGS="-L$MPI_PREFIX/lib -Wl,-rpath,$MPI_PREFIX/lib -lmpi"
```

Use this same MPI installation for the cgo helper, launcher, and providers.
`mpitest` obtains the actual native `MPI_Comm` representation and its size,
duplicates `MPI_COMM_WORLD`, and keeps the duplicate alive until library handles
are destroyed. Never substitute numeric MPI handle values. Each test initializes
MPI on a locked OS thread and keeps MPI calls on that thread.

The cuStateVec test passes a NULL soname to `NewCommunicator`. Its cgo helper
links MPI into the executable at startup, so cuStateVec resolves the already
loaded Open MPI symbols. The test does not exercise explicit soname loading.

Build the cuTensorNet and cuDensityMat providers from the official SDK sources,
using the same MPI headers and library. These are provider libraries, separate
from the SDK's main shared libraries.

```sh
for package in cutensornet cudensitymat; do
  gcc -shared -std=c99 -fPIC \
    -I"$CUQUANTUM_ROOT/include" -I"$CUDA_ROOT/include" \
    -I"$MPI_PREFIX/include" \
    "$CUQUANTUM_ROOT/distributed_interfaces/${package}_distributed_interface_mpi.c" \
    -L"$MPI_PREFIX/lib" -Wl,-rpath,"$MPI_PREFIX/lib" -lmpi \
    -o "$TEST_OUT/lib${package}_mpi.so"
done
export CUTENSORNET_COMM_LIB="$TEST_OUT/libcutensornet_mpi.so"
export CUDENSITYMAT_COMM_LIB="$TEST_OUT/libcudensitymat_mpi.so"

for package in custatevec cutensornet cudensitymat; do
  go test -mod=readonly -tags=nvidiampi -c \
    -o "$TEST_OUT/$package.test" "./$package"
done
```

Launch each compiled test once per rank; do not run `go test` itself under MPI.
The following Open MPI transport selection matches the bounded single-node
TCP validation setup. Other MPI transports require their own qualification.
GNU `timeout` bounds the launcher; retain logs and exit status, and verify that
all rank processes exit afterward.

```sh
CUSTATEVEC_DISTRIBUTED_TEST=1 timeout -k 5s 75s \
  mpiexec --mca btl self,tcp --mca pml ob1 -n 2 \
  "$TEST_OUT/custatevec.test" -test.v -test.run='^TestDistributedSwap$' -test.timeout=60s

CUTENSORNET_DISTRIBUTED_TEST=1 timeout -k 5s 75s \
  mpiexec --mca btl self,tcp --mca pml ob1 -n 2 \
  "$TEST_OUT/cutensornet.test" -test.v -test.run='^TestDistributedContraction$' -test.timeout=60s

CUDENSITYMAT_DISTRIBUTED_TEST=1 timeout -k 5s 75s \
  mpiexec --mca btl self,tcp --mca pml ob1 -n 2 \
  "$TEST_OUT/cudensitymat.test" -test.v -test.run='^TestDistributedNorm$' -test.timeout=60s
```

For a negative oracle control, repeat each successful command with its matching
`CUSTATEVEC_BAD_ORACLE=1`, `CUTENSORNET_BAD_ORACLE=1`, or
`CUDENSITYMAT_BAD_ORACLE=1` environment assignment. Require a nonzero test exit
and the intentional numerical mismatch in the log; a prerequisite error or
timeout does not qualify that control. Remove the assignment before positive
runs.

The cuStateVec missing-rank control deliberately holds rank 1 before the swap.
Run it only with an external deadline shorter than the hold:

```sh
CUSTATEVEC_DISTRIBUTED_TEST=1 CUSTATEVEC_MISSING_RANK=1 timeout -k 2s 8s \
  mpiexec --mca btl self,tcp --mca pml ob1 -n 2 \
  "$TEST_OUT/custatevec.test" -test.v -test.run='^TestDistributedSwap$' -test.timeout=60s
```

Expect external termination and check that no rank remains. This control proves
the harness bounds a stalled job; it does not establish library fault recovery
or orderly native cleanup after process termination.

The numerical scope is one local/global index-bit swap of eight complex
amplitudes, one sliced 2-by-3 times 3-by-2 contraction, and one partitioned
pure-state squared norm with global index coverage. These tests do not cover
all distributed APIs, MPI implementations, node topologies, data types, or
partial-failure paths. Record source and executable hashes, SDK/MPI/provider
versions, GPU identities, command environments, logs, exit codes, and cleanup
errors with each result. Reset/rank-query success alone is not a numerical pass.
