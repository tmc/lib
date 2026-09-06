# Initcheck and synccheck fixtures

`checkers.cu` supplies clean and intentionally invalid cases for the existing
Go runner. Initcheck's bad case reads a newly allocated, unwritten global input;
its output is copied back so the compiler must preserve the load. Synccheck's
bad case calls `__syncwarp` from lanes excluded by its mask. All named lanes
participate, avoiding a deliberately absent participant or divergent block-wide
barrier. Invalid synchronization is still an error: run only under synccheck and
an external deadline, never as a standalone correctness test.

These patterns follow NVIDIA's [Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html),
particularly initcheck and synccheck's invalid-arguments classification. No native
result is established by checking in this source or compiling the Go tests.

The command below was validated with CUDA 13.0 on GB10. From the `nvidia`
module, on an authorized CUDA GPU host:

```sh
export CHECKER_OUT="$HOME/tmp/cudasanitizer-checkers"
mkdir -p "$CHECKER_OUT"
# Set to the actual GPU target supported by the installed nvcc.
export CUDA_ARCH=sm_121
nvcc -lineinfo -arch="$CUDA_ARCH" -o "$CHECKER_OUT/checkers" \
  x/cudasanitizer/testdata/checkers.cu
LIB_CUDA_HARDWARE_TEST=1 \
CUDASANITIZER_TEST_CHECKERS_PROG="$CHECKER_OUT/checkers" \
GOWORK=off timeout -k 5s 240s go test -count=1 -v \
  -run '^TestAdditionalCheckersCUDA$' -timeout=220s ./x/cudasanitizer
```

Each checker runs the same executable with `clean` and `bad` arguments. Clean
cases verify all 32 output values and checked CUDA cleanup before printing their
completion marker. Bad cases require a successful launch marker, a nonzero
parsed sanitizer summary, and the expected finding family. They do not require
successful CUDA synchronization after an intentionally invalid kernel: the
checker may terminate erroneous work. Such a run does not prove clean teardown
of invalid GPU work. The runner applies a 45-second deadline per invocation;
the outer deadline bounds the complete four-case test.

Without `LIB_CUDA_HARDWARE_TEST`, tests explicitly skip. With it enabled, missing
tooling or the fixture path fails. Retain nvcc/tool versions, GPU/driver identity,
source and executable hashes, command/logs, outcomes and skipped cases. Keep the
binary outside Git. These are checker acceptance fixtures, not application code.
