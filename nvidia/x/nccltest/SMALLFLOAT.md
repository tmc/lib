# Reduced-precision datatype work

The half, bfloat16, E4M3, and E5M2 codecs are implemented and tested. These
datatypes are accepted by Case and both commands, with native element widths,
NCCL enums, buffers, and premultiplication scalars. Local tests and single-rank native collective/omission tests pass for all four
datatypes. Completed matched upstream matrices remain pending; see DGX.md.

`floatformat.go` implements round-to-nearest-even, subnormals, signed zero,
IEEE overflow for half/bfloat16, and CUDA's finite saturation for FP8. E4M3's
extended finite exponent range and sole NaN magnitude are handled separately.

`TestCUDAFloatCodec` builds a CPU-only fixture against actual CUDA headers:

```sh
NCCLTEST_CODEC_HEADERS=/path/to/cuda-headers GOWORK=off go test ./x/nccltest -run '^TestCUDAFloatCodec$' -count=1 -v
```

The header root uses the cuda_cudart, cuda_crt, and cuda_cccl directory layout.
Without the variable, this optional native conversion test skips. When set,
missing headers, compiler failures, timeouts, and mismatches fail the test.
No CUDA device or runtime is used by this fixture.

The retained run checked 498,146 distinct double input bit patterns: every
finite value in each small format, adjacent-value rounding midpoints and their
double neighbors, both signs, overflow boundaries, infinities, and NaNs.
There were 1,992,584 encoding checks and as many independent CUDA decoding
checks. Non-NaN results agree bit for bit; NaNs agree by classification.
All finite small-format bit patterns also pass local round-trip tests.
`receipts/cuda-float-codec-final.log` and
`receipts/cuda-float-codec-provenance.json` retain results, compiler arguments,
and hashes of the source, raw inputs/outputs, and 20 CUDA header dependencies.

That fixture used local CUDA 13.3 headers with Clang. The identical source and
input corpus also ran on the DGX using GCC and its installed CUDA 13.0 headers.
The complete native output matched byte for byte. The DGX commands, hashes, and
20 header dependencies are retained in `cuda-float-codec-dgx.txt` and
`cuda-float-codec-dgx-headers.json`. Both runs exercise host conversion routines.

## Runner integration

`smallfloat.go` supplies rank-aware payloads and exact numeric checks. Copies
and min/max use varied finite normal encodings; receive poison is NaN.
Sums activate up to four ranks per element in a rotating window, contributing
signed ones. Pre-multiplied sums use rank scalars one or two. Products use
signed powers of two with magnitude at most sixteen. These restrictions keep
all intermediate results representable regardless of reduction order.

Average activates at most two ranks. NCCL v2.31.2-1 converts the reciprocal
rank count to float32, then to the collective datatype, and performs PreMulSum
(`src/enqueue/enqueue.cc`, commit
`7b83616df3ae082a1f32bb74c27458bfe8153a13`). The oracle uses that quantized
scale q; expected results are zero, signed q, or signed 2q. E4M3 q rounds to
zero at 1024 ranks. No broad tolerance hides these rounding rules.

`TestSmallAverageScale` checks fixed conversion vectors. `TestSmallChecker`
rejects corruption and poison. `TestSmallReductionTrees` exercises all six
reductions across 1 through 1024 ranks with randomized reduction trees and
checks that every intermediate remains exactly representable.

`TestSmallCopyRouting` independently assembles four-rank receive buffers for
seven copy operations, including root three, and rejects misplaced distinct
elements. `TestGatherRankPermutation` rejects swapped rank blocks for all four
formats. These controls pass locally, along with race tests and vet.

`TestSmallVariableRouting` also passes fixed spread-one traffic checks and
wrong-destination rejection. Remaining evidence includes completed matched
upstream datatype matrices and physical multi-rank execution.
One-rank CUDA runs cannot prove multi-rank arithmetic or transport. Codec
conformance and host oracle tests are separate from GPU execution evidence.
