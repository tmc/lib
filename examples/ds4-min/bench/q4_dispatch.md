# Scope-A Q4_K dispatch

CUDA dispatch now distinguishes the final-six-layer Q4_K expert splice from
the existing IQ2_XXS/Q2_K layout. All-Q4 and MXFP4 scope-B checkpoints remain
unsupported. Model admission remains fail-closed without a registered
checkpoint manifest; synthetic kernel qualification does not admit a model.

Q4_K gate/up consume one common Q8_K input. Weighted SwiGLU produces six
slot-specific Q8_K inputs for down. The selected expert ID is read by slot,
not output row. Scalar and batch evaluation use the same selected kernel.
Q4 source weights remain resident rather than entering the IQ2/Q2 artifact
repacker. Additional activation buffers are allocated only for models with
Q4 layers, and snapshot, restore and persistence own their contents.

Signed source commits:

- `b81dd9eea`: selected Q4_K CUDA dispatch and scope-A host plan.
- `a5aca4edf`: IQ2 scratch allocation follows actual IQ2 artifact needs.
- `1024b8cde`: compile the actual test mutation, not an unused source string.
- `bfa868e16`: initialize the CPU resource fixture.

## Synthetic device gate

The GB10 gate uses 256 experts, six selected slots, five output rows and two
Q4 blocks per row. Packed high scale/minimum bits and slot-specific inputs
are populated. Independent host Q4 decoding and float64 accumulation are the
reference; production common and slotted routes pass with maximum absolute
errors `0.000244140625` and `0.00048828125`. A wrong selected expert and a
device high-scale-bit mutation must fail that same comparator.

The sealed run passes on NVIDIA GB10 sm_121, driver 580.173.02, under
`gpu-lock`, including CPU resource and actual-compilation-input preflight.
It releases rc0. Binary SHA256, verified before acquire and after exit:
`3ade3cd03c0bc96ef2d2087c487e3dc8a43a8e51dfcc85c84c3aa8c632fdfcd7`.
Raw sealed log SHA256:
`0f37817d0f8dd79655556471b2d1cd3fd1a8d4f40cf088987825f4f03de12d98`.

Candidate actual compilation-input SHA256:
`feb44730ab9f437fdbb3f0a63e973053840c893191176c9ee0b9176dabe55d0e`.
Mutant actual compilation-input SHA256:
`f96091e83c61b2c213d6515559471515d5005cc1a3fc0efb1d36cdd58132d5c5`.
The test binary uses the pre-existing examples module overlay, including
purego v0.11.1; it is not called a clean source-archive module build.

Earlier attempts are retained, not overwritten: a lock invocation error ran
zero tests; an unused-source mutant failed the sensitivity gate; a nil CPU
fixture panicked before device arithmetic; and an unsealed upload-race PASS
was diagnostic only. The final sealed rerun changes no production arithmetic
or tolerance from those tests.

Raw receipts are outside the repo at
`~/tmp/ds4-min/20261007-q4-dispatch/` on the Mac and
`~/tmp/ds4-lib-workflow/20261007-q4-dispatch/` on DGX.

## Remaining gates

Default Q2 nonregression is HOLD at baseline admission, not a candidate
failure. Clean baseline `b0b8f9adc513409294a5debefc103a98f393303e`
(binary SHA256
`a9cbdbb6872c74913517c574bcd5d92437326ee965cb03097321551db1cfd645`)
returns prefill digest
`53b076ad8d3dcb6a7d69685703e9d546526b7bce87d3ff9d6055347b1e35763c`
instead of registered
`6d2ad5c9a07fd2d27619e864fbdd6e24de9532b67125b1ea3052c75629d40ce5`.
The corrected registered State-bridge fixture uses context 65665, 2048
prompt tokens, two decode tokens, temperature zero, no thinking, graphs off
and no warm prefill. The locked process exits 1 at the prefill assertion.
Candidate evaluation and performance ABBA never start. This supersedes the
earlier incorrectly configured context-4096 smoke, which omitted explicit
temperature and thinking settings. No digest waiver or numerical retry was
made. Raw files are `q2-f1-smoke.sh`, `q2-f1-smoke-gate.log` and
`q2-baseline-f1.stderr` in the receipt directories above.

This matches the registered workload, not the registered build. The retained
September 28 receipt names source `6e7151e870f90dd35e4153c15cdeadfd76d4c0c9`,
clean overlay `9b71fc209436340f125d73268347f0c58f6ead69` and binary
`e1a8dbfc29691fa72aebaab63c90974b51c69d65b0c673793a8d2f1541db99d1`.
That overlay was recovered from DGX `~/tmp/ds4-state-f1-repo` and imported
from a scoped bundle, SHA256
`41db6695ced109e091825ddba9ebc20d5146245440117a364d5060f3c98a24be`.
The retained executable's full hash matches the old receipt, and its original
lock log ends rc0 with the registered digests. Current baseline is a different build and includes
later State, independent request-stream and checkpoint work. Current math
compilation requests device SM architecture, fast math, line information and
extra device vectorization; the recovered old source requests the same flags.
The old receipt records no resolved compiler/NVRTC library hash. The old and current records agree on model SHA, prompt token hash,
GB10 and driver 580.173.02. Current baseline logs resident replacement of 129
expert tensors (72.56 GiB), 43 MoE artifact layers and 43 dense artifact layers;
the recovered old raw log reports the same residency configuration, and
`cuda_resident_linux.go` is byte-identical between both commits.
The October 7 shell resolves `/usr/local/cuda` to CUDA 13.4 and exposes an
explicit CUDA 13.0 `NVRTC_LIB_PATH`; the run does not retain a resolved-library
hash, so neither toolkit path is assigned as its measured compiler identity.
One concrete prefill-sensitive source difference is commit `e3ddbbdf6`:
three Q8_1 quantizers now use `__fdiv_rn(value, d)` instead of `value / d`,
and expert Q8 reciprocal operations also use round-to-nearest division.
With fast math, those operations need not produce identical quantized bytes.
This is a falsifiable source candidate, not measured attribution; the later
F16 double-to-float accumulation change concerns decode. The Go version is
1.27.1 and purego v0.11.0 in both clean binary build records. Stable `53b076ad`
is not a new registered digest; admission needs an authoritative baseline
for the newer source or a separately authorized causal control.

Real Q2/Q4 model provenance, serving admission and full-model correctness and
performance are open; DGX has insufficient disk space for the missing model.
Metal Q4 dispatch and physical qualification are separate work.
Default Q2 digest and warm-prefill A/B regression gates remain required
because adding a CUDA source fragment can change module code generation.
