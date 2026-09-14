# Native disassembly fixture

nvdisasm-13.0-sm121.json is actual output from nvdisasm V13.0.85 on
spark-31db, generated on 2026-09-05 from this CUDA source:

    extern "C" __global__ void addone(float *x) { x[threadIdx.x] += 1; }

Compiled with CUDA 13.0 nvcc using `-cubin -arch=sm_121`, then passed through
Disassembler with JSON mode. The producer and ptxas identities are retained
in the JSON metadata. This fixture tests real schema parsing; it does not
replace live invocation or GPU execution tests.

# Toolchain note fixture

tkinfo-ptxas-13.0-sm121.cubin.hex is a hex-encoded ELF64 CUDA image whose only
content section is the .note.nv.tkinfo section copied byte-for-byte out of
../../cudaoccupancy/testdata/kernels.cubin.hex, a real cubin ptxas produced for
sm_121 under CUDA 13.0. The ELF header and section table around it are
synthesized so the fixture stays under a kilobyte; the 176 note bytes under
test are unmodified ptxas output, as is e_flags.

The stamp it carries reads:

    tki_toolName     ptxas
    tki_toolVersion  Cuda compilation tools, release 13.0, V13.0.88
    tki_toolBranch   Build system must define TOOLS_VERSION_EXTENDED
    tki_toolOptions  -lineinfo  -arch sm_121

tki_toolBranch is the unexpanded placeholder shown above in this image, while
nvdisasm-13.0-sm121.json records a real branch string for a differently built
cubin of the same release. The field is not parsed for that reason.

# Instruction encoding fixture

sm121a-encoding.sass is actual output from nvdisasm V13.0.85 on spark-31db,
generated on 2026-09-14 from this CUDA source:

    extern "C" __global__ void addk(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] * b[i] + 1.0f;
    }

Compiled with CUDA 13.0 nvcc using `-cubin -arch=sm_121a`, then disassembled
with `--print-code --print-instruction-encoding`. It carries the two encoding
words of every instruction, which TestEncodingRoundTrip feeds back through
`nvdisasm --binary SM121a` where nvdisasm is installed.

tkinfo-mixed-13.0-13.1.cubin.hex is the same image with the note section
carrying that descriptor twice, the second copy's release digits edited to
13.1 / V13.1.55 in place. It stands in for an image linked from objects two
toolkits produced, which is the condition ReadToolchains exists to detect. No
such image was available on this host to capture directly.

# Loop fixture (backward branches)

sm121a-loop-13.3.sass and sm121a-loop-13.4.sass are actual output from nvdisasm
V13.4.49 on spark-31db, generated on 2026-09-14 from this CUDA source:

    extern "C" __global__ void k4(const double* a, double* b, int n){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      double s = 0.0;
      for (int j=0;j<n;j++) s = fma(s, a[i+j], 1.0);
      if (i<n) b[i] = s;
    }

Compiled with `-cubin -arch=sm_121a` under two toolkits, then disassembled with
`--print-code --print-instruction-encoding`. The ptxas identity in each image:

    sm121a-loop-13.3.sass   release 13.3, V13.3.73   -arch sm_121a -m 64
    sm121a-loop-13.4.sass   release 13.4, V13.4.59   -arch sm_121a -m 64

These exist because every other SASS fixture in the repository has **no backward
branch**: across the seven that carry labels, all 37 branches are either
self-branches (27) or forward (10). Loop-membership logic tested only against
those passes an implementation that reports "not in a loop" unconditionally.

Each of these files carries all four label shapes in one kernel:

    2  backward branches  (.L_x_2 at 0x170, .L_x_5) — real loops
    7  forward branches
    1  self-branch        (.L_x_6, the post-EXIT trap idiom)
    1  trailing label     (.L_x_7, no following instruction)

`BSSY` does not appear here, but note it takes a label operand in the
x/cudalowering fixtures while not being a control transfer; match branches on
opcode, not on the presence of a label operand.

The pair is also the reproduction of the hot-loop padding result. Over the main
loop body (backward branch at 0x760 targeting 0x170 under 13.3, 0x3c0 targeting
0x170 under 13.4):

    toolkit  body  NOP  LDG.E.64  DFMA  other
    13.3       96   58        16    16      6
    13.4       38    0        16    16      6

13.4 removes every NOP from the hot loop — 58 of 96 instructions, 60% of issue
slots per iteration — while leaving all 16 loads and all 16 DFMAs in place.
Whole-.text totals go 232/110 NOP to 136/14 NOP with LDG and DFMA unchanged at
29 each. A section-level census reports this as a 41% instruction-count drop and
attributes it to the wrong thing; the loop-scoped census is the claim that
survives. This is why the corpus diff has to be taken over the non-NOP stream
and why loop membership has to be computed rather than assumed.

# Reconvergence fixture (non-branch label operands)

sm121a-reconvergence.sass is actual output from nvdisasm V13.4.49 on spark-31db,
generated on 2026-09-14 from this CUDA source:

    extern "C" __global__ void reconv(const int* c, float* b, int n){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      float s = 0.0f;
      int m = c[i];
      for (int j=0;j<m;j++){ s += j; if (s > 100.0f) { b[i+j] = s; s = 0.0f; } }
      b[i] = s;
    }

Compiled with `-cubin -arch=sm_121a` under ptxas 13.4.59 (release 13.4, V13.4.59)
and disassembled with `--print-code --print-instruction-encoding`.

It exists as the negative control the loop fixtures lack. Those contain only
BRA and BRA.U, so every label operand in them is a branch and a classifier that
matches any instruction carrying a label operand scores perfectly. Here the
census is:

    branches      8 forward, 1 backward, 1 self
    non-branches  6 forward  (BSSY.RECONVERGENT, BSSY.RELIABLE)
    trailing      1 (.L_x_13)

BSSY takes a label operand and is not a control transfer; it sets up the
reconvergence point for a divergent region. A classifier must match on opcode.
The per-thread loop bound and the conditional store are both load-bearing: a
predicable if/else is compiled without any reconvergence instruction at all.

One limit to state plainly, because it decides what this fixture can and cannot
prove. Across every fixture available here — these, the loop pair, and the six
in x/cudalowering/testdata — all 14 non-branch label operands point **forward**,
and none points backward. That is structural: BSSY names a reconvergence point
ahead of the divergence it guards. A forward target never creates a loop, so
misclassifying BSSY as a branch does not change a backward-branch-derived
InLoop result on any input we can produce. This fixture therefore discriminates
only against an assertion made on the branch classification itself. An InLoop
table alone cannot fail on it, and should not be claimed to cover it.
