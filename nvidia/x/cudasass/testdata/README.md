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
