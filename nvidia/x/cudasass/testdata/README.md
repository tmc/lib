# Native disassembly fixture

nvdisasm-13.0-sm121.json is actual output from nvdisasm V13.0.85 on
gpu-host, generated on 2026-09-05 from this CUDA source:

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
