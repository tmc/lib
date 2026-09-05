# Native disassembly fixture

nvdisasm-13.0-sm121.json is actual output from nvdisasm V13.0.85 on
spark-31db, generated on 2026-09-05 from this CUDA source:

    extern "C" __global__ void addone(float *x) { x[threadIdx.x] += 1; }

Compiled with CUDA 13.0 nvcc using `-cubin -arch=sm_121`, then passed through
Disassembler with JSON mode. The producer and ptxas identities are retained
in the JSON metadata. This fixture tests real schema parsing; it does not
replace live invocation or GPU execution tests.
