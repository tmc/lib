/* Stub CUPTI headers for the producer control harness. They exist only so
   cupti_agent.c can be compiled and driven deterministically on a host with
   no CUDA toolkit and no GPU. Layouts are self-consistent with the stub
   implementation, not with NVIDIA's ABI, except where cupti_agent.c asserts a
   layout outright. */
#ifndef CUDAPROF_TESTCUPTI_VERSION_H
#define CUDAPROF_TESTCUPTI_VERSION_H

#define CUPTI_API_VERSION 27

#endif
