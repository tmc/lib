  __shared__ __align__(8) unsigned long long cta_barrier;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
        "r"((unsigned)__cvta_generic_to_shared(&cta_barrier)), "r"(blockDim.x) : "memory");
  }
  __syncthreads();
  const unsigned cta_barrier_addr = (unsigned)__cvta_generic_to_shared(&cta_barrier);
