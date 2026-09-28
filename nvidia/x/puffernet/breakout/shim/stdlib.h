// NVRTC stand-in for <stdlib.h>. See cuda_runtime.h.
#pragma once
#define calloc(n, size) 0
#define free(p) ((void)0)
