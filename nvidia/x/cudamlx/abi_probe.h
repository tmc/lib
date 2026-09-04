// Probe records measured by gen_abi.go with clanglayout (via appledocs
// clangdump) against a live MLX checkout. MAX_NDIM comes from MLX's real
// mlx/backend/cuda/device/config.h, so the generated Go mirrors track
// the C++ source.
//
// Shape and Strides are cuda::std::array<int32_t/int64_t, MAX_NDIM> in
// mlx/backend/cuda/device/utils.cuh; libcu++ defines std::array as a
// single C array member, so these plain-struct probes share its layout.

#include "mlx/backend/cuda/device/config.h"

typedef struct {
  int val[MAX_NDIM];
} mlx_shape_abi;

typedef struct {
  long long val[MAX_NDIM];
} mlx_strides_abi;

typedef struct {
  unsigned x, y, z;
} mlx_dim3_abi;
