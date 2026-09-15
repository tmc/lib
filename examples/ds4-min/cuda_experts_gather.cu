
// DS4_MOE_MAX_EXPERTS bounds the counting sort below. The Go side rejects
// wider tensors, so the bound is checked before a launch, not after.
#define DS4_MOE_MAX_EXPERTS 256

// DS4_MOE_GATHER_TILE is the number of (token, slot) pairs one warp keeps in
// flight against a single expert weight row. The pigeonhole bound on a group
// of W tokens is 6W selections over at most 256 experts, so a wide group
// reads every selected row about 6W/min(6W,256) times when the pairs are
// unbatched. Tiling reads it ceil(count/TILE) times instead. The Go constant
// cudaMoEGatherTile mirrors this and TestCUDAMoEGatherTileConstant fails if
// the two drift apart.
#ifndef DS4_MOE_GATHER_TILE
#define DS4_MOE_GATHER_TILE 4
#endif

// ds4_moe_group_csr groups a token group's selected expert ids by expert.
// ids is token major with slots entries per token, so pair p covers token
// p / slots and slot p % slots. The result is a CSR listing: pairs holds every
// pair id sorted by bin and, within a bin, in ascending pair order, bin b
// occupying pairs[offsets[b]] through pairs[offsets[b+1]]; active lists the
// non-empty bins in ascending bin order and count is their number. Bin
// experts collects ids the per token kernels reject, which the gather kernels
// poison exactly as those kernels do.
//
// One block, one thread per bin. Each thread scans every id twice, once to
// count its own bin and once to append to it, so there is no atomic and the
// listing does not depend on thread scheduling. Every thread reads the same
// id in the same step, which the load path broadcasts, so the two scans cost
// about 2*npairs L1 hits and one 257 element serial scan.
extern "C" __global__ void ds4_moe_group_csr(
    const int *ids, int *offsets, int *pairs, int *active, int *count,
    unsigned npairs, unsigned experts) {
  __shared__ int cursor[DS4_MOE_MAX_EXPERTS + 1];
  const unsigned bins = experts + 1;
  for (unsigned bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    int n = 0;
    for (unsigned p = 0; p < npairs; p++) {
      int id = ids[p];
      unsigned b = (id < 0 || unsigned(id) >= experts) ? experts : unsigned(id);
      if (b == bin) n++;
    }
    cursor[bin] = n;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0, n = 0;
    for (unsigned bin = 0; bin < bins; bin++) {
      int c = cursor[bin];
      offsets[bin] = total;
      cursor[bin] = total;
      total += c;
      if (c) active[n++] = int(bin);
    }
    offsets[bins] = total;
    *count = n;
  }
  __syncthreads();
  for (unsigned bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    int at = cursor[bin];
    for (unsigned p = 0; p < npairs; p++) {
      int id = ids[p];
      unsigned b = (id < 0 || unsigned(id) >= experts) ? experts : unsigned(id);
      if (b == bin) pairs[at++] = int(p);
    }
  }
}

// ds4_iq2_xxs_pair_selected_q8_1_gather is ds4_iq2_xxs_pair_selected_q8_1 with
// the expert weight row read once for DS4_MOE_GATHER_TILE pairs routed to that
// expert instead of once per pair. blockIdx.y selects an active bin and
// blockIdx.x a range of rows, so a block owns one (expert, rows) pair and
// walks only the pairs the CSR listing routed there.
//
// Every output keeps its own accumulator and its own shfl_down reduction over
// the same ascending lane strided part order, reading the same weight bytes
// and the same activation block as the per pair kernel, so each output float32
// is bit identical to that kernel's. Tiling changes only which loop is outer
// and which warp computes an output, never the order of a single sum.
extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_1_gather(
    const unsigned char *gate, const unsigned char *up,
    const int *offsets, const int *pairs, const int *active, const int *count,
    const ds4_q8_1_activation *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  if (blockIdx.y >= unsigned(*count)) return;
  const unsigned bin = unsigned(active[blockIdx.y]);
  const int begin = offsets[bin], end = offsets[bin + 1];
  const unsigned lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bin >= experts) {
    if (row < rows && lane == 0) {
      for (int p = begin; p < end; p++) {
        unsigned long long o = (unsigned long long)unsigned(pairs[p]) * rows + row;
        gate_out[o] = __int_as_float(0x7fffffff);
        up_out[o] = __int_as_float(0x7fffffff);
      }
    }
    return;
  }

  __shared__ unsigned int s_grid_lo[256];
  __shared__ unsigned int s_grid_hi[256];
  __shared__ unsigned char s_ksigns[128];
  for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
    unsigned long long grid = ds4_cuda_iq2xxs_grid[i];
    s_grid_lo[i] = unsigned(grid);
    s_grid_hi[i] = unsigned(grid >> 32);
  }
  for (unsigned i = threadIdx.x; i < 32; i += blockDim.x) {
    reinterpret_cast<unsigned int *>(s_ksigns)[i] = reinterpret_cast<const unsigned int *>(ds4_cuda_ksigns_iq2xs)[i];
  }
  __syncthreads();
  if (row >= rows) return;

  const unsigned blocks = cols / 256;
  const unsigned long long offset = ((unsigned long long)bin * rows + row) * blocks * 66;
  for (int base = begin; base < end; base += DS4_MOE_GATHER_TILE) {
    const unsigned n = unsigned(end - base) < DS4_MOE_GATHER_TILE ? unsigned(end - base) : DS4_MOE_GATHER_TILE;
    unsigned pair[DS4_MOE_GATHER_TILE];
    const ds4_q8_1_activation *token[DS4_MOE_GATHER_TILE];
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
      pair[t] = t < n ? unsigned(pairs[base + t]) : 0;
      token[t] = x + (unsigned long long)(pair[t] / slots) * (cols / 32);
    }
    float g[DS4_MOE_GATHER_TILE] = {0.0f}, u[DS4_MOE_GATHER_TILE] = {0.0f};
    for (unsigned part = lane; part < cols / 32; part += 32) {
      const unsigned b = part / 8, sub = part % 8;
      const unsigned char *wg = gate + offset + (unsigned long long)b * 66;
      const unsigned char *wu = up + offset + (unsigned long long)b * 66;
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
        if (t < n) {
          g[t] += ds4_iq2_part_q8_1(wg, token[t] + part, sub, s_grid_lo, s_grid_hi, s_ksigns);
          u[t] += ds4_iq2_part_q8_1(wu, token[t] + part, sub, s_grid_lo, s_grid_hi, s_ksigns);
        }
      }
    }
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
      if (t < n) {
        float gt = g[t], ut = u[t];
        for (unsigned stride = 16; stride; stride >>= 1) {
          gt += __shfl_down_sync(0xffffffff, gt, stride);
          ut += __shfl_down_sync(0xffffffff, ut, stride);
        }
        if (lane == 0) {
          unsigned long long o = (unsigned long long)pair[t] * rows + row;
          gate_out[o] = gt;
          up_out[o] = ut;
        }
      }
    }
  }
}

// ds4_q2_k_selected_q8_1_gather is ds4_q2_k_selected_q8_1 with the expert
// weight row read once for DS4_MOE_GATHER_TILE pairs instead of once per pair.
// Q2 takes a per pair activation row, so tiling reuses the weight bytes only;
// the activation reads stay one per pair, which is the small side of the
// traffic. Each output keeps its own accumulator over the same lane/16 strided
// block order and the same shfl_xor reduction, so it is bit identical to the
// per pair kernel's.
extern "C" __global__ void ds4_q2_k_selected_q8_1_gather(
    const unsigned char *weights,
    const int *offsets, const int *pairs, const int *active, const int *count,
    const ds4_q8_1_activation *x, float *out,
    unsigned cols, unsigned rows, unsigned experts) {
  if (blockIdx.y >= unsigned(*count)) return;
  const unsigned bin = unsigned(active[blockIdx.y]);
  const int begin = offsets[bin], end = offsets[bin + 1];
  const unsigned lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  if (bin >= experts) {
    if (lane == 0) {
      for (int p = begin; p < end; p++)
        out[(unsigned long long)unsigned(pairs[p]) * rows + row] = __int_as_float(0x7fffffff);
    }
    return;
  }

  const unsigned blocks = cols / 256;
  const unsigned long long offset = ((unsigned long long)bin * rows + row) * blocks * 84;
  for (int base = begin; base < end; base += DS4_MOE_GATHER_TILE) {
    const unsigned n = unsigned(end - base) < DS4_MOE_GATHER_TILE ? unsigned(end - base) : DS4_MOE_GATHER_TILE;
    unsigned pair[DS4_MOE_GATHER_TILE];
    const ds4_q8_1_activation *input[DS4_MOE_GATHER_TILE];
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
      pair[t] = t < n ? unsigned(pairs[base + t]) : 0;
      input[t] = x + (unsigned long long)pair[t] * (cols / 32);
    }
    float sum[DS4_MOE_GATHER_TILE] = {0.0f};
    for (unsigned b = lane / 16; b < blocks; b += 2) {
      const unsigned char *w = weights + offset + (unsigned long long)b * 84;
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
        if (t < n) sum[t] += ds4_q2_window_q8_1(w, input[t] + b * 8, lane % 16);
      }
    }
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
      if (t < n) {
        float s = sum[t];
        for (unsigned stride = 16; stride; stride >>= 1)
          s += __shfl_xor_sync(0xffffffff, s, stride);
        if (lane == 0) out[(unsigned long long)pair[t] * rows + row] = s;
      }
    }
  }
}
