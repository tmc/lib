
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
#define DS4_MOE_GATHER_TILE 16
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
      ds4_iq2_hoisted_part hg, hu;
      ds4_iq2_hoist_part(wg, sub, s_grid_lo, s_grid_hi, s_ksigns, &hg);
      ds4_iq2_hoist_part(wu, sub, s_grid_lo, s_grid_hi, s_ksigns, &hu);
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
        if (t < n) {
          g[t] += ds4_iq2_eval_hoisted(hg, token[t] + part);
          u[t] += ds4_iq2_eval_hoisted(hu, token[t] + part);
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
    const unsigned iqs = lane % 16;
    const unsigned qoffset = 16 + iqs * 4;
    const unsigned scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const unsigned scale_base = scale_offset & ~1;
    const unsigned b_init = lane / 16;
    const unsigned char *w = weights + offset + (unsigned long long)b_init * 84;
    const ds4_q8_1_activation *cur_input[DS4_MOE_GATHER_TILE];
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
      cur_input[t] = input[t] + b_init * 8;
    }
    for (unsigned b = b_init; b < blocks; b += 2) {
      unsigned v = *reinterpret_cast<const unsigned int *>(w + qoffset);
      unsigned long long w_sc = reinterpret_cast<unsigned long long>(w + scale_base);
      unsigned int sc0, sc1;
      if ((w_sc & 7) == 0) {
        uint2 sc_vec = *reinterpret_cast<const uint2 *>(w + scale_base);
        sc0 = sc_vec.x;
        sc1 = sc_vec.y;
      } else {
        sc0 = *reinterpret_cast<const unsigned int *>(w + scale_base);
        sc1 = *reinterpret_cast<const unsigned int *>(w + scale_base + 4);
      }
      unsigned char scales[4];
      if ((scale_offset & 1) == 0) {
        scales[0] = (unsigned char)(sc0 & 0xff);
        scales[1] = (unsigned char)((sc0 >> 16) & 0xff);
        scales[2] = (unsigned char)(sc1 & 0xff);
        scales[3] = (unsigned char)((sc1 >> 16) & 0xff);
      } else {
        scales[0] = (unsigned char)((sc0 >> 8) & 0xff);
        scales[1] = (unsigned char)((sc0 >> 24) & 0xff);
        scales[2] = (unsigned char)((sc1 >> 8) & 0xff);
        scales[3] = (unsigned char)((sc1 >> 24) & 0xff);
      }
      unsigned d_pair = *reinterpret_cast<const unsigned int *>(w + 80);
      float d = ds4_half_to_float(d_pair & 0xffff);
      float dm = ds4_half_to_float(d_pair >> 16);

      int vi[4], scale_lo[4], scale_hi[4];
      #pragma unroll
      for (unsigned i = 0; i < 4; i++) {
        vi[i] = (v >> (2 * i)) & 0x03030303;
        scale_lo[i] = int(scales[i] & 15);
        scale_hi[i] = int(scales[i] >> 4);
      }
      const unsigned iqs_div_8 = 4 * (iqs / 8);
      const unsigned qs_off = (iqs % 8) * 4;

      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
        if (t < n) {
          const ds4_q8_1_activation *q_base = cur_input[t] + iqs_div_8;
          float sum_d = 0.0f, sum_m = 0.0f;
          #pragma unroll
          for (unsigned i = 0; i < 4; i++) {
            const ds4_q8_1_activation *q = q_base + i;
            int q_packed = *reinterpret_cast<const int *>(&q->qs[qs_off]);
            int dot = __dp4a(vi[i], q_packed, 0);
            int minimum = scale_hi[i] * __dp4a(0x01010101, q_packed, 0);
            float d8 = ds4_half_to_float(q->d);
            sum_d += d8 * float(dot * scale_lo[i]);
            sum_m += d8 * float(minimum);
          }
          sum[t] += d * sum_d - dm * sum_m;
        }
      }
      w += 168;
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_GATHER_TILE; t++) {
        cur_input[t] += 16;
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
