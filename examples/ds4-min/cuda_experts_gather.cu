
// DS4_MOE_MAX_EXPERTS bounds the counting sort below. The Go side rejects
// wider tensors, so the bound is checked before a launch, not after.
#define DS4_MOE_MAX_EXPERTS 256

// The three-pass CSR path gives each expert bin its own block while preserving
// stable pair order.
extern "C" __global__ void ds4_moe_group_count(
    const int *ids, int *counts, unsigned npairs, unsigned experts) {
  const unsigned bin = blockIdx.x;
  if (bin > experts) return;
  __shared__ unsigned totals[1];
  if (threadIdx.x == 0) totals[0] = 0;
  __syncthreads();
  unsigned local = 0;
  for (unsigned base = 0; base < npairs; base += blockDim.x) {
    const unsigned p = base + threadIdx.x;
    const bool hit = p < npairs &&
        ((ids[p] < 0 || unsigned(ids[p]) >= experts) ? experts : unsigned(ids[p])) == bin;
    const unsigned mask = __ballot_sync(0xffffffff, hit);
    if ((threadIdx.x & 31) == 0) atomicAdd(totals, __popc(mask));
  }
  __syncthreads();
  if (threadIdx.x == 0) counts[bin] = int(totals[0]);
}

extern "C" __global__ void ds4_moe_group_prefix(
    const int *counts, int *offsets, int *active, int *count, unsigned experts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int total = 0, n = 0;
  for (unsigned bin = 0; bin <= experts; bin++) {
    const int c = counts[bin];
    offsets[bin] = total;
    total += c;
    if (c) active[n++] = int(bin);
  }
  offsets[experts + 1] = total;
  *count = n;
}

extern "C" __global__ void ds4_moe_group_fill(
    const int *ids, const int *offsets, int *pairs, unsigned npairs, unsigned experts) {
  const unsigned bin = blockIdx.x;
  if (bin > experts) return;
  __shared__ unsigned warp_counts[8];
  __shared__ unsigned tile_base;
  if (threadIdx.x == 0) tile_base = 0;
  __syncthreads();
  for (unsigned base = 0; base < npairs; base += blockDim.x) {
    const unsigned p = base + threadIdx.x;
    const bool hit = p < npairs &&
        ((ids[p] < 0 || unsigned(ids[p]) >= experts) ? experts : unsigned(ids[p])) == bin;
    const unsigned mask = __ballot_sync(0xffffffff, hit);
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) warp_counts[warp] = __popc(mask);
    __syncthreads();
    if (hit) {
      unsigned rank = 0;
      for (unsigned w = 0; w < warp; w++) rank += warp_counts[w];
      rank += __popc(mask & (lane == 0 ? 0u : ((1u << lane) - 1u)));
      pairs[offsets[bin] + tile_base + rank] = int(p);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned n = 0;
      for (unsigned w = 0; w < (blockDim.x + 31) / 32; w++) n += warp_counts[w];
      tile_base += n;
    }
    __syncthreads();
  }
}

// ds4_moe_ids_helper computes CSR expert_bounds, pair mapping ids_dst, and
// token mapping ids_src in a single pass over ids.
// One warp (32 threads) per expert block.
extern "C" __global__ void ds4_moe_ids_helper(
    const int * __restrict__ ids,
    int * __restrict__ ids_src,
    int * __restrict__ ids_dst,
    int * __restrict__ expert_bounds,
    unsigned n_tokens,
    unsigned n_experts,
    unsigned n_expert_used) {
  const unsigned expert = blockIdx.x;
  int nex_prev = 0;
  int it_compact_count = 0;

  if (n_expert_used == 6) {
    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
      int it_compact = 0;
      for (unsigned it0 = 0; it0 < n_tokens; it0 += 4) {
        const unsigned it = it0 + (threadIdx.x >> 3);
        const unsigned iex = threadIdx.x & 7;
        const bool valid = (iex < 6) && (it < n_tokens);
        const int expert_used = valid ? ids[it * 6 + iex] : -1;

        if (pass == 0) {
          bool is_less = false;
          if (valid && expert_used >= 0) {
            if (expert < n_experts) {
              is_less = ((unsigned)expert_used < expert);
            } else {
              is_less = ((unsigned)expert_used < n_experts);
            }
          }
          const unsigned less_mask = __ballot_sync(0xffffffff, is_less);
          nex_prev += __popc(less_mask);
        }

        bool is_match = false;
        if (valid) {
          if (expert < n_experts) {
            is_match = (expert_used == (int)expert);
          } else {
            is_match = (expert_used < 0 || (unsigned)expert_used >= n_experts);
          }
        }
        const unsigned match_mask = __ballot_sync(0xffffffff, is_match);
        if (pass == 1 && is_match) {
          const int rank = __popc(match_mask & ((1u << threadIdx.x) - 1u));
          const int itc = it_compact + rank;
          if (ids_src != nullptr) ids_src[nex_prev + itc] = (int)it;
          if (ids_dst != nullptr) ids_dst[nex_prev + itc] = (int)(it * 6 + iex);
        }
        it_compact += __popc(match_mask);
      }
      if (pass == 0) {
        it_compact_count = it_compact;
      }
    }
  } else {
    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
      int it_compact = 0;
      for (unsigned it = 0; it < n_tokens; ++it) {
        int iex_used = -1;
        for (unsigned iex = threadIdx.x; iex < n_expert_used; iex += 32) {
          const int expert_used = ids[it * n_expert_used + iex];
          if (pass == 0) {
            bool is_less = false;
            if (expert_used >= 0) {
              if (expert < n_experts) {
                is_less = ((unsigned)expert_used < expert);
              } else {
                is_less = ((unsigned)expert_used < n_experts);
              }
            }
            if (is_less) nex_prev++;
          }
          bool is_match = false;
          if (expert < n_experts) {
            is_match = (expert_used == (int)expert);
          } else {
            is_match = (expert_used < 0 || (unsigned)expert_used >= n_experts);
          }
          if (is_match) iex_used = (int)iex;
        }
        const unsigned match_mask = __ballot_sync(0xffffffff, iex_used != -1);
        if (pass == 1 && iex_used != -1) {
          const int rank = __popc(match_mask & ((1u << threadIdx.x) - 1u));
          const int itc = it_compact + rank;
          if (ids_src != nullptr) ids_src[nex_prev + itc] = (int)it;
          if (ids_dst != nullptr) ids_dst[nex_prev + itc] = (int)(it * n_expert_used + iex_used);
        }
        it_compact += __popc(match_mask);
      }
      if (pass == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          nex_prev += __shfl_xor_sync(0xffffffff, nex_prev, offset);
        }
        it_compact_count = it_compact;
      }
    }
  }

  if (threadIdx.x == 0 && expert_bounds != nullptr) {
    expert_bounds[expert] = nex_prev;
    if (expert == gridDim.x - 1) {
      expert_bounds[gridDim.x] = nex_prev + it_compact_count;
    }
  }
}

// DS4_MOE_IQ2_GATHER_TILE is the number of (token, slot) pairs one warp keeps in
// flight against a single expert weight row for IQ2_XXS gate and up projections.
#ifndef DS4_MOE_IQ2_GATHER_TILE
#define DS4_MOE_IQ2_GATHER_TILE 8
#endif

// DS4_MOE_Q2_GATHER_TILE is the number of (token, slot) pairs one warp keeps in
// flight against a single expert weight row for Q2_K down projections.
#ifndef DS4_MOE_Q2_GATHER_TILE
#define DS4_MOE_Q2_GATHER_TILE 4
#endif

#ifndef DS4_MOE_GATHER_TILE
#define DS4_MOE_GATHER_TILE 16
#endif

// ds4_iq2_xxs_pair_selected_q8_1_gather is ds4_iq2_xxs_pair_selected_q8_1 with
// the expert weight row read once for DS4_MOE_IQ2_GATHER_TILE pairs routed to that
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
    const unsigned char * __restrict__ gate, const unsigned char * __restrict__ up,
    const int * __restrict__ offsets, const int * __restrict__ pairs, const int * __restrict__ active, const int * __restrict__ count,
    const ds4_q8_1_activation * __restrict__ x, float * __restrict__ gate_out, float * __restrict__ up_out,
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
  for (int base = begin; base < end; base += DS4_MOE_IQ2_GATHER_TILE) {
    const unsigned n = unsigned(end - base) < DS4_MOE_IQ2_GATHER_TILE ? unsigned(end - base) : DS4_MOE_IQ2_GATHER_TILE;
    const ds4_q8_1_activation *token[DS4_MOE_IQ2_GATHER_TILE];
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_IQ2_GATHER_TILE; t++) {
      unsigned p = t < n ? unsigned(pairs[base + t]) : 0;
      token[t] = x + (unsigned long long)(p / slots) * (cols / 32);
    }
    float g[DS4_MOE_IQ2_GATHER_TILE] = {0.0f}, u[DS4_MOE_IQ2_GATHER_TILE] = {0.0f};
    for (unsigned part = lane; part < cols / 32; part += 32) {
      const unsigned b = part / 8, sub = part % 8;
      const unsigned char *wg = gate + offset + (unsigned long long)b * 66;
      const unsigned char *wu = up + offset + (unsigned long long)b * 66;
      ds4_iq2_hoisted_part hg, hu;
      ds4_iq2_hoist_part(wg, sub, s_grid_lo, s_grid_hi, s_ksigns, &hg);
      ds4_iq2_hoist_part(wu, sub, s_grid_lo, s_grid_hi, s_ksigns, &hu);
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_IQ2_GATHER_TILE; t++) {
        if (t < n) {
          g[t] += ds4_iq2_eval_hoisted(hg, token[t] + part);
          u[t] += ds4_iq2_eval_hoisted(hu, token[t] + part);
        }
      }
    }
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_IQ2_GATHER_TILE; t++) {
      if (t < n) {
        float gt = g[t], ut = u[t];
        for (unsigned stride = 16; stride; stride >>= 1) {
          gt += __shfl_down_sync(0xffffffff, gt, stride);
          ut += __shfl_down_sync(0xffffffff, ut, stride);
        }
        if (lane == 0) {
          unsigned p = unsigned(pairs[base + t]);
          unsigned long long o = (unsigned long long)p * rows + row;
          gate_out[o] = gt;
          up_out[o] = ut;
        }
      }
    }
  }
}

// ds4_q2_k_selected_q8_1_gather is ds4_q2_k_selected_q8_1 with the expert
// weight row read once for DS4_MOE_Q2_GATHER_TILE pairs instead of once per pair.
// Q2 takes a per pair activation row, so tiling reuses the weight bytes only;
// the activation reads stay one per pair, which is the small side of the
// traffic. Each output keeps its own accumulator over the same lane/16 strided
// block order and the same shfl_xor reduction, so it is bit identical to the
// per pair kernel's.
extern "C" __global__ void ds4_q2_k_selected_q8_1_gather(
    const unsigned char * __restrict__ weights,
    const int * __restrict__ offsets, const int * __restrict__ pairs, const int * __restrict__ active, const int * __restrict__ count,
    const ds4_q8_1_activation * __restrict__ x, float * __restrict__ out,
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
  const unsigned b_init = lane / 16;
  const unsigned iqs = lane % 16;
  const unsigned qoffset = 16 + iqs * 4;
  const unsigned scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
  const unsigned scale_base = scale_offset & ~1;
  const unsigned sc_shift_base = (scale_offset & 1) << 3;
  const unsigned iqs_div_8 = 4 * (iqs / 8);
  const unsigned qs_off = (iqs % 8) * 4;

  for (int base = begin; base < end; base += DS4_MOE_Q2_GATHER_TILE) {
    const unsigned n = unsigned(end - base) < DS4_MOE_Q2_GATHER_TILE ? unsigned(end - base) : DS4_MOE_Q2_GATHER_TILE;
    const ds4_q8_1_activation *token_x[DS4_MOE_Q2_GATHER_TILE];
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_Q2_GATHER_TILE; t++) {
      unsigned p = t < n ? unsigned(pairs[base + t]) : 0;
      token_x[t] = x + (unsigned long long)p * (cols / 32);
    }
    float sum[DS4_MOE_Q2_GATHER_TILE] = {0.0f};
    const unsigned char *w = weights + offset + (unsigned long long)b_init * 84;
    for (unsigned b = b_init; b < blocks; b += 2) {
      unsigned v = *reinterpret_cast<const unsigned int *>(w + qoffset);
      unsigned int sc0 = *reinterpret_cast<const unsigned int *>(w + scale_base) >> sc_shift_base;
      unsigned int sc1 = *reinterpret_cast<const unsigned int *>(w + scale_base + 4) >> sc_shift_base;
      unsigned d_pair = *reinterpret_cast<const unsigned int *>(w + 80);
      float d = ds4_half_to_float(d_pair & 0xffff);
      float dm = ds4_half_to_float(d_pair >> 16);

      int v_sub[4];
      int s_lo[4];
      int s_hi[4];
      #pragma unroll
      for (unsigned i = 0; i < 4; i++) {
        v_sub[i] = (v >> (2 * i)) & 0x03030303;
        unsigned sc = (i < 2) ? sc0 : sc1;
        unsigned sc_shift = (i & 1) ? 16 : 0;
        s_lo[i] = int((sc >> sc_shift) & 0x0f);
        s_hi[i] = int((sc >> (sc_shift + 4)) & 0x0f);
      }

      const unsigned act_off = b * 8 + iqs_div_8;
      #pragma unroll
      for (unsigned t = 0; t < DS4_MOE_Q2_GATHER_TILE; t++) {
        if (t < n) {
          const ds4_q8_1_activation *q_base = token_x[t] + act_off;
          float sum_d = 0.0f, sum_m = 0.0f;
          #pragma unroll
          for (unsigned i = 0; i < 4; i++) {
            const ds4_q8_1_activation *q = q_base + i;
            int q_packed = *reinterpret_cast<const int *>(&q->qs[qs_off]);
            int dot = __dp4a(v_sub[i], q_packed, 0);
            int minimum = s_hi[i] * __dp4a(0x01010101, q_packed, 0);
            float d8 = ds4_half_to_float(q->d);
            sum_d += d8 * float(dot * s_lo[i]);
            sum_m += d8 * float(minimum);
          }
          sum[t] += d * sum_d - dm * sum_m;
        }
      }
      w += 168;
    }
    #pragma unroll
    for (unsigned t = 0; t < DS4_MOE_Q2_GATHER_TILE; t++) {
      if (t < n) {
        float s = sum[t];
        for (unsigned stride = 16; stride; stride >>= 1)
          s += __shfl_xor_sync(0xffffffff, s, stride);
        if (lane == 0) {
          unsigned p = unsigned(pairs[base + t]);
          out[(unsigned long long)p * rows + row] = s;
        }
      }
    }
  }
}
