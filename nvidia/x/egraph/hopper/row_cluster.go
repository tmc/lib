package hopper

import (
	"fmt"
	"strings"
)

const (
	// ClusterSize is the number of CTAs a cluster row kernel splits each row over.
	ClusterSize = 8

	// ClusterMaxRows is the default largest row count for which a launcher
	// chooses a kernel's 8-CTA cluster variant; above it the launcher chooses
	// the single-block wide variant.
	ClusterMaxRows = 64
)

// ClusterPrelude holds the device helpers for 8-CTA cluster and wide row
// kernels: per-CTA shared memory macros, PTX cluster barriers and DSMEM
// rank mapping, write-back 16-byte global stores (st.global.wb.v4.u32), and
// cluster/wide reductions with matching floating-point association order.
const ClusterPrelude = `#ifndef EG_CLUSTER_PRELUDE
#define EG_CLUSTER_PRELUDE

#if defined(EG_HOST_EMU)
#define EG_SHARED(T, name, dims) EG_EMU_SHARED(T, name, dims)
#define EG_REGISTER_CLUSTER(fn) \
    static const bool eg_reg_##fn = (eg_emu_cluster_kernels.insert(reinterpret_cast<const void*>(fn)), true);
#else
#define EG_SHARED(T, name, dims) alignas(16) __shared__ T name dims
#define EG_REGISTER_CLUSTER(fn)
#endif

EG_DEV int eg_cluster_rank() {
#if defined(EG_HOST_EMU)
    return eg_emu_cluster_rank;
#else
    uint32_t r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return (int)r;
#endif
}

EG_DEV void eg_cluster_sync() {
#if defined(EG_HOST_EMU)
    eg_emu_cluster_sync();
#else
    asm volatile("barrier.cluster.arrive.aligned;\n"
                 "barrier.cluster.wait.aligned;" ::: "memory");
#endif
}

EG_DEV float* eg_map_shared_rank(float* p, int r) {
#if defined(EG_HOST_EMU)
    return eg_emu_map_shared_rank(p, r);
#else
    float* m;
    asm("mapa.u64 %0, %1, %2;" : "=l"(m) : "l"(p), "r"((uint32_t)r));
    return m;
#endif
}

// eg_stwb16 writes 16 bytes with st.global.wb.v4.u32 so nvcc never splits a
// 16-byte store into scalar stores when alignment is not proved statically.
EG_DEV void eg_stwb16(void* p, eg_u32x4 v) {
#if defined(EG_HOST_EMU)
    *reinterpret_cast<eg_u32x4*>(p) = v;
#else
    asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#endif
}

template <int V> EG_DEV void eg_store_wb(float* p, const float* v) {
    if constexpr (V % 4 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 4) {
            eg_stwb16(p + i, eg_u32x4{
                __float_as_uint(v[i]), __float_as_uint(v[i + 1]),
                __float_as_uint(v[i + 2]), __float_as_uint(v[i + 3])});
        }
    } else {
        eg_store<V>(p, v);
    }
}

template <int V> EG_DEV void eg_store_wb(uint16_t* p, const float* v) {
    if constexpr (V % 8 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 8) {
            eg_stwb16(p + i, eg_u32x4{
                eg_pack_bf16x2(v[i], v[i + 1]), eg_pack_bf16x2(v[i + 2], v[i + 3]),
                eg_pack_bf16x2(v[i + 4], v[i + 5]), eg_pack_bf16x2(v[i + 6], v[i + 7])});
        }
    } else {
        eg_store<V>(p, v);
    }
}

template <int V> EG_DEV void eg_store_wb(uint8_t* p, const float* v) {
    eg_store<V>(p, v);
}

// eg_cluster_sum sums v over an 8-CTA cluster: warp butterfly, the CTA's
// warps in order, then an xor butterfly (4, 2, 1) across the 8 CTA ranks
// via map_shared_rank. red is [2][32] of EG_SHARED memory; slot alternates
// so back-to-back calls do not race on one buffer.
EG_DEV float eg_cluster_sum(float v, float (*red)[32], int slot) {
    v = eg_warp_sum(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    if (lane == 0) red[slot][warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int j = 0; j < nwarps; ++j) s += red[slot][j];
        red[slot][31] = s;
    }
    eg_cluster_sync();
    float s = lane < 8 ? eg_map_shared_rank(&red[slot][31], lane)[0] : 0.0f;
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    return __shfl_sync(0xffffffffu, s, 0);
}

// eg_block_sum8 sums v over one wide CTA of 8*chunks threads in the exact
// association order of eg_cluster_sum over 8 CTAs of chunks threads.
EG_DEV float eg_block_sum8(float v, float (*red)[32], int slot) {
    v = eg_warp_sum(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, per = blockDim.x >> 8;
    if (lane == 0) red[slot][warp] = v;
    __syncthreads();
    float s = 0.0f;
    if (lane < 8) {
        for (int j = 0; j < per; ++j) s += red[slot][lane * per + j];
    }
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    return __shfl_sync(0xffffffffu, s, 0);
}

EG_DEV float eg_cluster_max(float v, float (*red)[32], int slot) {
    v = eg_warp_max(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    if (lane == 0) red[slot][warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = red[slot][0];
        for (int j = 1; j < nwarps; ++j) s = fmaxf(s, red[slot][j]);
        red[slot][31] = s;
    }
    eg_cluster_sync();
    float s = lane < 8 ? eg_map_shared_rank(&red[slot][31], lane)[0] : -__uint_as_float(0x7f800000u);
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, m));
    return __shfl_sync(0xffffffffu, s, 0);
}

EG_DEV float eg_block_max8(float v, float (*red)[32], int slot) {
    v = eg_warp_max(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, per = blockDim.x >> 8;
    if (lane == 0) red[slot][warp] = v;
    __syncthreads();
    float s = -__uint_as_float(0x7f800000u);
    if (lane < 8) {
        s = red[slot][lane * per];
        for (int j = 1; j < per; ++j) s = fmaxf(s, red[slot][lane * per + j]);
    }
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, m));
    return __shfl_sync(0xffffffffu, s, 0);
}

// eg_cluster_sum_slots reduces K slot sums at once across an 8-CTA cluster of
// NW warps per CTA, adding the 8 ranks in order 0..7 into tot[k].
template <int K, int NW>
EG_DEV void eg_cluster_sum_slots(const float (&ss)[K], float* red, float (*part)[K], float* tot) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float v = eg_warp_sum(ss[k]);
        if (lane == 0) part[warp][k] = v;
    }
    __syncthreads();
    if ((int)threadIdx.x < K) {
        float s = part[0][threadIdx.x];
#pragma unroll
        for (int j = 1; j < NW; ++j) s += part[j][threadIdx.x];
        red[threadIdx.x] = s;
    }
    eg_cluster_sync();
    if ((int)threadIdx.x < K) {
        float s = 0.0f;
        for (int r = 0; r < 8; ++r) s += eg_map_shared_rank(&red[threadIdx.x], r)[0];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
}

// eg_block_sum8_slots reduces K slot sums over one wide CTA of 8*NW warps in
// the exact association order of eg_cluster_sum_slots.
template <int K, int NW>
EG_DEV void eg_block_sum8_slots(const float (&ss)[K], float (*part)[K], float* tot) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float v = eg_warp_sum(ss[k]);
        if (lane == 0) part[warp][k] = v;
    }
    __syncthreads();
    if ((int)threadIdx.x < K) {
        float t = 0.0f;
        for (int r = 0; r < 8; ++r) {
            float s = part[r * NW][threadIdx.x];
#pragma unroll
            for (int j = 1; j < NW; ++j) s += part[r * NW + j][threadIdx.x];
            t += s;
        }
        tot[threadIdx.x] = t;
    }
    __syncthreads();
}

#endif // EG_CLUSTER_PRELUDE
`

func (k RowKernel) maxClusterRows() int {
	if k.ClusterMaxRows > 0 {
		return k.ClusterMaxRows
	}
	return ClusterMaxRows
}

// clusterGeometry returns the per-CTA thread count, wide-block thread count,
// elements per thread, and vector width for k's cluster and wide variants.
func (k RowKernel) clusterGeometry() (ctaThreads, wideThreads, per, vec int) {
	if k.D >= 2048 && k.D%2048 == 0 && k.D <= 8192 {
		ctaThreads = k.D / 64
		wideThreads = ClusterSize * ctaThreads
		return ctaThreads, wideThreads, 8, 8
	}
	if k.D > 8192 && k.D%2048 == 0 {
		ctaThreads = 128
		wideThreads = 1024
		return ctaThreads, wideThreads, k.D / 1024, 8
	}
	ctaThreads = 32
	for _, t := range []int{128, 96, 64, 32} {
		if k.D%(ClusterSize*t*2) == 0 && k.D/(ClusterSize*t) <= 8 {
			ctaThreads = t
			break
		}
	}
	wideThreads = ClusterSize * ctaThreads
	per = k.D / wideThreads
	vec = per
	if vec > 8 {
		vec = 8
	}
	for per%vec != 0 {
		vec /= 2
	}
	return ctaThreads, wideThreads, per, vec
}

// CanCluster reports whether k can be emitted as an 8-CTA cluster kernel.
func (k RowKernel) CanCluster() bool {
	if k.D < 256 || k.D%256 != 0 {
		return false
	}
	c := k
	c.Cluster = true
	_, err := c.check()
	return err == nil
}

// HasWide reports whether k has a single-block wide variant (8*ctaThreads <= 1024).
func (k RowKernel) HasWide() bool {
	if !k.CanCluster() {
		return false
	}
	_, wideThreads, _, _ := k.clusterGeometry()
	return wideThreads <= 1024
}

// ClusterGrid returns the grid and block for k's 8-CTA cluster variant.
func (k RowKernel) ClusterGrid(rows int) (grid, block [3]int) {
	ctaThreads, _, _, _ := k.clusterGeometry()
	return [3]int{ClusterSize, rows, 1}, [3]int{ctaThreads, 1, 1}
}

// WideGrid returns the grid and block for k's single-block wide variant.
func (k RowKernel) WideGrid(rows int) (grid, block [3]int) {
	_, wideThreads, _, _ := k.clusterGeometry()
	return [3]int{rows, 1, 1}, [3]int{wideThreads, 1, 1}
}

func (k RowKernel) checkCluster(syms map[string]rowSym) error {
	if k.Slots < 0 || k.Slots > 32 {
		return fmt.Errorf("row kernel %s: slots %d must be in 0..32", k.Name, k.Slots)
	}
	if k.SlotIDs != "" && k.Slots == 0 {
		return fmt.Errorf("row kernel %s: SlotIDs set without Slots", k.Name)
	}
	if k.SlotIDs != "" {
		s, ok := syms[k.SlotIDs]
		if !ok || s.tensor == nil || s.tensor.Kind != RowScalar || (s.tensor.DType != I32 && s.tensor.DType != I64) {
			return fmt.Errorf("row kernel %s: SlotIDs %q must be an I32 or I64 RowScalar input", k.Name, k.SlotIDs)
		}
	}
	if k.Slots > 0 {
		reds := 0
		for _, v := range k.Values {
			if v.Reduce != "" {
				reds++
				if v.Reduce != "sum" {
					return fmt.Errorf("row kernel %s: slot reduction %s must be sum", k.Name, v.Name)
				}
			}
		}
		if reds > 1 {
			return fmt.Errorf("row kernel %s: at most one reduction supported with Slots", k.Name)
		}
	}
	if !k.Cluster {
		return nil
	}
	if k.D < 256 || k.D%256 != 0 {
		return fmt.Errorf("row kernel %s: cluster width %d must be a multiple of 256", k.Name, k.D)
	}
	_, _, _, cvec := k.clusterGeometry()
	for _, t := range append(append([]RowTensor(nil), k.Inputs...), k.Outputs...) {
		gs, rs := t.strides(k.D)
		if t.Kind == RowData && (t.Offset%cvec != 0 || gs%cvec != 0 || rs%cvec != 0 || t.Padded%cvec != 0) {
			return fmt.Errorf("row kernel %s: %s offset and strides must be multiples of %d", k.Name, t.Name, cvec)
		}
	}
	return nil
}

// ClusterCUDA returns k's 8-CTA cluster kernel (<Name>_cluster).
func (k RowKernel) ClusterCUDA() (string, error) {
	return k.emitSplit(false)
}

// WideCUDA returns k's single-block wide kernel (<Name>_wide).
func (k RowKernel) WideCUDA() (string, error) {
	return k.emitSplit(true)
}

func (k RowKernel) emitSplit(wide bool) (string, error) {
	c := k
	c.Cluster = true
	syms, err := c.check()
	if err != nil {
		return "", err
	}
	ctaThreads, wideThreads, per, vec := c.clusterGeometry()
	if wide && wideThreads > 1024 {
		return "", fmt.Errorf("row kernel %s: wide block %d exceeds 1024 threads", k.Name, wideThreads)
	}
	var b strings.Builder
	params := "\n    " + strings.Join(k.params(), ",\n    ")
	if wide {
		fmt.Fprintf(&b, "// %s_wide: %s_cluster's %d CTAs as one block, grid (rows), %d threads.\n",
			k.Name, k.Name, ClusterSize, wideThreads)
		fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s_wide(%s) {\n",
			wideThreads, k.Name, params)
	} else {
		fmt.Fprintf(&b, "// %s_cluster: %s split over an %d-CTA cluster, grid (%d, rows), %d threads.\n",
			k.Name, k.Name, ClusterSize, ClusterSize, ctaThreads)
		// Without minBlocksPerMultiprocessor = 1, ptxas caps __cluster_dims__
		// kernels at 80 registers.
		fmt.Fprintf(&b, "__global__ void __cluster_dims__(%d, 1, 1) __launch_bounds__(%d, 1) %s_cluster(%s) {\n",
			ClusterSize, ctaThreads, k.Name, params)
	}
	if k.Slots > 0 {
		mode := 1
		if wide {
			mode = 2
		}
		body, err := c.slotBody(syms, mode)
		if err != nil {
			return "", err
		}
		b.WriteString(body)
		b.WriteString("}\n")
		if !wide {
			fmt.Fprintf(&b, "EG_REGISTER_CLUSTER(%s_cluster)\n", k.Name)
		}
		return b.String(), nil
	}
	b.WriteString("    EG_SHARED(float, eg_red, [2][32]);\n")
	if wide {
		b.WriteString("    const int r = blockIdx.x, g = threadIdx.x;\n")
	} else {
		b.WriteString("    const int r = blockIdx.y, g = blockIdx.x * blockDim.x + threadIdx.x;\n")
	}
	fmt.Fprintf(&b, "    constexpr int PER = %d, VEC = %d;\n", per, vec)
	b.WriteString("#define EG_COL(j) (g * PER + (j))\n")

	addr := func(t RowTensor, group string) string {
		gs, rs := t.strides(k.D)
		return fmt.Sprintf("(size_t)%d + (size_t)(%s) * %d + (size_t)(r %% %d) * %d",
			t.Offset, group, gs, t.group(), rs)
	}
	// Hoist every global load before the first reduction / cluster barrier so
	// no load stalls behind a cluster barrier.
	for _, in := range k.Inputs {
		switch in.Kind {
		case RowData:
			grp := fmt.Sprintf("r / %d", in.group())
			if in.Padded > 0 {
				fmt.Fprintf(&b, "    float %s_v[PER] = {};\n    {\n        const %s* p = %s + %s;\n",
					in.Name, in.DType.cType(), in.buffer(), addr(in, grp))
				fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; j += VEC) {\n")
				fmt.Fprintf(&b, "            if (EG_COL(j) < %d) eg_load<VEC>(p + EG_COL(j), &%s_v[j]);\n        }\n    }\n",
					in.Padded, in.Name)
			} else {
				fmt.Fprintf(&b, "    float %s_v[PER];\n    {\n        const %s* p = %s + %s;\n",
					in.Name, in.DType.cType(), in.buffer(), addr(in, grp))
				fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; j += VEC) {\n")
				fmt.Fprintf(&b, "            eg_load<VEC>(p + EG_COL(j), &%s_v[j]);\n        }\n    }\n", in.Name)
			}
		case ColVector:
			fmt.Fprintf(&b, "    float %s_v[PER];\n#pragma unroll\n", in.Name)
			fmt.Fprintf(&b, "    for (int j = 0; j < PER; j += VEC) {\n")
			fmt.Fprintf(&b, "        eg_load<VEC>(%s + %d + EG_COL(j), &%s_v[j]);\n    }\n",
				in.buffer(), in.Offset, in.Name)
		case RowScalar:
			ld := scalarLoad(in.DType, fmt.Sprintf("%s[%d + r / %d]", in.buffer(), in.Offset, in.group()))
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", in.Name, ld)
		case Scalar:
			ld := scalarLoad(in.DType, fmt.Sprintf("%s[%d]", in.buffer(), in.Offset))
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", in.Name, ld)
		}
	}
	for _, o := range k.Outputs {
		if o.Index != "" {
			fmt.Fprintf(&b, "    const long long %s_idx = (long long)%s[r / %d];\n",
				o.Name, o.Index, o.group())
		}
	}

	computed := make(map[string]bool)
	for _, in := range k.Inputs {
		computed[in.Name] = true
	}
	stored := make([]bool, len(k.Outputs))
	// Emit each output store as soon as its value is computed (e.g. stream_out
	// before the second cluster reduction).
	flushStores := func() {
		for i, o := range k.Outputs {
			if stored[i] || !computed[o.Value] {
				continue
			}
			stored[i] = true
			group := fmt.Sprintf("r / %d", o.group())
			if o.Index != "" {
				fmt.Fprintf(&b, "    if (%s_idx >= 0) {\n", o.Name)
				group = o.Name + "_idx"
			} else {
				b.WriteString("    {\n")
			}
			fmt.Fprintf(&b, "        %s* p = %s + %s;\n", o.DType.cType(), o.Name, addr(o, group))
			fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; j += VEC) {\n")
			fmt.Fprintf(&b, "            eg_store_wb<VEC>(p + EG_COL(j), &%s_v[j]);\n        }\n    }\n", o.Value)
		}
	}

	ref := func(a string) string {
		switch a {
		case "col":
			return "(float)EG_COL(j)"
		case "sgn":
			return "((EG_COL(j) & 1) ? 1.0f : -1.0f)"
		}
		if strings.HasPrefix(a, "~") {
			return a[1:] + "_v[j ^ 1]"
		}
		if syms[a].vector {
			return a + "_v[j]"
		}
		return a + "_s"
	}

	slot := 0
	for _, v := range k.Values {
		args := make([]string, len(v.Args))
		for i, a := range v.Args {
			args[i] = ref(a)
		}
		expr, err := v.Body.CUDA(args...)
		if err != nil {
			return "", err
		}
		switch {
		case v.Reduce != "":
			sumFn := "eg_cluster_sum"
			if wide {
				sumFn = "eg_block_sum8"
			}
			if v.Reduce == "max" {
				maxFn := "eg_cluster_max"
				if wide {
					maxFn = "eg_block_max8"
				}
				fmt.Fprintf(&b, "    float %s_s = -__uint_as_float(0x7f800000u);\n#pragma unroll\n", v.Name)
				fmt.Fprintf(&b, "    for (int j = 0; j < PER; ++j) %s_s = fmaxf(%s_s, %s);\n",
					v.Name, v.Name, expr)
				fmt.Fprintf(&b, "    %s_s = %s(%s_s, eg_red, %d);\n", v.Name, maxFn, v.Name, slot)
			} else {
				fmt.Fprintf(&b, "    float %s_s = 0.0f;\n#pragma unroll\n", v.Name)
				fmt.Fprintf(&b, "    for (int j = 0; j < PER; ++j) %s_s += %s;\n", v.Name, expr)
				fmt.Fprintf(&b, "    %s_s = %s(%s_s, eg_red, %d);\n", v.Name, sumFn, v.Name, slot)
			}
			slot ^= 1
		case syms[v.Name].vector:
			fmt.Fprintf(&b, "    float %s_v[PER];\n#pragma unroll\n", v.Name)
			fmt.Fprintf(&b, "    for (int j = 0; j < PER; ++j) {\n        %s_v[j] = %s;\n    }\n",
				v.Name, expr)
		default:
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", v.Name, expr)
		}
		computed[v.Name] = true
		flushStores()
	}
	flushStores()
	b.WriteString("#undef EG_COL\n")
	if !wide {
		b.WriteString("    eg_cluster_sync();\n")
	}
	b.WriteString("}\n")
	if !wide {
		fmt.Fprintf(&b, "EG_REGISTER_CLUSTER(%s_cluster)\n", k.Name)
	}
	return b.String(), nil
}

// slotBody emits the body of a multi-slot RowKernel (Slots > 0).
// mode is 0 for the 256-thread block kernel, 1 for cluster, 2 for wide.
func (k RowKernel) slotBody(syms map[string]rowSym, mode int) (string, error) {
	var b strings.Builder
	K := k.Slots
	var per, vec, nw int
	switch mode {
	case 0:
		threads, p, v := k.geometry()
		per, vec = p, v
		b.WriteString("    const int r = blockIdx.x, lane = threadIdx.x;\n")
		b.WriteString("    __shared__ float eg_red[8];\n")
		fmt.Fprintf(&b, "    __shared__ float eg_tot[%d];\n", K)
		fmt.Fprintf(&b, "    constexpr int PER = %d, VEC = %d, THREADS = %d;\n", per, vec, threads)
		b.WriteString("#define EG_COL(j) (((j) / VEC) * (THREADS * VEC) + lane * VEC + (j) % VEC)\n")
	case 1:
		ctaThreads, _, p, v := k.clusterGeometry()
		per, vec, nw = p, v, ctaThreads/32
		b.WriteString("    EG_SHARED(float, eg_red, [2][32]);\n")
		fmt.Fprintf(&b, "    EG_SHARED(float, eg_part, [%d][%d]);\n", nw, K)
		fmt.Fprintf(&b, "    EG_SHARED(float, eg_tot, [%d]);\n", K)
		b.WriteString("    const int r = blockIdx.y, g = blockIdx.x * blockDim.x + threadIdx.x;\n")
		fmt.Fprintf(&b, "    constexpr int PER = %d, VEC = %d;\n", per, vec)
		b.WriteString("#define EG_COL(j) (g * PER + (j))\n")
	case 2:
		ctaThreads, _, p, v := k.clusterGeometry()
		per, vec, nw = p, v, ctaThreads/32
		b.WriteString("    EG_SHARED(float, eg_red, [2][32]);\n")
		fmt.Fprintf(&b, "    EG_SHARED(float, eg_part, [%d][%d]);\n", ClusterSize*nw, K)
		fmt.Fprintf(&b, "    EG_SHARED(float, eg_tot, [%d]);\n", K)
		b.WriteString("    const int r = blockIdx.x, g = threadIdx.x;\n")
		fmt.Fprintf(&b, "    constexpr int PER = %d, VEC = %d;\n", per, vec)
		b.WriteString("#define EG_COL(j) (g * PER + (j))\n")
	}

	// Hoist ColVector, Scalar, and per-slot RowScalar loads before the slot
	// loop so all global loads complete before the first barrier.
	for _, in := range k.Inputs {
		switch in.Kind {
		case ColVector:
			fmt.Fprintf(&b, "    float %s_v[PER];\n#pragma unroll\n", in.Name)
			fmt.Fprintf(&b, "    for (int j = 0; j < PER; j += VEC) {\n")
			fmt.Fprintf(&b, "        eg_load<VEC>(%s + %d + EG_COL(j), &%s_v[j]);\n    }\n",
				in.buffer(), in.Offset, in.Name)
		case Scalar:
			ld := scalarLoad(in.DType, fmt.Sprintf("%s[%d]", in.buffer(), in.Offset))
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", in.Name, ld)
		case RowScalar:
			if in.Name == k.SlotIDs {
				fmt.Fprintf(&b, "    int64_t %s_slot[%d];\n#pragma unroll\n", in.Name, K)
				fmt.Fprintf(&b, "    for (int k = 0; k < %d; ++k) {\n", K)
				fmt.Fprintf(&b, "        %s_slot[k] = (int64_t)%s[%d + (r * %d + k) / %d];\n    }\n",
					in.Name, in.buffer(), in.Offset, K, in.group())
			} else {
				fmt.Fprintf(&b, "    float %s_slot[%d];\n#pragma unroll\n", in.Name, K)
				fmt.Fprintf(&b, "    for (int k = 0; k < %d; ++k) {\n", K)
				ld := scalarLoad(in.DType, fmt.Sprintf("%s[%d + (r * %d + k) / %d]",
					in.buffer(), in.Offset, K, in.group()))
				fmt.Fprintf(&b, "        %s_slot[k] = %s;\n    }\n", in.Name, ld)
			}
		}
	}

	redIdx := -1
	for i, v := range k.Values {
		if v.Reduce != "" {
			redIdx = i
			break
		}
	}

	// Find pre-reduction vector values used after redIdx.
	needAfter := make(map[string]bool)
	for _, v := range k.Values[redIdx+1:] {
		for _, a := range v.Args {
			needAfter[strings.TrimPrefix(a, "~")] = true
		}
	}
	for _, o := range k.Outputs {
		needAfter[o.Value] = true
	}
	savedVec := make(map[string]bool)
	if redIdx >= 0 {
		for _, in := range k.Inputs {
			if in.Kind == RowData && needAfter[in.Name] {
				savedVec[in.Name] = true
				fmt.Fprintf(&b, "    float %s_slot[%d][PER];\n", in.Name, K)
			}
		}
		for _, v := range k.Values[:redIdx] {
			if syms[v.Name].vector && needAfter[v.Name] {
				savedVec[v.Name] = true
				fmt.Fprintf(&b, "    float %s_slot[%d][PER];\n", v.Name, K)
			}
		}
		fmt.Fprintf(&b, "    float %s_slot[%d];\n", k.Values[redIdx].Name, K)
	}

	addrSlot := func(t RowTensor, rowVar string) string {
		gs, rs := t.strides(k.D)
		return fmt.Sprintf("(size_t)%d + (size_t)(%s / %d) * %d + (size_t)(%s %% %d) * %d",
			t.Offset, rowVar, t.group(), gs, rowVar, t.group(), rs)
	}

	ref := func(a string, postRed bool) string {
		switch a {
		case "col":
			return "(float)EG_COL(j)"
		case "sgn":
			return "((EG_COL(j) & 1) ? 1.0f : -1.0f)"
		}
		if strings.HasPrefix(a, "~") {
			name := a[1:]
			if postRed && savedVec[name] {
				return name + "_slot[k][j ^ 1]"
			}
			return name + "_v[j ^ 1]"
		}
		if syms[a].vector {
			if postRed && savedVec[a] {
				return a + "_slot[k][j]"
			}
			return a + "_v[j]"
		}
		return a + "_s"
	}

	if redIdx >= 0 {
		redVal := k.Values[redIdx]
		fmt.Fprintf(&b, "#pragma unroll\n    for (int k = 0; k < %d; ++k) {\n", K)
		fmt.Fprintf(&b, "        %s_slot[k] = 0.0f;\n", redVal.Name)
		if k.SlotIDs != "" {
			fmt.Fprintf(&b, "        if (%s_slot[k] < 0) continue;\n", k.SlotIDs)
		}
		fmt.Fprintf(&b, "        const int sr = r * %d + k;\n", K)
		for _, in := range k.Inputs {
			switch in.Kind {
			case RowData:
				if in.Padded > 0 {
					fmt.Fprintf(&b, "        float %s_v[PER] = {};\n        {\n            const %s* p = %s + %s;\n",
						in.Name, in.DType.cType(), in.buffer(), addrSlot(in, "sr"))
					fmt.Fprintf(&b, "#pragma unroll\n            for (int j = 0; j < PER; j += VEC) {\n")
					fmt.Fprintf(&b, "                if (EG_COL(j) < %d) eg_load<VEC>(p + EG_COL(j), &%s_v[j]);\n            }\n        }\n",
						in.Padded, in.Name)
				} else {
					fmt.Fprintf(&b, "        float %s_v[PER];\n        {\n            const %s* p = %s + %s;\n",
						in.Name, in.DType.cType(), in.buffer(), addrSlot(in, "sr"))
					fmt.Fprintf(&b, "#pragma unroll\n            for (int j = 0; j < PER; j += VEC) {\n")
					fmt.Fprintf(&b, "                eg_load<VEC>(p + EG_COL(j), &%s_v[j]);\n            }\n        }\n", in.Name)
				}
				if savedVec[in.Name] {
					fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; ++j) %s_slot[k][j] = %s_v[j];\n",
						in.Name, in.Name)
				}
			case RowScalar:
				fmt.Fprintf(&b, "        const float %s_s = (float)%s_slot[k];\n", in.Name, in.Name)
			}
		}
		for _, v := range k.Values[:redIdx] {
			args := make([]string, len(v.Args))
			for i, a := range v.Args {
				args[i] = ref(a, false)
			}
			expr, err := v.Body.CUDA(args...)
			if err != nil {
				return "", err
			}
			if syms[v.Name].vector {
				fmt.Fprintf(&b, "        float %s_v[PER];\n#pragma unroll\n", v.Name)
				fmt.Fprintf(&b, "        for (int j = 0; j < PER; ++j) {\n            %s_v[j] = %s;\n", v.Name, expr)
				if savedVec[v.Name] {
					fmt.Fprintf(&b, "            %s_slot[k][j] = %s_v[j];\n", v.Name, v.Name)
				}
				b.WriteString("        }\n")
			} else {
				fmt.Fprintf(&b, "        const float %s_s = %s;\n", v.Name, expr)
			}
		}
		rargs := make([]string, len(redVal.Args))
		for i, a := range redVal.Args {
			rargs[i] = ref(a, false)
		}
		rexpr, err := redVal.Body.CUDA(rargs...)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; ++j) %s_slot[k] += %s;\n",
			redVal.Name, rexpr)
		b.WriteString("    }\n")

		switch mode {
		case 0:
			fmt.Fprintf(&b, "    for (int k = 0; k < %d; ++k) {\n", K)
			fmt.Fprintf(&b, "        float v = eg_warp_sum(%s_slot[k]);\n", redVal.Name)
			b.WriteString("        eg_tot[k] = eg_block_combine<0>(v, eg_red);\n    }\n")
		case 1:
			fmt.Fprintf(&b, "    eg_cluster_sum_slots<%d, %d>(%s_slot, eg_red[0], eg_part, eg_tot);\n",
				K, nw, redVal.Name)
		case 2:
			fmt.Fprintf(&b, "    eg_block_sum8_slots<%d, %d>(%s_slot, eg_part, eg_tot);\n",
				K, nw, redVal.Name)
		}
	}

	for _, o := range k.Outputs {
		fmt.Fprintf(&b, "    float %s_acc[PER] = {};\n", o.Name)
	}
	fmt.Fprintf(&b, "#pragma unroll\n    for (int k = 0; k < %d; ++k) {\n", K)
	if k.SlotIDs != "" {
		fmt.Fprintf(&b, "        if (%s_slot[k] < 0) continue;\n", k.SlotIDs)
	}
	if redIdx < 0 {
		fmt.Fprintf(&b, "        const int sr = r * %d + k;\n", K)
		for _, in := range k.Inputs {
			if in.Kind == RowData {
				fmt.Fprintf(&b, "        float %s_v[PER];\n        {\n            const %s* p = %s + %s;\n",
					in.Name, in.DType.cType(), in.buffer(), addrSlot(in, "sr"))
				fmt.Fprintf(&b, "#pragma unroll\n            for (int j = 0; j < PER; j += VEC) {\n")
				fmt.Fprintf(&b, "                eg_load<VEC>(p + EG_COL(j), &%s_v[j]);\n            }\n        }\n", in.Name)
			}
		}
	} else {
		fmt.Fprintf(&b, "        const float %s_s = eg_tot[k];\n", k.Values[redIdx].Name)
	}
	for _, in := range k.Inputs {
		if in.Kind == RowScalar {
			fmt.Fprintf(&b, "        const float %s_s = (float)%s_slot[k];\n", in.Name, in.Name)
		}
	}
	for _, v := range k.Values[redIdx+1:] {
		args := make([]string, len(v.Args))
		for i, a := range v.Args {
			args[i] = ref(a, redIdx >= 0)
		}
		expr, err := v.Body.CUDA(args...)
		if err != nil {
			return "", err
		}
		if syms[v.Name].vector {
			fmt.Fprintf(&b, "        float %s_v[PER];\n#pragma unroll\n", v.Name)
			fmt.Fprintf(&b, "        for (int j = 0; j < PER; ++j) %s_v[j] = %s;\n", v.Name, expr)
		} else {
			fmt.Fprintf(&b, "        const float %s_s = %s;\n", v.Name, expr)
		}
	}
	for _, o := range k.Outputs {
		valExpr := ref(o.Value, redIdx >= 0)
		fmt.Fprintf(&b, "#pragma unroll\n        for (int j = 0; j < PER; ++j) %s_acc[j] += %s;\n",
			o.Name, valExpr)
	}
	b.WriteString("    }\n")

	storeFn := "eg_store_wb"
	if mode == 0 {
		storeFn = "eg_store"
	}
	for _, o := range k.Outputs {
		gs, rs := o.strides(k.D)
		group := fmt.Sprintf("r / %d", o.group())
		if o.Index != "" {
			fmt.Fprintf(&b, "    {\n        const long long g = (long long)%s[r / %d];\n        if (g >= 0) {\n",
				o.Index, o.group())
			group = "g"
		} else {
			b.WriteString("    {\n        {\n")
		}
		fmt.Fprintf(&b, "            %s* p = %s + (size_t)%d + (size_t)(%s) * %d + (size_t)(r %% %d) * %d;\n",
			o.DType.cType(), o.Name, o.Offset, group, gs, o.group(), rs)
		fmt.Fprintf(&b, "#pragma unroll\n            for (int j = 0; j < PER; j += VEC) {\n")
		fmt.Fprintf(&b, "                %s<VEC>(p + EG_COL(j), &%s_acc[j]);\n            }\n        }\n    }\n",
			storeFn, o.Name)
	}
	b.WriteString("#undef EG_COL\n")
	if mode == 1 {
		b.WriteString("    eg_cluster_sync();\n")
	}
	return b.String(), nil
}

func (k RowKernel) clusterLauncher() string {
	params := k.params()
	names := paramNames(params)
	ctaThreads, wideThreads, _, _ := k.clusterGeometry()
	maxRows := k.maxClusterRows()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(\n    %s,\n    cudaStream_t stream) {\n",
		k.Name, strings.Join(params, ",\n    "))
	b.WriteString("    if (rows <= 0) return;\n")
	fmt.Fprintf(&b, "    if (rows <= %d) {\n", maxRows)
	fmt.Fprintf(&b, "        const dim3 grid(%d, rows), block(%d);\n", ClusterSize, ctaThreads)
	fmt.Fprintf(&b, "        EG_LAUNCH(%s_cluster, grid, block, 0, stream, %s);\n",
		k.Name, strings.Join(names, ", "))
	if k.HasWide() {
		b.WriteString("    } else {\n")
		fmt.Fprintf(&b, "        const dim3 grid(rows), block(%d);\n", wideThreads)
		fmt.Fprintf(&b, "        EG_LAUNCH(%s_wide, grid, block, 0, stream, %s);\n    }\n",
			k.Name, strings.Join(names, ", "))
	} else {
		b.WriteString("    } else {\n")
		b.WriteString("        const dim3 grid(rows), block(256);\n")
		fmt.Fprintf(&b, "        EG_LAUNCH(%s, grid, block, 0, stream, %s);\n    }\n",
			k.Name, strings.Join(names, ", "))
	}
	b.WriteString("}\n")
	return b.String()
}
