package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// DynSparseTarget names the runtime tensor that is pruned to 2:4 structured
// sparsity just before it is consumed as operand A of a tcgen05.mma.sp.
type DynSparseTarget uint8

const (
	// DynSparsePostSwiGLU prunes h = SiLU(gate) * up before the dense down_proj.
	DynSparsePostSwiGLU DynSparseTarget = iota
	// DynSparseKVCachePages prunes K or V pages along the MMA reduction
	// dimension (head dim for K in S^T = K Q^T, tokens for V in O^T = V^T P^T).
	DynSparseKVCachePages
	// DynSparseAttnProbP prunes the post-softmax probabilities P before P V.
	DynSparseAttnProbP
)

// String returns the name of t.
func (t DynSparseTarget) String() string {
	switch t {
	case DynSparsePostSwiGLU:
		return "post-swiglu"
	case DynSparseKVCachePages:
		return "kv-cache-pages"
	case DynSparseAttnProbP:
		return "attn-prob-p"
	default:
		return fmt.Sprintf("DynSparseTarget(%d)", uint8(t))
	}
}

// DynSparseTile is a [Rows, DenseCols] tensor pruned at runtime to 2:4
// structured sparsity along its columns.
//
// Values and Meta use the layout of [SpCompress2to4F32] applied row by row:
// Values holds the two kept elements of each 4-element group in ascending
// column order and Meta packs one 4-bit (idx0 | idx1<<2) nibble per group,
// eight groups per uint32 word.
type DynSparseTile struct {
	Target    DynSparseTarget
	Rows      int
	DenseCols int

	Values []float32 // [Rows, DenseCols/2] kept values (dequantized when quantized)
	Meta   []uint32  // [Rows, DenseCols/32] 2:4 index metadata

	// PackedE2M1 and ScalesUE8M0 are set only when the kept values were
	// quantized to MXFP4: two E2M1 codes per byte, one UE8M0 scale per 32
	// kept values ([Rows, DenseCols/4] and [Rows, DenseCols/64]).
	PackedE2M1  []uint8
	ScalesUE8M0 []uint8

	// RetainedL2Energy is ||prune(x)||^2 / ||x||^2 of the pruning step alone
	// (1 when x is all zeros).
	RetainedL2Energy float64
}

// DynamicPrune2To4 prunes the row-major [rows, cols] matrix x to 2:4
// structured sparsity along its columns, keeping in every 4-element group the
// two elements of largest magnitude. Ties go to the lower column index, the
// MAXABS rule of PTX ISA 9.4 spcompress and tcgen05.ld.spcompress, so for any
// x the result retains the largest l2 energy of any 2:4 pattern.
//
// cols must be a positive multiple of 32. When quantize is true, cols must be
// a multiple of 64 and the kept values are further quantized to MXFP4 (E2M1
// with one UE8M0 scale per 32 kept values).
func DynamicPrune2To4(target DynSparseTarget, x []float32, rows, cols int, quantize bool) (DynSparseTile, error) {
	if rows < 1 || cols < 32 || cols%32 != 0 {
		return DynSparseTile{}, fmt.Errorf("tcgen05 dynamic prune 2:4: invalid shape rows=%d cols=%d (want rows>=1, cols a positive multiple of 32)", rows, cols)
	}
	if quantize && cols%64 != 0 {
		return DynSparseTile{}, fmt.Errorf("tcgen05 dynamic prune 2:4: cols=%d must be a multiple of 64 to quantize", cols)
	}
	if len(x) != rows*cols {
		return DynSparseTile{}, fmt.Errorf("tcgen05 dynamic prune 2:4: len(x)=%d, want %d", len(x), rows*cols)
	}
	if target > DynSparseAttnProbP {
		return DynSparseTile{}, fmt.Errorf("tcgen05 dynamic prune 2:4: invalid target %d", uint8(target))
	}
	t := DynSparseTile{
		Target:    target,
		Rows:      rows,
		DenseCols: cols,
		Values:    make([]float32, rows*cols/2),
		Meta:      make([]uint32, rows*cols/32),
	}
	var kept, total float64
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		for _, v := range row {
			total += float64(v) * float64(v)
		}
		c, m, err := SpCompress2to4F32(row, SpOpMaxAbs)
		if err != nil {
			return DynSparseTile{}, fmt.Errorf("tcgen05 dynamic prune 2:4: row %d: %w", r, err)
		}
		for _, v := range c {
			kept += float64(v) * float64(v)
		}
		copy(t.Values[r*cols/2:], c)
		copy(t.Meta[r*cols/32:], m)
	}
	t.RetainedL2Energy = 1
	if total > 0 {
		t.RetainedL2Energy = kept / total
	}
	if quantize {
		t.PackedE2M1, t.ScalesUE8M0 = QuantizeMXFP4E2M1(t.Values)
		t.Values = DequantizeMXFP4E2M1(t.PackedE2M1, t.ScalesUE8M0)
	}
	return t, nil
}

// Dense returns the [Rows, DenseCols] dense matrix that t represents, with
// pruned positions set to zero.
func (t DynSparseTile) Dense() ([]float32, error) {
	if len(t.Values) != t.Rows*t.DenseCols/2 || len(t.Meta) != t.Rows*t.DenseCols/32 {
		return nil, fmt.Errorf("tcgen05 dyn sparse tile: inconsistent lengths values=%d meta=%d for %dx%d", len(t.Values), len(t.Meta), t.Rows, t.DenseCols)
	}
	out := make([]float32, 0, t.Rows*t.DenseCols)
	half, words := t.DenseCols/2, t.DenseCols/32
	for r := range t.Rows {
		row, err := SpDecompress2to4F32(t.Values[r*half:(r+1)*half], t.Meta[r*words:(r+1)*words], t.DenseCols)
		if err != nil {
			return nil, fmt.Errorf("tcgen05 dyn sparse tile: row %d: %w", r, err)
		}
		out = append(out, row...)
	}
	return out, nil
}

// StorageBytes returns the bytes needed to hold t: the kept values at
// valueBits each (4 when t is quantized, ignoring valueBits), the 2:4
// metadata at 4 bits per 4-element group, and one byte per UE8M0 scale.
// A dense tensor of the same shape needs Rows*DenseCols*valueBits/8 bytes,
// so for BF16 the ratio is 9/16 and for 8-bit formats 5/8.
func (t DynSparseTile) StorageBytes(valueBits int) int64 {
	if t.PackedE2M1 != nil {
		valueBits = 4
	}
	kept := int64(t.Rows) * int64(t.DenseCols) / 2
	meta := int64(t.Rows) * int64(t.DenseCols) / 8
	return kept*int64(valueBits)/8 + meta + int64(len(t.ScalesUE8M0))
}

// SparseMMAActivationDenseWeight returns D = A * W^T for the 2:4 sparse
// activation A = act [Rows, DenseCols] and the dense weight w [n, DenseCols]
// (row-major, one output feature per row, the layout of a down_proj weight).
// It is the host reference of tcgen05.mma.sp with a runtime-pruned operand A
// and an unmodified dense operand B. Accumulation is float32 in group order.
func SparseMMAActivationDenseWeight(act DynSparseTile, w []float32, n int) ([]float32, error) {
	k := act.DenseCols
	if n < 1 || len(w) != n*k {
		return nil, fmt.Errorf("tcgen05 sparse act x dense weight: len(w)=%d, want n*%d with n=%d>=1", len(w), k, n)
	}
	if len(act.Values) != act.Rows*k/2 || len(act.Meta) != act.Rows*k/32 {
		return nil, fmt.Errorf("tcgen05 sparse act x dense weight: inconsistent tile lengths values=%d meta=%d", len(act.Values), len(act.Meta))
	}
	out := make([]float32, act.Rows*n)
	half, words := k/2, k/32
	for r := range act.Rows {
		vals := act.Values[r*half : (r+1)*half]
		meta := act.Meta[r*words : (r+1)*words]
		for j := range n {
			wr := w[j*k : (j+1)*k]
			var acc float32
			for g := range k / 4 {
				nib := (meta[g>>3] >> (4 * (g & 7))) & 0xf
				i0, i1 := int(nib&3), int(nib>>2)
				if i0 >= i1 {
					return nil, fmt.Errorf("tcgen05 sparse act x dense weight: row %d group %d invalid metadata (%d, %d)", r, g, i0, i1)
				}
				acc += vals[2*g]*wr[4*g+i0] + vals[2*g+1]*wr[4*g+i1]
			}
			out[r*n+j] = acc
		}
	}
	return out, nil
}

// SwiGLU returns SiLU(gate) * up elementwise in float32, the activation
// pruned by [TCGen05DynSparseSwiGLU].
func SwiGLU(gate, up []float32) []float32 {
	h := make([]float32, len(gate))
	for i, g := range gate {
		h[i] = g / (1 + float32(math.Exp(-float64(g)))) * up[i]
	}
	return h
}

// TCGen05DynSparseSwiGLU configures a single-CTA kernel that fuses a SwiGLU
// epilogue, runtime 2:4 pruning of h = SiLU(gate) * up, and a 2:4 sparse
// down_proj tcgen05.mma.sp against an unmodified dense weight.
//
// gate and up [M, I] are staged into TMEM with tcgen05.st (standing in for
// the gate_proj and up_proj accumulators), h is computed in registers and
// written back over the gate columns, and each 4-column group of h is pruned:
// on sm_107a the kernel issues
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2,
// which also returns the running row absmax used for UE8M0 scaling; on other
// targets the selection runs in registers. The compressed operand and its
// metadata then feed tcgen05.mma.sp.cta_group::1.kind::tf32 with the dense
// down_proj weight [N, I] as operand B.
//
// The device instruction paths are emitted for sm_100a, sm_103a and sm_107a
// and use a structural instruction descriptor and shared-memory descriptors;
// they have not been executed on tcgen05 hardware. When EG_EMU is defined
// (cudahostemu and sm_12x devices) the same kernel runs an exact C++
// fallback, which is what the tests check against [TCGen05DynSparseSwiGLU.Reference].
type TCGen05DynSparseSwiGLU struct {
	Name   string
	M      int    // token rows; must be 128
	I      int    // intermediate size (MMA K); 32, 64 or 128 (default 64)
	N      int    // down_proj output features; 32, 64 or 128 and <= 2*I (default 64)
	Target string // "sm_100a", "sm_103a", "sm_107a" or "sm_107f" (default "sm_107a")
}

func (k TCGen05DynSparseSwiGLU) norm() (TCGen05DynSparseSwiGLU, error) {
	if k.Name == "" {
		k.Name = "tcgen05_dynsparse_swiglu"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 dynsparse swiglu: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.I == 0 {
		k.I = 64
	}
	if k.N == 0 {
		k.N = 64
	}
	if k.Target == "" {
		k.Target = "sm_107a"
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 dynsparse swiglu %s: unsupported target %q", k.Name, k.Target)
	}
	if k.M != 128 {
		return k, fmt.Errorf("tcgen05 dynsparse swiglu %s: M=%d, want 128", k.Name, k.M)
	}
	if k.I != 32 && k.I != 64 && k.I != 128 {
		return k, fmt.Errorf("tcgen05 dynsparse swiglu %s: I=%d, want 32, 64 or 128", k.Name, k.I)
	}
	if (k.N != 32 && k.N != 64 && k.N != 128) || k.N > 2*k.I {
		return k, fmt.Errorf("tcgen05 dynsparse swiglu %s: N=%d, want 32, 64 or 128 and <= 2*I=%d", k.Name, k.N, 2*k.I)
	}
	if b := k.sharedBytes(); b > 48<<10 {
		return k, fmt.Errorf("tcgen05 dynsparse swiglu %s: %d bytes of static shared memory exceed 48 KiB (I=%d, N=%d)", k.Name, b, k.I, k.N)
	}
	return k, nil
}

func (k TCGen05DynSparseSwiGLU) sharedBytes() int {
	return 4 * (k.M*k.I/2 + k.M*k.I/32 + k.N*k.I)
}

// Validate reports whether k describes a supported kernel.
func (k TCGen05DynSparseSwiGLU) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 128-thread block.
func (k TCGen05DynSparseSwiGLU) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// InstDescriptor returns the tcgen05.mma.sp kind::tf32 instruction
// descriptor with the 2:4 sparsity bit set.
func (k TCGen05DynSparseSwiGLU) InstDescriptor() (uint32, error) {
	k, err := k.norm()
	if err != nil {
		return 0, err
	}
	d := InstDesc{
		Kind:   KindTF32,
		Sparse: true,
		DType:  ElemF32,
		AType:  ElemTF32,
		BType:  ElemTF32,
		M:      k.M,
		N:      k.N,
		K:      16,
	}
	if err := d.ValidateForTarget(k.Target); err != nil {
		return 0, err
	}
	return d.Encode()
}

// TMEM column layout: gate|up at 0 (2*I columns, h overwrites gate),
// accumulator at 2*I (N columns), 2:4 metadata at 2*I+N (32 columns).
func (k TCGen05DynSparseSwiGLU) tmemLayout() (gu, d, e uint32) {
	return 0, uint32(2 * k.I), uint32(2*k.I + k.N)
}

// TraceEvents returns the tcgen05 event sequence of the kernel for
// cudatmem.VerifyTrace.
func (k TCGen05DynSparseSwiGLU) TraceEvents() []cudatmem.TraceEvent {
	k, err := k.norm()
	if err != nil {
		return nil
	}
	gu, d, e := k.tmemLayout()
	ev := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Offset: gu, Columns: uint32(2 * k.I), Rows: 128},
		{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Offset: d, Columns: uint32(k.N), Rows: 128},
		{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Offset: e, Columns: 32, Rows: 128},
	}
	perWarp := func(kind cudatmem.EventKind, off uint32, cols int) {
		for w := range 4 {
			ev = append(ev, cudatmem.TraceEvent{Kind: kind, Warp: w, ActiveThreads: 32, CTAGroup: 1, Offset: off, Columns: uint32(cols), RowStart: uint32(32 * w), Rows: 32})
		}
	}
	perWarp(cudatmem.EventSt, gu, 2*k.I) // stage gate, up
	perWarp(cudatmem.EventLd, gu, 2*k.I) // load gate, up
	perWarp(cudatmem.EventSt, gu, k.I)   // h over gate
	perWarp(cudatmem.EventLd, gu, k.I)   // ld.red.spcompress
	ev = append(ev,
		cudatmem.TraceEvent{Kind: cudatmem.EventCp, ActiveThreads: 1, CTAGroup: 1, Offset: e, Columns: 32, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventMMA, ActiveThreads: 1, CTAGroup: 1, Offset: d, Columns: uint32(k.N), Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventCommit, ActiveThreads: 1, CTAGroup: 1, Offset: d, Columns: uint32(k.N), Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventWait, ActiveThreads: 32, CTAGroup: 1, Offset: d, Columns: uint32(k.N), Rows: 128},
	)
	perWarp(cudatmem.EventLd, d, k.N)
	for _, a := range []struct{ off, cols uint32 }{{gu, uint32(2 * k.I)}, {d, uint32(k.N)}, {e, 32}} {
		ev = append(ev, cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Offset: a.off, Columns: a.cols, Rows: 128})
	}
	return ev
}

// CUDA returns the kernel source without Prelude. The kernel signature is
//
//	NAME(const float* gate, const float* up, const float* w_down,
//	     float* out_h_comp, uint32_t* out_meta, float* out_row_absmax, float* out)
//
// with gate, up [M, I], w_down [N, I], out_h_comp [M, I/2],
// out_meta [M, I/32], out_row_absmax [M] and out [M, N].
func (k TCGen05DynSparseSwiGLU) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idesc, err := k.InstDescriptor()
	if err != nil {
		return "", err
	}
	gu, d, e := k.tmemLayout()
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	fmt.Fprintf(&b, `
// %[1]s: %[2]s fused SwiGLU + runtime 2:4 prune + tcgen05.mma.sp down_proj (M=%[3]d, I=%[4]d, N=%[5]d).
__global__ void __launch_bounds__(128) %[1]s(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ w_down,
    float* __restrict__ out_h_comp,
    uint32_t* __restrict__ out_meta,
    float* __restrict__ out_row_absmax,
    float* __restrict__ out) {
    constexpr int M = %[3]d, I = %[4]d, N = %[5]d;
    constexpr int I_HALF = I / 2, I_META = I / 32, I_GROUPS = I / 4;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = 32 * warp + lane;

    EG_TCGEN05_SHARED(float, smem_a_comp, [M * I_HALF]);
    EG_TCGEN05_SHARED(uint32_t, smem_a_meta, [M * I_META]);
    EG_TCGEN05_SHARED(float, smem_w, [N * I]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot, [3]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate gate|up (%[6]d cols), accumulator (N cols), metadata (32 cols).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot[0], %[7]du, %[6]du);
        eg_tcgen05_alloc_cta1(tmem, &slot[1], %[8]du, (uint32_t)N);
        eg_tcgen05_alloc_cta1(tmem, &slot[2], %[9]du, 32u);
    }
    __syncthreads();
    const uint32_t gu_taddr = slot[0], d_taddr = slot[1], e_taddr = slot[2];

    // 2. Stage the gate_proj and up_proj results into TMEM.
    for (int c = 0; c < I; ++c) {
        eg_tcgen05_st_row(tmem, warp, lane, gu_taddr + (uint32_t)c, gate[row * I + c]);
        eg_tcgen05_st_row(tmem, warp, lane, gu_taddr + (uint32_t)(I + c), up[row * I + c]);
    }
    eg_tcgen05_wait_st();
    __syncwarp();

    // 3. h = SiLU(gate) * up in registers, written back over the gate columns.
    for (int c = 0; c < I; ++c) {
        const float g = eg_tcgen05_ld_row(tmem, warp, lane, gu_taddr + (uint32_t)c);
        const float u = eg_tcgen05_ld_row(tmem, warp, lane, gu_taddr + (uint32_t)(I + c));
        eg_tcgen05_wait_ld();
        eg_tcgen05_st_row(tmem, warp, lane, gu_taddr + (uint32_t)c, g / (1.0f + expf(-g)) * u);
    }
    eg_tcgen05_wait_st();
    __syncwarp();

    // 4. Runtime 2:4 prune of h: top-2 |h| per 4-column group, ties to the lower index.
    for (int w = 0; w < I_META; ++w) smem_a_meta[row * I_META + w] = 0u;
    float absmax = 0.0f;
    for (int g = 0; g < I_GROUPS; ++g) {
        float c0, c1;
        uint32_t nib;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070 && !defined(EG_EMU)
        {
            uint32_t mdata = 0u;
            float redval = 0.0f;
            const uint32_t taddr = ((uint32_t)(32 * warp) << 16) | ((gu_taddr + (uint32_t)(4 * g)) & 0xffffu);
            asm volatile(
                "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2 {%%0}, {%%1, %%2}, %%3, [%%4];\n"
                "tcgen05.wait::ld.sync.aligned;"
                : "=r"(mdata), "=f"(c0), "=f"(c1), "=f"(redval)
                : "r"(taddr)
                : "memory");
            nib = mdata & 0xfu;
            absmax = fmaxf(absmax, redval);
        }
#else
        {
            float v[4];
            for (int i = 0; i < 4; ++i) {
                v[i] = eg_tcgen05_ld_row(tmem, warp, lane, gu_taddr + (uint32_t)(4 * g + i));
                absmax = fmaxf(absmax, fabsf(v[i]));
            }
            int order[4] = {0, 1, 2, 3};
            for (int a = 0; a < 4; ++a) {
                for (int bb = a + 1; bb < 4; ++bb) {
                    const float fa = fabsf(v[order[a]]), fb = fabsf(v[order[bb]]);
                    if (fb > fa || (fb == fa && order[bb] < order[a])) {
                        const int t = order[a];
                        order[a] = order[bb];
                        order[bb] = t;
                    }
                }
            }
            int i0 = order[0], i1 = order[1];
            if (i0 > i1) {
                const int t = i0;
                i0 = i1;
                i1 = t;
            }
            c0 = v[i0];
            c1 = v[i1];
            nib = (uint32_t)i0 | ((uint32_t)i1 << 2);
        }
#endif
        smem_a_comp[row * I_HALF + 2 * g + 0] = c0;
        smem_a_comp[row * I_HALF + 2 * g + 1] = c1;
        smem_a_meta[row * I_META + (g >> 3)] |= nib << (4 * (g & 7));
        out_h_comp[row * I_HALF + 2 * g + 0] = c0;
        out_h_comp[row * I_HALF + 2 * g + 1] = c1;
    }
    eg_tcgen05_wait_ld();
    for (int w = 0; w < I_META; ++w) out_meta[row * I_META + w] = smem_a_meta[row * I_META + w];
    out_row_absmax[row] = absmax;

    // 5. Stage the dense down_proj weight [N, I] (operand B, K-major).
    for (int i = tid; i < N * I; i += 128) smem_w[i] = w_down[i];
    __syncthreads();

    // 6. One elected thread stages the 2:4 metadata and issues tcgen05.mma.sp.
    if (warp == 0 && eg_elect_one_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
        const uint64_t desc_meta = eg_wgmma_desc(smem_a_meta, 16, 128);
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%%0], %%1;"
                     :: "r"(e_taddr), "l"(desc_meta) : "memory");
        const uint64_t desc_a = eg_wgmma_desc(smem_a_comp, (uint32_t)(I_HALF * 4), (uint32_t)(8 * I_HALF * 4));
        const uint64_t desc_b = eg_wgmma_desc(smem_w, (uint32_t)(I * 4), (uint32_t)(8 * I * 4));
        const uint32_t idesc = 0x%08[10]xu;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.sp.cta_group::1.kind::tf32 [%%0], %%1, %%2, [%%3], %%4, p;\n"
            "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(e_taddr), "r"(idesc) : "memory");
#else
        (void)e_taddr;
        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int g = 0; g < I_GROUPS; ++g) {
                    const uint32_t nib = (smem_a_meta[r * I_META + (g >> 3)] >> (4 * (g & 7))) & 0xfu;
                    const int k0 = 4 * g + (int)(nib & 0x3u), k1 = 4 * g + (int)(nib >> 2);
                    acc += smem_a_comp[r * I_HALF + 2 * g] * smem_w[n * I + k0] +
                           smem_a_comp[r * I_HALF + 2 * g + 1] * smem_w[n * I + k1];
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
#endif
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 7. 4-warp 32-row partitioned epilogue.
    for (int n = 0; n < N; ++n) {
        out[row * N + n] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)n);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(gu_taddr, %[6]du);
        eg_tcgen05_dealloc_cta1(d_taddr, (uint32_t)N);
        eg_tcgen05_dealloc_cta1(e_taddr, 32u);
    }
}
`, k.Name, k.Target, k.M, k.I, k.N, 2*k.I, gu, d, e, idesc)
	return b.String(), nil
}

// CUDAFile returns the self-contained translation unit (Prelude plus CUDA).
func (k TCGen05DynSparseSwiGLU) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// DynSparseSwiGLUResult holds the outputs of [TCGen05DynSparseSwiGLU.Reference].
type DynSparseSwiGLUResult struct {
	H         DynSparseTile // pruned h [M, I]
	RowAbsMax []float32     // [M] max |h| per row (pre-pruning)
	Out       []float32     // [M, N] = prune(h) * w_down^T
}

// Reference computes the kernel's outputs on the host.
func (k TCGen05DynSparseSwiGLU) Reference(gate, up, wDown []float32) (DynSparseSwiGLUResult, error) {
	k, err := k.norm()
	if err != nil {
		return DynSparseSwiGLUResult{}, err
	}
	if len(gate) != k.M*k.I || len(up) != k.M*k.I || len(wDown) != k.N*k.I {
		return DynSparseSwiGLUResult{}, fmt.Errorf("tcgen05 dynsparse swiglu %s reference: lengths gate=%d up=%d w_down=%d, want %d, %d, %d",
			k.Name, len(gate), len(up), len(wDown), k.M*k.I, k.M*k.I, k.N*k.I)
	}
	h := SwiGLU(gate, up)
	tile, err := DynamicPrune2To4(DynSparsePostSwiGLU, h, k.M, k.I, false)
	if err != nil {
		return DynSparseSwiGLUResult{}, err
	}
	absmax := make([]float32, k.M)
	for r := range k.M {
		for _, v := range h[r*k.I : (r+1)*k.I] {
			absmax[r] = max(absmax[r], float32(math.Abs(float64(v))))
		}
	}
	out, err := SparseMMAActivationDenseWeight(tile, wDown, k.N)
	if err != nil {
		return DynSparseSwiGLUResult{}, err
	}
	return DynSparseSwiGLUResult{H: tile, RowAbsMax: absmax, Out: out}, nil
}
