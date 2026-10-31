package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// MoEBackwardCombineConfig configures the MoE backward combine: each token's
// input gradient dX_t = dX_shared + sum_k g_{t,k} dX_{t,e_k} and the gate
// gradients, with the routed sum optionally reduced in the NVLink fabric.
type MoEBackwardCombineConfig struct {
	Target       string // "sm_100a" or "sm_107a" for the emitted epilogue (default sm_100a)
	NumRanks     int
	NumExperts   int
	TopK         int
	HiddenDim    int
	HasSharedExp bool
	UseSwitchRed bool // expert ranks pre-scale by g and reduce onto the home rank's buffer in flight
	ElemBytes    int  // bytes per gradient element for traffic accounting (default 2, BF16)
}

func (c MoEBackwardCombineConfig) norm() (MoEBackwardCombineConfig, error) {
	if c.Target == "" {
		c.Target = "sm_100a"
	}
	if c.ElemBytes == 0 {
		c.ElemBytes = 2
	}
	switch {
	case c.Target != "sm_100a" && c.Target != "sm_107a":
		return c, fmt.Errorf("moe backward: target %q unsupported (want sm_100a or sm_107a)", c.Target)
	case c.NumRanks < 1 || c.NumExperts < 1:
		return c, fmt.Errorf("moe backward: NumRanks %d and NumExperts %d must be positive", c.NumRanks, c.NumExperts)
	case c.TopK < 1 || c.TopK > c.NumExperts:
		return c, fmt.Errorf("moe backward: TopK %d must be in 1..%d", c.TopK, c.NumExperts)
	case c.HiddenDim < 1:
		return c, fmt.Errorf("moe backward: HiddenDim %d must be positive", c.HiddenDim)
	case c.ElemBytes < 1:
		return c, fmt.Errorf("moe backward: ElemBytes %d must be positive", c.ElemBytes)
	}
	return c, nil
}

// MoEBackwardResult holds the combined gradients and the home rank's
// traffic for one layer.
type MoEBackwardResult struct {
	GradX          [][]float32 // [tokens][HiddenDim]
	GradGate       [][]float32 // [tokens][TopK] dL/dg = <dY_t, Y_{t,e_k}>
	GradGateLogits [][]float32 // [tokens][TopK] dL/du through g = sigmoid(u)/sum sigmoid(u)

	HomeRankHBMCombineReads int64 // bytes the home rank reads to combine; 0 with the switch
	HomeRankIngressBytes    int64 // bytes arriving at the home rank over NVLink
	SwitchReducedBytes      int64 // bytes the fabric reduces in flight
}

// ComputeMoEBackwardCombine computes the MoE backward combine and gate
// gradients for routing weights g_{t,k} = sigmoid(u_{t,k}) / sum_j
// sigmoid(u_{t,j}) given the gate logits u.
//
//	dX_t   = dX_shared,t + sum_k g_{t,k} dX_{t,e_k}
//	dg_t,k = <dY_t, Y_{t,e_k}>
//	du_t,k = s_k (1 - s_k) / S * (dg_k - sum_j g_j dg_j),  s = sigmoid(u), S = sum s
//
// expertGradX holds the unscaled dX_{t,e_k} and expertFwdY the forward
// expert outputs, both [tokens][TopK][HiddenDim]. dXShared may be nil when
// cfg.HasSharedExp is false.
//
// The sum is accumulated in a fixed order, shared expert first and then k
// order, in both modes, so pre-scaling at the expert gives exactly the
// home-rank result. The fabric's reduction order is not specified; on
// hardware the two agree only to rounding. Traffic assumes the switch
// combines the TopK contributions so that one copy reaches the home rank.
func ComputeMoEBackwardCombine(dY, dXShared [][]float32, expertGradX, expertFwdY [][][]float32, gateLogits [][]float32, cfg MoEBackwardCombineConfig) (MoEBackwardResult, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return MoEBackwardResult{}, err
	}
	T, K, D := len(dY), cfg.TopK, cfg.HiddenDim
	if T == 0 {
		return MoEBackwardResult{}, fmt.Errorf("moe backward: no tokens")
	}
	if len(expertGradX) != T || len(expertFwdY) != T || len(gateLogits) != T {
		return MoEBackwardResult{}, fmt.Errorf("moe backward: %d tokens of dY but %d/%d/%d of expert grads, outputs, logits",
			T, len(expertGradX), len(expertFwdY), len(gateLogits))
	}
	if cfg.HasSharedExp && len(dXShared) != T {
		return MoEBackwardResult{}, fmt.Errorf("moe backward: %d shared-expert rows, want %d", len(dXShared), T)
	}
	for t := range T {
		if len(dY[t]) != D || len(gateLogits[t]) != K || len(expertGradX[t]) != K || len(expertFwdY[t]) != K {
			return MoEBackwardResult{}, fmt.Errorf("moe backward: token %d has wrong shape", t)
		}
		for k := range K {
			if len(expertGradX[t][k]) != D || len(expertFwdY[t][k]) != D {
				return MoEBackwardResult{}, fmt.Errorf("moe backward: token %d expert %d has wrong width", t, k)
			}
		}
		if cfg.HasSharedExp && len(dXShared[t]) != D {
			return MoEBackwardResult{}, fmt.Errorf("moe backward: token %d shared gradient has width %d", t, len(dXShared[t]))
		}
	}

	res := MoEBackwardResult{
		GradX:          make([][]float32, T),
		GradGate:       make([][]float32, T),
		GradGateLogits: make([][]float32, T),
	}
	for t := range T {
		s, g := gateWeights(gateLogits[t])
		acc := make([]float32, D)
		if cfg.HasSharedExp {
			copy(acc, dXShared[t])
		}
		dg := make([]float32, K)
		for k := range K {
			for d, x := range expertGradX[t][k] {
				acc[d] += float32(g[k] * x)
			}
			var dot float32
			for d, y := range expertFwdY[t][k] {
				dot += float32(dY[t][d] * y)
			}
			dg[k] = dot
		}
		var gdg float32
		for k := range K {
			gdg += float32(g[k] * dg[k])
		}
		var S float32
		for _, x := range s {
			S += x
		}
		du := make([]float32, K)
		for k := range K {
			du[k] = s[k] * (1 - s[k]) / S * (dg[k] - gdg)
		}
		res.GradX[t], res.GradGate[t], res.GradGateLogits[t] = acc, dg, du
	}

	row := int64(T) * int64(D) * int64(cfg.ElemBytes)
	routed := int64(K) * row
	if cfg.UseSwitchRed {
		res.SwitchReducedBytes = routed
		res.HomeRankIngressBytes = row
	} else {
		res.HomeRankIngressBytes = routed
		res.HomeRankHBMCombineReads = routed
		if cfg.HasSharedExp {
			res.HomeRankHBMCombineReads += row
		}
	}
	return res, nil
}

// gateWeights returns s = sigmoid(u) and g = s / sum(s).
func gateWeights(u []float32) (s, g []float32) {
	s = make([]float32, len(u))
	g = make([]float32, len(u))
	var S float32
	for k, x := range u {
		s[k] = 1 / (1 + float32(math.Exp(float64(-x))))
		S += s[k]
	}
	for k := range s {
		g[k] = s[k] / S
	}
	return s, g
}

// EmitSwitchMoEBackwardEpiloguePTX returns the PTX schedule of the expert
// backward epilogue for one 128-row dX tile: load the tile from TMEM, scale
// each row by its token's g in registers, reduce dg = <dY, Y> with a warp
// shuffle tree, stage the tile in shared memory, and reduce it onto the
// home rank's buffer: fabric.try_red.async.tensor.multimem on sm_107a,
// cp.reduce.async.bulk.tensor to the home rank's NVLink address on sm_100a.
//
// PTX ISA 9.4 tcgen05.ld.red reduces with min or max only, so dg is not a
// tcgen05.ld.red.add; it is a shfl.sync.bfly sum. The listing has symbolic
// operands and is not assembled; [SwitchMoEBackwardKernel] is the executable
// form.
func EmitSwitchMoEBackwardEpiloguePTX(cfg MoEBackwardCombineConfig, kernelName string) (string, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return "", err
	}
	if !cIdent(kernelName) {
		return "", fmt.Errorf("moe backward: bad kernel name %q", kernelName)
	}
	red := "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%rd_home_tmap, {%r_col, %r_row}], [%r_smem_dx];\n\tcp.async.bulk.commit_group;"
	if cfg.Target == "sm_107a" {
		inst, err := RubinFabricAsyncTensorKernel{Op: FabricTensorTryRed, Multimem: true, Target: "sm_107a"}.PTXInstruction()
		if err != nil {
			return "", err
		}
		red = "$retry_red:\n\t" + inst + " %r_status, [%rd_home_le_id, %rd_home_off, {%r_col, %r_row}], [%r_smem_dx], [%rd_mbar_le_id, %rd_mbar_off];\n" +
			"\tsetp.ne.u32 %p_retry, %r_status, 0;\n\t@%p_retry bra $retry_red;"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: MoE backward expert epilogue (%s), top-%d, hidden %d, %s.\n", kernelName, cfg.Target, cfg.TopK, cfg.HiddenDim,
		map[bool]string{true: "switch reduction", false: "home-rank combine"}[cfg.UseSwitchRed])
	fmt.Fprintf(&b, ".visible .entry %s(.param .u64 gate, .param .u64 dy, .param .u64 y, .param .u64 dg_out)\n{\n", kernelName)
	b.WriteString("\ttcgen05.wait::ld.sync.aligned;\n")
	b.WriteString("\ttcgen05.ld.sync.aligned.32x32b.x4.b32 {%f_dx0, %f_dx1, %f_dx2, %f_dx3}, [%r_tmem_dx];\n")
	b.WriteString("\tld.global.f32 %f_g, [%rd_gate_row];\n")
	if cfg.UseSwitchRed {
		b.WriteString("\tmul.f32 %f_dx0, %f_dx0, %f_g;\n\tmul.f32 %f_dx1, %f_dx1, %f_g;\n\tmul.f32 %f_dx2, %f_dx2, %f_g;\n\tmul.f32 %f_dx3, %f_dx3, %f_g;\n")
	}
	b.WriteString("\t// dg = <dY_t, Y_t,e>: per-lane partial products, then a butterfly sum\n")
	b.WriteString("\tfma.rn.f32 %f_dot, %f_dy0, %f_y0, %f_dot;\n")
	for _, m := range []int{16, 8, 4, 2, 1} {
		fmt.Fprintf(&b, "\tshfl.sync.bfly.b32 %%f_t, %%f_dot, %d, 31, -1;\n\tadd.f32 %%f_dot, %%f_dot, %%f_t;\n", m)
	}
	b.WriteString("\tst.global.f32 [%rd_dg_out], %f_dot;\n")
	b.WriteString("\tst.shared.v4.f32 [%r_smem_dx], {%f_dx0, %f_dx1, %f_dx2, %f_dx3};\n")
	b.WriteString("\tfence.proxy.async.shared::cta;\n\tbar.sync 0;\n")
	if cfg.UseSwitchRed {
		b.WriteString("\t" + red + "\n")
	} else {
		b.WriteString("\tcp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd_home_staging_tmap, {%r_col, %r_row}], [%r_smem_dx];\n")
	}
	b.WriteString("\tret;\n}\n")
	return b.String(), nil
}

// SwitchMoEBackwardKernel emits a multi-rank CUDA kernel of the switch
// combine: every rank serves the (token, slot) pairs routed to it, computes
// dg = <dY, Y> with a warp reduction and stores it on the token's home rank,
// and atomically adds g * dX onto the home rank's GradX buffer, which holds
// dX_shared before launch. Token t's home rank is t mod NumRanks.
//
// The device path adds with red.global.add.f32 (atomicAdd) through a peer
// pointer; under cudahostemu each rank is a host thread and the peer write
// lands in the target rank's slice of the symmetric buffer. Floating-point
// add order across ranks is not fixed, so results match the reference to
// rounding.
type SwitchMoEBackwardKernel struct {
	Name   string
	Config MoEBackwardCombineConfig
	Tokens int
}

func (k SwitchMoEBackwardKernel) norm() (SwitchMoEBackwardKernel, error) {
	if k.Name == "" {
		k.Name = "nvl72_switch_moe_bwd"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("moe backward: bad kernel name %q", k.Name)
	}
	if k.Tokens < 1 {
		return k, fmt.Errorf("moe backward: Tokens %d must be positive", k.Tokens)
	}
	c, err := k.Config.norm()
	if err != nil {
		return k, err
	}
	k.Config = c
	return k, nil
}

// Grid returns one 128-thread CTA per rank.
func (k SwitchMoEBackwardKernel) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// CUDA returns the kernel without the preludes. Every buffer is symmetric
// with the same per-rank size: dy [T][D], y and dx [T][K][D], gate [T][K]
// (g), route [T][K] (serving rank), grad_x [T][D], dg [T][K].
func (k SwitchMoEBackwardKernel) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	c := k.Config
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank MoE backward switch combine, %d tokens, top-%d, hidden %d.\n",
		k.Name, c.NumRanks, k.Tokens, c.TopK, c.HiddenDim)
	b.WriteString(`// eg_moe_bwd_peer returns rank's slice of the symmetric buffer base for
// remote atomic adds and stores. The kernel never reads a peer slice, so
// under emulation it addresses the slice directly rather than through
// eg_emu_peer_ptr, whose checker treats every peer access as a read.
template <typename T>
EG_DEV T* eg_moe_bwd_peer(T* base, int rank, int me, size_t bytes) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    (void)me;
    return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + (size_t)rank * bytes);
#else
    return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + ((ptrdiff_t)rank - (ptrdiff_t)me) * (ptrdiff_t)bytes);
#endif
}
`)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n", k.Name)
	b.WriteString(`    const float* __restrict__ dy, const float* __restrict__ y, const float* __restrict__ dx,
    const float* __restrict__ gate, const int32_t* __restrict__ route,
    float* __restrict__ grad_x, float* __restrict__ dg) {
`)
	fmt.Fprintf(&b, "    constexpr int P = %d, T = %d, K = %d, D = %d;\n", c.NumRanks, k.Tokens, c.TopK, c.HiddenDim)
	b.WriteString(`    const int me = eg_nvlink_rank();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const float* ldy = eg_nvlink_local_ptr(dy, (size_t)T * D);
    const float* ly = eg_nvlink_local_ptr(y, (size_t)T * K * D);
    const float* ldx = eg_nvlink_local_ptr(dx, (size_t)T * K * D);
    const float* lgate = eg_nvlink_local_ptr(gate, (size_t)T * K);
    const int32_t* lroute = eg_nvlink_local_ptr(route, (size_t)T * K);
    for (int tk = warp; tk < T * K; tk += warps) {
        if (lroute[tk] != me) continue;
        const int t = tk / K, home = t % P;
        const float g = lgate[tk];
        float* hx = eg_moe_bwd_peer(grad_x, home, me, (size_t)T * D * sizeof(float));
        float* hdg = eg_moe_bwd_peer(dg, home, me, (size_t)T * K * sizeof(float));
        float dot = 0.0f;
        for (int d = lane; d < D; d += 32) {
            const float p = ldy[(size_t)t * D + d] * ly[(size_t)tk * D + d];
            dot += p;
            const float v = g * ldx[(size_t)tk * D + d];
            atomicAdd(&hx[(size_t)t * D + d], v);
        }
        dot = eg_warp_sum(dot);
        if (lane == 0) hdg[tk] = dot;
    }
}
`)
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude] and the kernel.
func (k SwitchMoEBackwardKernel) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + src, nil
}
