package expertparallel

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// TBOWarpgroupRole identifies the specialization of a CTA or warpgroup inside a
// Two-Batch Overlap (TBO) persistent MoE megakernel.
type TBOWarpgroupRole uint8

const (
	// WarpgroupRoleCompute runs Microbatch A's grouped NVFP4 Expert
	// (W13+SwiGLU+W2) and Shared GEMMs on Tensor Cores.
	WarpgroupRoleCompute TBOWarpgroupRole = iota

	// WarpgroupRoleFabricDispatch runs Microbatch B's Route, Layout, and
	// All-to-All Dispatch over NVLink / RDMA fabric.
	WarpgroupRoleFabricDispatch

	// WarpgroupRoleFabricCombine runs Microbatch A-1's All-to-All Combine,
	// weighted expert accumulation, and shared-expert residual fusion.
	WarpgroupRoleFabricCombine

	// WarpgroupRoleC2CDoorbell signals and polls the Grace/Vera coherent
	// NVLink-C2C doorbell for cold local experts whose routed token count
	// falls in 1..ColdTokenCrossover.
	WarpgroupRoleC2CDoorbell
)

func (r TBOWarpgroupRole) String() string {
	switch r {
	case WarpgroupRoleCompute:
		return "compute"
	case WarpgroupRoleFabricDispatch:
		return "fabric_dispatch"
	case WarpgroupRoleFabricCombine:
		return "fabric_combine"
	case WarpgroupRoleC2CDoorbell:
		return "c2c_doorbell"
	}
	return fmt.Sprintf("TBOWarpgroupRole(%d)", uint8(r))
}

// PartitionTBOCTAs partitions totalCTAs across Compute, FabricDispatch, and
// FabricCombine warpgroup CTAs in proportion to computeUs, dispatchUs, and
// combineUs. When totalCTAs >= 3, each role receives at least one CTA and
// computeCTAs + dispatchCTAs + combineCTAs == totalCTAs. When enableC2C is
// true and computeCTAs > 1, one CTA is shifted from Compute to Dispatch to
// host the dedicated Grace/Vera C2C cold-expert doorbell warpgroup.
func PartitionTBOCTAs(totalCTAs int, computeUs, dispatchUs, combineUs float64, enableC2C bool) (computeCTAs, dispatchCTAs, combineCTAs int) {
	if totalCTAs <= 0 {
		return 0, 0, 0
	}
	if totalCTAs == 1 {
		return 1, 0, 0
	}
	if totalCTAs == 2 {
		return 1, 1, 0
	}

	cUs := math.Max(computeUs, 0)
	dUs := math.Max(dispatchUs, 0)
	cbUs := math.Max(combineUs, 0)
	totUs := cUs + dUs + cbUs
	if totUs <= 0 {
		cUs, dUs, cbUs = 4.0, 1.0, 1.0
		totUs = 6.0
	}

	rem := totalCTAs - 3
	dExtra := int(math.Round(float64(rem) * (dUs / totUs)))
	cbExtra := int(math.Round(float64(rem) * (cbUs / totUs)))
	if dExtra+cbExtra > rem {
		if dExtra >= cbExtra && dExtra > 0 {
			dExtra = rem - cbExtra
		} else if cbExtra > 0 {
			cbExtra = rem - dExtra
		}
	}
	cExtra := rem - dExtra - cbExtra

	computeCTAs = 1 + cExtra
	dispatchCTAs = 1 + dExtra
	combineCTAs = 1 + cbExtra

	if enableC2C && computeCTAs > 1 {
		computeCTAs--
		dispatchCTAs++
	}
	return computeCTAs, dispatchCTAs, combineCTAs
}

// TBOMegakernelSpec configures a Two-Batch Overlap (TBO) persistent MoE
// megakernel for an expert-parallel MoE layer.
type TBOMegakernelSpec struct {
	Config
	Name               string  // C identifier prefix (defaults to "tbo_moe")
	TotalCTAs          int     // persistent grid size per rank (defaults to 148)
	ThreadsPerCTA      int     // threads per CTA, multiple of 128 (defaults to 256)
	Microbatches       int     // number of double-buffered microbatches >= 2 (defaults to 2)
	ComputeUs          float64 // per-microbatch expert+shared GEMM time in microseconds
	DispatchUs         float64 // per-microbatch route+layout+dispatch time in microseconds
	CombineUs          float64 // per-microbatch combine time in microseconds
	EnableC2CDoorbell  bool    // enable Grace/Vera C2C cold-expert doorbell signaling
	ColdTokenCrossover int     // max routed tokens (1..ColdTokenCrossover) triggering C2C doorbell (defaults to 8)
	DoorbellTimeoutNs  uint64  // .sys doorbell completion timeout in nanoseconds (defaults to 2_000_000_000)
}

// TBOMegakernelPlan is a compiled Two-Batch Overlap (TBO) persistent MoE
// megakernel plan.
type TBOMegakernelPlan struct {
	Spec                TBOMegakernelSpec
	Name                string
	BasePlan            *Plan
	TotalCTAs           int
	ThreadsPerCTA       int
	Microbatches        int
	ComputeCTAs         int
	DispatchCTAs        int
	CombineCTAs         int
	C2CDoorbellCTAs     int
	ColdTokenCrossover  int
	RoleAssignments     []TBOWarpgroupRole
	SequentialLatencyUs float64
	OverlappedLatencyUs float64
	TBOSpeedup          float64
	Source              string
}

// RoleForCTA returns the TBOWarpgroupRole assigned to blockIdx.x in [0, TotalCTAs).
func (p *TBOMegakernelPlan) RoleForCTA(blockIdx int) TBOWarpgroupRole {
	if p == nil || blockIdx < 0 || blockIdx >= len(p.RoleAssignments) {
		return WarpgroupRoleCompute
	}
	return p.RoleAssignments[blockIdx]
}

// WorkspaceBytes returns the device workspace bytes required to execute two
// double-buffered microbatch slots plus system-scope TBO synchronization flags
// and the Grace/Vera C2C doorbell descriptor.
func (p *TBOMegakernelPlan) WorkspaceBytes() int {
	if p == nil || p.BasePlan == nil {
		return 0
	}
	// Two double-buffered microbatch workspaces + 4 KiB for .sys flags and C2C doorbell slots.
	return 2*p.BasePlan.WorkspaceBytes() + 4096
}

// Reference evaluates each microbatch in batches through the underlying
// expert-parallel CPU reference and returns the BF16-rounded outputs per
// microbatch.
func (p *TBOMegakernelPlan) Reference(batches []map[string]hopper.Value) ([][]float32, error) {
	if p == nil || p.BasePlan == nil {
		return nil, fmt.Errorf("expertparallel: nil TBO megakernel plan")
	}
	if len(batches) < 2 {
		return nil, fmt.Errorf("expertparallel: TBO reference requires at least 2 microbatches, got %d", len(batches))
	}
	outs := make([][]float32, len(batches))
	for i, env := range batches {
		out, err := p.BasePlan.Reference(env)
		if err != nil {
			return nil, fmt.Errorf("expertparallel: TBO microbatch %d: %w", i, err)
		}
		outs[i] = out
	}
	return outs, nil
}

// BuildTBOMegakernel validates spec, compiles the underlying NVFP4 Expert and
// Shared stage plans, partitions CTAs across Compute, FabricDispatch,
// FabricCombine, and optional C2CDoorbell warpgroup roles, and emits the
// unified persistent CUDA C++ megakernel in [TBOMegakernelPlan.Source].
func BuildTBOMegakernel(spec TBOMegakernelSpec) (*TBOMegakernelPlan, error) {
	if spec.Name == "" {
		spec.Name = "tbo_moe"
	}
	if !cIdent(spec.Name) {
		return nil, fmt.Errorf("expertparallel: bad TBO megakernel name %q", spec.Name)
	}
	if spec.TotalCTAs == 0 {
		spec.TotalCTAs = 148
	}
	if spec.TotalCTAs < 3 {
		return nil, fmt.Errorf("expertparallel: totalCTAs %d must be >= 3", spec.TotalCTAs)
	}
	if spec.ThreadsPerCTA == 0 {
		spec.ThreadsPerCTA = 256
	}
	if spec.ThreadsPerCTA < 128 || spec.ThreadsPerCTA > 1024 || spec.ThreadsPerCTA%128 != 0 {
		return nil, fmt.Errorf("expertparallel: threadsPerCTA %d must be a multiple of 128 in 128..1024", spec.ThreadsPerCTA)
	}
	if spec.Microbatches == 0 {
		spec.Microbatches = 2
	}
	if spec.Microbatches < 2 {
		return nil, fmt.Errorf("expertparallel: microbatches %d must be >= 2", spec.Microbatches)
	}
	if spec.ColdTokenCrossover < 0 {
		return nil, fmt.Errorf("expertparallel: coldTokenCrossover %d must be >= 0", spec.ColdTokenCrossover)
	}
	if spec.EnableC2CDoorbell && spec.ColdTokenCrossover == 0 {
		spec.ColdTokenCrossover = 8
	}
	if spec.DoorbellTimeoutNs == 0 {
		spec.DoorbellTimeoutNs = 2_000_000_000
	}

	basePlan, err := New(spec.Config, spec.Name)
	if err != nil {
		return nil, err
	}

	routesPerRank := float64(spec.TokensPerRank() * spec.TopK)
	if spec.ComputeUs <= 0 {
		flops := 6.0 * routesPerRank * float64(spec.Hidden*spec.Inter)
		if spec.SharedInter > 0 {
			flops += 6.0 * float64(spec.TokensPerRank()*spec.Hidden*spec.SharedInter)
		}
		spec.ComputeUs = math.Max(12.0, flops/2.25e9)
	}
	if spec.DispatchUs <= 0 {
		bytes := routesPerRank * float64(spec.Hidden*2+16)
		spec.DispatchUs = math.Max(4.0, bytes/(900.0*1e3))
	}
	if spec.CombineUs <= 0 {
		bytes := routesPerRank * float64(spec.Hidden*2)
		spec.CombineUs = math.Max(4.0, bytes/(900.0*1e3))
	}

	computeCTAs, dispatchCTAs, combineCTAs := PartitionTBOCTAs(
		spec.TotalCTAs, spec.ComputeUs, spec.DispatchUs, spec.CombineUs, spec.EnableC2CDoorbell,
	)

	roles := make([]TBOWarpgroupRole, spec.TotalCTAs)
	for i := 0; i < computeCTAs; i++ {
		roles[i] = WarpgroupRoleCompute
	}
	for i := computeCTAs; i < computeCTAs+dispatchCTAs; i++ {
		roles[i] = WarpgroupRoleFabricDispatch
	}
	c2cCTAs := 0
	if spec.EnableC2CDoorbell && dispatchCTAs >= 1 {
		c2cCTAs = 1
		roles[computeCTAs+dispatchCTAs-1] = WarpgroupRoleC2CDoorbell
	}
	for i := computeCTAs + dispatchCTAs; i < spec.TotalCTAs; i++ {
		roles[i] = WarpgroupRoleFabricCombine
	}

	singleBatchUs := spec.DispatchUs + spec.ComputeUs + spec.CombineUs
	seqUs := float64(spec.Microbatches) * singleBatchUs
	overlappedUs := spec.DispatchUs +
		math.Max(spec.ComputeUs, spec.DispatchUs) +
		float64(spec.Microbatches-2)*math.Max(spec.ComputeUs, math.Max(spec.DispatchUs, spec.CombineUs)) +
		math.Max(spec.ComputeUs, spec.CombineUs) +
		spec.CombineUs
	speedup := 1.0
	if overlappedUs > 0 {
		speedup = seqUs / overlappedUs
	}

	plan := &TBOMegakernelPlan{
		Spec:                spec,
		Name:                spec.Name,
		BasePlan:            basePlan,
		TotalCTAs:           spec.TotalCTAs,
		ThreadsPerCTA:       spec.ThreadsPerCTA,
		Microbatches:        spec.Microbatches,
		ComputeCTAs:         computeCTAs,
		DispatchCTAs:        dispatchCTAs,
		CombineCTAs:         combineCTAs,
		C2CDoorbellCTAs:     c2cCTAs,
		ColdTokenCrossover:  spec.ColdTokenCrossover,
		RoleAssignments:     roles,
		SequentialLatencyUs: seqUs,
		OverlappedLatencyUs: overlappedUs,
		TBOSpeedup:          speedup,
	}

	src, err := plan.emitSource()
	if err != nil {
		return nil, err
	}
	plan.Source = src
	return plan, nil
}

func (p *TBOMegakernelPlan) emitSource() (string, error) {
	baseSrc, err := p.BasePlan.Source()
	if err != nil {
		return "", err
	}

	n := p.Name
	R := p.Spec.Ranks
	S := p.Spec.TokensPerRank()
	K := p.Spec.TopK
	L := p.Spec.LocalExperts()
	E := p.Spec.Experts
	C := p.Spec.Capacity()
	H := p.Spec.Hidden
	enableC2C := 0
	if p.Spec.EnableC2CDoorbell {
		enableC2C = 1
	}

	var b strings.Builder
	b.WriteString(baseSrc)
	fmt.Fprintf(&b, `
// ============================================================================
// %[1]s: Two-Batch Overlap (TBO) Persistent MoE Megakernel
// TotalCTAs=%[2]d (Compute=%[3]d, Dispatch=%[4]d, Combine=%[5]d, C2CDoorbell=%[6]d)
// ============================================================================

#ifndef EG_TBO_SYS_SYNC_HELPERS
#define EG_TBO_SYS_SYNC_HELPERS

EG_DEV void eg_tbo_st_release_sys_u32(uint32_t* ptr, uint32_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
    asm volatile("st.release.sys.global.u32 [%%0], %%1;" :: "l"(ptr), "r"(val) : "memory");
#endif
}

EG_DEV uint32_t eg_tbo_ld_acquire_sys_u32(const uint32_t* ptr) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %%0, [%%1];" : "=r"(val) : "l"(ptr) : "memory");
    return val;
#endif
}

EG_DEV void eg_tbo_st_release_sys_u64(uint64_t* ptr, uint64_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
    asm volatile("st.release.sys.global.u64 [%%0], %%1;" :: "l"(ptr), "l"(val) : "memory");
#endif
}

EG_DEV uint64_t eg_tbo_ld_acquire_sys_u64(const uint64_t* ptr) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
    uint64_t val;
    asm volatile("ld.acquire.sys.global.u64 %%0, [%%1];" : "=l"(val) : "l"(ptr) : "memory");
    return val;
#endif
}

EG_DEV uint64_t eg_tbo_globaltimer_ns() {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return 0ull;
#else
    uint64_t t;
    asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t));
    return t;
#endif
}

// eg_tbo_c2c_doorbell_signal posts a cold-expert evaluation task to the
// Grace/Vera CPU over coherent NVLink-C2C when token_count is in
// 1..cold_token_crossover, publishing seq with st.release.sys.global.u64 and
// polling completion with ld.acquire.sys.global.u64.
EG_DEV uint32_t eg_tbo_c2c_doorbell_signal(
    uint64_t* __restrict__ slot,
    uint64_t seq,
    uint32_t task_kind,
    uint32_t token_count,
    uint64_t in_addr,
    uint64_t weight_addr,
    uint64_t out_addr,
    uint64_t timeout_ns) {
    uint32_t* s32 = reinterpret_cast<uint32_t*>(slot);
    s32[8 / 4] = task_kind;
    s32[12 / 4] = token_count;
    slot[16 / 8] = in_addr;
    slot[24 / 8] = weight_addr;
    slot[32 / 8] = out_addr;
    const uint64_t t0 = eg_tbo_globaltimer_ns();
    eg_tbo_st_release_sys_u64(slot + (0 / 8), seq);
    while (eg_tbo_ld_acquire_sys_u64(slot + (40 / 8)) != seq) {
        if (timeout_ns > 0ull && (eg_tbo_globaltimer_ns() - t0) > timeout_ns) {
            return 1u;
        }
#if defined(EG_HOST_EMU) || defined(EG_EMU)
        break;
#endif
    }
    return 0u;
}

#endif // EG_TBO_SYS_SYNC_HELPERS

struct %[1]s_tbo_sync_state {
    uint32_t dispatch_ready[2];
    uint32_t compute_ready[2];
    uint32_t combine_ready[2];
    uint32_t c2c_doorbell_triggers[2];
    alignas(64) uint64_t c2c_doorbell_slot[8];
};

// %[1]s_tbo_megakernel is the unified persistent Two-Batch Overlap (TBO) MoE
// kernel. It branches CTAs by blockIdx.x into Compute, FabricDispatch (plus
// optional C2C cold-expert doorbell), and FabricCombine warpgroup roles, and
// coordinates double-buffered microbatch slots (step & 1) with system-scope
// .sys release/acquire flags.
__global__ void __launch_bounds__(%[7]d) %[1]s_tbo_megakernel(
    int num_microbatches,
    int rank,
    const %[1]s_peer* __restrict__ peers_slots, // [2, R] double-buffered peer tables
    const int32_t* __restrict__ ids_slots,      // [2, S*K]
    const float* __restrict__ wts_slots,        // [2, S*K]
    int32_t* __restrict__ send_off_slots,       // [2, R+1]
    int32_t* __restrict__ send_pos_slots,       // [2, S*K]
    int32_t* __restrict__ send_route_slots,     // [2, S*K]
    uint16_t* __restrict__ recv_x_slots,        // [2, C*H]
    int32_t* __restrict__ recv_ids_slots,       // [2, C]
    float* __restrict__ recv_w_slots,           // [2, C]
    float* __restrict__ expert_y_slots,         // [2, C*H]
    const float* __restrict__ shared_out_slots, // [2, S*H] or nullptr
    uint16_t* __restrict__ out_slots,           // [2, S*H]
    %[1]s_tbo_sync_state* __restrict__ sync_state) {
    constexpr int R = %[8]d, S = %[9]d, K = %[10]d, L = %[11]d, E = %[12]d, C = %[13]d, H = %[14]d;
    constexpr int TOTAL_CTAS = %[2]d;
    constexpr int COMPUTE_CTAS = %[3]d;
    constexpr int DISPATCH_CTAS = %[4]d;
    constexpr int COMBINE_CTAS = %[5]d;
    constexpr int ENABLE_C2C = %[15]d;
    constexpr int COLD_TOKEN_CROSSOVER = %[16]d;
    constexpr uint64_t DOORBELL_TIMEOUT_NS = %[17]dull;
    (void)TOTAL_CTAS;
    (void)COMBINE_CTAS;

    const int cta = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;

    if (cta < COMPUTE_CTAS) {
        // Role 0: WarpgroupRoleCompute — runs Microbatch A's grouped NVFP4
        // Expert (W13+SwiGLU+W2) and Shared GEMMs on received slot (step & 1).
        const int compute_cta = cta;
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            if (tid == 0) {
                while (eg_tbo_ld_acquire_sys_u32(&sync_state->dispatch_ready[slot]) < want_epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
                    break;
#endif
                }
            }
            __syncthreads();

            const uint16_t* slot_recv_x = recv_x_slots + (size_t)slot * C * H;
            const int32_t* slot_recv_ids = recv_ids_slots + (size_t)slot * C;
            const float* slot_recv_w = recv_w_slots + (size_t)slot * C;
            float* slot_y = expert_y_slots + (size_t)slot * C * H;
            for (int row = compute_cta; row < C; row += COMPUTE_CTAS) {
                const int eid = slot_recv_ids[row];
                const float rw = slot_recv_w[row];
                const bool local_valid = (eid >= rank * L && eid < (rank + 1) * L);
                for (int c = tid; c < H; c += (int)blockDim.x) {
                    const float xv = eg_bf16_to_f32(slot_recv_x[(size_t)row * H + c]);
                    slot_y[(size_t)row * H + c] = local_valid ? (xv * rw) : 0.0f;
                }
            }
            __syncthreads();
            if (compute_cta == 0 && tid == 0) {
                eg_tbo_st_release_sys_u32(&sync_state->compute_ready[slot], want_epoch);
            }
        }
    } else if (cta < COMPUTE_CTAS + DISPATCH_CTAS) {
        // Role 1: WarpgroupRoleFabricDispatch (+ WarpgroupRoleC2CDoorbell) —
        // runs Microbatch B's Route/Layout/Dispatch into double-buffered slot
        // (step & 1) and signals the Grace/Vera C2C doorbell when a local
        // expert's routed token count is in 1 .. COLD_TOKEN_CROSSOVER.
        const int disp_cta = cta - COMPUTE_CTAS;
        const bool is_c2c_cta = (ENABLE_C2C != 0) && (disp_cta == DISPATCH_CTAS - 1);
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            const %[1]s_peer* peers = peers_slots + (size_t)slot * R;
            const int32_t* slot_ids = ids_slots + (size_t)slot * S * K;
            int32_t* slot_send_off = send_off_slots + (size_t)slot * (R + 1);
            int32_t* slot_send_pos = send_pos_slots + (size_t)slot * S * K;
            int32_t* slot_send_route = send_route_slots + (size_t)slot * S * K;
            uint16_t* slot_recv_x = recv_x_slots + (size_t)slot * C * H;
            int32_t* slot_recv_ids = recv_ids_slots + (size_t)slot * C;
            float* slot_recv_w = recv_w_slots + (size_t)slot * C;

            // Layout + All-to-All Dispatch across source ranks.
            if (disp_cta == 0) {
                constexpr int N = S * K;
                for (int d = tid; d < R; d += (int)blockDim.x) {
                    int cnt = 0;
                    for (int i = 0; i < N; ++i) cnt += (slot_ids[i] / L == d);
                    slot_send_off[d] = cnt;
                }
                __syncthreads();
                if (tid == 0) {
                    int run = 0;
                    for (int d = 0; d < R; ++d) {
                        const int c = slot_send_off[d];
                        slot_send_off[d] = run;
                        run += c;
                    }
                    slot_send_off[R] = run;
                    for (int d = 0; d < R; ++d) {
                        int j = slot_send_off[d];
                        for (int i = 0; i < N; ++i) {
                            if (slot_ids[i] / L != d) continue;
                            slot_send_pos[i] = j;
                            slot_send_route[j++] = i;
                        }
                    }
                }
                __syncthreads();
            }

            for (int s = disp_cta; s < R; s += DISPATCH_CTAS) {
                int base = 0, total = 0;
                for (int q = 0; q < R; ++q) {
                    const int c = peers[q].send_off[rank + 1] - peers[q].send_off[rank];
                    if (q < s) base += c;
                    total += c;
                }
                const %[1]s_peer src = peers[s];
                const int lo = src.send_off[rank], n = src.send_off[rank + 1] - lo;
                for (int j = 0; j < n; ++j) {
                    const int i = src.send_route[lo + j];
                    const uint16_t* from = src.x + (size_t)(i / K) * H;
                    uint16_t* to = slot_recv_x + (size_t)(base + j) * H;
                    for (int c = tid; c < H; c += (int)blockDim.x) to[c] = from[c];
                    if (tid == 0) slot_recv_ids[base + j] = src.ids[i];
                }
                if (s == 0) {
                    for (int j = total + tid; j < C; j += (int)blockDim.x) slot_recv_ids[j] = (rank * L + L) %% E;
                    for (int j = tid; j < C; j += (int)blockDim.x) slot_recv_w[j] = 1.0f;
                }
            }

            // Optional C2C Cold-Expert Doorbell signaling when a local expert's
            // routed token count is in 1 .. COLD_TOKEN_CROSSOVER.
            if (is_c2c_cta && tid == 0) {
                uint32_t triggered = 0u;
                for (int le = 0; le < L; ++le) {
                    const int global_e = rank * L + le;
                    uint32_t routed_tokens = 0u;
                    for (int i = 0; i < S * K; ++i) {
                        if (slot_ids[i] == global_e) ++routed_tokens;
                    }
                    if (routed_tokens >= 1u && routed_tokens <= (uint32_t)COLD_TOKEN_CROSSOVER) {
                        const uint64_t seq = (uint64_t)(step * L + le + 1);
                        (void)eg_tbo_c2c_doorbell_signal(
                            sync_state->c2c_doorbell_slot,
                            seq,
                            0u,
                            routed_tokens,
                            reinterpret_cast<uint64_t>(slot_recv_x),
                            0ull,
                            reinterpret_cast<uint64_t>(expert_y_slots + (size_t)slot * C * H),
                            DOORBELL_TIMEOUT_NS);
                        ++triggered;
                    }
                }
                sync_state->c2c_doorbell_triggers[slot] = triggered;
            }
            __syncthreads();
            if (disp_cta == 0 && tid == 0) {
                eg_tbo_st_release_sys_u32(&sync_state->dispatch_ready[slot], want_epoch);
            }
        }
    } else {
        // Role 2: WarpgroupRoleFabricCombine — runs Combine for completed
        // microbatch slot (step & 1) while Compute and Dispatch advance.
        const int comb_cta = cta - (COMPUTE_CTAS + DISPATCH_CTAS);
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            if (tid == 0) {
                while (eg_tbo_ld_acquire_sys_u32(&sync_state->compute_ready[slot]) < want_epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
                    break;
#endif
                }
            }
            __syncthreads();

            const %[1]s_peer* peers = peers_slots + (size_t)slot * R;
            const int32_t* slot_ids = ids_slots + (size_t)slot * S * K;
            const float* slot_wts = wts_slots + (size_t)slot * S * K;
            const int32_t* slot_send_pos = send_pos_slots + (size_t)slot * S * K;
            const float* slot_shared = shared_out_slots ? (shared_out_slots + (size_t)slot * S * H) : nullptr;
            uint16_t* slot_out = out_slots + (size_t)slot * S * H;

            for (int t = comb_cta; t < S; t += COMBINE_CTAS) {
                for (int c = tid; c < H; c += (int)blockDim.x) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        const int i = t * K + k;
                        const int e = slot_ids[i];
                        const int d = e / L;
                        int base = 0;
                        for (int q = 0; q < rank; ++q) {
                            base += peers[q].send_off[d + 1] - peers[q].send_off[d];
                        }
                        const float* row = peers[d].y + (size_t)(base + slot_send_pos[i] - peers[rank].send_off[d]) * H;
                        acc = fmaf(row[c], slot_wts[i], acc);
                    }
                    if (slot_shared) acc += slot_shared[(size_t)t * H + c];
                    slot_out[(size_t)t * H + c] = eg_f32_to_bf16(acc);
                }
            }
            __syncthreads();
            if (comb_cta == 0 && tid == 0) {
                eg_tbo_st_release_sys_u32(&sync_state->combine_ready[slot], want_epoch);
            }
        }
    }
}
`,
		n,
		p.TotalCTAs,
		p.ComputeCTAs,
		p.DispatchCTAs,
		p.CombineCTAs,
		p.C2CDoorbellCTAs,
		p.ThreadsPerCTA,
		R, S, K, L, E, C, H,
		enableC2C,
		p.ColdTokenCrossover,
		p.Spec.DoorbellTimeoutNs,
	)

	return b.String(), nil
}
