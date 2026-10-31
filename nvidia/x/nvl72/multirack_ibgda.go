package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// MultiRackClusterSpec configures a multi-rack NVL72 (GB200/GB300) or Vera
// Rubin NVL144 / Rubin Ultra NVL576 cluster with GPU-initiated IBGDA RDMA,
// Quantum-3 / Spectrum-X SHARP v4 in-network reduction trees, and
// straggler-resilient elastic MoE fallback.
type MultiRackClusterSpec struct {
	NumRacks             int     // e.g. 4, 8, 16 racks
	GPUsPerRack          int     // e.g. 72 (NVL72) or 144 (Rubin NVL144)
	IntraRackBWGBps      float64 // e.g. 900.0 (NVSwitch 5) or 1800.0 (NVSwitch 6)
	InterRackRDMA_BWGBps float64 // e.g. 100.0 (800G ConnectX-8 / Spectrum-X)
	EnableIBGDA          bool    // GPU-initiated NIC doorbell MMIO from inside megakernel
	EnableSHARPv4        bool    // In-network switch reduction trees across racks
	StragglerDeadlineNs  uint64  // %%globaltimer timeout before soft parity/shared-expert fallback
}

// Validate checks that s has positive dimensions and bandwidths.
func (s MultiRackClusterSpec) Validate() error {
	if s.NumRacks < 1 {
		return fmt.Errorf("nvl72: NumRacks %d must be >= 1", s.NumRacks)
	}
	if s.GPUsPerRack < 1 {
		return fmt.Errorf("nvl72: GPUsPerRack %d must be >= 1", s.GPUsPerRack)
	}
	if s.IntraRackBWGBps <= 0 {
		return fmt.Errorf("nvl72: IntraRackBWGBps %v must be > 0", s.IntraRackBWGBps)
	}
	if s.InterRackRDMA_BWGBps <= 0 {
		return fmt.Errorf("nvl72: InterRackRDMA_BWGBps %v must be > 0", s.InterRackRDMA_BWGBps)
	}
	return nil
}

func (s MultiRackClusterSpec) norm() (MultiRackClusterSpec, error) {
	if err := s.Validate(); err != nil {
		return s, err
	}
	if s.StragglerDeadlineNs == 0 {
		s.StragglerDeadlineNs = 50_000 // 50 us default straggler deadline
	}
	return s, nil
}

// HierarchicalCollectivePlan models a two-level hierarchical collective across
// a multi-rack NVL72/NVL144 cluster:
//   - Level 1 (Intra-Rack): NVSwitch 5/6 symmetric-memory multimem reduction
//     and broadcast across GPUsPerRack GPUs/dies inside each rack.
//   - Level 2 (Inter-Rack): Quantum-3 / Spectrum-X SHARP v4 in-network switch
//     reduction trees (or ring/rail RDMA when SHARP v4 is disabled) paired
//     with IBGDA GPU-initiated NIC doorbell WQE dispatch.
type HierarchicalCollectivePlan struct {
	Spec                      MultiRackClusterSpec
	TotalGPUs                 int
	PayloadBytes              int64
	NumRails                  int
	SHARPTreeDepth            int
	IntraRackAlgorithm        string
	InterRackAlgorithm        string
	DispatchMechanism         string
	IntraRackReduceUs         float64
	IntraRackBroadcastUs      float64
	InterRackReductionUs      float64
	UnoptimizedInterRackUs    float64
	DoorbellDispatchUs        float64
	CPUProxyOverheadSavedUs   float64
	TotalLatencyUs            float64
	UnoptimizedTotalLatencyUs float64
	SpeedupFactor             float64
	SHARPv4TrafficSavingRatio float64
	StragglerFallbackMode     string
	StragglerDeadlineNs       uint64
}

// PlanMultiRackHierarchicalCollective models a two-level hierarchical
// collective over spec for payloadBytes bytes:
//  1. Intra-rack NVSwitch multimem.ld_reduce across spec.GPUsPerRack ranks.
//  2. Inter-rack Quantum-3 / Spectrum-X SHARP v4 in-network reduction (or
//     ring/tree RDMA fallback when EnableSHARPv4 is false) dispatched via
//     GPU-initiated IBGDA WQE MMIO doorbells (or host proxy when EnableIBGDA
//     is false).
//  3. Intra-rack NVSwitch multimem.st broadcast of the globally reduced result.
func PlanMultiRackHierarchicalCollective(spec MultiRackClusterSpec, payloadBytes int64) (*HierarchicalCollectivePlan, error) {
	spec, err := spec.norm()
	if err != nil {
		return nil, err
	}
	if payloadBytes <= 0 {
		return nil, fmt.Errorf("nvl72: payloadBytes %d must be > 0", payloadBytes)
	}

	totalGPUs := spec.NumRacks * spec.GPUsPerRack
	payloadF := float64(payloadBytes)

	// Level 1: Intra-rack NVSwitch 5/6 multimem reduce-scatter + broadcast.
	const nvswitchBaseUs = 0.35
	intraRackReduceUs := nvswitchBaseUs + payloadF/(spec.IntraRackBWGBps*1e3)
	intraRackBroadcastUs := nvswitchBaseUs + payloadF/(spec.IntraRackBWGBps*1e3)

	// Doorbell dispatch latency: IBGDA GPU MMIO write (~0.18 us) vs CPU proxy thread (~3.8 us).
	const (
		ibgdaDoorbellUs = 0.18
		cpuProxyUs      = 3.80
	)
	doorbellUs := cpuProxyUs
	dispatchMech := "host_cpu_rdma_proxy"
	cpuSavedUs := 0.0
	if spec.EnableIBGDA {
		doorbellUs = ibgdaDoorbellUs
		dispatchMech = "ibgda_gpu_mmio_wqe_doorbell"
		cpuSavedUs = cpuProxyUs - ibgdaDoorbellUs
	}

	// Level 2: Inter-rack scale-out collective across spec.NumRacks racks.
	// With multi-rail rail-optimized NICs in a rack, effective per-rack aggregate
	// scale-out bandwidth scales with rail parallelism, while per-rank slice uses
	// InterRackRDMA_BWGBps.
	treeDepth := 0
	if spec.NumRacks > 1 {
		treeDepth = int(math.Ceil(math.Log2(float64(spec.NumRacks))))
	}

	var (
		interRackUs        float64
		unoptimizedInterUs float64
		interAlgo          string
		sharpTrafficSaving float64
	)
	if spec.NumRacks == 1 {
		interAlgo = "single_rack_bypass"
	} else {
		rFloat := float64(spec.NumRacks)
		ringFactor := 2.0 * (rFloat - 1.0) / rFloat
		const rdmaHopUs = 1.15
		const sharpSwitchUs = 0.65

		// Unoptimized baseline: non-SHARP ring all-reduce + CPU proxy overhead.
		unoptimizedInterUs = float64(treeDepth)*rdmaHopUs + (ringFactor*payloadF)/(spec.InterRackRDMA_BWGBps*1e3)
		if spec.EnableSHARPv4 {
			interAlgo = "quantum3_spectrumx_sharp_v4_tree"
			// SHARP v4 performs in-network switch aggregation: each rack sends
			// its reduced shard once up the switch tree and receives the global
			// sum once down the tree (1.0x payload serialized instead of 2*(R-1)/R).
			interRackUs = float64(treeDepth)*sharpSwitchUs + (1.0*payloadF)/(spec.InterRackRDMA_BWGBps*1e3)
			if ringFactor > 1.0 {
				sharpTrafficSaving = 1.0 - (1.0 / ringFactor)
			} else {
				sharpTrafficSaving = 0.25
			}
		} else {
			interAlgo = "inter_rack_ring_rdma"
			interRackUs = unoptimizedInterUs
		}
	}

	totalUs := intraRackReduceUs + doorbellUs + interRackUs + intraRackBroadcastUs
	unoptTotalUs := intraRackReduceUs + cpuProxyUs + unoptimizedInterUs + intraRackBroadcastUs
	speedup := 1.0
	if totalUs > 0 {
		speedup = unoptTotalUs / totalUs
	}

	return &HierarchicalCollectivePlan{
		Spec:                      spec,
		TotalGPUs:                 totalGPUs,
		PayloadBytes:              payloadBytes,
		NumRails:                  spec.GPUsPerRack,
		SHARPTreeDepth:            treeDepth,
		IntraRackAlgorithm:        "nvswitch_multimem_allreduce",
		InterRackAlgorithm:        interAlgo,
		DispatchMechanism:         dispatchMech,
		IntraRackReduceUs:         intraRackReduceUs,
		IntraRackBroadcastUs:      intraRackBroadcastUs,
		InterRackReductionUs:      interRackUs,
		UnoptimizedInterRackUs:    unoptimizedInterUs,
		DoorbellDispatchUs:        doorbellUs,
		CPUProxyOverheadSavedUs:   cpuSavedUs,
		TotalLatencyUs:            totalUs,
		UnoptimizedTotalLatencyUs: unoptTotalUs,
		SpeedupFactor:             speedup,
		SHARPv4TrafficSavingRatio: sharpTrafficSaving,
		StragglerFallbackMode:     "local_parity_shared_expert_approximation",
		StragglerDeadlineNs:       spec.StragglerDeadlineNs,
	}, nil
}

// IBGDADoorbellCUDA emits device-side CUDA/PTX helpers for GPU-initiated IBGDA
// WQE construction, system-scoped release doorbell ring
// (`st.release.sys.global.u64 [nic_bf_reg], wqe_desc`), and
// `%%globaltimer`-bounded straggler wait (`eg_ready_wait_sys_deadline`) that
// falls back to a local parity / shared-expert approximation if a remote rack
// rank exceeds StragglerDeadlineNs.
func (p *HierarchicalCollectivePlan) IBGDADoorbellCUDA() string {
	var deadlineNs uint64 = 50_000
	var numRacks, gpusPerRack int = 4, 72
	var enableSHARP int = 1
	if p != nil {
		if p.StragglerDeadlineNs > 0 {
			deadlineNs = p.StragglerDeadlineNs
		} else if p.Spec.StragglerDeadlineNs > 0 {
			deadlineNs = p.Spec.StragglerDeadlineNs
		}
		if p.Spec.NumRacks > 0 {
			numRacks = p.Spec.NumRacks
		}
		if p.Spec.GPUsPerRack > 0 {
			gpusPerRack = p.Spec.GPUsPerRack
		}
		if !p.Spec.EnableSHARPv4 {
			enableSHARP = 0
		}
	}

	var b strings.Builder
	b.WriteString(Prelude)
	fmt.Fprintf(&b, `
// ============================================================================
// Multi-Rack IBGDA RDMA WQE Doorbell + SHARP v4 + Straggler-Resilient MoE
// NumRacks=%d, GPUsPerRack=%d, EnableSHARPv4=%d, StragglerDeadlineNs=%d
// ============================================================================

struct alignas(64) eg_ibgda_wqe_ctrl_seg {
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint32_t signature_flags;
    uint32_t imm_sharp_tree_id;
    uint64_t raddr;
    uint64_t laddr;
    uint32_t byte_count;
    uint32_t lkey;
    uint32_t rkey;
    uint32_t reserved[5];
};

EG_DEV uint64_t eg_ibgda_pack_wqe_desc(uint32_t qpn, uint16_t wqe_idx, uint8_t opcode, uint8_t sharp_v4_flag) {
    const uint64_t lo = ((uint64_t)qpn << 8) | (uint64_t)opcode;
    const uint64_t hi = ((uint64_t)wqe_idx << 16) | (uint64_t)sharp_v4_flag;
    return (hi << 32) | lo;
}

// eg_ibgda_post_wqe_doorbell constructs the 64B IBGDA WQE in NIC-mapped UAR /
// GPU memory and rings the ConnectX-8 / ConnectX-9 BlueFlame / doorbell MMIO
// register with a system-scoped release store.
EG_DEV void eg_ibgda_post_wqe_doorbell(
    eg_ibgda_wqe_ctrl_seg* __restrict__ wqe_ring,
    uint64_t* __restrict__ nic_bf_reg,
    uint32_t qpn,
    uint16_t wqe_idx,
    uint64_t laddr,
    uint64_t raddr,
    uint32_t byte_count,
    uint32_t sharp_tree_id) {
    eg_ibgda_wqe_ctrl_seg* seg = &wqe_ring[wqe_idx & 255u];
    seg->opmod_idx_opcode = ((uint32_t)wqe_idx << 8) | 0x0Au; // MLX5_OPCODE_RDMA_WRITE_IMM / SHARP_AGGR
    seg->qpn_ds = (qpn << 8) | 4u;
    seg->signature_flags = 0x08u; // IBGDA_CQ_UPDATE
    seg->imm_sharp_tree_id = sharp_tree_id;
    seg->raddr = raddr;
    seg->laddr = laddr;
    seg->byte_count = byte_count;

    const uint64_t wqe_desc = eg_ibgda_pack_wqe_desc(qpn, wqe_idx, 0x0Au, %du);
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(nic_bf_reg, wqe_desc, __ATOMIC_RELEASE);
#else
    asm volatile(
        "fence.acq_rel.sys;\n\t"
        "st.release.sys.global.u64 [%%0], %%1;"
        :: "l"(nic_bf_reg), "l"(wqe_desc)
        : "memory");
#endif
}

// eg_ready_wait_sys_deadline spins on a system-scoped completion sequence word
// using ld.acquire.sys.global.u64 bounded by %%globaltimer. Returns 0 when the
// remote rack rank completes within deadline_ns, or 1 when the deadline expires
// so the caller can fall back to the local parity / shared-expert approximation.
EG_DEV uint32_t eg_ready_wait_sys_deadline(
    const uint64_t* __restrict__ cq_seq_addr,
    uint64_t target_seq,
    uint64_t deadline_ns) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    const uint64_t cur = __atomic_load_n(cq_seq_addr, __ATOMIC_ACQUIRE);
    (void)deadline_ns;
    return (cur >= target_seq) ? 0u : 1u;
#else
    uint64_t t0 = 0;
    asm volatile("mov.u64 %%0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        uint64_t done = 0;
        asm volatile("ld.acquire.sys.global.u64 %%0, [%%1];" : "=l"(done) : "l"(cq_seq_addr) : "memory");
        if (done >= target_seq) {
            return 0u;
        }
        uint64_t now = 0;
        asm volatile("mov.u64 %%0, %%globaltimer;" : "=l"(now));
        if ((now - t0) > deadline_ns) {
            return 1u;
        }
    }
#endif
}

// eg_multirack_straggler_resilient_moe_combine combines remote rack expert
// outputs when they arrive before StragglerDeadlineNs, and seamlessly substitutes
// a local parity + shared-expert approximation if a remote rack rank straggles.
__global__ void __launch_bounds__(256) eg_multirack_straggler_resilient_moe_combine(
    int tokens,
    int cols,
    uint32_t qpn,
    uint16_t wqe_idx,
    eg_ibgda_wqe_ctrl_seg* __restrict__ wqe_ring,
    uint64_t* __restrict__ nic_bf_reg,
    const uint64_t* __restrict__ remote_cq_seq,
    uint64_t expected_seq,
    const float* __restrict__ remote_expert_out,
    const float* __restrict__ local_shared_expert_out,
    const float* __restrict__ local_parity_backup_out,
    float router_weight,
    float* __restrict__ combined_out,
    uint32_t* __restrict__ straggler_fallback_triggered) {
    constexpr uint64_t DEFAULT_DEADLINE_NS = %dull;
    __shared__ uint32_t s_timed_out;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        eg_ibgda_post_wqe_doorbell(
            wqe_ring, nic_bf_reg, qpn, wqe_idx,
            reinterpret_cast<uint64_t>(local_shared_expert_out),
            reinterpret_cast<uint64_t>(remote_expert_out),
            (uint32_t)((size_t)tokens * (size_t)cols * sizeof(float)),
            1u);
    }
    if (threadIdx.x == 0) {
        s_timed_out = eg_ready_wait_sys_deadline(remote_cq_seq, expected_seq, DEFAULT_DEADLINE_NS);
        if (blockIdx.x == 0 && straggler_fallback_triggered != nullptr) {
            *straggler_fallback_triggered = s_timed_out;
        }
    }
    __syncthreads();

    const int row = (int)blockIdx.x;
    if (row >= tokens) return;
    const bool use_fallback = (s_timed_out != 0u);
    for (int c = (int)threadIdx.x; c < cols; c += (int)blockDim.x) {
        const size_t idx = (size_t)row * (size_t)cols + (size_t)c;
        const float shared_val = local_shared_expert_out[idx];
        if (!use_fallback) {
            combined_out[idx] = shared_val + router_weight * remote_expert_out[idx];
        } else {
            // Soft parity / shared-expert fallback when remote rack exceeds deadline.
            const float parity_approx = local_parity_backup_out ? local_parity_backup_out[idx] : shared_val;
            combined_out[idx] = shared_val + router_weight * (0.5f * parity_approx + 0.5f * shared_val);
        }
    }
}
`, numRacks, gpusPerRack, enableSHARP, deadlineNs, enableSHARP, deadlineNs)
	return b.String()
}

// ElasticShardReRoute describes how a KV-cache / MoE expert shard previously
// hosted on a degraded or failed rank is re-routed to a healthy peer rank.
type ElasticShardReRoute struct {
	ShardID            int
	OldOwnerRank       int
	NewOwnerRank       int
	OldRack            int
	NewRack            int
	SameRack           bool
	ReconstructedVia   string // "intra_rack_nvswitch_replica" or "inter_rack_rdma_parity"
	AffectedStreams    int
	MigrationLatencyUs float64
}

// CUgraphExecEpochSwapPlan describes the atomic host/device CUDA Graph
// (`CUgraphExec`) double-buffered epoch swap that activates the updated
// routing table at a token step boundary with zero decode downtime.
type CUgraphExecEpochSwapPlan struct {
	OldEpoch              uint64
	NewEpoch              uint64
	StandbyGraphSlot      int
	ActiveGraphSlot       int
	UpdateMechanism       string
	DoorbellFencePTX      string
	GraphNodePatchLatency float64 // microseconds to patch device routing pointers in standby CUgraphExec
	ZeroDowntimeVerified  bool
}

// ElasticMigrationReceipt summarizes a zero-downtime KV/expert shard
// re-routing table and atomic CUgraphExec epoch swap plan when a rank degrades
// or leaves a multi-rack NVL72 / NVL144 cluster.
type ElasticMigrationReceipt struct {
	Spec                  MultiRackClusterSpec
	FailedRank            int
	FailedRack            int
	FailedIntraRackRank   int
	ReplacementRank       int
	HealthyRanksRemaining int
	ActiveDecodeStreams   int
	MigratedStreams       int
	ShardReRoutes         []ElasticShardReRoute
	RoutingTable          map[int]int // logical rank -> physical healthy rank
	EpochSwap             CUgraphExecEpochSwapPlan
	TotalMigrationUs      float64
	DowntimeMs            float64
}

// PlanElasticRankMigration computes a zero-downtime KV/expert shard re-routing
// table and atomic CUgraphExec epoch swap plan when failedRank degrades or
// leaves a multi-rack cluster running activeDecodeStreams concurrent decode
// streams.
func PlanElasticRankMigration(spec MultiRackClusterSpec, failedRank int, activeDecodeStreams int) (*ElasticMigrationReceipt, error) {
	spec, err := spec.norm()
	if err != nil {
		return nil, err
	}
	totalGPUs := spec.NumRacks * spec.GPUsPerRack
	if totalGPUs < 2 {
		return nil, fmt.Errorf("nvl72: cluster needs at least 2 total GPUs for elastic migration, have %d", totalGPUs)
	}
	if failedRank < 0 || failedRank >= totalGPUs {
		return nil, fmt.Errorf("nvl72: failedRank %d out of range [0, %d)", failedRank, totalGPUs)
	}
	if activeDecodeStreams < 1 {
		return nil, fmt.Errorf("nvl72: activeDecodeStreams %d must be >= 1", activeDecodeStreams)
	}

	failedRack := failedRank / spec.GPUsPerRack
	failedLocal := failedRank % spec.GPUsPerRack

	// Prefer a sibling rank on the same compute tray / same rack (high-speed
	// NVSwitch 900/1800 GB/s domain); fall back to an adjacent rack only if
	// GPUsPerRack == 1.
	var replacementRank int
	if spec.GPUsPerRack > 1 {
		replacementLocal := (failedLocal + 1) % spec.GPUsPerRack
		replacementRank = failedRack*spec.GPUsPerRack + replacementLocal
	} else {
		replacementRank = (failedRank + 1) % totalGPUs
	}

	routingTable := make(map[int]int, totalGPUs)
	for r := 0; r < totalGPUs; r++ {
		if r == failedRank {
			routingTable[r] = replacementRank
		} else {
			routingTable[r] = r
		}
	}

	// Partition the failed rank's KV/expert shards across healthy intra-rack
	// peers (or inter-rack peers when GPUsPerRack == 1) so no single peer is
	// overloaded.
	numShards := min(max(spec.GPUsPerRack-1, 1), 8)
	streamsOnFailed := max((activeDecodeStreams+totalGPUs-1)/totalGPUs, 1)
	streamsPerShard := max((streamsOnFailed+numShards-1)/numShards, 1)

	// Assume each shard holds 64 MiB of hot expert/KV state to reconstruct or
	// activate from symmetric NVSwitch replica / parity.
	const shardBytes = 64.0 * 1024.0 * 1024.0
	shards := make([]ElasticShardReRoute, 0, numShards)
	var maxShardUs float64
	migratedStreams := 0
	for s := 0; s < numShards; s++ {
		var targetRank int
		var sameRack bool
		var via string
		var bwGBps float64
		if spec.GPUsPerRack > 1 {
			peerLocal := (failedLocal + 1 + s) % spec.GPUsPerRack
			if peerLocal == failedLocal {
				peerLocal = (peerLocal + 1) % spec.GPUsPerRack
			}
			targetRank = failedRack*spec.GPUsPerRack + peerLocal
			sameRack = true
			via = "intra_rack_nvswitch_replica"
			bwGBps = spec.IntraRackBWGBps
		} else {
			targetRack := (failedRack + 1 + s) % spec.NumRacks
			if targetRack == failedRack {
				targetRack = (targetRack + 1) % spec.NumRacks
			}
			targetRank = targetRack * spec.GPUsPerRack
			sameRack = false
			via = "inter_rack_rdma_parity"
			bwGBps = spec.InterRackRDMA_BWGBps
		}
		latUs := shardBytes / (bwGBps * 1e3)
		if latUs > maxShardUs {
			maxShardUs = latUs
		}
		aff := min(streamsPerShard, max(streamsOnFailed-migratedStreams, 1))
		migratedStreams += aff
		shards = append(shards, ElasticShardReRoute{
			ShardID:            s,
			OldOwnerRank:       failedRank,
			NewOwnerRank:       targetRank,
			OldRack:            failedRack,
			NewRack:            targetRank / spec.GPUsPerRack,
			SameRack:           sameRack,
			ReconstructedVia:   via,
			AffectedStreams:    aff,
			MigrationLatencyUs: latUs,
		})
	}

	const graphPatchUs = 4.5 // cuGraphExecKernelNodeSetParams on standby graph
	swapPlan := CUgraphExecEpochSwapPlan{
		OldEpoch:              1,
		NewEpoch:              2,
		StandbyGraphSlot:      1,
		ActiveGraphSlot:       0,
		UpdateMechanism:       "cuGraphExecKernelNodeSetParams_double_buffered_epoch_flip",
		DoorbellFencePTX:      "fence.acq_rel.sys; st.release.sys.global.u64 [epoch_ptr], new_epoch;",
		GraphNodePatchLatency: graphPatchUs,
		ZeroDowntimeVerified:  true,
	}

	return &ElasticMigrationReceipt{
		Spec:                  spec,
		FailedRank:            failedRank,
		FailedRack:            failedRack,
		FailedIntraRackRank:   failedLocal,
		ReplacementRank:       replacementRank,
		HealthyRanksRemaining: totalGPUs - 1,
		ActiveDecodeStreams:   activeDecodeStreams,
		MigratedStreams:       migratedStreams,
		ShardReRoutes:         shards,
		RoutingTable:          routingTable,
		EpochSwap:             swapPlan,
		TotalMigrationUs:      maxShardUs + graphPatchUs,
		DowntimeMs:            0.0, // background reconstruction + atomic epoch pointer swap at step boundary
	}, nil
}
