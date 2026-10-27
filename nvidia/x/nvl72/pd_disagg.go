package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// AttentionCacheKind identifies the KV cache representation transferred
// between disaggregated prefill and decode workers.
type AttentionCacheKind string

const (
	// CacheGQA selects Grouped-Query Attention (GQA) KV cache storage
	// (2 * NumKVHeads * HeadDim * 2 bytes per token per layer in BF16).
	CacheGQA AttentionCacheKind = "gqa"

	// CacheMLA selects Multi-head Latent Attention (MLA) compressed latent
	// cache storage (KVLoRARank FP8 bytes + RoPEDim * 2 BF16 bytes per token
	// per layer, e.g. 512 + 64*2 = 640 bytes/token/layer).
	CacheMLA AttentionCacheKind = "mla"
)

// Transfer transport identifiers for disaggregated prefill-to-decode KV/MLA
// cache streaming across the 4-tier NVL72 pod hierarchy.
const (
	TransportNVLink5IntraRack = "nvlink5-intra-rack"
	TransportConnectX7RDMA    = "connectx7-400g-rdma"
	TransportConnectX8RDMA    = "connectx8-800g-rdma"
)

// Per-GPU FP4 Tensor Core throughput (PFLOPS) and Special Function Unit (SFU)
// attention softmax scaling for GB200 (sm_100a) and GB300 (sm_103a).
const (
	gb200FP4PFLOPS = 10.0
	gb300FP4PFLOPS = 15.0
	gb200SFUScale  = 1.0
	gb300SFUScale  = 2.0
)

const pdFabricPrelude = `#ifndef EG_PD_FABRIC_HELPERS
#define EG_PD_FABRIC_HELPERS

// eg_fabric_peer_ptr maps a symmetric memory base pointer to target rank's
// address space across the 72-GPU NVLink 5 fabric.
template <typename T>
EG_DEV T* eg_fabric_peer_ptr(T* base, int dst_rank, int my_rank, size_t bytes_per_rank) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    (void)my_rank;
    return reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(base) + (size_t)dst_rank * bytes_per_rank);
#else
    return reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(base) +
        ((ptrdiff_t)dst_rank - (ptrdiff_t)my_rank) * (ptrdiff_t)bytes_per_rank);
#endif
}

// eg_st_global_v4_u32 issues a 128-bit vectorized peer fabric store
// (st.global.v4.u32) to NVLink 5 symmetric memory or Grace C2C staging memory.
EG_DEV void eg_st_global_v4_u32(void* dst, eg_u32x4 v) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    std::memcpy(dst, &v, sizeof(eg_u32x4));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
#else
    *reinterpret_cast<eg_u32x4*>(dst) = v;
#endif
}

// eg_st_release_sys_u32 issues a system-scope release fence (fence.AcqRel.sys /
// fence.acq_rel.sys) followed by a system-scope release store
// (st.release.sys.global.u32) to signal page readiness across the NVLink 5 /
// NVLink-C2C coherence domain.
EG_DEV void eg_st_release_sys_u32(uint32_t* addr, uint32_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    std::atomic_thread_fence(std::memory_order_acq_rel);
    std::atomic_ref<uint32_t>(*addr).store(val, std::memory_order_release);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.acq_rel.sys;\n\tst.release.sys.global.u32 [%0], %1;"
                 :: "l"(addr), "r"(val) : "memory");
#else
    *addr = val;
#endif
}

// eg_ld_acquire_sys_u32 performs a system-scope acquire load
// (ld.acquire.sys.global.u32) on a page-ready flag written by a peer rank.
EG_DEV uint32_t eg_ld_acquire_sys_u32(uint32_t* addr) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    const uint32_t v = std::atomic_ref<uint32_t>(*addr).load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_acq_rel);
    return v;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(val) : "l"(addr) : "memory");
    return val;
#else
    return *addr;
#endif
}

// eg_kv_transfer_barrier synchronizes all participating ranks between the
// prefill page push phase and the decode page verification/consumption phase.
EG_DEV void eg_kv_transfer_barrier(int num_ranks, uint32_t epoch) {
    __syncthreads();
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_nvlink_barrier(nullptr, num_ranks, epoch);
#else
    (void)num_ranks;
    (void)epoch;
    __threadfence_system();
#endif
    __syncthreads();
}

#endif // EG_PD_FABRIC_HELPERS
`

// NVL72KVPageTransfer emits a zero-copy NVLink 5 and Grace NVLink-C2C paged
// KV / MLA latent cache transfer kernel across up to 72 Blackwell GPUs.
//
// In Phase 1, the prefill rank streams NumPages pages (each of PageBytes bytes,
// indexed by src_page_ids[p] in src_pages) directly into the target decode
// rank's paged KV pool (dst_kv_pool at dst_page_ids[p] on dst_rank) using
// 128-bit peer fabric stores (st.global.v4.u32 via eg_fabric_peer_ptr),
// optionally stages through the target compute tray's 450 GB/s Grace LPDDR5X
// NVLink-C2C buffer (c2c_staging_pool), and issues a system-scope release
// fence (fence.AcqRel.sys / st.release.sys.global.u32) setting
// dst_page_ready[dst_page_id] = 1.
//
// In Phase 2, the target decode rank observes dst_page_ready[dst_page_id] == 1
// via ld.acquire.sys.global.u32 and consumes/promotes the received KV or MLA
// latent page in the same multi-rank launch.
type NVL72KVPageTransfer struct {
	Name               string
	NumPages           int  // number of KV or MLA latent pages to stream from prefill rank to decode rank
	PageBytes          int  // bytes per page (e.g. MLA: PageSize*(512+64*2); GQA: PageSize*2*NumKVHeads*HeadDim*ElemBytes)
	UseMLALatent       bool // when true, packs compressed KVLoRARank FP8 latent + RoPEDim BF16 RoPE per token
	UseGraceC2CStaging bool // when true, stages cold/overflow KV pages through the tray's 450 GB/s Grace LPDDR5X NVLink-C2C pool
	NumRanks           int  // number of ranks in the NVL72 domain (e.g. 8 or 72)
}

func (k NVL72KVPageTransfer) norm() NVL72KVPageTransfer {
	if k.Name == "" || !cIdent(k.Name) {
		k.Name = "nvl72_kv_page_transfer"
	}
	if k.NumPages < 1 {
		k.NumPages = 1
	}
	if k.PageBytes < 16 {
		if k.UseMLALatent {
			k.PageBytes = 640
		} else {
			k.PageBytes = 256
		}
	}
	if k.NumRanks < 1 {
		k.NumRanks = 8
	}
	if k.NumRanks > NVL72MaxRanks {
		k.NumRanks = NVL72MaxRanks
	}
	return k
}

func (k NVL72KVPageTransfer) threads() int {
	k = k.norm()
	if k.NumRanks > 16 {
		return 32
	}
	return 64
}

// Grid returns the launch grid dimensions [NumPages, 1, 1].
func (k NVL72KVPageTransfer) Grid() [3]int {
	k = k.norm()
	return [3]int{k.NumPages, 1, 1}
}

// Block returns the CTA block dimensions [Threads, 1, 1].
func (k NVL72KVPageTransfer) Block() [3]int {
	return [3]int{k.threads(), 1, 1}
}

func (k NVL72KVPageTransfer) params() []string {
	return []string{
		"int total_pool_pages",
		"const uint8_t* __restrict__ src_pages",
		"const int32_t* __restrict__ src_page_ids",
		"const int32_t* __restrict__ dst_page_ids",
		"uint8_t* __restrict__ dst_kv_pool",
		"uint32_t* __restrict__ dst_page_ready",
		"uint8_t* __restrict__ c2c_staging_pool",
	}
}

// CUDA returns the device kernel definition for NVL72KVPageTransfer.
func (k NVL72KVPageTransfer) CUDA() string {
	k = k.norm()
	threads := k.threads()
	mlaLit := "false"
	if k.UseMLALatent {
		mlaLit = "true"
	}
	c2cLit := "false"
	if k.UseGraceC2CStaging {
		c2cLit = "true"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank NVLink 5 / Grace C2C zero-copy KV/MLA page transfer (%d pages, %d B/page, MLA=%s, C2C=%s).\n",
		k.Name, k.NumRanks, k.NumPages, k.PageBytes, mlaLit, c2cLit)
	fmt.Fprintf(&b, "// PTX ordering: st.global.v4.u32 -> fence.AcqRel.sys -> st.release.sys.global.u32 -> ld.acquire.sys.global.u32.\n")
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int NUM_PAGES = %d, PAGE_BYTES = %d, NUM_RANKS = %d, THREADS = %d;\n",
		k.NumPages, k.PageBytes, k.NumRanks, threads)
	fmt.Fprintf(&b, "    constexpr bool USE_MLA_LATENT = %s, USE_GRACE_C2C_STAGING = %s;\n",
		mlaLit, c2cLit)
	b.WriteString(`    (void)USE_MLA_LATENT;
    const int my_rank = eg_nvlink_rank();
    const int tid = threadIdx.x;
    const int pages_per_rank = (total_pool_pages >= NUM_RANKS && (total_pool_pages % NUM_RANKS) == 0)
        ? (total_pool_pages / NUM_RANKS)
        : total_pool_pages;
    const size_t pool_bytes_per_rank = (pages_per_rank < total_pool_pages)
        ? ((size_t)pages_per_rank * (size_t)PAGE_BYTES)
        : 0u;
    const size_t ready_bytes_per_rank = (pages_per_rank < total_pool_pages)
        ? ((size_t)pages_per_rank * sizeof(uint32_t))
        : 0u;

    for (int p = (int)blockIdx.x; p < NUM_PAGES; p += (int)gridDim.x) {
        const int src_page_id = src_page_ids ? src_page_ids[p] : p;
        const int dst_page_id = dst_page_ids ? dst_page_ids[p] : p;
        const bool valid_page = (src_page_id >= 0 && src_page_id < NUM_PAGES &&
                                 dst_page_id >= 0 && dst_page_id < total_pool_pages);
        const int dst_rank = (valid_page && pages_per_rank < total_pool_pages)
            ? (dst_page_id / pages_per_rank)
            : (valid_page ? (dst_page_id % NUM_RANKS) : 0);
        const int local_page_id = (valid_page && pages_per_rank < total_pool_pages)
            ? (dst_page_id % pages_per_rank)
            : dst_page_id;
        const int dst_tray_id = dst_rank / 4;
        (void)dst_tray_id;

        // Phase 1: Prefill rank (rank 0) pushes the KV or MLA latent page over
        // NVLink 5 peer fabric pointers using 128-bit st.global.v4.u32 stores
        // (and stages to Grace LPDDR5X C2C pool when USE_GRACE_C2C_STAGING is enabled),
        // then issues fence.AcqRel.sys + st.release.sys.global.u32 on dst_page_ready.
        if (valid_page && my_rank == 0) {
            const uint8_t* src_ptr = src_pages + (size_t)src_page_id * (size_t)PAGE_BYTES;
            uint8_t* peer_pool_base = eg_fabric_peer_ptr(dst_kv_pool, dst_rank, my_rank, pool_bytes_per_rank);
            uint8_t* peer_page_ptr = peer_pool_base + (size_t)local_page_id * (size_t)PAGE_BYTES;
            uint8_t* c2c_page_ptr = nullptr;
            if (c2c_staging_pool != nullptr) {
                uint8_t* peer_c2c_base = eg_fabric_peer_ptr(c2c_staging_pool, dst_rank, my_rank, pool_bytes_per_rank);
                c2c_page_ptr = peer_c2c_base + (size_t)local_page_id * (size_t)PAGE_BYTES;
            }

            constexpr int VEC16 = PAGE_BYTES / 16;
            for (int v = tid; v < VEC16; v += THREADS) {
                eg_u32x4 quad;
                std::memcpy(&quad, src_ptr + (size_t)v * 16u, sizeof(eg_u32x4));
                if (USE_GRACE_C2C_STAGING && c2c_page_ptr != nullptr) {
                    eg_st_global_v4_u32(c2c_page_ptr + (size_t)v * 16u, quad);
                } else {
                    eg_st_global_v4_u32(peer_page_ptr + (size_t)v * 16u, quad);
                }
            }
            for (int rem = VEC16 * 16 + tid; rem < PAGE_BYTES; rem += THREADS) {
                if (USE_GRACE_C2C_STAGING && c2c_page_ptr != nullptr) {
                    c2c_page_ptr[rem] = src_ptr[rem];
                } else {
                    peer_page_ptr[rem] = src_ptr[rem];
                }
            }
        }
        __syncthreads();
        if (valid_page && my_rank == 0 && tid == 0) {
            uint32_t* peer_ready_base = eg_fabric_peer_ptr(dst_page_ready, dst_rank, my_rank, ready_bytes_per_rank);
            eg_st_release_sys_u32(peer_ready_base + local_page_id, 1u);
        }

        // Cross-rank synchronization so target decode rank observes the
        // completed NVLink 5 / C2C page write and release flag in this launch.
        eg_kv_transfer_barrier(NUM_RANKS, (uint32_t)(p * 2 + 1));

        // Phase 2: Target decode rank (my_rank == dst_rank) acquires
        // dst_page_ready[dst_page_id] via ld.acquire.sys.global.u32 and
        // consumes/promotes the transferred KV or MLA latent page.
        if (valid_page && my_rank == dst_rank) {
            uint32_t* local_ready_base = eg_fabric_peer_ptr(dst_page_ready, dst_rank, my_rank, ready_bytes_per_rank);
            const uint32_t ready = eg_ld_acquire_sys_u32(local_ready_base + local_page_id);
            if (ready == 1u) {
                uint8_t* local_pool_base = eg_fabric_peer_ptr(dst_kv_pool, dst_rank, my_rank, pool_bytes_per_rank);
                uint8_t* local_page_ptr = local_pool_base + (size_t)local_page_id * (size_t)PAGE_BYTES;
                if (c2c_staging_pool != nullptr) {
                    uint8_t* local_c2c_base = eg_fabric_peer_ptr(c2c_staging_pool, dst_rank, my_rank, pool_bytes_per_rank);
                    uint8_t* local_c2c_ptr = local_c2c_base + (size_t)local_page_id * (size_t)PAGE_BYTES;
                    constexpr int VEC16 = PAGE_BYTES / 16;
                    for (int v = tid; v < VEC16; v += THREADS) {
                        eg_u32x4 quad;
                        if (USE_GRACE_C2C_STAGING) {
                            std::memcpy(&quad, local_c2c_ptr + (size_t)v * 16u, sizeof(eg_u32x4));
                            eg_st_global_v4_u32(local_page_ptr + (size_t)v * 16u, quad);
                        } else {
                            std::memcpy(&quad, local_page_ptr + (size_t)v * 16u, sizeof(eg_u32x4));
                            eg_st_global_v4_u32(local_c2c_ptr + (size_t)v * 16u, quad);
                        }
                    }
                    for (int rem = VEC16 * 16 + tid; rem < PAGE_BYTES; rem += THREADS) {
                        if (USE_GRACE_C2C_STAGING) {
                            local_page_ptr[rem] = local_c2c_ptr[rem];
                        } else {
                            local_c2c_ptr[rem] = local_page_ptr[rem];
                        }
                    }
                }
            }
        }

        eg_kv_transfer_barrier(NUM_RANKS, (uint32_t)(p * 2 + 2));
    }
}
`)
	return b.String()
}

// Launcher returns the host launcher function for NVL72KVPageTransfer.
func (k NVL72KVPageTransfer) Launcher() string {
	k = k.norm()
	grid := k.Grid()
	block := k.Block()
	params := k.params()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n",
		k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    if (total_pool_pages > 0) EG_LAUNCH(%s, dim3(%d), dim3(%d), 0, stream, %s);\n}\n",
		k.Name, grid[0], block[0], strings.Join(paramNames(params), ", "))
	return b.String()
}

// Source returns the standalone translation unit for NVL72KVPageTransfer,
// including [Prelude], [CollectivePrelude], fabric helpers, kernel, and
// launcher.
func (k NVL72KVPageTransfer) Source() string {
	return Prelude + "\n" + CollectivePrelude + "\n" + pdFabricPrelude + "\n" + k.CUDA() + "\n" + k.Launcher()
}

// Reference computes the expected paged KV pool contents and page-ready flags
// after transferring k.NumPages pages of size pageBytes from srcPages into
// dstPageIDs within a pool of totalPoolPages pages.
func (k NVL72KVPageTransfer) Reference(srcPages []byte, dstPageIDs []int32, totalPoolPages int, pageBytes int) (pool []byte, ready []uint32) {
	k = k.norm()
	if pageBytes <= 0 {
		pageBytes = k.PageBytes
	}
	if totalPoolPages < 0 {
		totalPoolPages = 0
	}
	pool = make([]byte, totalPoolPages*pageBytes)
	ready = make([]uint32, totalPoolPages)

	numPages := k.NumPages
	if len(dstPageIDs) < numPages {
		numPages = len(dstPageIDs)
	}
	for p := 0; p < numPages; p++ {
		srcStart := p * pageBytes
		srcEnd := srcStart + pageBytes
		if srcEnd > len(srcPages) {
			break
		}
		dstPage := int(dstPageIDs[p])
		if dstPage < 0 || dstPage >= totalPoolPages {
			continue
		}
		dstStart := dstPage * pageBytes
		copy(pool[dstStart:dstStart+pageBytes], srcPages[srcStart:srcEnd])
		ready[dstPage] = 1
	}
	return pool, ready
}

// PDModelSpec describes the attention cache layout and MoE architecture of a
// frontier model for prefill/decode disaggregation planning.
type PDModelSpec struct {
	NumLayers            int
	HiddenSize           int
	NumQHeads            int
	NumKVHeads           int
	HeadDim              int
	CacheKind            AttentionCacheKind // CacheGQA or CacheMLA
	KVLoRARank           int                // MLA compressed latent rank in FP8 bytes (e.g. 512)
	RoPEDim              int                // MLA decoupled RoPE dimension in BF16 elements (e.g. 64)
	TotalExperts         int
	TopK                 int
	ActiveParamsBillions float64
	TotalParamsBillions  float64
}

// PDSLA specifies the workload geometry and latency targets for disaggregated
// prefill and decode serving.
type PDSLA struct {
	PromptTokens      int     // input prompt length in tokens (e.g. 4096 or 32768)
	DecodeBatchPerGPU int     // concurrent decoding sequences per decode GPU (e.g. 64 or 128)
	MaxTTFTMs         float64 // target Time-To-First-Token SLA in milliseconds
	MaxTPOTMs         float64 // target Time-Per-Output-Token SLA in milliseconds
}

// PDDisaggPlan reports the selected prefill/decode rack and GPU split, KV/MLA
// transfer footprint, latency breakdown, and cluster token throughput.
type PDDisaggPlan struct {
	Generation           string
	TotalRacks           int
	PrefillRacks         int
	DecodeRacks          int
	PrefillGPUs          int
	DecodeGPUs           int
	KVBytesPerToken      int64
	TotalKVTransferBytes int64
	PrefillComputeMs     float64
	KVTransferMs         float64
	EstimatedTTFTMs      float64
	EstimatedTPOTMs      float64
	TransferTransport    string
	MeetsSLA             bool
	TokensPerSecCluster  float64
}

func (m PDModelSpec) kvBytesPerTokenPerLayer() (int64, error) {
	switch m.CacheKind {
	case CacheMLA:
		if m.KVLoRARank <= 0 {
			return 0, fmt.Errorf("nvl72: MLA KVLoRARank %d must be > 0", m.KVLoRARank)
		}
		if m.RoPEDim < 0 {
			return 0, fmt.Errorf("nvl72: MLA RoPEDim %d must be >= 0", m.RoPEDim)
		}
		// Compressed KVLoRARank stored in FP8 (1 byte/elem) + RoPEDim stored in BF16 (2 bytes/elem).
		return int64(m.KVLoRARank + m.RoPEDim*2), nil
	case CacheGQA, "":
		if m.NumKVHeads <= 0 || m.HeadDim <= 0 {
			return 0, fmt.Errorf("nvl72: GQA NumKVHeads (%d) and HeadDim (%d) must be > 0", m.NumKVHeads, m.HeadDim)
		}
		// K and V projections stored in BF16 (2 bytes/elem).
		return int64(2 * m.NumKVHeads * m.HeadDim * 2), nil
	default:
		return 0, fmt.Errorf("nvl72: unsupported AttentionCacheKind %q (want %q or %q)", m.CacheKind, CacheGQA, CacheMLA)
	}
}

// PlanPDDisaggregation evaluates candidate prefill/decode splits across
// totalRacks NVL72 racks (or compute-tray GPU splits within a single 72-GPU
// rack when totalRacks == 1) on GB200 or GB300 hardware and returns the split
// that satisfies sla while maximizing TokensPerSecCluster.
func PlanPDDisaggregation(gen string, totalRacks int, sla PDSLA, model PDModelSpec) (PDDisaggPlan, error) {
	spec, err := LookupPodSpec(gen)
	if err != nil {
		return PDDisaggPlan{}, err
	}
	if totalRacks < 1 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: totalRacks %d must be >= 1", totalRacks)
	}
	if sla.PromptTokens < 1 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: PromptTokens %d must be >= 1", sla.PromptTokens)
	}
	if sla.DecodeBatchPerGPU < 1 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: DecodeBatchPerGPU %d must be >= 1", sla.DecodeBatchPerGPU)
	}
	if sla.MaxTTFTMs <= 0 || sla.MaxTPOTMs <= 0 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: MaxTTFTMs (%v) and MaxTPOTMs (%v) must be > 0", sla.MaxTTFTMs, sla.MaxTPOTMs)
	}
	if model.NumLayers < 1 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: NumLayers %d must be >= 1", model.NumLayers)
	}
	if model.HiddenSize < 2 || model.HiddenSize%2 != 0 {
		return PDDisaggPlan{}, fmt.Errorf("nvl72: HiddenSize %d must be a positive multiple of 2", model.HiddenSize)
	}

	kvBytesPerToken, err := model.kvBytesPerTokenPerLayer()
	if err != nil {
		return PDDisaggPlan{}, err
	}
	totalKVTransferBytes := int64(sla.PromptTokens) * int64(model.NumLayers) * kvBytesPerToken

	genLower := strings.ToLower(spec.Generation)
	fp4PFLOPS := gb200FP4PFLOPS
	sfuScale := gb200SFUScale
	if spec.Generation == GenGB300 {
		fp4PFLOPS = gb300FP4PFLOPS
		sfuScale = gb300SFUScale
	}

	transport := TransportNVLink5IntraRack
	transportBWGBps := spec.Tier2NVLink5UniGBps // 900 GB/s intra-rack NVLink 5
	if totalRacks > 1 {
		transportBWGBps = spec.Tier4ScaleOutGBps // 50 GB/s ConnectX-7 vs 100 GB/s ConnectX-8
		if spec.Generation == GenGB300 {
			transport = TransportConnectX8RDMA
		} else {
			transport = TransportConnectX7RDMA
		}
	}

	// KV/MLA cache transfer latency in ms: bytes / (GB/s * 1e6).
	kvTransferMs := float64(totalKVTransferBytes) / (transportBWGBps * 1e6)

	topK := model.TopK
	if topK < 1 {
		topK = 8
	}
	totalExperts := model.TotalExperts
	if totalExperts < topK {
		totalExperts = max(topK, 64)
	}
	numQHeads := model.NumQHeads
	if numQHeads < 1 {
		numQHeads = 64
	}
	headDim := model.HeadDim
	if headDim < 1 {
		headDim = 128
	}
	activeParams := model.ActiveParamsBillions * 1e9
	if activeParams <= 0 {
		activeParams = float64(model.NumLayers) * 12.0 * float64(model.HiddenSize) * float64(model.HiddenSize)
	}
	totalParams := model.TotalParamsBillions * 1e9
	if totalParams < activeParams {
		totalParams = activeParams * float64(max(totalExperts/topK, 1))
	}

	type splitCandidate struct {
		prefillRacks int
		decodeRacks  int
		prefillGPUs  int
		decodeGPUs   int
	}
	var candidates []splitCandidate
	if totalRacks == 1 {
		// Evaluate compute-tray splits (4 GPUs/tray) within the 72-GPU rack.
		for prefillTrays := 1; prefillTrays < spec.TraysPerRack; prefillTrays++ {
			pGPUs := prefillTrays * NVL72GPUsPerTray
			dGPUs := spec.GPUsPerRack - pGPUs
			candidates = append(candidates, splitCandidate{
				prefillRacks: 1,
				decodeRacks:  1,
				prefillGPUs:  pGPUs,
				decodeGPUs:   dGPUs,
			})
		}
	} else {
		for pRacks := 1; pRacks < totalRacks; pRacks++ {
			dRacks := totalRacks - pRacks
			candidates = append(candidates, splitCandidate{
				prefillRacks: pRacks,
				decodeRacks:  dRacks,
				prefillGPUs:  pRacks * spec.GPUsPerRack,
				decodeGPUs:   dRacks * spec.GPUsPerRack,
			})
		}
	}

	var (
		bestPlan     PDDisaggPlan
		haveFeasible bool
		bestViolation = math.MaxFloat64
	)

	for _, c := range candidates {
		// 1. Prefill compute latency on an intra-rack prefill group (up to 72 GPUs)
		// with multi-rack queueing/pipeline load factor (1 + 0.15/prefillRacks).
		instanceGPUs := float64(min(c.prefillGPUs, spec.GPUsPerRack))
		loadFactor := 1.0 + 0.15/float64(c.prefillRacks)

		linearFLOPs := 2.0 * float64(sla.PromptTokens) * activeParams
		attnFLOPs := 2.0 * float64(model.NumLayers) * float64(sla.PromptTokens) * float64(sla.PromptTokens) * float64(numQHeads*headDim)
		sfuOps := float64(model.NumLayers) * float64(sla.PromptTokens) * float64(sla.PromptTokens) * float64(numQHeads) * 16.0

		tensorRate := instanceGPUs * fp4PFLOPS * 1e15 * 0.55
		sfuRate := instanceGPUs * 0.35e15 * sfuScale

		prefillEP, err := EstimateMultiRackPodEP(
			spec.Generation,
			1,
			max(sla.PromptTokens/int(instanceGPUs), 1),
			model.HiddenSize,
			topK,
			totalExperts,
		)
		if err != nil {
			return PDDisaggPlan{}, err
		}
		prefillCommMs := float64(model.NumLayers) * (prefillEP.TBOTwoBatchNs / 2.0) / 1e6
		prefillComputeMs := (((linearFLOPs+attnFLOPs)/tensorRate + sfuOps/sfuRate) * 1e3 + prefillCommMs) * loadFactor
		estimatedTTFTMs := prefillComputeMs + kvTransferMs

		// 2. Decode TPOT latency across decodeRacks / decodeGPUs using EstimateMultiRackPodEP.
		decodeEP, err := EstimateMultiRackPodEP(
			spec.Generation,
			c.decodeRacks,
			sla.DecodeBatchPerGPU,
			model.HiddenSize,
			topK,
			totalExperts,
		)
		if err != nil {
			return PDDisaggPlan{}, err
		}
		rackScale := float64(c.decodeRacks*spec.GPUsPerRack) / float64(c.decodeGPUs)
		epStepMs := float64(model.NumLayers) * (decodeEP.TBOTwoBatchNs / 2.0) / 1e6 * rackScale

		// Per-step HBM3e weight streaming + KV/MLA cache scan across DecodeBatchPerGPU sequences.
		weightBytesPerGPU := (totalParams * 0.5) / float64(c.decodeGPUs) // FP4 0.5 B/param sharded across decodeGPUs
		kvScanBytesPerGPU := float64(sla.DecodeBatchPerGPU) * float64(totalKVTransferBytes) / math.Min(float64(c.decodeGPUs), float64(spec.GPUsPerRack))
		hbmScanMs := (weightBytesPerGPU + kvScanBytesPerGPU) / (spec.Tier1HBM3eGBps * 1e6)

		// Decode attention softmax SFU term and HBM3e capacity pressure (192 GiB GB200 vs 288 GiB GB300).
		decodeSFUMs := (float64(model.NumLayers) * float64(sla.DecodeBatchPerGPU) * float64(sla.PromptTokens) * float64(numQHeads) * 16.0) /
			(float64(c.decodeGPUs) * 0.35e15 * sfuScale) * 1e3
		usedHBMGB := (weightBytesPerGPU + kvScanBytesPerGPU) / 1e9
		capPressure := 1.0
		if usedHBMGB > float64(spec.HBM3eCapacityGB)*0.85 {
			// Overflow pages spill to Tier 3 Grace LPDDR5X NVLink-C2C (450 GB/s).
			overflowFrac := (usedHBMGB - float64(spec.HBM3eCapacityGB)*0.85) / float64(spec.HBM3eCapacityGB)
			capPressure = 1.0 + overflowFrac*(spec.Tier1HBM3eGBps/spec.Tier3GraceC2CGBps)*0.25
		}
		estimatedTPOTMs := (epStepMs + hbmScanMs + decodeSFUMs) * capPressure

		// 3. Coupled cluster token throughput across prefill and decode pools.
		prefillPoolTokPerSec := (float64(c.prefillGPUs) / float64(spec.GPUsPerRack)) * float64(sla.PromptTokens) * (1000.0 / estimatedTTFTMs)
		decodePoolTokPerSec := float64(c.decodeGPUs*sla.DecodeBatchPerGPU) * (1000.0 / estimatedTPOTMs)
		tokensPerSecCluster := (2.0 * prefillPoolTokPerSec * decodePoolTokPerSec) / (prefillPoolTokPerSec + decodePoolTokPerSec)

		meetsSLA := estimatedTTFTMs <= sla.MaxTTFTMs && estimatedTPOTMs <= sla.MaxTPOTMs
		plan := PDDisaggPlan{
			Generation:           genLower,
			TotalRacks:           totalRacks,
			PrefillRacks:         c.prefillRacks,
			DecodeRacks:          c.decodeRacks,
			PrefillGPUs:          c.prefillGPUs,
			DecodeGPUs:           c.decodeGPUs,
			KVBytesPerToken:      kvBytesPerToken,
			TotalKVTransferBytes: totalKVTransferBytes,
			PrefillComputeMs:     prefillComputeMs,
			KVTransferMs:         kvTransferMs,
			EstimatedTTFTMs:      estimatedTTFTMs,
			EstimatedTPOTMs:      estimatedTPOTMs,
			TransferTransport:    transport,
			MeetsSLA:             meetsSLA,
			TokensPerSecCluster:  tokensPerSecCluster,
		}

		if meetsSLA {
			if !haveFeasible || plan.TokensPerSecCluster > bestPlan.TokensPerSecCluster {
				bestPlan = plan
				haveFeasible = true
			}
		} else if !haveFeasible {
			ttftExceed := math.Max(0.0, (estimatedTTFTMs-sla.MaxTTFTMs)/sla.MaxTTFTMs)
			tpotExceed := math.Max(0.0, (estimatedTPOTMs-sla.MaxTPOTMs)/sla.MaxTPOTMs)
			violation := ttftExceed + tpotExceed
			if violation < bestViolation || (math.Abs(violation-bestViolation) < 1e-9 && plan.TokensPerSecCluster > bestPlan.TokensPerSecCluster) {
				bestViolation = violation
				bestPlan = plan
			}
		}
	}

	return bestPlan, nil
}
