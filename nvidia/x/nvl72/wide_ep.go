package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// GB200 NVL72 rack topology and 3-tier bandwidth constants.
const (
	// NVL72MaxRanks is the maximum number of Blackwell GPUs in one GB200 NVL72
	// NVLink 5 domain (18 compute trays x 4 GPUs per tray).
	NVL72MaxRanks = 72

	// NVL72NumTrays is the number of compute trays in a GB200 NVL72 rack.
	NVL72NumTrays = 18

	// NVL72GPUsPerTray is the number of Blackwell GPUs per compute tray.
	NVL72GPUsPerTray = 4

	// NVL72HBM3eGBps is the per-GPU HBM3e bandwidth in GB/s (8.0 TB/s).
	NVL72HBM3eGBps = 8000.0

	// NVL72NVLink5UniGBps is the per-GPU unidirectional NVLink 5 bandwidth in
	// GB/s (900 GB/s unidirectional, 1.8 TB/s bidirectional).
	NVL72NVLink5UniGBps = 900.0

	// NVL72GraceC2CGBps is the per-GPU Grace-Blackwell NVLink-C2C coherent memory
	// bandwidth in GB/s (450 GB/s unidirectional, 900 GB/s bidirectional).
	NVL72GraceC2CGBps = 450.0

	// NVL72BlackwellTensorTFLOPS is the per-GPU dense BF16/FP8 tensor-core
	// compute ceiling in TFLOPS on GB200.
	NVL72BlackwellTensorTFLOPS = 2250.0
)

const nvl72FabricPrelude = `#ifndef EG_NVL72_FABRIC_HELPERS
#define EG_NVL72_FABRIC_HELPERS

#if defined(EG_HOST_EMU) || defined(EG_EMU)
inline thread_local int eg_nvl72_emu_rank_tls = 0;
inline int eg_nvl72_emu_num_ranks = 1;

struct eg_nvl72_emu_bar_key {
    const void* ptr;
    unsigned bx, by, bz;
    uint32_t epoch;
    bool operator<(const eg_nvl72_emu_bar_key& o) const noexcept {
        if (ptr != o.ptr) return ptr < o.ptr;
        if (bx != o.bx) return bx < o.bx;
        if (by != o.by) return by < o.by;
        if (bz != o.bz) return bz < o.bz;
        return epoch < o.epoch;
    }
};

struct eg_nvl72_emu_bar_state {
    int arrived = 0;
    bool completed = false;
    std::condition_variable cv;
};

inline std::mutex eg_nvl72_emu_mu;
inline std::map<eg_nvl72_emu_bar_key, std::unique_ptr<eg_nvl72_emu_bar_state>> eg_nvl72_emu_bars;

inline void eg_nvl72_emu_reset() {
    std::lock_guard<std::mutex> lock(eg_nvl72_emu_mu);
    eg_nvl72_emu_bars.clear();
}

inline void eg_nvl72_emu_barrier(
    uint32_t* mc_barrier_ptr,
    volatile uint32_t* local_barrier_ptr,
    int num_ranks,
    uint32_t target_epoch) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    const int want_ranks = num_ranks > 0 ? num_ranks : eg_nvl72_emu_num_ranks;
    std::unique_lock<std::mutex> lock(eg_nvl72_emu_mu);
    if (mc_barrier_ptr) {
        for (int r = 0; r < want_ranks; ++r) {
            mc_barrier_ptr[r] += 1u;
        }
    } else if (local_barrier_ptr) {
        *local_barrier_ptr += (uint32_t)want_ranks;
    }
    const eg_nvl72_emu_bar_key key{mc_barrier_ptr ? (const void*)mc_barrier_ptr : (const void*)local_barrier_ptr,
                                   blockIdx.x, blockIdx.y, blockIdx.z, target_epoch};
    auto& st_ptr = eg_nvl72_emu_bars[key];
    if (!st_ptr) st_ptr = std::make_unique<eg_nvl72_emu_bar_state>();
    eg_nvl72_emu_bar_state* st = st_ptr.get();
    if (++st->arrived == want_ranks) {
        st->completed = true;
        st->cv.notify_all();
    } else {
        const bool ok = st->cv.wait_for(lock, std::chrono::milliseconds(5000), [&] { return st->completed; });
        if (!ok) {
            std::fprintf(stderr, "eg_nvl72_emu_barrier timeout at epoch %u (%d/%d ranks arrived)\n",
                         target_epoch, st->arrived, want_ranks);
            std::fflush(stderr);
            std::_Exit(3);
        }
    }
    if (local_barrier_ptr && *local_barrier_ptr < (uint32_t)want_ranks * target_epoch) {
        *local_barrier_ptr = (uint32_t)want_ranks * target_epoch;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

template <typename F>
inline void eg_nvl72_emu_run(int num_ranks, dim3 grid, dim3 block, size_t smem, F body) {
    eg_emu_last_grid = grid;
    const long n = (long)block.x * block.y * block.z;
    if (num_ranks < 1 || num_ranks > 72 || n <= 0 || n > 1024 ||
        grid.x == 0 || grid.y == 0 || grid.z == 0) {
        eg_emu_fail(cudaErrorInvalidConfiguration);
        return;
    }
    gridDim = grid;
    blockDim = block;
    eg_nvl72_emu_num_ranks = num_ranks;
    eg_nvl72_emu_reset();

    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        std::vector<std::thread> rank_threads;
        rank_threads.reserve((size_t)num_ranks);
        for (int gpu_r = 0; gpu_r < num_ranks; ++gpu_r) {
            rank_threads.emplace_back([&, gpu_r, bz, by, bx] {
                eg_emu_cluster cl((int)n);
                eg_emu_block b((int)n);
                b.smem.assign(smem + 256, eg_emu_smem_fill);
                for (int w = 0; w * 32 < n; ++w) {
                    b.warps.emplace_back(new eg_emu_warp(std::min(32, (int)n - 32 * w)));
                    b.warps.back()->stalled = &b.warp_stalled;
                }
                cl.blocks.push_back(&b);
                std::vector<std::thread> threads;
                threads.reserve((size_t)n);
                for (int i = 0; i < n; ++i) {
                    threads.emplace_back([&, gpu_r, i] {
                        threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                        blockIdx = dim3(bx, by, bz);
                        eg_emu_cur_block = &b;
                        eg_emu_cur_warp = b.warps[i / 32].get();
                        eg_emu_cur_cluster = &cl;
                        eg_emu_lane = i % 32;
                        eg_emu_rank = 0;
                        eg_nvl72_emu_rank_tls = gpu_r;
                        eg_emu_cp_pending.clear();
                        eg_emu_cp_committed = 0;
                        eg_emu_wg_async = false;
                        eg_emu_wg_pending.clear();
                        eg_emu_wg_accs.clear();
                        eg_emu_wg_committed = 0;
                        body();
                        eg_emu_cur_warp->drop(eg_emu_lane);
                        {
                            std::lock_guard<std::mutex> lock(b.bar_mu);
                            --b.live_threads;
                            if (b.sync_arrived > 0 && b.sync_arrived == b.live_threads) {
                                eg_emu_sync_release(&b, b.sync_arrived);
                            }
                            eg_emu_check_deadlock(&b, "thread exit");
                            eg_emu_release_delayers(&b);
                        }
                        cl.bar.arrive_and_drop();
                    });
                }
                for (auto& t : threads) t.join();
            });
        }
        for (auto& rt : rank_threads) rt.join();
    }
}

#define EG_NVL72_SHARED(T, name, dims) EG_EMU_SHARED(T, name, dims)
#else
#define EG_NVL72_SHARED(T, name, dims) __shared__ T name dims
#endif

EG_DEV int eg_nvl72_rank() {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return eg_nvl72_emu_rank_tls;
#else
    return 0;
#endif
}

// eg_nvl72_tray_id returns the compute tray index (0..17) for rank in a
// GB200 NVL72 rack (18 trays x 4 Blackwell GPUs per tray).
EG_DEV int eg_nvl72_tray_id(int rank) {
    return rank / 4;
}

// eg_nvl72_tray_local_rank returns the intra-tray GPU index (0..3) for rank.
EG_DEV int eg_nvl72_tray_local_rank(int rank) {
    return rank % 4;
}

// eg_nvl72_same_tray reports whether rank_a and rank_b reside on the same
// 4-GPU compute tray.
EG_DEV bool eg_nvl72_same_tray(int rank_a, int rank_b) {
    return (rank_a / 4) == (rank_b / 4);
}

// eg_nvl72_multicast_st_i32 broadcasts a 32-bit signed integer to all ranks
// bound to the NVSwitch multicast pointer mc_ptr in a single instruction.
EG_DEV void eg_nvl72_multicast_st_i32(int32_t* mc_ptr, int32_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    std::atomic_ref<int32_t>(*mc_ptr).store(val, std::memory_order_release);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("multimem.st.global.s32 [%0], %1;" :: "l"(mc_ptr), "r"(val) : "memory");
#else
    *mc_ptr = val;
#endif
}

// eg_nvl72_multicast_st_f32 broadcasts an FP32 value to all ranks bound to
// the NVSwitch multicast pointer mc_ptr.
EG_DEV void eg_nvl72_multicast_st_f32(float* mc_ptr, float val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    std::atomic_ref<float>(*mc_ptr).store(val, std::memory_order_release);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("multimem.st.global.f32 [%0], %1;" :: "l"(mc_ptr), "f"(val) : "memory");
#else
    *mc_ptr = val;
#endif
}

// eg_nvl72_multicast_barrier performs an O(1) cross-rack NVSwitch multicast
// barrier across num_ranks GPUs (up to 72 GPUs across 18 trays).
// Thread 0 issues a system-scoped release fence followed by
// multimem.red.release.sys.global.add.u32 on mc_barrier_ptr, which the
// NVSwitch fabric fans out to every rank's local HBM simultaneously, and
// spins on ld.acquire.sys.global.u32 of local_barrier_ptr until it reaches
// num_ranks * target_epoch.
EG_DEV void eg_nvl72_multicast_barrier(
    uint32_t* mc_barrier_ptr,
    volatile uint32_t* local_barrier_ptr,
    int num_ranks,
    uint32_t target_epoch) {
    __syncthreads();
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_nvl72_emu_barrier(mc_barrier_ptr, local_barrier_ptr, num_ranks, target_epoch);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        asm volatile("fence.acq_rel.sys;\n\tmultimem.red.release.sys.global.add.u32 [%0], 1;"
                     :: "l"(mc_barrier_ptr) : "memory");
        const uint32_t want = (uint32_t)num_ranks * target_epoch;
        uint32_t got = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                         : "=r"(got) : "l"(local_barrier_ptr) : "memory");
        } while (got < want);
    }
#else
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        (void)mc_barrier_ptr;
        *local_barrier_ptr = (uint32_t)num_ranks * target_epoch;
    }
#endif
    __syncthreads();
}

#endif // EG_NVL72_FABRIC_HELPERS
`

// NVL72FabricHelpers returns the device and host-emulation C++/PTX helpers for
// GB200 NVL72 18-tray / 72-GPU rack topology, O(1) NVSwitch multicast barriers
// (multimem.red.release.sys.global.add.u32 + ld.acquire.sys.global.u32),
// multicast stores (multimem.st.global.s32), and the 72-rank cooperative host
// emulator runner (eg_nvl72_emu_run).
func NVL72FabricHelpers() string {
	return nvl72FabricPrelude
}

// NVL72WideEPPlan computes the deterministic MoonEP-compatible routing plan
// across R <= 72 ranks and E total experts (E % R == 0). Given topk_ids
// [R, S, K], it counts tokens_per_expert [R, E], computes per-rank local
// expert segment offsets expert_offsets [R, E/R + 1] and destination slot
// indices dest_slots [R, S, K], and broadcasts the routing plan across all R
// ranks via NVSwitch multicast stores (eg_nvl72_multicast_st_i32) followed by
// eg_nvl72_multicast_barrier.
type NVL72WideEPPlan struct {
	Name          string
	TokensPerRank int // S: tokens per rank
	TopK          int // K: routed experts per token
	TotalExperts  int // E: total experts across all R ranks (divisible by Ranks)
	Ranks         int // R: number of EP ranks (1..72)
	Threads       int // threads per CTA (0 defaults to 64 for R<=16, 32 for R>16)
}

func (k NVL72WideEPPlan) norm() (NVL72WideEPPlan, error) {
	if k.Name == "" {
		k.Name = "nvl72_wide_ep_plan"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("nvl72 wide ep plan: bad name %q", k.Name)
	}
	if k.TokensPerRank < 1 {
		return k, fmt.Errorf("nvl72 wide ep plan %s: tokensPerRank %d must be >= 1", k.Name, k.TokensPerRank)
	}
	if k.TopK < 1 || k.TopK > 32 {
		return k, fmt.Errorf("nvl72 wide ep plan %s: topK %d must be in 1..32", k.Name, k.TopK)
	}
	if k.Ranks < 1 || k.Ranks > NVL72MaxRanks {
		return k, fmt.Errorf("nvl72 wide ep plan %s: ranks %d must be in 1..%d", k.Name, k.Ranks, NVL72MaxRanks)
	}
	if k.TotalExperts < k.Ranks || k.TotalExperts%k.Ranks != 0 {
		return k, fmt.Errorf("nvl72 wide ep plan %s: totalExperts %d must be a positive multiple of ranks %d", k.Name, k.TotalExperts, k.Ranks)
	}
	if k.Threads == 0 {
		if k.Ranks > 16 {
			k.Threads = 32
		} else {
			k.Threads = 64
		}
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("nvl72 wide ep plan %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

// MaxSlotsPerRank returns the worst-case number of dispatched token slots a
// destination rank can receive across all R ranks (R * S * K).
func (k NVL72WideEPPlan) MaxSlotsPerRank() int {
	return k.Ranks * k.TokensPerRank * k.TopK
}

func (k NVL72WideEPPlan) params() []string {
	return []string{
		"const int32_t* __restrict__ topk_ids",
		"int32_t* __restrict__ tokens_per_expert",
		"int32_t* __restrict__ expert_offsets",
		"int32_t* __restrict__ dest_slots",
		"uint32_t* __restrict__ mc_barrier",
		"uint32_t base_epoch",
	}
}

// Grid returns the per-rank cooperative launch grid and block.
func (k NVL72WideEPPlan) Grid() (grid, block [3]int) {
	normed, err := k.norm()
	if err != nil {
		return [3]int{1, 1, 1}, [3]int{32, 1, 1}
	}
	return [3]int{1, 1, 1}, [3]int{normed.Threads, 1, 1}
}

// CUDA returns the kernel definition for NVL72WideEPPlan.
func (k NVL72WideEPPlan) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	epn := k.TotalExperts / k.Ranks
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank GB200 NVL72 Wide-EP routing plan (%d experts, top-%d, S=%d).\n",
		k.Name, k.Ranks, k.TotalExperts, k.TopK, k.TokensPerRank)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int R = %d, S = %d, K = %d, E = %d, EPN = %d, THREADS = %d, N = S * K;\n",
		k.Ranks, k.TokensPerRank, k.TopK, k.TotalExperts, epn, k.Threads)
	b.WriteString(`    const int r = eg_nvl72_rank();
    const int tid = threadIdx.x;
    const int tray_id = eg_nvl72_tray_id(r);
    const int tray_local = eg_nvl72_tray_local_rank(r);
    (void)tray_id;
    (void)tray_local;

    // Phase 1: each rank r counts tokens_per_expert[r, e] for its local routes
    // and broadcasts its row across all R ranks via NVSwitch multicast stores.
    const int32_t* my_ids = topk_ids + (size_t)r * N;
    for (int e = tid; e < E; e += THREADS) {
        int32_t count = 0;
        for (int i = 0; i < N; ++i) {
            if (my_ids[i] == e) {
                ++count;
            }
        }
        eg_nvl72_multicast_st_i32(tokens_per_expert + (size_t)r * E + e, count);
    }

    // O(1) NVSwitch multicast barrier: all 72 ranks observe the complete [R, E]
    // tokens_per_expert matrix before computing prefix offsets.
    eg_nvl72_multicast_barrier(mc_barrier, mc_barrier + r, R, base_epoch + 1u);

    // Phase 2a: each rank r computes its local expert segment offsets
    // expert_offsets[r, 0..EPN] and broadcasts them via multicast store.
    if (tid == 0) {
        int32_t running = 0;
        eg_nvl72_multicast_st_i32(expert_offsets + (size_t)r * (EPN + 1) + 0, 0);
        for (int le = 0; le < EPN; ++le) {
            const int e = r * EPN + le;
            int32_t tot_e = 0;
            for (int src_r = 0; src_r < R; ++src_r) {
                tot_e += tokens_per_expert[(size_t)src_r * E + e];
            }
            running += tot_e;
            eg_nvl72_multicast_st_i32(expert_offsets + (size_t)r * (EPN + 1) + le + 1, running);
        }
    }

    // Phase 2b: each rank r computes deterministic destination slot offsets
    // (dest_rank, dest_expert_local, slot_idx) for its local routes (r, s, k)
    // and broadcasts dest_slots[r, s, k] via multicast store.
    for (int i = tid; i < N; i += THREADS) {
        const int e = my_ids[i];
        int32_t slot_idx = -1;
        if (e >= 0 && e < E) {
            const int dest_rank = e / EPN;
            const int dest_expert_local = e % EPN;
            int32_t base_off = 0;
            for (int le = 0; le < dest_expert_local; ++le) {
                const int prev_e = dest_rank * EPN + le;
                for (int src_r = 0; src_r < R; ++src_r) {
                    base_off += tokens_per_expert[(size_t)src_r * E + prev_e];
                }
            }
            int32_t rank_off = 0;
            for (int src_r = 0; src_r < r; ++src_r) {
                rank_off += tokens_per_expert[(size_t)src_r * E + e];
            }
            int32_t local_off = 0;
            for (int j = 0; j < i; ++j) {
                if (my_ids[j] == e) {
                    ++local_off;
                }
            }
            slot_idx = base_off + rank_off + local_off;
        }
        eg_nvl72_multicast_st_i32(dest_slots + (size_t)r * N + i, slot_idx);
    }

    // O(1) NVSwitch multicast barrier: ensure routing table and per-expert
    // counts are visible across all R ranks before token dispatch.
    eg_nvl72_multicast_barrier(mc_barrier, mc_barrier + r, R, base_epoch + 2u);
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for NVL72WideEPPlan.
func (k NVL72WideEPPlan) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := strings.Join(paramNames(params), ", ")
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "#if defined(EG_HOST_EMU) || defined(EG_EMU)\n")
	fmt.Fprintf(&b, "    (void)stream;\n")
	fmt.Fprintf(&b, "    eg_nvl72_emu_run(%d, dim3(1), dim3(%d), 0, [&] { %s(%s); });\n", k.Ranks, k.Threads, k.Name, names)
	fmt.Fprintf(&b, "#else\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, dim3(1), dim3(%d), 0, stream, %s);\n", k.Name, k.Threads, names)
	fmt.Fprintf(&b, "#endif\n}\n")
	return b.String(), nil
}

// Source returns a complete translation unit for NVL72WideEPPlan.
func (k NVL72WideEPPlan) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + NVL72FabricHelpers() + "\n" + src + "\n" + l, nil
}

// Reference computes the Go reference outputs for NVL72WideEPPlan:
//   - tokensPerExpert [R * E]
//   - expertOffsets   [R * (E/R + 1)]
//   - destSlots       [R * S * K]
func (k NVL72WideEPPlan) Reference(topkIDs []int32) (tokensPerExpert, expertOffsets, destSlots []int32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	n := k.TokensPerRank * k.TopK
	if len(topkIDs) != k.Ranks*n {
		return nil, nil, nil, fmt.Errorf("nvl72 wide ep plan %s reference: topkIDs len %d != %d", k.Name, len(topkIDs), k.Ranks*n)
	}
	epn := k.TotalExperts / k.Ranks
	tokensPerExpert = make([]int32, k.Ranks*k.TotalExperts)
	expertOffsets = make([]int32, k.Ranks*(epn+1))
	destSlots = make([]int32, k.Ranks*n)

	for r := 0; r < k.Ranks; r++ {
		for i := 0; i < n; i++ {
			e := int(topkIDs[r*n+i])
			if e >= 0 && e < k.TotalExperts {
				tokensPerExpert[r*k.TotalExperts+e]++
			}
		}
	}

	for destRank := 0; destRank < k.Ranks; destRank++ {
		var running int32
		expertOffsets[destRank*(epn+1)+0] = 0
		for le := 0; le < epn; le++ {
			e := destRank*epn + le
			var tot int32
			for srcRank := 0; srcRank < k.Ranks; srcRank++ {
				tot += tokensPerExpert[srcRank*k.TotalExperts+e]
			}
			running += tot
			expertOffsets[destRank*(epn+1)+le+1] = running
		}
	}

	for r := 0; r < k.Ranks; r++ {
		for i := 0; i < n; i++ {
			e := int(topkIDs[r*n+i])
			if e < 0 || e >= k.TotalExperts {
				destSlots[r*n+i] = -1
				continue
			}
			destRank := e / epn
			localE := e % epn
			baseOff := expertOffsets[destRank*(epn+1)+localE]
			var rankOff int32
			for srcRank := 0; srcRank < r; srcRank++ {
				rankOff += tokensPerExpert[srcRank*k.TotalExperts+e]
			}
			var localOff int32
			for j := 0; j < i; j++ {
				if int(topkIDs[r*n+j]) == e {
					localOff++
				}
			}
			destSlots[r*n+i] = baseOff + rankOff + localOff
		}
	}
	return tokensPerExpert, expertOffsets, destSlots, nil
}

// NVL72WideEPDispatch pushes each rank's local input tokens x [R, S, H] over
// NVLink 5 directly into dest_rank's symmetric receive buffer
// sym_recv_x [R, MaxSlots, H] with optional fused quantization scaling, while
// recording provenance src_info [R, MaxSlots] = (r * S + s) * K + k and
// recv_weights [R, MaxSlots] = topk_weights[r, s, k].
type NVL72WideEPDispatch struct {
	Name          string
	TokensPerRank int     // S: tokens per rank
	HiddenSize    int     // H: hidden dimension
	TopK          int     // K: routed experts per token
	TotalExperts  int     // E: total experts across all R ranks
	Ranks         int     // R: number of EP ranks (1..72)
	DType         DType   // F32 or BF16
	QuantizeScale float32 // fused scale applied on dispatch (0 defaults to 1.0)
	Threads       int     // threads per CTA
}

func (k NVL72WideEPDispatch) norm() (NVL72WideEPDispatch, error) {
	if k.Name == "" {
		k.Name = "nvl72_wide_ep_dispatch"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("nvl72 wide ep dispatch: bad name %q", k.Name)
	}
	if k.TokensPerRank < 1 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: tokensPerRank %d must be >= 1", k.Name, k.TokensPerRank)
	}
	if k.HiddenSize < 2 || k.HiddenSize%2 != 0 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: hiddenSize %d must be a positive multiple of 2", k.Name, k.HiddenSize)
	}
	if k.TopK < 1 || k.TopK > 32 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: topK %d must be in 1..32", k.Name, k.TopK)
	}
	if k.Ranks < 1 || k.Ranks > NVL72MaxRanks {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: ranks %d must be in 1..%d", k.Name, k.Ranks, NVL72MaxRanks)
	}
	if k.TotalExperts < k.Ranks || k.TotalExperts%k.Ranks != 0 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: totalExperts %d must be a positive multiple of ranks %d", k.Name, k.TotalExperts, k.Ranks)
	}
	if k.DType == 0 {
		k.DType = F32
	}
	if k.DType != F32 && k.DType != BF16 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: dtype %v must be F32 or BF16", k.Name, k.DType)
	}
	if k.QuantizeScale == 0 {
		k.QuantizeScale = 1.0
	}
	if k.Threads == 0 {
		if k.Ranks > 16 {
			k.Threads = 32
		} else {
			k.Threads = 64
		}
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("nvl72 wide ep dispatch %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

func (k NVL72WideEPDispatch) params() []string {
	ct := k.DType.cType()
	return []string{
		"const " + ct + "* __restrict__ x",
		"const int32_t* __restrict__ topk_ids",
		"const float* __restrict__ topk_weights",
		"const int32_t* __restrict__ dest_slots",
		ct + "* __restrict__ sym_recv_x",
		"int32_t* __restrict__ src_info",
		"float* __restrict__ recv_weights",
		"uint32_t* __restrict__ mc_barrier",
		"uint32_t target_epoch",
	}
}

// Grid returns the per-rank cooperative launch grid and block.
func (k NVL72WideEPDispatch) Grid() (grid, block [3]int) {
	normed, err := k.norm()
	if err != nil {
		return [3]int{1, 1, 1}, [3]int{32, 1, 1}
	}
	return [3]int{1, 1, 1}, [3]int{normed.Threads, 1, 1}
}

// CUDA returns the kernel definition for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ct := k.DType.cType()
	epn := k.TotalExperts / k.Ranks
	maxSlots := k.Ranks * k.TokensPerRank * k.TopK
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank GB200 NVL72 Wide-EP NVLink 5 token push (S=%d, H=%d, K=%d).\n",
		k.Name, k.Ranks, k.TokensPerRank, k.HiddenSize, k.TopK)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int R = %d, S = %d, H = %d, K = %d, E = %d, EPN = %d, MAX_SLOTS = %d, THREADS = %d;\n",
		k.Ranks, k.TokensPerRank, k.HiddenSize, k.TopK, k.TotalExperts, epn, maxSlots, k.Threads)
	fmt.Fprintf(&b, "    constexpr float QSCALE = %s;\n", floatLiteral(k.QuantizeScale))
	b.WriteString(`    const int r = eg_nvl72_rank();
    const int tid = threadIdx.x;

    for (int s = 0; s < S; ++s) {
        const ` + ct + `* token_x = x + ((size_t)r * S + (size_t)s) * H;
        for (int k = 0; k < K; ++k) {
            const size_t route_idx = ((size_t)r * S + (size_t)s) * K + (size_t)k;
            const int e = topk_ids[route_idx];
            const int slot = dest_slots[route_idx];
            if (e < 0 || e >= E || slot < 0 || slot >= MAX_SLOTS) {
                continue;
            }
            const int dest_rank = e / EPN;
            const size_t dst_slot_idx = (size_t)dest_rank * MAX_SLOTS + (size_t)slot;
            ` + ct + `* dst_row = sym_recv_x + dst_slot_idx * H;
            for (int c = tid; c < H; c += THREADS) {
`)
	if k.DType == F32 {
		b.WriteString(`                const float v = token_x[c] * QSCALE;
                dst_row[c] = v;
`)
	} else {
		b.WriteString(`                const float v = eg_bf16_to_f32(token_x[c]) * QSCALE;
                dst_row[c] = eg_f32_to_bf16(v);
`)
	}
	b.WriteString(`            }
            if (tid == 0) {
                src_info[dst_slot_idx] = (int32_t)route_idx;
                recv_weights[dst_slot_idx] = topk_weights[route_idx];
            }
        }
    }

    // O(1) NVSwitch multicast barrier: ensure all pushed token activations and
    // provenance metadata have landed in every destination rank's HBM3e.
    eg_nvl72_multicast_barrier(mc_barrier, mc_barrier + r, R, target_epoch);
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := strings.Join(paramNames(params), ", ")
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "#if defined(EG_HOST_EMU) || defined(EG_EMU)\n")
	fmt.Fprintf(&b, "    (void)stream;\n")
	fmt.Fprintf(&b, "    eg_nvl72_emu_run(%d, dim3(1), dim3(%d), 0, [&] { %s(%s); });\n", k.Ranks, k.Threads, k.Name, names)
	fmt.Fprintf(&b, "#else\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, dim3(1), dim3(%d), 0, stream, %s);\n", k.Name, k.Threads, names)
	fmt.Fprintf(&b, "#endif\n}\n")
	return b.String(), nil
}

// Source returns a complete translation unit for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + NVL72FabricHelpers() + "\n" + src + "\n" + l, nil
}

// Reference computes the Go float64 reference for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Reference(
	x []float64,
	topkIDs []int32,
	topkWeights []float32,
	destSlots []int32,
) (symRecvX []float64, srcInfo []int32, recvWeights []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	n := k.TokensPerRank * k.TopK
	maxSlots := k.Ranks * n
	if len(x) != k.Ranks*k.TokensPerRank*k.HiddenSize {
		return nil, nil, nil, fmt.Errorf("nvl72 wide ep dispatch %s reference: x len %d != %d", k.Name, len(x), k.Ranks*k.TokensPerRank*k.HiddenSize)
	}
	if len(topkIDs) != k.Ranks*n || len(topkWeights) != k.Ranks*n || len(destSlots) != k.Ranks*n {
		return nil, nil, nil, fmt.Errorf("nvl72 wide ep dispatch %s reference: route slice length mismatch", k.Name)
	}
	epn := k.TotalExperts / k.Ranks
	symRecvX = make([]float64, k.Ranks*maxSlots*k.HiddenSize)
	srcInfo = make([]int32, k.Ranks*maxSlots)
	for i := range srcInfo {
		srcInfo[i] = -1
	}
	recvWeights = make([]float32, k.Ranks*maxSlots)

	for r := 0; r < k.Ranks; r++ {
		for s := 0; s < k.TokensPerRank; s++ {
			for rt := 0; rt < k.TopK; rt++ {
				routeIdx := (r*k.TokensPerRank+s)*k.TopK + rt
				e := int(topkIDs[routeIdx])
				slot := int(destSlots[routeIdx])
				if e < 0 || e >= k.TotalExperts || slot < 0 || slot >= maxSlots {
					continue
				}
				destRank := e / epn
				dstSlotIdx := destRank*maxSlots + slot
				srcInfo[dstSlotIdx] = int32(routeIdx)
				recvWeights[dstSlotIdx] = topkWeights[routeIdx]
				for c := 0; c < k.HiddenSize; c++ {
					xv := k.DType.Round(float32(x[(r*k.TokensPerRank+s)*k.HiddenSize+c]))
					scaled := k.DType.Round(xv * k.QuantizeScale)
					symRecvX[dstSlotIdx*k.HiddenSize+c] = float64(scaled)
				}
			}
		}
	}
	return symRecvX, srcInfo, recvWeights, nil
}

// NVL72WideEPExpertScale applies per-expert transformation on each rank's
// received token slots [expert_offsets[r, le] .. expert_offsets[r, le+1]),
// writing hot experts (cold_expert_mask[e] == 0) to HBM3e expert_out and cold
// experts (cold_expert_mask[e] != 0) to Grace CPU LPDDR5X coherent memory
// c2c_cold_expert_out for NVLink-C2C fallback.
type NVL72WideEPExpertScale struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	DType         DType
	Threads       int
}

func (k NVL72WideEPExpertScale) norm() (NVL72WideEPExpertScale, error) {
	if k.Name == "" {
		k.Name = "nvl72_wide_ep_expert_scale"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("nvl72 wide ep expert scale: bad name %q", k.Name)
	}
	if k.TokensPerRank < 1 || k.HiddenSize < 2 || k.HiddenSize%2 != 0 {
		return k, fmt.Errorf("nvl72 wide ep expert scale %s: invalid S=%d H=%d", k.Name, k.TokensPerRank, k.HiddenSize)
	}
	if k.TopK < 1 || k.TopK > 32 || k.Ranks < 1 || k.Ranks > NVL72MaxRanks {
		return k, fmt.Errorf("nvl72 wide ep expert scale %s: invalid K=%d R=%d", k.Name, k.TopK, k.Ranks)
	}
	if k.TotalExperts < k.Ranks || k.TotalExperts%k.Ranks != 0 {
		return k, fmt.Errorf("nvl72 wide ep expert scale %s: E=%d must be divisible by R=%d", k.Name, k.TotalExperts, k.Ranks)
	}
	if k.DType == 0 {
		k.DType = F32
	}
	if k.DType != F32 && k.DType != BF16 {
		return k, fmt.Errorf("nvl72 wide ep expert scale %s: dtype %v must be F32 or BF16", k.Name, k.DType)
	}
	if k.Threads == 0 {
		if k.Ranks > 16 {
			k.Threads = 32
		} else {
			k.Threads = 64
		}
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("nvl72 wide ep expert scale %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

func (k NVL72WideEPExpertScale) params() []string {
	ct := k.DType.cType()
	return []string{
		"const " + ct + "* __restrict__ sym_recv_x",
		"const int32_t* __restrict__ expert_offsets",
		"const " + ct + "* __restrict__ expert_scales",
		"const int32_t* __restrict__ cold_expert_mask",
		ct + "* __restrict__ expert_out",
		ct + "* __restrict__ c2c_cold_expert_out",
		"uint32_t* __restrict__ mc_barrier",
		"uint32_t target_epoch",
	}
}

// CUDA returns the kernel definition for NVL72WideEPExpertScale.
func (k NVL72WideEPExpertScale) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ct := k.DType.cType()
	epn := k.TotalExperts / k.Ranks
	maxSlots := k.Ranks * k.TokensPerRank * k.TopK
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank local expert scale + Grace NVLink-C2C cold-expert offload.\n", k.Name, k.Ranks)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int R = %d, H = %d, EPN = %d, MAX_SLOTS = %d, THREADS = %d;\n",
		k.Ranks, k.HiddenSize, epn, maxSlots, k.Threads)
	b.WriteString(`    const int r = eg_nvl72_rank();
    const int tid = threadIdx.x;

    for (int le = 0; le < EPN; ++le) {
        const int e = r * EPN + le;
        const int32_t start_slot = expert_offsets[(size_t)r * (EPN + 1) + le];
        const int32_t end_slot = expert_offsets[(size_t)r * (EPN + 1) + le + 1];
        const bool is_cold = (cold_expert_mask != nullptr) && (cold_expert_mask[e] != 0);
        const ` + ct + `* scale_row = expert_scales + (size_t)e * H;
        for (int slot = start_slot; slot < end_slot; ++slot) {
            const size_t slot_idx = (size_t)r * MAX_SLOTS + (size_t)slot;
            const ` + ct + `* in_row = sym_recv_x + slot_idx * H;
            ` + ct + `* out_row = (is_cold ? c2c_cold_expert_out : expert_out) + slot_idx * H;
            for (int c = tid; c < H; c += THREADS) {
`)
	if k.DType == F32 {
		b.WriteString(`                out_row[c] = in_row[c] * scale_row[c];
`)
	} else {
		b.WriteString(`                const float xv = eg_bf16_to_f32(in_row[c]);
                const float sv = eg_bf16_to_f32(scale_row[c]);
                out_row[c] = eg_f32_to_bf16(xv * sv);
`)
	}
	b.WriteString(`            }
        }
    }

    eg_nvl72_multicast_barrier(mc_barrier, mc_barrier + r, R, target_epoch);
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for NVL72WideEPExpertScale.
func (k NVL72WideEPExpertScale) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := strings.Join(paramNames(params), ", ")
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "#if defined(EG_HOST_EMU) || defined(EG_EMU)\n")
	fmt.Fprintf(&b, "    (void)stream;\n")
	fmt.Fprintf(&b, "    eg_nvl72_emu_run(%d, dim3(1), dim3(%d), 0, [&] { %s(%s); });\n", k.Ranks, k.Threads, k.Name, names)
	fmt.Fprintf(&b, "#else\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, dim3(1), dim3(%d), 0, stream, %s);\n", k.Name, k.Threads, names)
	fmt.Fprintf(&b, "#endif\n}\n")
	return b.String(), nil
}

// NVL72WideEPCombine gathers the K routed expert outputs for each token (r, s)
// across up to 72 ranks over NVLink 5 (900 GB/s per GPU) or from Grace CPU
// LPDDR5X coherent memory c2c_cold_expert_out over NVLink-C2C (450 GB/s per
// GPU) when an expert is flagged cold/offloaded, multiplies by
// topk_weights[r, s, k], adds residual[r, s, H], and applies fused RMSNorm into
// norm_out[r, s, H].
type NVL72WideEPCombine struct {
	Name          string
	TokensPerRank int     // S: tokens per rank
	HiddenSize    int     // H: hidden dimension
	TopK          int     // K: routed experts per token
	TotalExperts  int     // E: total experts across all R ranks
	Ranks         int     // R: number of EP ranks (1..72)
	Eps           float32 // RMSNorm epsilon (0 defaults to 1e-5)
	DType         DType   // F32 or BF16
	Threads       int     // threads per CTA
}

func (k NVL72WideEPCombine) norm() (NVL72WideEPCombine, error) {
	if k.Name == "" {
		k.Name = "nvl72_wide_ep_combine"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("nvl72 wide ep combine: bad name %q", k.Name)
	}
	if k.TokensPerRank < 1 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: tokensPerRank %d must be >= 1", k.Name, k.TokensPerRank)
	}
	if k.HiddenSize < 2 || k.HiddenSize%2 != 0 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: hiddenSize %d must be a positive multiple of 2", k.Name, k.HiddenSize)
	}
	if k.TopK < 1 || k.TopK > 32 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: topK %d must be in 1..32", k.Name, k.TopK)
	}
	if k.Ranks < 1 || k.Ranks > NVL72MaxRanks {
		return k, fmt.Errorf("nvl72 wide ep combine %s: ranks %d must be in 1..%d", k.Name, k.Ranks, NVL72MaxRanks)
	}
	if k.TotalExperts < k.Ranks || k.TotalExperts%k.Ranks != 0 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: totalExperts %d must be a positive multiple of ranks %d", k.Name, k.TotalExperts, k.Ranks)
	}
	if k.Eps == 0 {
		k.Eps = 1e-5
	}
	if k.DType == 0 {
		k.DType = F32
	}
	if k.DType != F32 && k.DType != BF16 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: dtype %v must be F32 or BF16", k.Name, k.DType)
	}
	if k.Threads == 0 {
		if k.Ranks > 16 {
			k.Threads = 32
		} else {
			k.Threads = 64
		}
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("nvl72 wide ep combine %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

func (k NVL72WideEPCombine) params() []string {
	ct := k.DType.cType()
	return []string{
		"const " + ct + "* __restrict__ expert_out",
		"const " + ct + "* __restrict__ c2c_cold_expert_out",
		"const int32_t* __restrict__ cold_expert_mask",
		"const int32_t* __restrict__ topk_ids",
		"const float* __restrict__ topk_weights",
		"const int32_t* __restrict__ dest_slots",
		"const " + ct + "* __restrict__ residual",
		"const " + ct + "* __restrict__ rms_weight",
		ct + "* __restrict__ residual_out",
		ct + "* __restrict__ norm_out",
		"uint32_t* __restrict__ mc_barrier",
		"uint32_t target_epoch",
	}
}

// Grid returns the per-rank cooperative launch grid and block.
func (k NVL72WideEPCombine) Grid() (grid, block [3]int) {
	normed, err := k.norm()
	if err != nil {
		return [3]int{1, 1, 1}, [3]int{32, 1, 1}
	}
	return [3]int{1, 1, 1}, [3]int{normed.Threads, 1, 1}
}

// CUDA returns the kernel definition for NVL72WideEPCombine.
func (k NVL72WideEPCombine) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ct := k.DType.cType()
	epn := k.TotalExperts / k.Ranks
	maxSlots := k.Ranks * k.TokensPerRank * k.TopK
	per := (k.HiddenSize + k.Threads - 1) / k.Threads
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank GB200 NVL72 Wide-EP NVLink 5 + Grace C2C combine + residual RMSNorm.\n",
		k.Name, k.Ranks)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int R = %d, S = %d, H = %d, K = %d, E = %d, EPN = %d, MAX_SLOTS = %d, THREADS = %d, PER = %d;\n",
		k.Ranks, k.TokensPerRank, k.HiddenSize, k.TopK, k.TotalExperts, epn, maxSlots, k.Threads, per)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    const int r = eg_nvl72_rank();
    const int tid = threadIdx.x;
    EG_NVL72_SHARED(float, eg_red, [32]);

    for (int s = 0; s < S; ++s) {
        const size_t token_idx = (size_t)r * S + (size_t)s;
        const ` + ct + `* res_row = residual + token_idx * H;
        ` + ct + `* res_out_row = residual_out + token_idx * H;
        ` + ct + `* norm_out_row = norm_out + token_idx * H;

        float h_reg[PER];
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int c = tid + j * THREADS;
            if (c < H) {
                float acc = 0.0f;
#pragma unroll
                for (int rt = 0; rt < K; ++rt) {
                    const size_t route_idx = token_idx * K + (size_t)rt;
                    const int e = topk_ids[route_idx];
                    const int slot = dest_slots[route_idx];
                    if (e < 0 || e >= E || slot < 0 || slot >= MAX_SLOTS) {
                        continue;
                    }
                    const int dest_rank = e / EPN;
                    const float w = topk_weights[route_idx];
                    const bool is_cold = (cold_expert_mask != nullptr) && (cold_expert_mask[e] != 0);
                    const ` + ct + `* src_buf = is_cold ? c2c_cold_expert_out : expert_out;
                    const size_t elem_off = ((size_t)dest_rank * MAX_SLOTS + (size_t)slot) * H + (size_t)c;
`)
	if k.DType == F32 {
		b.WriteString(`                    const float ev = src_buf[elem_off];
                    acc += ev * w;
`)
	} else {
		b.WriteString(`                    const float ev = eg_bf16_to_f32(src_buf[elem_off]);
                    acc += ev * w;
`)
	}
	b.WriteString(`                }
`)
	if k.DType == F32 {
		b.WriteString(`                const float res_v = res_row[c];
                const float h = acc + res_v;
                res_out_row[c] = h;
                h_reg[j] = h;
                sq += h * h;
`)
	} else {
		b.WriteString(`                const float res_v = eg_bf16_to_f32(res_row[c]);
                const float h_raw = acc + res_v;
                const uint16_t h_bf16 = eg_f32_to_bf16(h_raw);
                res_out_row[c] = h_bf16;
                const float h = eg_bf16_to_f32(h_bf16);
                h_reg[j] = h;
                sq += h * h;
`)
	}
	b.WriteString(`            } else {
                h_reg[j] = 0.0f;
            }
        }

        sq = eg_warp_sum(sq);
        if constexpr (THREADS > 32) {
            sq = eg_block_combine<0>(sq, eg_red);
        }
        const float inv_rms = rsqrtf(sq / (float)H + EPS);

#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int c = tid + j * THREADS;
            if (c < H) {
`)
	if k.DType == F32 {
		b.WriteString(`                const float rw = rms_weight[c];
                norm_out_row[c] = h_reg[j] * inv_rms * rw;
`)
	} else {
		b.WriteString(`                const float rw = eg_bf16_to_f32(rms_weight[c]);
                norm_out_row[c] = eg_f32_to_bf16(h_reg[j] * inv_rms * rw);
`)
	}
	b.WriteString(`            }
        }
    }

    eg_nvl72_multicast_barrier(mc_barrier, mc_barrier + r, R, target_epoch);
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := strings.Join(paramNames(params), ", ")
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "#if defined(EG_HOST_EMU) || defined(EG_EMU)\n")
	fmt.Fprintf(&b, "    (void)stream;\n")
	fmt.Fprintf(&b, "    eg_nvl72_emu_run(%d, dim3(1), dim3(%d), 0, [&] { %s(%s); });\n", k.Ranks, k.Threads, k.Name, names)
	fmt.Fprintf(&b, "#else\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, dim3(1), dim3(%d), 0, stream, %s);\n", k.Name, k.Threads, names)
	fmt.Fprintf(&b, "#endif\n}\n")
	return b.String(), nil
}

// Source returns a complete translation unit for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + NVL72FabricHelpers() + "\n" + src + "\n" + l, nil
}

// Reference computes the Go float64 reference for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Reference(
	expertOut []float64,
	c2cColdExpertOut []float64,
	coldExpertMask []int32,
	topkIDs []int32,
	topkWeights []float32,
	destSlots []int32,
	residual []float64,
	rmsWeight []float64,
) (residualOut, normOut []float64, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	n := k.TokensPerRank * k.TopK
	maxSlots := k.Ranks * n
	totalTokenElems := k.Ranks * k.TokensPerRank * k.HiddenSize
	if len(expertOut) != k.Ranks*maxSlots*k.HiddenSize {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: expertOut len %d != %d", k.Name, len(expertOut), k.Ranks*maxSlots*k.HiddenSize)
	}
	if len(c2cColdExpertOut) != 0 && len(c2cColdExpertOut) != k.Ranks*maxSlots*k.HiddenSize {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: c2cColdExpertOut len %d != %d", k.Name, len(c2cColdExpertOut), k.Ranks*maxSlots*k.HiddenSize)
	}
	if len(coldExpertMask) != 0 && len(coldExpertMask) != k.TotalExperts {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: coldExpertMask len %d != %d", k.Name, len(coldExpertMask), k.TotalExperts)
	}
	if len(topkIDs) != k.Ranks*n || len(topkWeights) != k.Ranks*n || len(destSlots) != k.Ranks*n {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: route slice length mismatch", k.Name)
	}
	if len(residual) != totalTokenElems {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: residual len %d != %d", k.Name, len(residual), totalTokenElems)
	}
	if len(rmsWeight) != k.HiddenSize {
		return nil, nil, fmt.Errorf("nvl72 wide ep combine %s reference: rmsWeight len %d != %d", k.Name, len(rmsWeight), k.HiddenSize)
	}

	epn := k.TotalExperts / k.Ranks
	residualOut = make([]float64, totalTokenElems)
	normOut = make([]float64, totalTokenElems)

	for r := 0; r < k.Ranks; r++ {
		for s := 0; s < k.TokensPerRank; s++ {
			tokenIdx := r*k.TokensPerRank + s
			hRow := make([]float32, k.HiddenSize)
			var sqSum float32
			for c := 0; c < k.HiddenSize; c++ {
				var acc float32
				for rt := 0; rt < k.TopK; rt++ {
					routeIdx := tokenIdx*k.TopK + rt
					e := int(topkIDs[routeIdx])
					slot := int(destSlots[routeIdx])
					if e < 0 || e >= k.TotalExperts || slot < 0 || slot >= maxSlots {
						continue
					}
					destRank := e / epn
					w := topkWeights[routeIdx]
					isCold := len(coldExpertMask) > 0 && coldExpertMask[e] != 0
					srcBuf := expertOut
					if isCold {
						srcBuf = c2cColdExpertOut
					}
					elemOff := (destRank*maxSlots+slot)*k.HiddenSize + c
					ev := k.DType.Round(float32(srcBuf[elemOff]))
					acc += ev * w
				}
				resV := k.DType.Round(float32(residual[tokenIdx*k.HiddenSize+c]))
				hRaw := acc + resV
				hRounded := k.DType.Round(hRaw)
				residualOut[tokenIdx*k.HiddenSize+c] = float64(hRounded)
				hRow[c] = hRounded
				sqSum += hRounded * hRounded
			}
			meanSq := sqSum/float32(k.HiddenSize) + k.Eps
			invRMS := float32(1.0) / float32(math.Sqrt(float64(meanSq)))
			for c := 0; c < k.HiddenSize; c++ {
				rw := k.DType.Round(float32(rmsWeight[c]))
				y := k.DType.Round(hRow[c] * invRMS * rw)
				normOut[tokenIdx*k.HiddenSize+c] = float64(y)
			}
		}
	}
	return residualOut, normOut, nil
}

// NVL72WideEP bundles the complete 4-stage GB200 NVL72 Wide-EP pipeline:
//
//	NVL72WideEPPlan -> NVL72WideEPDispatch -> NVL72WideEPExpertScale -> NVL72WideEPCombine
type NVL72WideEP struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	Eps           float32
	DType         DType
	QuantizeScale float32
	Threads       int
}

// Stages returns the four constituent kernel descriptors for w.
func (w NVL72WideEP) Stages() (NVL72WideEPPlan, NVL72WideEPDispatch, NVL72WideEPExpertScale, NVL72WideEPCombine, error) {
	prefix := w.Name
	if prefix == "" {
		prefix = "nvl72_wide_ep"
	}
	plan, err := (NVL72WideEPPlan{
		Name:          prefix + "_plan",
		TokensPerRank: w.TokensPerRank,
		TopK:          w.TopK,
		TotalExperts:  w.TotalExperts,
		Ranks:         w.Ranks,
		Threads:       w.Threads,
	}).norm()
	if err != nil {
		return NVL72WideEPPlan{}, NVL72WideEPDispatch{}, NVL72WideEPExpertScale{}, NVL72WideEPCombine{}, err
	}
	disp, err := (NVL72WideEPDispatch{
		Name:          prefix + "_dispatch",
		TokensPerRank: w.TokensPerRank,
		HiddenSize:    w.HiddenSize,
		TopK:          w.TopK,
		TotalExperts:  w.TotalExperts,
		Ranks:         w.Ranks,
		DType:         w.DType,
		QuantizeScale: w.QuantizeScale,
		Threads:       w.Threads,
	}).norm()
	if err != nil {
		return NVL72WideEPPlan{}, NVL72WideEPDispatch{}, NVL72WideEPExpertScale{}, NVL72WideEPCombine{}, err
	}
	scale, err := (NVL72WideEPExpertScale{
		Name:          prefix + "_expert_scale",
		TokensPerRank: w.TokensPerRank,
		HiddenSize:    w.HiddenSize,
		TopK:          w.TopK,
		TotalExperts:  w.TotalExperts,
		Ranks:         w.Ranks,
		DType:         w.DType,
		Threads:       w.Threads,
	}).norm()
	if err != nil {
		return NVL72WideEPPlan{}, NVL72WideEPDispatch{}, NVL72WideEPExpertScale{}, NVL72WideEPCombine{}, err
	}
	comb, err := (NVL72WideEPCombine{
		Name:          prefix + "_combine",
		TokensPerRank: w.TokensPerRank,
		HiddenSize:    w.HiddenSize,
		TopK:          w.TopK,
		TotalExperts:  w.TotalExperts,
		Ranks:         w.Ranks,
		Eps:           w.Eps,
		DType:         w.DType,
		Threads:       w.Threads,
	}).norm()
	if err != nil {
		return NVL72WideEPPlan{}, NVL72WideEPDispatch{}, NVL72WideEPExpertScale{}, NVL72WideEPCombine{}, err
	}
	return plan, disp, scale, comb, nil
}

// Source returns a single translation unit containing [Prelude],
// [NVL72FabricHelpers], and all four pipeline kernels and their launchers.
func (w NVL72WideEP) Source() (string, error) {
	plan, disp, scale, comb, err := w.Stages()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteByte('\n')
	b.WriteString(NVL72FabricHelpers())
	b.WriteByte('\n')
	for _, fn := range []func() (string, error){
		plan.CUDA, plan.Launcher,
		disp.CUDA, disp.Launcher,
		scale.CUDA, scale.Launcher,
		comb.CUDA, comb.Launcher,
	} {
		s, err := fn()
		if err != nil {
			return "", err
		}
		b.WriteString(s)
		b.WriteByte('\n')
	}
	return b.String(), nil
}

// Reference executes the full 4-stage float64 reference pipeline:
//
//	Plan -> Dispatch -> ExpertScale (HBM3e hot + Grace C2C cold) -> Combine + ResidualRMSNorm
func (w NVL72WideEP) Reference(
	x []float64,
	topkIDs []int32,
	topkWeights []float32,
	expertScales []float64,
	coldExpertMask []int32,
	residual []float64,
	rmsWeight []float64,
) (
	tokensPerExpert, expertOffsets, destSlots, srcInfo []int32,
	residualOut, normOut []float64,
	err error,
) {
	plan, disp, _, comb, err := w.Stages()
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	if len(expertScales) != w.TotalExperts*w.HiddenSize {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("nvl72 wide ep reference: expertScales len %d != %d", len(expertScales), w.TotalExperts*w.HiddenSize)
	}
	tokensPerExpert, expertOffsets, destSlots, err = plan.Reference(topkIDs)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	symRecvX, srcInfo, _, err := disp.Reference(x, topkIDs, topkWeights, destSlots)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	epn := w.TotalExperts / w.Ranks
	maxSlots := w.Ranks * w.TokensPerRank * w.TopK
	expertOut := make([]float64, w.Ranks*maxSlots*w.HiddenSize)
	c2cColdOut := make([]float64, w.Ranks*maxSlots*w.HiddenSize)
	for r := 0; r < w.Ranks; r++ {
		for le := 0; le < epn; le++ {
			e := r*epn + le
			startSlot := int(expertOffsets[r*(epn+1)+le])
			endSlot := int(expertOffsets[r*(epn+1)+le+1])
			isCold := len(coldExpertMask) > 0 && coldExpertMask[e] != 0
			for slot := startSlot; slot < endSlot; slot++ {
				slotIdx := r*maxSlots + slot
				for c := 0; c < w.HiddenSize; c++ {
					inV := comb.DType.Round(float32(symRecvX[slotIdx*w.HiddenSize+c]))
					scV := comb.DType.Round(float32(expertScales[e*w.HiddenSize+c]))
					prod := comb.DType.Round(inV * scV)
					if isCold {
						c2cColdOut[slotIdx*w.HiddenSize+c] = float64(prod)
					} else {
						expertOut[slotIdx*w.HiddenSize+c] = float64(prod)
					}
				}
			}
		}
	}

	residualOut, normOut, err = comb.Reference(
		expertOut, c2cColdOut, coldExpertMask, topkIDs, topkWeights, destSlots, residual, rmsWeight,
	)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	return tokensPerExpert, expertOffsets, destSlots, srcInfo, residualOut, normOut, nil
}

// NVL72CostEstimate models the 3-tier memory/fabric traffic and Two-Batch
// Overlap (TBO) latency of GB200 NVL72 Wide Expert Parallelism across
// HBM3e (8.0 TB/s), NVLink 5 (900 GB/s unidirectional per GPU), and
// Grace-Blackwell NVLink-C2C (450 GB/s per GPU).
type NVL72CostEstimate struct {
	TokensPerRank   int
	HiddenSize      int
	TopK            int
	TotalExperts    int
	NumRanks        int
	NumTrays        int
	C2CColdFraction float64

	// Per-GPU traffic in bytes across the three GB200 memory/fabric tiers.
	MulticastPlanBytes  uint64 // NVSwitch multicast routing plan bytes
	DispatchNVLinkBytes uint64 // Outbound NVLink 5 token push bytes per GPU
	CombineNVLinkBytes  uint64 // Inbound NVLink 5 hot-expert gather bytes per GPU
	CombineC2CBytes     uint64 // Grace NVLink-C2C cold-expert fallback bytes per GPU
	LocalHBMBytes       uint64 // Local HBM3e expert MLP + RMSNorm bytes per GPU

	// Per-microbatch stage latencies in nanoseconds.
	PlanNs          float64 // O(1) NVSwitch multicast barrier + plan broadcast latency
	DispatchNs      float64 // NVLink 5 token dispatch latency
	ExpertComputeNs float64 // Local tcgen05 / HBM3e expert MLP latency
	CombineNVLinkNs float64 // NVLink 5 hot-expert gather latency
	CombineC2CNs    float64 // Grace NVLink-C2C cold-expert fallback latency
	CombineTotalNs  float64 // Total combine + fused residual RMSNorm latency

	// Two-Batch Overlap (TBO) metrics across a 2-microbatch window.
	SequentialTwoBatchNs float64 // 2 microbatches without overlap
	TBOTwoBatchNs        float64 // 2 microbatches overlapping comm (MB0/MB1) with tcgen05 compute
	TBOOverlapSavingsNs  float64 // SequentialTwoBatchNs - TBOTwoBatchNs
	TBOSpeedup           float64 // SequentialTwoBatchNs / TBOTwoBatchNs
}

// EstimateNVL72WideEP models the GB200 NVL72 3-tier roofline (8.0 TB/s HBM3e,
// 900 GB/s unidirectional NVLink 5 per GPU, 450 GB/s Grace NVLink-C2C per GPU)
// and computes the latency savings from Two-Batch Overlap (TBO), where
// microbatch 0's NVLink Dispatch/Combine overlaps with microbatch 1's tcgen05
// expert compute.
func EstimateNVL72WideEP(tokensPerRank, hiddenSize, topK, totalExperts, numRanks int, c2cColdFraction float64) NVL72CostEstimate {
	s := max(tokensPerRank, 1)
	h := max(hiddenSize, 1)
	k := max(topK, 1)
	r := min(max(numRanks, 1), NVL72MaxRanks)
	e := max(totalExperts, r)
	coldFrac := math.Max(0.0, math.Min(1.0, c2cColdFraction))
	hotFrac := 1.0 - coldFrac
	trays := (r + NVL72GPUsPerTray - 1) / NVL72GPUsPerTray

	routesPerGPU := s * k
	// BF16 token payload (2B/elem) + 8B provenance/weight metadata per route.
	dispatchBytesPerRoute := h*2 + 8
	combineBytesPerRoute := h * 2

	multicastPlanBytes := uint64(r*(e+routesPerGPU)*4 + r*4)
	dispatchNVLinkBytes := uint64(routesPerGPU * dispatchBytesPerRoute)
	combineNVLinkBytes := uint64(math.Round(float64(routesPerGPU*combineBytesPerRoute) * hotFrac))
	combineC2CBytes := uint64(math.Round(float64(routesPerGPU*combineBytesPerRoute) * coldFrac))

	// Local HBM3e traffic: active local expert weights + token activations + residual RMSNorm.
	expertsPerRank := max(e/r, 1)
	expertMLPActBytes := uint64(routesPerGPU * h * 2 * 6)
	expertWeightBytes := uint64(expertsPerRank) * uint64(h) * uint64(h) * 2
	rmsNormBytes := uint64(s * h * 6)
	localHBMBytes := expertMLPActBytes + expertWeightBytes + rmsNormBytes

	// Stage latencies (bytes / (GB/s) = nanoseconds).
	const nvswitchMulticastBaseNs = 350.0
	planNs := nvswitchMulticastBaseNs + float64(multicastPlanBytes)/NVL72NVLink5UniGBps
	dispatchNs := float64(dispatchNVLinkBytes) / NVL72NVLink5UniGBps

	expertFLOPs := float64(2 * 3 * routesPerGPU * h * h)
	hbmComputeNs := float64(localHBMBytes) / NVL72HBM3eGBps
	tensorComputeNs := expertFLOPs / (NVL72BlackwellTensorTFLOPS * 1000.0)
	expertComputeNs := math.Max(hbmComputeNs, tensorComputeNs)

	combineNVLinkNs := float64(combineNVLinkBytes) / NVL72NVLink5UniGBps
	combineC2CNs := float64(combineC2CBytes) / NVL72GraceC2CGBps
	rmsNormNs := float64(rmsNormBytes) / NVL72HBM3eGBps
	combineTotalNs := combineNVLinkNs + combineC2CNs + rmsNormNs

	singleBatchNs := planNs + dispatchNs + expertComputeNs + combineTotalNs
	seqTwoBatchNs := 2.0 * singleBatchNs

	// Two-Batch Overlap (TBO) schedule:
	//   Step 1: MB0 Plan + Dispatch
	//   Step 2: overlap(MB0 ExpertCompute, MB1 Plan + Dispatch)
	//   Step 3: overlap(MB1 ExpertCompute, MB0 Combine)
	//   Step 4: MB1 Combine
	commDispatchNs := planNs + dispatchNs
	tboTwoBatchNs := commDispatchNs +
		math.Max(expertComputeNs, commDispatchNs) +
		math.Max(expertComputeNs, combineTotalNs) +
		combineTotalNs
	savingsNs := seqTwoBatchNs - tboTwoBatchNs
	speedup := 1.0
	if tboTwoBatchNs > 0 {
		speedup = seqTwoBatchNs / tboTwoBatchNs
	}

	return NVL72CostEstimate{
		TokensPerRank:        s,
		HiddenSize:           h,
		TopK:                 k,
		TotalExperts:         e,
		NumRanks:             r,
		NumTrays:             trays,
		C2CColdFraction:      coldFrac,
		MulticastPlanBytes:   multicastPlanBytes,
		DispatchNVLinkBytes:  dispatchNVLinkBytes,
		CombineNVLinkBytes:   combineNVLinkBytes,
		CombineC2CBytes:      combineC2CBytes,
		LocalHBMBytes:        localHBMBytes,
		PlanNs:               planNs,
		DispatchNs:           dispatchNs,
		ExpertComputeNs:      expertComputeNs,
		CombineNVLinkNs:      combineNVLinkNs,
		CombineC2CNs:         combineC2CNs,
		CombineTotalNs:       combineTotalNs,
		SequentialTwoBatchNs: seqTwoBatchNs,
		TBOTwoBatchNs:        tboTwoBatchNs,
		TBOOverlapSavingsNs:  savingsNs,
		TBOSpeedup:           speedup,
	}
}
