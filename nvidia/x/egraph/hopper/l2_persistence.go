package hopper

import (
	"fmt"
	"strings"
)

// L2EvictionPolicy specifies the Hopper/Blackwell L2 cache eviction priority
// for a global memory buffer accessed by a persistent megakernel.
type L2EvictionPolicy string

// Supported L2 cache eviction policies on Hopper/Blackwell (sm_90a+).
const (
	L2PolicyNormal     L2EvictionPolicy = "normal"
	L2PolicyEvictLast  L2EvictionPolicy = "evict_last"
	L2PolicyEvictFirst L2EvictionPolicy = "evict_first"
	L2PolicyStreaming  L2EvictionPolicy = "streaming"
)

// String returns the PTX/CUDA name of the eviction policy.
func (p L2EvictionPolicy) String() string {
	return string(p)
}

// L2BufferPolicy describes the classified L2 cache eviction policy, byte
// footprint, and target residency hit ratio for a single buffer in a
// [Plan] or [CrossLayerStack].
type L2BufferPolicy struct {
	Name       string
	Category   string
	Bytes      int64
	Policy     L2EvictionPolicy
	HitRatio   float64
	ReuseCount int
}

// L2PersistencePlan holds the L2 cache persistence classification and hardware
// access policy window configuration for a [CrossLayerStack] or [Plan].
type L2PersistencePlan struct {
	L2CapacityBytes      int64
	PersistedWindowBytes int64
	TotalPinnedBytes     int64
	TotalStreamBytes     int64
	ArenaPingPongBytes   int64
	ReadyBarrierBytes    int64
	Entries              []L2BufferPolicy
	ByName               map[string]L2BufferPolicy
}

// PlanL2Persistence analyzes a [CrossLayerStack] against l2CapacityBytes and
// classifies buffers into:
//   - Persist-in-L2 ([L2PolicyEvictLast], HitRatio=1.0): cross-layer ping-pong
//     arena slots ("eg_arena_even", "eg_arena_odd"), the "ready" barrier table,
//     cross-layer activation bridges, and shared hot MoE experts / KV prefix pages.
//   - Stream-Through ([L2PolicyEvictFirst], HitRatio=0.0): single-use cold layer
//     weight matrices at M=1 so weight streaming never thrashes L2-resident activations.
func PlanL2Persistence(stack *CrossLayerStack, l2CapacityBytes int64) (*L2PersistencePlan, error) {
	if stack == nil {
		return nil, fmt.Errorf("l2 persistence: nil cross-layer stack")
	}
	plans := stack.Plans()
	if len(plans) == 0 {
		return nil, fmt.Errorf("l2 persistence: empty cross-layer stack")
	}
	if l2CapacityBytes <= 0 {
		return nil, fmt.Errorf("l2 persistence: l2CapacityBytes (%d) must be positive", l2CapacityBytes)
	}
	return buildL2PersistencePlan(plans, stack.CrossMegakernels, stack.FinalMegakernel, int64(stack.ArenaBytes), l2CapacityBytes)
}

// PlanL2PersistenceForPlan analyzes a standalone [Plan] against l2CapacityBytes.
func PlanL2PersistenceForPlan(p *Plan, l2CapacityBytes int64) (*L2PersistencePlan, error) {
	if p == nil || len(p.Kernels) == 0 {
		return nil, fmt.Errorf("l2 persistence: empty plan")
	}
	if l2CapacityBytes <= 0 {
		return nil, fmt.Errorf("l2 persistence: l2CapacityBytes (%d) must be positive", l2CapacityBytes)
	}
	return buildL2PersistencePlan([]*Plan{p}, nil, nil, 0, l2CapacityBytes)
}

func buildL2PersistencePlan(plans []*Plan, megas []Megakernel, finalMega *Megakernel, arenaBytes, l2CapacityBytes int64) (*L2PersistencePlan, error) {
	if arenaBytes <= 0 {
		if arena, err := PlanArena(plans, 0, false); err == nil && arena.Bytes > 0 {
			arenaBytes = int64(arena.Bytes)
		}
		if arenaBytes <= 0 {
			arenaBytes = 4096
		}
	}

	var readyBytes int64 = 128
	for _, mk := range megas {
		if rb := int64(mk.ReadyBytes()); rb > readyBytes {
			readyBytes = rb
		}
	}
	if finalMega != nil {
		if rb := int64(finalMega.ReadyBytes()); rb > readyBytes {
			readyBytes = rb
		}
	}
	for _, p := range plans {
		if need := int64(8 * (len(p.Kernels) + 1)); need > readyBytes {
			readyBytes = need
		}
	}
	readyBytes = (readyBytes + 127) &^ 127

	lp := &L2PersistencePlan{
		L2CapacityBytes:    l2CapacityBytes,
		ArenaPingPongBytes: arenaBytes,
		ReadyBarrierBytes:  readyBytes,
		ByName:             make(map[string]L2BufferPolicy),
	}

	addEntry := func(e L2BufferPolicy) {
		if existing, ok := lp.ByName[e.Name]; ok {
			existing.ReuseCount += e.ReuseCount
			if e.Policy == L2PolicyEvictLast && existing.Policy != L2PolicyEvictLast {
				existing.Policy = L2PolicyEvictLast
				existing.HitRatio = 1.0
				existing.Category = e.Category
			}
			lp.ByName[e.Name] = existing
			for i := range lp.Entries {
				if lp.Entries[i].Name == e.Name {
					lp.Entries[i] = existing
					break
				}
			}
			return
		}
		lp.Entries = append(lp.Entries, e)
		lp.ByName[e.Name] = e
	}

	// 1. Pin cross-layer ping-pong arena slots and ready barrier table in L2.
	addEntry(L2BufferPolicy{
		Name:       "eg_arena_even",
		Category:   "arena_ping_pong",
		Bytes:      arenaBytes,
		Policy:     L2PolicyEvictLast,
		HitRatio:   1.0,
		ReuseCount: max(len(plans), 1),
	})
	addEntry(L2BufferPolicy{
		Name:       "eg_arena_odd",
		Category:   "arena_ping_pong",
		Bytes:      arenaBytes,
		Policy:     L2PolicyEvictLast,
		HitRatio:   1.0,
		ReuseCount: max(len(plans), 1),
	})
	addEntry(L2BufferPolicy{
		Name:       "ready",
		Category:   "ready_barrier",
		Bytes:      readyBytes,
		Policy:     L2PolicyEvictLast,
		HitRatio:   1.0,
		ReuseCount: max(len(plans), 1),
	})

	// Count argument uses and detect low-M weight streaming across all plans.
	useCounts := make(map[string]int)
	weightStageUses := make(map[string]int)
	decodeWeightInputs := make(map[string]bool)

	for _, p := range plans {
		for _, k := range p.Kernels {
			isDecodeGEMM := (k.GEMM != nil && k.M <= 16) ||
				(k.MMA != nil && k.M <= 16) ||
				(k.Tile != nil && k.M <= 16) ||
				(k.UMMA != nil && k.M <= 16)
			for idx, arg := range k.Args {
				useCounts[arg]++
				if isDecodeGEMM && idx >= 1 {
					weightStageUses[arg]++
					decodeWeightInputs[arg] = true
				}
			}
		}
	}

	// 2. Classify plan inputs, temps, and outputs.
	for _, p := range plans {
		planUseCounts := make(map[string]int)
		planWeightUses := make(map[string]int)
		for _, k := range p.Kernels {
			isDecodeGEMM := (k.GEMM != nil && k.M <= 16) ||
				(k.MMA != nil && k.M <= 16) ||
				(k.Tile != nil && k.M <= 16) ||
				(k.UMMA != nil && k.M <= 16)
			for idx, arg := range k.Args {
				planUseCounts[arg]++
				if isDecodeGEMM && idx >= 1 {
					planWeightUses[arg]++
				}
			}
		}

		for _, tmp := range p.Temps {
			nbytes := tensorByteSize(tmp)
			addEntry(L2BufferPolicy{
				Name:       tmp.Name,
				Category:   "activation_bridge",
				Bytes:      nbytes,
				Policy:     L2PolicyEvictLast,
				HitRatio:   1.0,
				ReuseCount: max(useCounts[tmp.Name], 2),
			})
		}

		for _, in := range p.Inputs {
			nbytes := tensorByteSize(in)
			uses := max(planUseCounts[in.Name], 1)
			switch {
			case isHotMoEOrKVPagerBuffer(in.Name):
				addEntry(L2BufferPolicy{
					Name:       in.Name,
					Category:   "hot_moe_kv",
					Bytes:      nbytes,
					Policy:     L2PolicyEvictLast,
					HitRatio:   1.0,
					ReuseCount: max(uses, 2),
				})
			case isColdWeightTensor(in, decodeWeightInputs[in.Name], planWeightUses[in.Name], uses):
				addEntry(L2BufferPolicy{
					Name:       in.Name,
					Category:   "cold_weight",
					Bytes:      nbytes,
					Policy:     L2PolicyEvictFirst,
					HitRatio:   0.0,
					ReuseCount: uses,
				})
			case len(in.Shape) <= 1 || isActivationBridgeName(in.Name):
				addEntry(L2BufferPolicy{
					Name:       in.Name,
					Category:   "activation",
					Bytes:      nbytes,
					Policy:     L2PolicyEvictLast,
					HitRatio:   1.0,
					ReuseCount: uses,
				})
			default:
				addEntry(L2BufferPolicy{
					Name:       in.Name,
					Category:   "normal",
					Bytes:      nbytes,
					Policy:     L2PolicyNormal,
					HitRatio:   0.5,
					ReuseCount: uses,
				})
			}
		}

		if p.Output.Name != "" {
			nbytes := tensorByteSize(p.Output)
			addEntry(L2BufferPolicy{
				Name:       p.Output.Name,
				Category:   "activation",
				Bytes:      nbytes,
				Policy:     L2PolicyEvictLast,
				HitRatio:   1.0,
				ReuseCount: max(useCounts[p.Output.Name], 1),
			})
		}
	}

	var pinnedBytes, streamBytes int64
	for _, e := range lp.Entries {
		switch e.Policy {
		case L2PolicyEvictLast:
			pinnedBytes += e.Bytes
		case L2PolicyEvictFirst, L2PolicyStreaming:
			streamBytes += e.Bytes
		}
	}
	lp.TotalPinnedBytes = pinnedBytes
	lp.TotalStreamBytes = streamBytes
	lp.PersistedWindowBytes = min(pinnedBytes, l2CapacityBytes)
	return lp, nil
}

func tensorByteSize(t Tensor) int64 {
	if len(t.Shape) == 0 {
		return 0
	}
	b, err := arenaTensorBytes(t)
	if err != nil {
		return 0
	}
	return int64(b)
}

func isHotMoEOrKVPagerBuffer(name string) bool {
	lower := strings.ToLower(name)
	if strings.Contains(lower, "qkv") {
		return false
	}
	for _, pat := range []string{
		"kv_cache", "kv_prefix", "kv_page", "k_cache", "v_cache", "k_page", "v_page", "page_table", "pages",
		"prefix", "shared_expert", "hot_expert", "moe_shared", "router",
	} {
		if strings.Contains(lower, pat) {
			return true
		}
	}
	return false
}

func isActivationBridgeName(name string) bool {
	lower := strings.ToLower(name)
	for _, pat := range []string{"xn", "x_in", "attn", "hid", "residual", "bridge", "norm"} {
		if lower == pat || strings.HasPrefix(lower, pat+"_") || strings.HasSuffix(lower, "_"+pat) {
			return true
		}
	}
	return false
}

func isColdWeightTensor(t Tensor, inDecodeGEMM bool, weightUses, totalUses int) bool {
	if len(t.Shape) < 2 {
		return false
	}
	if isHotMoEOrKVPagerBuffer(t.Name) {
		return false
	}
	lower := strings.ToLower(t.Name)
	hasWeightName := strings.HasPrefix(lower, "w_") ||
		strings.Contains(lower, "_w_") ||
		strings.HasSuffix(lower, "_w") ||
		strings.Contains(lower, "weight") ||
		strings.Contains(lower, "qkv") ||
		strings.Contains(lower, "gate") ||
		strings.Contains(lower, "up") ||
		strings.Contains(lower, "down") ||
		strings.Contains(lower, "o_proj") ||
		strings.Contains(lower, "w_head")
	if hasWeightName && totalUses <= 2 {
		return true
	}
	if inDecodeGEMM && weightUses == 1 && totalUses == 1 && t.Shape[0] >= 16 && t.Shape[1] >= 16 {
		return true
	}
	return false
}

// CUDAPolicyPrelude emits the CUDA/PTX prelude defining L2 cache eviction
// policy descriptors (createpolicy.fractional.L2::evict_last.b64 and
// createpolicy.fractional.L2::evict_first.b64), cache-hinted global loads and
// stores (ld.global.L2::cache_hint and st.global.L2::cache_hint), and the host
// stream persisting window configurator for eg_arena_even, eg_arena_odd, and
// ready.
func (lp *L2PersistencePlan) CUDAPolicyPrelude() string {
	var capBytes, windowBytes, arenaBytes, readyBytes int64
	if lp != nil {
		capBytes = lp.L2CapacityBytes
		windowBytes = lp.PersistedWindowBytes
		arenaBytes = lp.ArenaPingPongBytes
		readyBytes = lp.ReadyBarrierBytes
	}

	var b strings.Builder
	b.WriteString("#ifndef EG_L2_PERSISTENCE_PRELUDE\n")
	b.WriteString("#define EG_L2_PERSISTENCE_PRELUDE\n\n")
	fmt.Fprintf(&b, "// L2 Persistence Plan: capacity=%d bytes, persisted_window=%d bytes, arena_slot=%d bytes, ready=%d bytes.\n",
		capBytes, windowBytes, arenaBytes, readyBytes)
	if lp != nil {
		for _, e := range lp.Entries {
			fmt.Fprintf(&b, "//   buffer %-28s category=%-18s policy=%-12s hit_ratio=%.2f bytes=%d\n",
				e.Name, e.Category, e.Policy, e.HitRatio, e.Bytes)
		}
	}

	b.WriteString(`
// PTX L2 cache policy descriptor builders (Hopper sm_90a / Blackwell sm_100a+).
EG_DEV uint64_t eg_l2_policy_evict_last(float fraction = 1.0f) {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n" : "=l"(policy) : "f"(fraction));
#else
    (void)fraction;
#endif
    return policy;
}

EG_DEV uint64_t eg_l2_policy_evict_first(float fraction = 1.0f) {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n" : "=l"(policy) : "f"(fraction));
#else
    (void)fraction;
#endif
    return policy;
}

// Cache-hinted global memory loads and stores using L2::cache_hint descriptors.
EG_DEV uint32_t eg_ld_global_l2_hint_b32(const uint32_t* ptr, uint64_t policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t val;
    asm volatile("ld.global.L2::cache_hint.b32 %0, [%1], %2;\n" : "=r"(val) : "l"(ptr), "l"(policy) : "memory");
    return val;
#else
    (void)policy;
    return *ptr;
#endif
}

EG_DEV uint64_t eg_ld_global_l2_hint_b64(const uint64_t* ptr, uint64_t policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint64_t val;
    asm volatile("ld.global.L2::cache_hint.b64 %0, [%1], %2;\n" : "=l"(val) : "l"(ptr), "l"(policy) : "memory");
    return val;
#else
    (void)policy;
    return *ptr;
#endif
}

EG_DEV void eg_st_global_l2_hint_b32(uint32_t* ptr, uint32_t val, uint64_t policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" :: "l"(ptr), "r"(val), "l"(policy) : "memory");
#else
    (void)policy;
    *ptr = val;
#endif
}

EG_DEV void eg_st_global_l2_hint_b64(uint64_t* ptr, uint64_t val, uint64_t policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile("st.global.L2::cache_hint.b64 [%0], %1, %2;\n" :: "l"(ptr), "l"(val), "l"(policy) : "memory");
#else
    (void)policy;
    *ptr = val;
#endif
}

// Convenience helpers for streaming single-use M=1 weight tiles (evict_first)
// vs pinning cross-layer ping-pong arena activations (evict_last).
EG_DEV uint32_t eg_ld_cold_weight_b32(const uint32_t* ptr) {
    return eg_ld_global_l2_hint_b32(ptr, eg_l2_policy_evict_first(1.0f));
}

EG_DEV uint32_t eg_ld_pinned_arena_b32(const uint32_t* ptr) {
    return eg_ld_global_l2_hint_b32(ptr, eg_l2_policy_evict_last(1.0f));
}

EG_DEV void eg_st_pinned_arena_b32(uint32_t* ptr, uint32_t val) {
    eg_st_global_l2_hint_b32(ptr, val, eg_l2_policy_evict_last(1.0f));
}

#endif // EG_L2_PERSISTENCE_PRELUDE
`)
	return b.String()
}
