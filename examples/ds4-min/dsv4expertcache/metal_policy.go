package dsv4expertcache

import (
	"errors"
	"math"
	"os"
	"strconv"
)

func metalSlabTarget() uint64 {
	const mib = uint64(1 << 20)
	target := uint64(4096) * mib
	s := skipSpace(os.Getenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB"))
	if s == "" {
		return target
	}
	negative := s[0] == '-'
	if s[0] == '+' || negative {
		s = s[1:]
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 || i != len(s) {
		return target
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		v = math.MaxUint64
	} else if negative {
		v = -v
	}
	if v == 0 {
		return target
	}
	if v > math.MaxUint64/mib {
		return math.MaxUint64
	}
	return v * mib
}

// MetalCachePlan separates resident non-routed weights, cache and transient
// prefill reserve. Bytes count logical triplets, excluding slab page padding.
type MetalCachePlan struct {
	TargetBytes, CacheBytes, EffectiveBytes, PrefillReserve uint64
	Slots                                                   uint32
}

// PlanMetalCache applies the upstream working-set percentage and model cap.
// DS4_SSD_AUTO_CACHE_PCT accepts 50..95; an invalid override uses percent.
func PlanMetalCache(recommended, modelLimit, nonRouted, expertBytes, maxExperts uint64, percent uint32) (MetalCachePlan, error) {
	var p MetalCachePlan
	if recommended == 0 || expertBytes == 0 || percent < 50 || percent > 95 {
		return p, errors.New("invalid Metal cache plan")
	}
	pct := uint64(percent)
	if s := os.Getenv("DS4_SSD_AUTO_CACHE_PCT"); s != "" {
		s = skipSpace(s)
		v, rest, ok := unsignedPrefix(s)
		if ok && rest == "" && v >= 50 && v <= 95 {
			pct = v
		}
	}
	if recommended > math.MaxUint64/pct {
		p.TargetBytes = math.MaxUint64
	} else {
		p.TargetBytes = recommended * pct / 100
	}
	if modelLimit != 0 {
		p.TargetBytes = min(p.TargetBytes, modelLimit)
	}
	if p.TargetBytes > nonRouted {
		p.CacheBytes = p.TargetBytes - nonRouted
	}
	n := max(uint64(1), p.CacheBytes/expertBytes)
	if maxExperts != 0 {
		n = min(n, maxExperts)
	}
	n = min(n, uint64(math.MaxUint32))
	if n > math.MaxUint64/expertBytes {
		return MetalCachePlan{}, errors.New("Metal effective cache extent overflow")
	}
	p.Slots = uint32(n)
	p.EffectiveBytes = n * expertBytes
	return p, nil
}

// ReserveMetalPrefill reserves up to two dominant routed layers in an explicit
// byte budget, leaving at least one logical expert when the reserve is reduced.
func ReserveMetalPrefill(bytes, expertBytes uint64, experts, layers uint32) (cacheBytes, reserve uint64, err error) {
	if expertBytes == 0 || experts == 0 || layers == 0 || expertBytes > math.MaxUint64/uint64(experts) {
		return 0, 0, errors.New("invalid Metal prefill reserve")
	}
	row := expertBytes * uint64(experts)
	n := uint64(min(layers, uint32(2)))
	if row > math.MaxUint64/n {
		return 0, 0, errors.New("Metal prefill reserve overflow")
	}
	reserve = row * n
	if reserve >= bytes {
		reserve = 0
		if bytes > expertBytes {
			reserve = bytes - expertBytes
		}
	}
	return bytes - reserve, reserve, nil
}

// MetalPrefillEnabled matches the pinned IQ2_XXS/Q2_K batch-selected policy.
// Quality mode and incompatible operator switches disable it. Capacity must
// cover a whole layer even when a particular request has fewer unique experts.
func MetalPrefillEnabled(tokens, experts, selected, gateType, downType, cacheSlots uint32, quality bool) bool {
	if tokens <= 1 || experts == 0 || selected != 6 || gateType != 16 || downType != 10 || quality || cacheSlots < experts {
		return false
	}
	for _, name := range []string{"DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE", "DS4_METAL_MOE_WRITE_CLAMPED_ACT", "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION"} {
		if _, set := os.LookupEnv(name); set {
			return false
		}
	}
	if _, set := os.LookupEnv("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"); set {
		return true
	}
	var lo, hi uint32
	if experts == 256 {
		lo, hi = 2, 760
	} else if experts == 384 {
		lo, hi = 2, 800
	}
	lo = metalTokenLimit("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN", lo)
	hi = metalTokenLimit("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX", hi)
	return hi != 0 && tokens >= lo && tokens <= hi
}

func metalTokenLimit(name string, fallback uint32) uint32 {
	s := skipSpace(os.Getenv(name))
	if s == "" {
		return fallback
	}
	i := 0
	if s[0] == '+' || s[0] == '-' {
		i++
	}
	start := i
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == start {
		return fallback
	}
	v, err := strconv.ParseInt(s[:i], 10, 64)
	if err != nil {
		if s[0] == '-' {
			return 0
		}
		return math.MaxUint32
	}
	if v <= 0 {
		return 0
	}
	if uint64(v) > math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(v)
}

// CompactMetalRoutes returns first-occurrence expert order and a detached
// token-major remap. Address tables must retain every returned expert resource.
func CompactMetalRoutes(ids []int32, experts uint32) (unique []int32, remap []int32, err error) {
	if len(ids) == 0 || experts == 0 {
		return nil, nil, errors.New("invalid Metal routes")
	}
	seen := make(map[int32]int32)
	for _, id := range ids {
		if id < 0 || uint32(id) >= experts {
			return nil, nil, errors.New("invalid selected Metal expert")
		}
		slot, ok := seen[id]
		if !ok {
			slot = int32(len(unique))
			seen[id] = slot
			unique = append(unique, id)
		}
		remap = append(remap, slot)
	}
	return unique, remap, nil
}
