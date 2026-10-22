//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

func (r *metalRuntime) attachStreaming(o streamingOptions, fileBytes uint64) error {
	triplets, err := streamingTriplets(r.model, len(r.layers))
	if err != nil {
		return err
	}
	base, cacheableLayers, err := metalStreamingDominant(triplets, len(r.layers))
	if err != nil {
		return err
	}
	strides := [3]uint64{base.Gate.Stride, base.Up.Stride, base.Down.Stride}
	var expertBytes, routedBytes, nonRouted uint64
	for _, n := range strides {
		if n > math.MaxUint64-expertBytes {
			return errors.New("expert stride overflow")
		}
		expertBytes += n
	}
	for _, tr := range triplets {
		for _, t := range []dsv4expertcache.Tensor{tr.Gate, tr.Up, tr.Down} {
			if t.Bytes > math.MaxUint64-routedBytes {
				return errors.New("routed weight extent overflow")
			}
			routedBytes += t.Bytes
		}
	}
	for _, t := range r.model.Tensors {
		n, e := t.Bytes()
		if e != nil || n > math.MaxUint64-nonRouted {
			return errors.New("model tensor extent overflow")
		}
		nonRouted += n
	}
	if routedBytes > nonRouted {
		return errors.New("routed weights exceed model")
	}
	nonRouted -= routedBytes
	maxExperts := uint64(cacheableLayers) * uint64(base.Gate.Experts)
	var slots uint64
	var budget uint64
	if o.budget != "" {
		b, e := dsv4expertcache.ParseBudget(o.budget)
		if e != nil {
			return e
		}
		if b.Bytes != 0 {
			budget, _, err = dsv4expertcache.ReserveMetalPrefill(b.Bytes, expertBytes, uint32(base.Gate.Experts), uint32(cacheableLayers))
			if err != nil {
				return err
			}
			slots = budget / expertBytes
		} else {
			slots = min(uint64(b.Slots), maxExperts)
			page := uint64(os.Getpagesize())
			if expertBytes > math.MaxUint64-(page-1) {
				return errors.New("expert physical extent overflow")
			}
			physical := (expertBytes + page - 1) &^ (page - 1)
			if slots > math.MaxUint64/physical {
				return errors.New("expert slab extent overflow")
			}
			budget = slots * physical
		}
	} else {
		// The context estimate is the pinned Flash graph estimate, not the
		// smaller scalar Go arena. Keep upstream's planning headroom.
		contextBytes, e := metalStreamingContextBytes(r.ctx)
		if e != nil {
			return e
		}
		recommended := uint64(r.dev.Raw().RecommendedMaxWorkingSetSize())
		target := uint64(math.MaxUint64)
		if recommended <= math.MaxUint64/7 {
			target = recommended * 7 / 8
		}
		limit := uint64(1 << 30)
		if target > contextBytes {
			limit = max(limit, (target-contextBytes)/(1<<30)*(1<<30))
		}
		p, e := dsv4expertcache.PlanMetalCache(recommended, limit, nonRouted, expertBytes, maxExperts, 86)
		if e != nil {
			return e
		}
		slots, budget = uint64(p.Slots), p.EffectiveBytes
	}
	slots = min(slots, maxExperts)
	if slots == 0 || slots > uint64(math.MaxInt) {
		return errors.New("empty streaming cache")
	}
	model, quant, err := r.stateIdentity.get(r.file, int64(fileBytes), r.model)
	if err != nil {
		return err
	}
	h := sha256.New()
	var word [8]byte
	for layer := range r.layers {
		tr := triplets[layer]
		for _, t := range []dsv4expertcache.Tensor{tr.Gate, tr.Up, tr.Down} {
			for _, n := range []uint64{t.Offset, t.Bytes, t.Stride, uint64(t.Experts)} {
				binary.LittleEndian.PutUint64(word[:], n)
				h.Write(word[:])
			}
		}
	}
	identity := dsv4expertcache.Identity{Model: model, Quant: quant}
	copy(identity.Layout[:], h.Sum(nil))
	source, err := dsv4expertcache.NewSource(r.file, fileBytes, identity, triplets)
	if err != nil {
		return err
	}
	cache, err := dsv4expertcache.New(identity, int(slots), expertBytes, budget)
	if err != nil {
		return err
	}
	// Publish before the error check, including partial native allocations.
	r.streaming, err = initMetalStreaming(r.dev, source, cache, strides, int(slots), budget, base.Gate.Experts)
	r.streaming.triplets = triplets
	if err != nil {
		return err
	}
	return r.streaming.preload(o)
}

func metalStreamingDominant(triplets map[int]dsv4expertcache.Triplet, layers int) (dsv4expertcache.Triplet, int, error) {
	var best dsv4expertcache.Triplet
	bestCount := 0
	for layer := range layers {
		candidate, ok := triplets[layer]
		if !ok {
			return best, 0, errors.New("streaming layer missing")
		}
		bytes, err := metalTripletBytes(candidate)
		if err != nil {
			return best, 0, err
		}
		count := 0
		for row := range layers {
			t := triplets[row]
			n, err := metalTripletBytes(t)
			if err != nil {
				return best, 0, err
			}
			if n == bytes {
				count++
			}
		}
		if count > bestCount {
			best, bestCount = candidate, count
		}
	}
	if bestCount == 0 {
		return best, 0, errors.New("streaming dominant class unavailable")
	}
	return best, bestCount, nil
}

func metalTripletBytes(t dsv4expertcache.Triplet) (uint64, error) {
	var bytes uint64
	for _, n := range []uint64{t.Gate.Stride, t.Up.Stride, t.Down.Stride} {
		if n == 0 || n > math.MaxUint64-bytes {
			return 0, errors.New("streaming class extent unavailable")
		}
		bytes += n
	}
	return bytes, nil
}

func metalStreamingContextBytes(context int) (uint64, error) {
	if context <= 0 || uint64(context) > math.MaxInt32 {
		return 0, errors.New("invalid streaming context")
	}
	ctx := uint64(context)
	prefill := ctx
	if s := os.Getenv("DS4_METAL_PREFILL_CHUNK"); s != "" {
		if v, ok := metalStreamingLong(s); ok && v > 0 {
			prefill = uint64(uint32(v))
		}
	} else {
		prefill = min(ctx, uint64(4096))
	}
	prefill = min(ctx, max(prefill, uint64(1)))
	raw := min(ctx, uint64(128)) + prefill
	raw = min(raw, ctx)
	raw = min((raw+255)&^uint64(255), uint64(8192))
	if s := os.Getenv("DS4_METAL_GRAPH_RAW_CAP"); s != "" {
		if v, ok := metalStreamingLong(s); ok && v > 0 {
			raw = min(uint64(uint32(v)), ctx, uint64(8192))
			raw = max(raw, min(ctx, uint64(128)))
		}
	}
	compressed := uint64(0)
	for layer := 2; layer < 43; layer++ {
		ratio := uint64(128)
		if layer%2 == 0 {
			ratio = 4
		}
		rows := ctx/ratio + 2
		compressed += rows * 512 * 2
		if ratio == 4 {
			compressed += rows * 128 * 4
		}
	}
	return 43*raw*512*4 + compressed + 2*(ctx/4+2)*prefill*4 + (prefill/4+2)*512*4, nil
}

// metalStreamingLong is strtol's decimal-prefix grammar on the 64-bit pinned
// oracle. The source ignores errno and casts positive overflow to uint32.
func metalStreamingLong(s string) (int64, bool) {
	s = strings.TrimLeft(s, " \t\n\r\v\f")
	i := 0
	if len(s) != 0 && (s[0] == '+' || s[0] == '-') {
		i++
	}
	start := i
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == start {
		return 0, false
	}
	v, err := strconv.ParseInt(s[:i], 10, 64)
	if err != nil {
		if s[0] == '-' {
			return math.MinInt64, true
		}
		return math.MaxInt64, true
	}
	return v, true
}

func (w *metalStreamingWeights) preload(o streamingOptions) error {
	if o.cold {
		return nil
	}
	if _, disabled := os.LookupEnv("DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST"); disabled {
		return nil
	}
	limit := metalStreamingPreloadCount(w.capacity, o.preload, os.Getenv("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP"))
	var entries []dsv4expertcache.HotEntry
	path := os.Getenv("DS4_METAL_STREAMING_EXPERT_HOTLIST")
	if path != "" {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		entries, err = dsv4expertcache.ReadHotlist(f, 43, 256, uint32(limit))
		err = errors.Join(err, f.Close())
		if err != nil {
			return err
		}
	} else {
		for rank, entry := range streamingFlashHotlist[:min(limit, len(streamingFlashHotlist))] {
			entries = append(entries, dsv4expertcache.HotEntry{Layer: uint32(entry >> 8), Expert: uint32(entry & 255), Priority: 1 + uint32(31*uint64(limit-rank)/uint64(limit))})
		}
	}
	for _, entry := range entries {
		t := w.triplets[int(entry.Layer)]
		if t.Gate.Stride != w.strides[0] || t.Up.Stride != w.strides[1] || t.Down.Stride != w.strides[2] {
			continue // mixed-size experts remain mapped, never mispacked
		}
		var err error
		w.selection, err = w.upload.Select(w.source, w.cache, int(entry.Layer), []int{int(entry.Expert)}, false)
		if err != nil {
			return err
		}
		if err := w.finish(); err != nil {
			return err
		}
		if err := w.upload.SeedPriority(int(entry.Layer), int(entry.Expert), entry.Priority); err != nil {
			return err
		}
	}
	return nil
}

// metalStreamingPreloadCount mirrors the Flash preload policy. strtoul accepts
// leading space and a sign, requires the entire remainder, and saturates on
// overflow even for a negative input.
func metalStreamingPreloadCount(capacity int, requested uint, env string) int {
	preload := uint64(requested)
	if preload == 0 {
		preload = uint64(capacity)
		cap := uint64(4096)
		s := strings.TrimLeft(env, " \t\n\r\v\f")
		if s != "" {
			negative := s[0] == '-'
			if s[0] == '+' || negative {
				s = s[1:]
			}
			valid := s != ""
			for i := range len(s) {
				valid = valid && s[i] >= '0' && s[i] <= '9'
			}
			if valid {
				v, err := strconv.ParseUint(s, 10, 64)
				if err != nil {
					v = math.MaxUint64
				} else if negative {
					v = -v
				}
				cap = min(v, uint64(math.MaxUint32))
			}
		}
		if cap != 0 {
			preload = min(preload, cap)
		}
	}
	return int(min(preload, uint64(capacity), uint64(43*256)))
}
