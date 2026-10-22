package dsv4expertcache

import (
	"errors"
	"math"
)

// MetalHotness counts routes, including misses, and halves counts every sixteen
// decode tokens. Calls must be serialized. Its zero value is ready to use.
type MetalHotness struct {
	routes        [80][384]uint32
	tokens, decay uint64
	first         int
	started       bool
}

// NoteToken advances the aging clock at the first routed layer only.
func (h *MetalHotness) NoteToken(layer int) error {
	if h == nil || layer < 0 || layer >= len(h.routes) {
		return errors.New("invalid Metal hotness layer")
	}
	if !h.started || layer < h.first {
		h.first, h.started = layer, true
	}
	if layer == h.first && h.tokens != math.MaxUint64 {
		h.tokens++
	}
	if h.decay == 0 {
		h.decay = h.tokens
	}
	for h.tokens-h.decay >= 16 {
		for layer := range h.routes {
			for expert := range h.routes[layer] {
				h.routes[layer][expert] >>= 1
			}
		}
		h.decay += 16
	}
	return nil
}

// Acquire protects the whole request and chooses victims by route hotness,
// then oldest use, then layer/expert order, matching the Metal global scan.
// The ordinary Cache LRU policy is unchanged.
func (h *MetalHotness) Acquire(c *Cache, keys []Key) (*Lease, error) {
	return h.acquire(c, keys, true)
}

func (h *MetalHotness) acquire(c *Cache, keys []Key, count bool) (*Lease, error) {
	if h == nil || c == nil || c.closed || c.active != nil || len(c.slots) == 0 || c.generation == math.MaxUint64 || len(keys) == 0 {
		return nil, errors.New("Metal cache unavailable")
	}
	unique := make(map[Key]int)
	for _, k := range keys {
		if k.Identity != c.identity || k.Layer < 0 || k.Layer >= 80 || k.Expert < 0 || k.Expert >= 384 {
			return nil, errors.New("invalid Metal expert key")
		}
		unique[k] = -1
	}
	if len(unique) > len(c.slots) {
		return nil, errors.New("Metal request exceeds slot capacity")
	}
	for _, k := range keys {
		if !count {
			break
		}
		p := &h.routes[k.Layer][k.Expert]
		if *p < math.MaxUint32 {
			*p++
		}
	}
	protected := make([]bool, len(c.slots))
	for i, s := range c.slots {
		if s.valid {
			if _, ok := unique[s.key]; ok {
				unique[s.key], protected[i] = i, true
			}
		}
	}
	l := &Lease{cache: c, generation: c.generation + 1, events: make(map[int]Completion)}
	for _, k := range keys {
		i := unique[k]
		if i < 0 {
			for j, s := range c.slots {
				if protected[j] {
					continue
				}
				if i < 0 || !s.valid || (c.slots[i].valid && h.less(s, c.slots[i])) {
					i = j
					if !s.valid {
						break
					}
				}
			}
			protected[i], unique[k] = true, i
			l.uploads = append(l.uploads, Upload{i, l.generation, k})
		}
		l.remap = append(l.remap, i)
	}
	c.generation, c.active = l.generation, l
	return l, nil
}

func (h *MetalHotness) less(a, b slot) bool {
	x, y := h.routes[a.key.Layer][a.key.Expert], h.routes[b.key.Layer][b.key.Expert]
	if x != y {
		return x < y
	}
	if a.used != b.used {
		return a.used < b.used
	}
	if a.key.Layer != b.key.Layer {
		return a.key.Layer < b.key.Layer
	}
	return a.key.Expert < b.key.Expert
}

// MetalPinCapacity computes the source pin-failure cap. The caller must drain
// active consumers before reducing cache capacity; zero means no new cap.
func MetalPinCapacity(entries, lockedSlots uint32, lockedBytes, expertBytes uint64) uint32 {
	cap := entries
	if lockedSlots != 0 && lockedSlots < cap {
		cap = lockedSlots
	}
	const gib = uint64(1 << 30)
	safe := lockedBytes / gib
	if safe > 1 {
		safe--
	}
	if safe != 0 && expertBytes != 0 {
		n := min(safe*gib/expertBytes, uint64(math.MaxUint32))
		if n != 0 && n < uint64(cap) {
			cap = uint32(n)
		}
	}
	return cap
}
