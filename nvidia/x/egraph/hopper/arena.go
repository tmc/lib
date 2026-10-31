package hopper

import (
	"fmt"
	"math"
	"sort"
)

// Arena describes temporary storage for plans executed sequentially on one
// stream. Plans must not overlap execution. Inputs, outputs not in Temps,
// and kernel workspaces are allocated separately.
type Arena struct {
	Offsets       []map[string]uint64 // tensor name to byte offset, indexed by plan
	Bytes         uint64              // allocation size, including alignment padding
	OriginalBytes uint64              // sum of temporary tensor sizes without padding or reuse
}

// PlanArena assigns 256-byte-aligned offsets to the temporary tensors in plans.
// A tensor is live from its first through its last kernel argument occurrence;
// tensors used in the same launch never share storage. A temporary that is also
// the plan output remains live until the plan finishes. Different plans share
// storage, so their temporary values must not survive into another plan.
// With reuse enabled, kernels must complete in sequence without pipelining.
// Disable reuse for persistent kernels or overlapping launches. Cross-plan
// sharing still requires non-overlapping program execution in either mode.
// A zero budget imposes no limit; otherwise Bytes must not exceed budget.
func PlanArena(plans []*Plan, budget uint64, reuse bool) (*Arena, error) {
	a := &Arena{Offsets: make([]map[string]uint64, len(plans))}
	for i, p := range plans {
		if p == nil {
			return nil, fmt.Errorf("plan arena: nil plan %d", i)
		}
		offsets, peak, original, err := planArena(p, reuse)
		if err != nil {
			return nil, fmt.Errorf("plan arena %q: %w", p.Name, err)
		}
		a.Offsets[i] = offsets
		if original > math.MaxUint64-a.OriginalBytes {
			return nil, fmt.Errorf("plan arena: total temporary bytes overflow")
		}
		a.OriginalBytes += original
		if peak > a.Bytes {
			a.Bytes = peak
		}
	}
	if budget != 0 && a.Bytes > budget {
		return nil, fmt.Errorf("plan arena: need %d bytes, budget %d", a.Bytes, budget)
	}
	return a, nil
}

type arenaTensor struct {
	name         string
	size, offset uint64
	first, last  int
}

func planArena(p *Plan, reuse bool) (map[string]uint64, uint64, uint64, error) {
	ts := make([]arenaTensor, 0, len(p.Temps))
	seen := make(map[string]bool)
	var original uint64
	for _, t := range p.Temps {
		if t.Name == "" || seen[t.Name] {
			return nil, 0, 0, fmt.Errorf("empty or duplicate temporary name %q", t.Name)
		}
		seen[t.Name] = true
		size, err := arenaTensorBytes(t)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("temporary %q: %w", t.Name, err)
		}
		if size > math.MaxUint64-original {
			return nil, 0, 0, fmt.Errorf("temporary bytes overflow")
		}
		original += size
		if size > math.MaxUint64-255 {
			return nil, 0, 0, fmt.Errorf("temporary %q: aligned size overflow", t.Name)
		}
		v := arenaTensor{name: t.Name, size: (size + 255) &^ 255, first: -1, last: -1}
		for i, k := range p.Kernels {
			for _, arg := range k.Args {
				if arg == t.Name {
					if v.first < 0 {
						v.first = i
					}
					v.last = i
				}
			}
		}
		// An unreferenced temporary cannot be proven dead from argument metadata.
		if v.first < 0 {
			v.first = 0
			v.last = len(p.Kernels)
		}
		if t.Name == p.Output.Name {
			v.last = len(p.Kernels)
		}
		ts = append(ts, v)
	}
	sort.SliceStable(ts, func(i, j int) bool { return ts[i].first < ts[j].first })
	offsets := make(map[string]uint64, len(ts))
	var placed []arenaTensor
	var peak uint64
	for _, t := range ts {
		// First-fit among the ranges live at this tensor's first launch. Sorting
		// by offset makes placement deterministic without depending on map order.
		var live []arenaTensor
		for _, v := range placed {
			if !reuse || v.last >= t.first {
				live = append(live, v)
			}
		}
		sort.Slice(live, func(i, j int) bool { return live[i].offset < live[j].offset })
		var off uint64
		for _, v := range live {
			if off <= v.offset && t.size <= v.offset-off {
				break
			}
			if end := v.offset + v.size; end > off {
				off = end
			}
		}
		if t.size > math.MaxUint64-off {
			return nil, 0, 0, fmt.Errorf("arena size overflow")
		}
		t.offset = off
		if end := off + t.size; end > peak {
			peak = end
		}
		offsets[t.name] = off
		placed = append(placed, t)
	}
	return offsets, peak, original, nil
}

func arenaTensorBytes(t Tensor) (uint64, error) {
	bits := t.DType.Bits()
	if bits == 0 {
		return 0, fmt.Errorf("invalid dtype %v", t.DType)
	}
	n := uint64(1)
	for _, d := range t.Shape {
		if d <= 0 {
			return 0, fmt.Errorf("nonpositive dimension %d", d)
		}
		if uint64(d) > math.MaxUint64/n {
			return 0, fmt.Errorf("shape size overflow")
		}
		n *= uint64(d)
	}
	// Avoid multiplying the element count by the bit size for packed types.
	if bits == 4 {
		return n/2 + n%2, nil
	}
	bytes := uint64(bits / 8)
	if n > math.MaxUint64/bytes {
		return 0, fmt.Errorf("storage size overflow")
	}
	return n * bytes, nil
}
