package timeline

import "sort"

// An Interval is a half-open device-busy window, in nanoseconds.
type Interval struct {
	Start, End uint64
}

// Union totals the time covered by at least one interval. Overlapping
// intervals are counted once.
func Union(in []Interval) uint64 {
	if len(in) == 0 {
		return 0
	}
	sorted := append([]Interval(nil), in...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Start < sorted[j].Start })
	var total uint64
	cur := sorted[0]
	for _, iv := range sorted[1:] {
		if iv.Start > cur.End {
			total += cur.End - cur.Start
			cur = iv
			continue
		}
		if iv.End > cur.End {
			cur.End = iv.End
		}
	}
	return total + cur.End - cur.Start
}
