package timeline

import "sort"

type rangeIdentity struct {
	journal         string
	process, thread uint32
}

type rangeIndex struct {
	groups    map[rangeIdentity]*rangeGroup
	crossings []bool
}

type rangeEntry struct {
	start, end int64
	index      int
}

// Each identity owns one start-sorted tree. maxEnd permits pruning subtrees
// that cannot contain the whole call; leaves retain original input indexes.
type rangeGroup struct {
	entries []rangeEntry
	maxEnd  []int64
}

func newRangeIndex(ranges []Range) *rangeIndex {
	x := &rangeIndex{groups: make(map[rangeIdentity]*rangeGroup), crossings: make([]bool, len(ranges))}
	for i, r := range ranges {
		key := rangeIdentity{r.Journal, r.Process, r.Thread}
		g := x.groups[key]
		if g == nil {
			g = &rangeGroup{}
			x.groups[key] = g
		}
		g.entries = append(g.entries, rangeEntry{r.Start, r.End, i})
	}
	for _, g := range x.groups {
		sort.Slice(g.entries, func(i, j int) bool {
			a, b := g.entries[i], g.entries[j]
			if a.start != b.start {
				return a.start < b.start
			}
			return a.index < b.index
		})
		g.maxEnd = make([]int64, 4*len(g.entries))
		g.build(1, 0, len(g.entries))
		// Crossing detection excludes zero-duration and reversed intervals, as the
		// reference predicate does. Both directional queries have strict bounds.
		valid := make([]rangeEntry, 0, len(g.entries))
		for _, e := range g.entries {
			if e.end > e.start {
				valid = append(valid, e)
			}
		}
		if len(valid) == 0 {
			continue
		}
		starts := make([]int64, len(valid))
		ends := make([]int64, len(valid))
		for i, e := range valid {
			starts[i] = e.start
			ends[i] = e.end
		}
		right := newRangeExtreme(ends, true)
		byEnd := append([]rangeEntry(nil), valid...)
		sort.Slice(byEnd, func(i, j int) bool { return byEnd[i].end < byEnd[j].end })
		endKeys := make([]int64, len(byEnd))
		startValues := make([]int64, len(byEnd))
		for i, e := range byEnd {
			endKeys[i] = e.end
			startValues[i] = e.start
		}
		left := newRangeExtreme(startValues, false)
		for _, e := range valid {
			lo := sort.Search(len(starts), func(i int) bool { return starts[i] > e.start })
			hi := sort.Search(len(starts), func(i int) bool { return starts[i] >= e.end })
			value, ok := right.query(lo, hi)
			if ok && value > e.end {
				x.crossings[e.index] = true
				continue
			}
			lo = sort.Search(len(endKeys), func(i int) bool { return endKeys[i] > e.start })
			hi = sort.Search(len(endKeys), func(i int) bool { return endKeys[i] >= e.end })
			value, ok = left.query(lo, hi)
			x.crossings[e.index] = ok && value < e.start
		}
	}
	return x
}

func (x *rangeIndex) enclosing(call Call) []int {
	g := x.groups[rangeIdentity{call.Journal, call.Process, call.Thread}]
	if g == nil {
		return nil
	}
	last := sort.Search(len(g.entries), func(i int) bool { return g.entries[i].start > call.Start })
	var indexes []int
	g.collect(1, 0, len(g.entries), last, call.End, &indexes)
	sort.Ints(indexes)
	return indexes
}

func (x *rangeIndex) crossed(index int) bool {
	return index >= 0 && index < len(x.crossings) && x.crossings[index]
}

func (g *rangeGroup) build(node, lo, hi int) int64 {
	if hi-lo == 1 {
		g.maxEnd[node] = g.entries[lo].end
		return g.maxEnd[node]
	}
	mid := lo + (hi-lo)/2
	a, b := g.build(node*2, lo, mid), g.build(node*2+1, mid, hi)
	if b > a {
		a = b
	}
	g.maxEnd[node] = a
	return a
}

func (g *rangeGroup) collect(node, lo, hi, last int, end int64, out *[]int) {
	if lo >= last || g.maxEnd[node] < end {
		return
	}
	if hi-lo == 1 {
		*out = append(*out, g.entries[lo].index)
		return
	}
	mid := lo + (hi-lo)/2
	g.collect(node*2, lo, mid, last, end, out)
	g.collect(node*2+1, mid, hi, last, end, out)
}

// rangeExtreme answers max or min queries over a half-open index interval.
// The iterative tree has O(n) storage and O(log n) query cost.
type rangeExtreme struct {
	values  []int64
	count   int
	maximum bool
}

func newRangeExtreme(values []int64, maximum bool) rangeExtreme {
	t := rangeExtreme{values: make([]int64, 2*len(values)), count: len(values), maximum: maximum}
	copy(t.values[t.count:], values)
	for i := t.count - 1; i > 0; i-- {
		t.values[i] = t.choose(t.values[2*i], t.values[2*i+1])
	}
	return t
}

func (t rangeExtreme) choose(a, b int64) int64 {
	if t.maximum && b > a || !t.maximum && b < a {
		return b
	}
	return a
}

func (t rangeExtreme) query(lo, hi int) (int64, bool) {
	var result int64
	found := false
	add := func(value int64) {
		if !found {
			result = value
			found = true
		} else {
			result = t.choose(result, value)
		}
	}
	for lo, hi = lo+t.count, hi+t.count; lo < hi; lo, hi = lo/2, hi/2 {
		if lo%2 != 0 {
			add(t.values[lo])
			lo++
		}
		if hi%2 != 0 {
			hi--
			add(t.values[hi])
		}
	}
	return result, found
}
