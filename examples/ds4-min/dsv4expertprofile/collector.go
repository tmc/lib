package dsv4expertprofile

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// Layer contains expert histograms, adjacent-position locality and LRU hits.
type Layer struct {
	Records       uint64    `json:"records"`
	Hits          []uint64  `json:"hits"`
	Weights       []float64 `json:"weights"`
	AdjacentPairs uint64    `json:"adjacent_pairs"`
	OverlapSum    float64   `json:"overlap_sum"`
	JaccardSum    float64   `json:"jaccard_sum"`
	CacheHits     []uint64  `json:"cache_hits"`
	CacheWeights  []float64 `json:"cache_weights"`
}

// Collector owns layer-local routing statistics.
type Collector struct {
	layers   []Layer
	experts  int
	caps     []int
	cache    [][][]int
	previous [][]int
	position []int
}

// Records returns the total number of collected layer positions.
func (c *Collector) Records() uint64 {
	if c == nil {
		return 0
	}
	var n uint64
	for _, l := range c.layers {
		n += l.Records
	}
	return n
}

// New creates bounded layer histograms and latest-N unique caches. Capacities
// must be distinct, positive, and no larger than experts.
func New(layers, experts int, capacities []int) (*Collector, error) {
	if layers <= 0 || layers > 1024 || experts <= 0 || experts > 4096 || len(capacities) > 16 {
		return nil, fmt.Errorf("expert profile: invalid dimensions")
	}
	seen := map[int]bool{}
	for _, n := range capacities {
		if n <= 0 || n > experts || seen[n] {
			return nil, fmt.Errorf("expert profile: invalid cache capacity")
		}
		seen[n] = true
	}
	c := &Collector{layers: make([]Layer, layers), experts: experts, caps: append([]int(nil), capacities...), cache: make([][][]int, layers), previous: make([][]int, layers), position: make([]int, layers)}
	for i := range c.layers {
		c.layers[i] = Layer{Hits: make([]uint64, experts), Weights: make([]float64, experts), CacheHits: make([]uint64, len(capacities)), CacheWeights: make([]float64, len(capacities))}
		c.cache[i] = make([][]int, len(capacities))
		c.position[i] = -1
	}
	return c, nil
}

// Observe records distinct selected experts and their nonnegative weights at
// a layer position. Invalid observations do not change statistics.
func (c *Collector) Observe(layer, position int, ids []int, weights []float32) error {
	if c == nil || layer < 0 || layer >= len(c.layers) || position < 0 || len(ids) == 0 || len(ids) != len(weights) || len(ids) > c.experts {
		return fmt.Errorf("expert profile: invalid observation")
	}
	seen := map[int]bool{}
	for i, id := range ids {
		if id < 0 || id >= c.experts || seen[id] || weights[i] < 0 || math.IsNaN(float64(weights[i])) || math.IsInf(float64(weights[i]), 0) {
			return fmt.Errorf("expert profile: invalid route")
		}
		seen[id] = true
	}
	l := &c.layers[layer]
	if l.Records == math.MaxUint64 {
		return fmt.Errorf("expert profile: record overflow")
	}
	for _, id := range ids {
		if l.Hits[id] == math.MaxUint64 {
			return fmt.Errorf("expert profile: hit overflow")
		}
	}
	for _, hits := range l.CacheHits {
		if hits > math.MaxUint64-uint64(len(ids)) {
			return fmt.Errorf("expert profile: cache hit overflow")
		}
	}
	l.Records++
	for i, id := range ids {
		l.Hits[id]++
		l.Weights[id] += float64(weights[i])
	}
	if c.position[layer]+1 == position && c.previous[layer] != nil {
		overlap := 0
		for _, id := range c.previous[layer] {
			if seen[id] {
				overlap++
			}
		}
		l.AdjacentPairs++
		l.OverlapSum += float64(overlap) / float64(len(ids))
		l.JaccardSum += float64(overlap) / float64(len(ids)+len(c.previous[layer])-overlap)
	}
	for k, capacity := range c.caps {
		entries := c.cache[layer][k]
		for slot, id := range ids {
			found := -1
			for i, e := range entries {
				if e == id {
					found = i
					break
				}
			}
			if found >= 0 {
				l.CacheHits[k]++
				l.CacheWeights[k] += float64(weights[slot])
				entries = append(entries[:found], entries[found+1:]...)
			}
			entries = append([]int{id}, entries...)
			if len(entries) > capacity {
				entries = entries[:capacity]
			}
		}
		c.cache[layer][k] = entries
	}
	c.previous[layer] = append(c.previous[layer][:0], ids...)
	c.position[layer] = position
	return nil
}

// Write writes JSON statistics with explicit cache capacities. It does not
// close w or expose mutable collector storage.
func (c *Collector) Write(w io.Writer) error {
	if c == nil || w == nil {
		return fmt.Errorf("expert profile: invalid output")
	}
	return json.NewEncoder(completeWriter{w}).Encode(struct {
		Source     string  `json:"source"`
		Experts    int     `json:"experts"`
		Capacities []int   `json:"capacities"`
		Layers     []Layer `json:"layers_detail"`
	}{"ds4-min", c.experts, c.caps, c.layers})
}
