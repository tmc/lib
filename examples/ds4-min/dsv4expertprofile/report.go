package dsv4expertprofile

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// Metadata identifies the producer and the router kind for every layer.
type Metadata struct {
	Source, Model string
	ExpertUsed    int
	HashRouter    []bool
}

type cacheReport struct {
	N               int     `json:"n"`
	Hits            uint64  `json:"hits"`
	Selections      uint64  `json:"selections,omitempty"`
	HitRate         float64 `json:"hit_rate"`
	WeightedHitRate float64 `json:"weighted_hit_rate"`
}
type expertReport struct {
	ID        int     `json:"id"`
	Count     uint64  `json:"count"`
	Pct       float64 `json:"pct"`
	Weight    float64 `json:"weight"`
	WeightPct float64 `json:"weight_pct"`
}
type layerReport struct {
	Layer      int            `json:"layer"`
	HashRouter bool           `json:"hash_router"`
	Records    uint64         `json:"records"`
	Selections uint64         `json:"selections"`
	Unique     int            `json:"unique_experts"`
	Overlap    float64        `json:"avg_adjacent_overlap"`
	Jaccard    float64        `json:"avg_adjacent_jaccard"`
	Top        []expertReport `json:"top_experts"`
	Cache      []cacheReport  `json:"cache"`
}

// WriteReport writes the upstream expert-locality JSON field contract. Numbers
// are emitted without the upstream display rounding; JSON bytes may differ.
func (c *Collector) WriteReport(w io.Writer, m Metadata) error {
	if c == nil || w == nil || m.Source == "" || m.Model == "" || m.ExpertUsed <= 0 || m.ExpertUsed > c.experts || len(m.HashRouter) != len(c.layers) {
		return fmt.Errorf("expert profile: invalid report metadata")
	}
	w = completeWriter{w}
	r := struct {
		Source     string        `json:"source"`
		Model      string        `json:"model"`
		Layers     int           `json:"layers"`
		Experts    int           `json:"experts"`
		Used       int           `json:"expert_used"`
		Records    uint64        `json:"layer_records"`
		Selections uint64        `json:"selections"`
		Caps       []int         `json:"cache_ns"`
		Cache      []cacheReport `json:"cache_summary"`
		Detail     []layerReport `json:"layers_detail"`
	}{Source: m.Source, Model: m.Model, Layers: len(c.layers), Experts: c.experts, Used: m.ExpertUsed, Caps: c.caps, Cache: make([]cacheReport, len(c.caps))}
	var totalWeight float64
	for il, l := range c.layers {
		d := layerReport{Layer: il, HashRouter: m.HashRouter[il], Records: l.Records, Top: []expertReport{}, Cache: make([]cacheReport, len(c.caps))}
		var weight float64
		for id, n := range l.Hits {
			d.Selections += n
			weight += l.Weights[id]
			if n != 0 {
				d.Top = append(d.Top, expertReport{ID: id, Count: n, Weight: l.Weights[id]})
			}
		}
		d.Unique = len(d.Top)
		sort.Slice(d.Top, func(i, j int) bool {
			a, b := d.Top[i], d.Top[j]
			if a.Count != b.Count {
				return a.Count > b.Count
			}
			if a.Weight != b.Weight {
				return a.Weight > b.Weight
			}
			return a.ID < b.ID
		})
		if len(d.Top) > 16 {
			d.Top = d.Top[:16]
		}
		for i := range d.Top {
			d.Top[i].Pct = 100 * ratio(float64(d.Top[i].Count), float64(d.Selections))
			d.Top[i].WeightPct = 100 * ratio(d.Top[i].Weight, weight)
		}
		d.Overlap = ratio(l.OverlapSum, float64(l.AdjacentPairs))
		d.Jaccard = ratio(l.JaccardSum, float64(l.AdjacentPairs))
		for i, n := range c.caps {
			d.Cache[i] = cacheReport{N: n, Hits: l.CacheHits[i], HitRate: ratio(float64(l.CacheHits[i]), float64(d.Selections)), WeightedHitRate: ratio(l.CacheWeights[i], weight)}
			r.Cache[i].N = n
			r.Cache[i].Hits += l.CacheHits[i]
			r.Cache[i].WeightedHitRate += l.CacheWeights[i]
		}
		r.Records += l.Records
		r.Selections += d.Selections
		totalWeight += weight
		r.Detail = append(r.Detail, d)
	}
	for i := range r.Cache {
		r.Cache[i].Selections = r.Selections
		r.Cache[i].HitRate = ratio(float64(r.Cache[i].Hits), float64(r.Selections))
		r.Cache[i].WeightedHitRate = ratio(r.Cache[i].WeightedHitRate, totalWeight)
	}
	return json.NewEncoder(w).Encode(r)
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

// WriteHotlist writes all layer/expert entries ordered by count, weight, layer
// and expert, including experts never selected.
func (c *Collector) WriteHotlist(w io.Writer, model string) error {
	if c == nil || w == nil || model == "" {
		return fmt.Errorf("expert profile: invalid hotlist output")
	}
	w = completeWriter{w}
	type entry struct {
		layer, id int
		count     uint64
		weight    float64
	}
	var entries []entry
	var records, selections uint64
	for layer, l := range c.layers {
		records += l.Records
		for id, n := range l.Hits {
			selections += n
			entries = append(entries, entry{layer, id, n, l.Weights[id]})
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.count != b.count {
			return a.count > b.count
		}
		if a.weight != b.weight {
			return a.weight > b.weight
		}
		if a.layer != b.layer {
			return a.layer < b.layer
		}
		return a.id < b.id
	})
	if _, err := fmt.Fprintf(w, "# ds4 expert hotlist v1\n# model %s\n# layers %d\n# experts %d\n# layer_records %d\n# selections %d\n# columns: layer expert hits weight\n", model, len(c.layers), c.experts, records, selections); err != nil {
		return err
	}
	for _, e := range entries {
		if _, err := fmt.Fprintf(w, "%d %d %d %.17g\n", e.layer, e.id, e.count, e.weight); err != nil {
			return err
		}
	}
	return nil
}

type completeWriter struct{ io.Writer }

func (w completeWriter) Write(p []byte) (int, error) {
	n, err := w.Writer.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	return n, err
}
