package dsv41

import (
	"fmt"
	"math"
	"sort"
)

// Candidates owns the layer-20 block mask and the selected compressed rows.
// Layers that do not refresh index scores reuse the preceding selection.
// The zero value starts an empty request; do not share it across requests.
type Candidates struct {
	mask     []bool
	selected []int
	position int
	valid    bool
}

// Reset discards the mask and selection.
func (c *Candidates) Reset() { *c = Candidates{} }

// Pick updates a source layer's selection and returns a copy of the selected
// row IDs. Scores must contain exactly the visible compressed rows. Layer 20
// admits at most 2048 eight-row blocks; the most recent block receives +Inf
// priority. Equal scores use lower row/block IDs first, including +Inf ties.
// Scores may include +/-Inf: upstream uses them for forced latest-block
// admission and masked rows. NaN has no defined ordering and is rejected.
// Invalid arguments leave the prior state unchanged.
func (c *Candidates) Pick(layer, position int, scores []float32) ([]int, error) {
	l, err := Schedule(layer)
	if err != nil {
		return nil, err
	}
	if position < 0 || position == math.MaxInt {
		return nil, fmt.Errorf("invalid candidate position")
	}
	if l.Ratio == 0 {
		return nil, nil
	}
	n := (position + 1) / l.Ratio
	if !l.RefreshIndex {
		if n == 0 {
			return nil, nil
		}
		if !c.valid || c.position != position {
			return nil, fmt.Errorf("missing index selection")
		}
		return append([]int(nil), c.selected...), nil
	}
	if len(scores) != n {
		return nil, fmt.Errorf("invalid index score count")
	}
	for _, v := range scores {
		if math.IsNaN(float64(v)) {
			return nil, fmt.Errorf("NaN index score")
		}
	}
	mask := c.mask
	if layer == 20 {
		blocks := (n + 7) / 8
		pooled := make([]float32, blocks)
		for b := range pooled {
			pooled[b] = float32(math.Inf(-1))
			for j := b * 8; j < min(n, (b+1)*8); j++ {
				pooled[b] = max(pooled[b], scores[j])
			}
		}
		if blocks > 0 {
			pooled[blocks-1] = float32(math.Inf(1))
		}
		mask = make([]bool, blocks)
		for _, b := range top(pooled, min(blocks, 2048)) {
			mask[b] = true
		}
	}
	filtered := append([]float32(nil), scores...)
	if layer > 20 {
		if len(mask) != (n+7)/8 {
			return nil, fmt.Errorf("missing candidate block mask")
		}
		for i := range filtered {
			if !mask[i/8] {
				filtered[i] = float32(math.Inf(-1))
			}
		}
	}
	selected := top(filtered, min(n, 512))
	c.mask = mask
	c.selected = selected
	c.position = position
	c.valid = true
	return append([]int(nil), selected...), nil
}

func top(scores []float32, n int) []int {
	ids := make([]int, len(scores))
	for i := range ids {
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool {
		a, b := ids[i], ids[j]
		if scores[a] == scores[b] {
			return a < b
		}
		return scores[a] > scores[b]
	})
	return ids[:n]
}
