package dsv41

import (
	"fmt"
	"math"
)

// BF16 rounds a float32 to BF16, retaining its float32 representation.
func BF16(x float32) float32 {
	b := math.Float32bits(x)
	if b&0x7f800000 != 0x7f800000 {
		b += 0x7fff + (b>>16)&1
	}
	return math.Float32frombits(b & 0xffff0000)
}

// Pool owns the unpaired row of one ratio-two KV source. It is request-local
// and is not safe for concurrent calls. Its zero value starts at position zero.
type Pool struct {
	position   int
	kv, scores []float32
}

// Reset discards carried values and restarts at position zero.
func (p *Pool) Reset() { *p = Pool{} }

// Append pools consecutive rows into dst. dst has ((rows+start%2)/2)*width
// elements. Shape errors leave both dst and the carry unchanged.
func (p *Pool) Append(dst, kv, scores []float32, width, start int) error {
	if width <= 0 || start < 0 || start != p.position || len(kv) == 0 || len(kv)%width != 0 || len(scores) != len(kv) {
		return fmt.Errorf("invalid pool shape or position")
	}
	rows := len(kv) / width
	if start > math.MaxInt-rows || len(dst) != ((rows+start%2)/2)*width || start%2 != 0 && len(p.kv) != width {
		return fmt.Errorf("invalid pool output or carry")
	}
	if len(p.kv) != 0 && len(p.kv) != width {
		return fmt.Errorf("pool width changed")
	}
	for pair := 0; pair < len(dst)/width; pair++ {
		row := pair*2 - start%2
		for col := 0; col < width; col++ {
			var ka, sa float32
			if row < 0 {
				ka, sa = p.kv[col], p.scores[col]
			} else {
				ka, sa = kv[row*width+col], scores[row*width+col]
			}
			b := (row+1)*width + col
			peak := max(sa, scores[b])
			ea := float32(math.Exp(float64(sa - peak)))
			eb := float32(math.Exp(float64(scores[b] - peak)))
			a := float32(ka * ea)
			z := float32(kv[b] * eb)
			dst[pair*width+col] = BF16(float32(float32(a+z) / float32(ea+eb)))
		}
	}
	last := (start + rows - 1) &^ 1
	if last >= start {
		if p.kv == nil {
			p.kv = make([]float32, width)
			p.scores = make([]float32, width)
		}
		copy(p.kv, kv[(last-start)*width:(last-start+1)*width])
		copy(p.scores, scores[(last-start)*width:(last-start+1)*width])
	}
	p.position += rows
	return nil
}
