package dsv4cpu

import (
	"errors"
	"math"
)

const negativeInfinity = float32(-1e30)

// Compressor owns one request's rolling projected KV and score rows.
// Ratio 4 keeps the preceding and current windows in two projection lanes;
// ratio 128 keeps one window. The zero value is not usable.
type Compressor struct {
	width, ratio, projectionWidth, position int
	kv, score                               []float32
}

// NewCompressor allocates rolling state, not model weights. Width is the
// emitted row width (512 for attention, 128 for the indexer).
func NewCompressor(width, ratio int) (*Compressor, error) {
	if width <= 0 || (ratio != 4 && ratio != 128) || width > int(^uint(0)>>1)/(ratio*4) {
		return nil, errors.New("invalid compressor shape")
	}
	lanes := 1
	if ratio == 4 {
		lanes = 2
	}
	c := &Compressor{width: width, ratio: ratio, projectionWidth: lanes * width, kv: make([]float32, lanes*ratio*lanes*width), score: make([]float32, lanes*ratio*lanes*width)}
	for i := range c.score {
		c.score[i] = negativeInfinity
	}
	return c, nil
}

// Push records one projected KV/score row and emits a pooled row at a ratio
// boundary. Score must already include the position-dependent bias. The
// emitted row still needs RMS normalization, RoPE and cache quantization.
// On false, dst is unchanged. Calls must follow consecutive positions.
func (c *Compressor) Push(dst, kv, score []float32, position int) (bool, error) {
	if c == nil || c.width == 0 || position != c.position || len(dst) != c.width || len(kv) != c.projectionWidth || len(score) != c.projectionWidth {
		return false, errors.New("invalid compressor update")
	}
	for i, v := range kv {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(score[i])) || math.IsInf(float64(score[i]), 0) {
			return false, errors.New("nonfinite compressor input")
		}
	}
	row := position % c.ratio
	if c.ratio == 4 {
		row += c.ratio
	}
	copy(c.kv[row*c.projectionWidth:], kv)
	copy(c.score[row*c.projectionWidth:], score)
	c.position++
	if c.position%c.ratio != 0 {
		return false, nil
	}
	c.pool(dst)
	if c.ratio == 4 {
		copy(c.kv[:c.ratio*c.projectionWidth], c.kv[c.ratio*c.projectionWidth:])
		copy(c.score[:c.ratio*c.projectionWidth], c.score[c.ratio*c.projectionWidth:])
		copy(c.kv[c.ratio*c.projectionWidth:], c.kv[:c.ratio*c.projectionWidth])
		copy(c.score[c.ratio*c.projectionWidth:], c.score[:c.ratio*c.projectionWidth])
	}
	return true, nil
}

func (c *Compressor) pool(out []float32) {
	for j := 0; j < c.width; j++ {
		maxScore := negativeInfinity
		for row := 0; row < c.ratio; row++ {
			s := c.score[row*c.projectionWidth+j]
			if s > maxScore {
				maxScore = s
			}
			if c.ratio == 4 {
				s = c.score[(c.ratio+row)*c.projectionWidth+c.width+j]
				if s > maxScore {
					maxScore = s
				}
			}
		}
		if maxScore <= negativeInfinity*0.5 {
			out[j] = 0
			continue
		}
		var denom, sum float32
		for row := 0; row < c.ratio; row++ {
			idx := row*c.projectionWidth + j
			wp := float32(math.Exp(float64(float32(c.score[idx] - maxScore))))
			if c.ratio == 4 {
				cur := (c.ratio+row)*c.projectionWidth + c.width + j
				wc := float32(math.Exp(float64(float32(c.score[cur] - maxScore))))
				denom += float32(wp + wc)
				sum += float32(wp * c.kv[idx])
				sum += float32(wc * c.kv[cur])
			} else {
				denom += wp
				sum += float32(wp * c.kv[idx])
			}
		}
		if denom > 0 {
			out[j] = sum / denom
		} else {
			out[j] = 0
		}
	}
}

// Reset clears all request history.
func (c *Compressor) Reset() error {
	if c == nil || c.width == 0 {
		return errors.New("invalid compressor")
	}
	clear(c.kv)
	for i := range c.score {
		c.score[i] = negativeInfinity
	}
	c.position = 0
	return nil
}
