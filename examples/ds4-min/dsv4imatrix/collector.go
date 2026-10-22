package dsv4imatrix

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Collector owns bounded, expert-major sums of squared activations.
type Collector struct {
	layers, experts, input, hidden int
	gate, down                     []float32
	counts                         []uint32
	chunks                         uint32
}

// New allocates a collector within maxBytes. All dimensions must be positive.
func New(layers, experts, input, hidden int, maxBytes uint64) (*Collector, error) {
	if layers <= 0 || experts <= 0 || input <= 0 || hidden <= 0 || layers > 1024 || experts > 4096 || input > 1<<20 || hidden > 1<<20 {
		return nil, fmt.Errorf("imatrix: invalid dimensions")
	}
	n := uint64(layers) * uint64(experts)
	if uint64(experts)*uint64(input) > math.MaxInt32 || uint64(experts)*uint64(hidden) > math.MaxInt32 {
		return nil, fmt.Errorf("imatrix: entry exceeds format limit")
	}
	bytes := n * (uint64(input) + uint64(hidden) + 1) * 4
	if bytes > maxBytes || bytes > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("imatrix: allocation exceeds byte limit")
	}
	return &Collector{layers: layers, experts: experts, input: input, hidden: hidden, gate: make([]float32, int(n)*input), down: make([]float32, int(n)*hidden), counts: make([]uint32, int(n))}, nil
}

// Observe collects one token's normalized input and slot-major, route-weighted
// SwiGLU values. Invalid observations leave the collector unchanged.
func (c *Collector) Observe(layer int, ids []int, input, mid []float32) error {
	if c == nil || layer < 0 || layer >= c.layers || len(ids) == 0 || len(ids) > c.experts || len(input) != c.input || len(mid) != len(ids)*c.hidden {
		return fmt.Errorf("imatrix: invalid observation shape")
	}
	seen := make(map[int]bool, len(ids))
	for _, id := range ids {
		if id < 0 || id >= c.experts || seen[id] {
			return fmt.Errorf("imatrix: invalid expert %d", id)
		}
		seen[id] = true
		if c.counts[layer*c.experts+id] == math.MaxUint32 {
			return fmt.Errorf("imatrix: count overflow")
		}
	}
	for _, values := range [][]float32{input, mid} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("imatrix: non-finite activation")
			}
		}
	}
	if c.chunks == math.MaxInt32 {
		return fmt.Errorf("imatrix: chunk overflow")
	}
	// Check arithmetic overflow before changing any sums. Match C's float32
	// square, accumulation and reciprocal-multiply order, not float64 means.
	for slot, id := range ids {
		e := layer*c.experts + id
		for i, v := range input {
			if math.IsInf(float64(c.gate[e*c.input+i]+v*v), 0) {
				return fmt.Errorf("imatrix: sum overflow")
			}
		}
		for i, v := range mid[slot*c.hidden : (slot+1)*c.hidden] {
			if math.IsInf(float64(c.down[e*c.hidden+i]+v*v), 0) {
				return fmt.Errorf("imatrix: sum overflow")
			}
		}
	}
	for slot, id := range ids {
		e := layer*c.experts + id
		c.counts[e]++
		for i, v := range input {
			c.gate[e*c.input+i] += float32(v * v)
		}
		for i, v := range mid[slot*c.hidden : (slot+1)*c.hidden] {
			c.down[e*c.hidden+i] += float32(v * v)
		}
	}
	c.chunks++
	return nil
}

// ObserveBatch collects token-major rows, each with slots experts. One batch
// contributes one chunk to the trailer, as in the upstream collector. A failed
// batch leaves the collector unchanged. Its transactional copy temporarily
// doubles the sum storage allocated by New.
func (c *Collector) ObserveBatch(layer, slots int, ids []int, input, mid []float32) error {
	if c == nil || layer < 0 || layer >= c.layers || slots <= 0 || slots > c.experts || len(ids) == 0 || len(ids)%slots != 0 || c.chunks == math.MaxInt32 {
		return fmt.Errorf("imatrix: invalid batch")
	}
	n := len(ids) / slots
	if uint64(len(input)) != uint64(n)*uint64(c.input) || uint64(len(mid)) != uint64(len(ids))*uint64(c.hidden) {
		return fmt.Errorf("imatrix: invalid batch shape")
	}
	staged := *c
	staged.gate = append([]float32(nil), c.gate...)
	staged.down = append([]float32(nil), c.down...)
	staged.counts = append([]uint32(nil), c.counts...)
	for row := 0; row < n; row++ {
		staged.chunks = 0
		if err := staged.Observe(layer, ids[row*slots:(row+1)*slots], input[row*c.input:(row+1)*c.input], mid[row*slots*c.hidden:(row+1)*slots*c.hidden]); err != nil {
			return err
		}
	}
	staged.chunks = c.chunks + 1
	*c = staged
	return nil
}

// MinSamples returns the smallest observation count across all layer experts.
// Unobserved experts therefore make the result zero.
func (c *Collector) MinSamples() uint32 {
	if c == nil || len(c.counts) == 0 {
		return 0
	}
	n := c.counts[0]
	for _, count := range c.counts[1:] {
		if count < n {
			n = count
		}
	}
	return n
}

// Chunks returns the number of accepted layer batches.
func (c *Collector) Chunks() uint32 {
	if c == nil {
		return 0
	}
	return c.chunks
}

// Write serializes activation means using the upstream legacy dat layout.
// It does not close w. The dataset name is recorded verbatim in the trailer.
func (c *Collector) Write(w io.Writer, dataset string) error {
	if c == nil || w == nil || len(dataset) > math.MaxInt32 {
		return fmt.Errorf("imatrix: invalid output")
	}
	w = completeWriter{w}
	put := func(v any) error { return binary.Write(w, binary.LittleEndian, v) }
	if err := put(int32(c.layers * 3)); err != nil {
		return fmt.Errorf("imatrix header: %w", err)
	}
	for layer := 0; layer < c.layers; layer++ {
		for _, kind := range []string{"gate", "up", "down"} {
			name := fmt.Sprintf("blk.%d.ffn_%s_exps.weight", layer, kind)
			cols, sums := c.input, c.gate
			if kind == "down" {
				cols, sums = c.hidden, c.down
			}
			if err := put(int32(len(name))); err != nil {
				return err
			}
			if _, err := io.WriteString(w, name); err != nil {
				return err
			}
			if err := put(int32(1)); err != nil {
				return err
			}
			if err := put(int32(c.experts * cols)); err != nil {
				return err
			}
			for expert := 0; expert < c.experts; expert++ {
				e := layer*c.experts + expert
				for col := 0; col < cols; col++ {
					var v float32
					if c.counts[e] != 0 {
						inv := float32(1) / float32(c.counts[e])
						v = sums[e*cols+col] * inv
					}
					if err := put(v); err != nil {
						return err
					}
				}
			}
		}
	}
	if err := put(int32(c.chunks)); err != nil {
		return err
	}
	if err := put(int32(len(dataset))); err != nil {
		return err
	}
	_, err := io.WriteString(w, dataset)
	return err
}

// completeWriter rejects a writer that silently accepts only a prefix.
type completeWriter struct{ io.Writer }

func (w completeWriter) Write(p []byte) (int, error) {
	n, err := w.Writer.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	return n, err
}
