package dsv4cpu

import (
	"errors"
	"math"
)

// Cache owns one request's raw sliding-window KV rows. It stores the same
// half-round-trip float values as the upstream scalar CPU cache. Inputs must
// already have undergone the model's rotary and non-rotary FP8 transforms.
// The zero value is not usable. A Cache must not be used concurrently.
type Cache struct {
	width, capacity, count int
	rows                   []float32
}

// NewCache allocates bounded request-local storage. Flash raw caches use
// width512 and capacity128.
func NewCache(width, capacity int) (*Cache, error) {
	if width <= 0 || capacity <= 0 || width > int(^uint(0)>>1)/capacity {
		return nil, errors.New("invalid KV cache shape")
	}
	return &Cache{width: width, capacity: capacity, rows: make([]float32, width*capacity)}, nil
}

// Push appends a row, evicting the oldest row if the sliding window is full.
func (c *Cache) Push(row []float32) error {
	if c == nil || c.width <= 0 || len(row) != c.width {
		return errors.New("invalid KV cache row")
	}
	if c.count == c.capacity {
		copy(c.rows, c.rows[c.width:])
	} else {
		c.count++
	}
	out := c.rows[(c.count-1)*c.width : c.count*c.width]
	for i, v := range row {
		out[i] = half(toHalf(v))
	}
	return nil
}

// Attend evaluates this request's current window.
func (c *Cache) Attend(dst, query, sinks []float32) error {
	if c == nil || c.width <= 0 {
		return errors.New("invalid KV cache")
	}
	return Attend(dst, query, c.rows[:c.count*c.width], sinks, c.width)
}

// Reset clears all request history.
func (c *Cache) Reset() {
	if c != nil {
		clear(c.rows)
		c.count = 0
	}
}

func toHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 16) & 0x8000
	exp := int32((b>>23)&255) - 127 + 15
	mant := b & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := mant >> shift
		round := (mant >> (shift - 1)) & 1
		sticky := mant & ((1 << (shift - 1)) - 1)
		if round != 0 && (sticky != 0 || h&1 != 0) {
			h++
		}
		return uint16(sign | h)
	}
	if exp >= 31 {
		if (b>>23)&255 == 255 && mant != 0 {
			return uint16(sign | 0x7e00)
		}
		return uint16(sign | 0x7c00)
	}
	h := sign | uint32(exp)<<10 | mant>>13
	round := mant & 0x1fff
	if round > 0x1000 || (round == 0x1000 && h&1 != 0) {
		h++
	}
	return uint16(h)
}
