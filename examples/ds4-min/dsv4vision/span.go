package dsv4vision

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Span owns a Flash Vision image block, including leading compression padding.
// The image-start marker through image-end marker uses the upstream visual
// attention bounds, not an ordinary causal window. Its zero value is empty.
type Span struct {
	start       int
	tokens      []int32
	values      []float32
	fingerprint [32]byte
}

// NewSpan arranges natural row-major encoder embeddings and sidecar sentinels
// into the pinned Flash token layout. All returned storage is independently
// owned. Encoder values must contain ceil(gridH/3)*ceil(gridW/3) rows of 4096.
// vocab is the language vocabulary size, not the sidecar tensor count.
func NewSpan(sidecar *Sidecar, gridH, gridW, start, vocab int, values []float32, fingerprint [32]byte) (Span, error) {
	var zero Span
	if start < 0 || vocab < 1 || vocab > math.MaxInt32-4 || gridH < 1 || gridW < 1 || gridH > MaxDimension || gridW > MaxDimension {
		return zero, fmt.Errorf("invalid image span dimensions or vocabulary")
	}
	h, w := (gridH+2)/3, (gridW+2)/3
	if len(values) != h*w*4096 {
		return zero, fmt.Errorf("invalid encoder embedding length")
	}
	types, perm, err := Layout(h, w, start)
	if err != nil {
		return zero, err
	}
	sentinels := make(map[TokenType][]float32)
	for typ, name := range map[TokenType]string{Start: "image_start", Pad: "image_pad", Newline: "image_newline", End: "image_end"} {
		r, err := sidecar.Reader(name)
		if err != nil {
			return zero, err
		}
		b := make([]byte, 4096*2)
		if _, err := io.ReadFull(r, b); err != nil {
			return zero, fmt.Errorf("read %s sentinel: %w", name, err)
		}
		v := make([]float32, 4096)
		for i := range v {
			v[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
		sentinels[typ] = v
	}
	s := Span{start: start, tokens: make([]int32, len(types)), values: make([]float32, len(types)*4096), fingerprint: fingerprint}
	imagePos := 0
	for i, typ := range types {
		s.tokens[i] = int32(vocab) + int32(typ)
		dst := s.values[i*4096 : (i+1)*4096]
		if typ == Pixel {
			if imagePos >= len(perm) || perm[imagePos] < 0 || perm[imagePos] >= h*w {
				return zero, fmt.Errorf("invalid image embedding permutation")
			}
			row := perm[imagePos]
			imagePos++
			copy(dst, values[row*4096:(row+1)*4096])
		} else {
			copy(dst, sentinels[typ])
		}
	}
	if imagePos != h*w {
		return zero, fmt.Errorf("invalid image embedding count")
	}
	return s, nil
}

// Bounds returns the half-open token interval occupied by the image block.
func (s Span) Bounds() (int, int) { return s.start, s.start + len(s.tokens) }

// Tokens returns a copy of the synthetic image token IDs.
func (s Span) Tokens() []int32 { return append([]int32(nil), s.tokens...) }

// Embeddings returns a copy of the layout-ordered 4096-wide embedding rows.
func (s Span) Embeddings() []float32 { return append([]float32(nil), s.values...) }

// Fingerprint returns the oriented image's dimension-and-pixel identity.
func (s Span) Fingerprint() [32]byte { return s.fingerprint }
