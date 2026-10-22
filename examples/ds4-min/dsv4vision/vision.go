// Package dsv4vision prepares RGB images and token layouts for DeepSeek V4 Flash
// Vision. It does not implement the vision encoder.
package dsv4vision

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
)

const MaxEncodedBytes = 64 << 20
const MaxDimension = 16384
const MaxPixels = 64 << 20

// Image is an oriented, tightly packed, eight-bit RGB image.
type Image struct {
	Width, Height int
	RGB           []byte
	Fingerprint   [32]byte
}

// NewImage copies RGB pixels and computes the upstream dimension-and-pixel hash.
func NewImage(width, height int, rgb []byte) (Image, error) {
	if width < 1 || height < 1 || width > MaxDimension || height > MaxDimension || int64(width)*int64(height) > MaxPixels || len(rgb) != width*height*3 {
		return Image{}, fmt.Errorf("invalid RGB dimensions or length")
	}
	i := Image{Width: width, Height: height, RGB: append([]byte(nil), rgb...)}
	h := sha256.New()
	var dims [8]byte
	binary.LittleEndian.PutUint32(dims[:4], uint32(width))
	binary.LittleEndian.PutUint32(dims[4:], uint32(height))
	h.Write(dims[:])
	h.Write(i.RGB)
	copy(i.Fingerprint[:], h.Sum(nil))
	return i, nil
}

// Patches contains row-major patches, each in channel, y, x order.
type Patches struct {
	ContentWidth, ContentHeight int
	PaddedWidth, PaddedHeight   int
	GridWidth, GridHeight       int
	LLMGridWidth, LLMGridHeight int
	Values                      []float32
}

func gridTokens(h, w int) (int, int, int) {
	h, w = (h/14+2)/3, (w/14+2)/3
	n := h*(w+1) + 2
	if h&1 != 0 {
		n += w + 1
	}
	if ((h+1)/2*(w+1))&1 != 0 {
		n += 2
	}
	return n, h, w
}

func solve(h, w, budget int) (int, int, bool) {
	ratio := float64(h) / float64(w)
	mw := math.Sqrt(float64(budget-2)/ratio+0.25) - 0.5
	mh := mw * ratio
	if mw < 1 {
		hh := (budget - 2) / 2 &^ 1
		return hh * 42, 42, hh != 0
	}
	if mh < 2 {
		ww := (budget-2)/2 - 1
		return 84, ww * 42, ww > 1
	}
	hh, ww := int(math.Floor(mh))&^1, int(math.Floor(mw))
	if hh == 0 || ww == 0 {
		return 0, 0, false
	}
	s := math.Min(float64(ww)*42/float64(w), float64(hh)*42/float64(h))
	hh, ww = int(math.Floor(float64(h)*s/14))*14, int(math.Floor(float64(w)*s/14))*14
	return hh, ww, hh != 0 && ww != 0
}

// Preprocess applies the pinned Flash Vision resize, padding and patch order.
func Preprocess(i Image) (Patches, error) {
	if i.Width < 1 || i.Height < 1 || i.Width > MaxDimension || i.Height > MaxDimension || int64(i.Width)*int64(i.Height) > MaxPixels || len(i.RGB) != i.Width*i.Height*3 {
		return Patches{}, fmt.Errorf("invalid RGB image")
	}
	w, h := i.Width, i.Height
	if w > h*8 {
		w = h * 8
	}
	if int64(w)*int64(h) < 147456 {
		s := math.Sqrt(147456 / float64(w*h))
		w, h = max(1, int(float64(w)*s)), max(1, int(float64(h)*s))
	}
	bw, bh := (w+13)/14*14, (h+13)/14*14
	n, lh, lw := gridTokens(bh, bw)
	for budget := 381; n > 381; budget-- {
		var ok bool
		if budget <= 4 {
			return Patches{}, fmt.Errorf("image does not fit vision budget")
		}
		bh, bw, ok = solve(h, w, budget)
		if !ok {
			return Patches{}, fmt.Errorf("image does not fit vision budget")
		}
		n, lh, lw = gridTokens(bh, bw)
	}
	if bw > MaxDimension || bh > MaxDimension || int64(bw)*int64(bh) > MaxPixels {
		return Patches{}, fmt.Errorf("resized image exceeds limits")
	}
	cw, ch, ox, oy := bw, bh, 0, 0
	if i.Width < i.Height*8 {
		s := math.Min(float64(bw)/float64(i.Width), float64(bh)/float64(i.Height))
		cw, ch = min(bw, max(1, int(math.RoundToEven(float64(i.Width)*s)))), min(bh, max(1, int(math.RoundToEven(float64(i.Height)*s))))
		ox, oy = int(math.RoundToEven(float64(bw-cw)*0.5)), int(math.RoundToEven(float64(bh-ch)*0.5))
	}
	canvas := make([]float32, bw*bh*3)
	for j := range canvas {
		canvas[j] = 127
	}
	resize(i, canvas[(oy*bw+ox)*3:], cw, ch, bw)
	for j, v := range canvas {
		canvas[j] = v/127.5 - 1
	}
	p := Patches{ContentWidth: cw, ContentHeight: ch, PaddedWidth: bw, PaddedHeight: bh, GridWidth: bw / 14, GridHeight: bh / 14, LLMGridWidth: lw, LLMGridHeight: lh, Values: make([]float32, len(canvas))}
	j := 0
	for py := 0; py < p.GridHeight; py++ {
		for px := 0; px < p.GridWidth; px++ {
			for c := 0; c < 3; c++ {
				for y := 0; y < 14; y++ {
					for x := 0; x < 14; x++ {
						p.Values[j] = canvas[((py*14+y)*bw+px*14+x)*3+c]
						j++
					}
				}
			}
		}
	}
	return p, nil
}

func cubic(x float64) float64 {
	x = math.Abs(x)
	if x < 1 {
		a := float64(float64(1.5*x) - 2.5)
		b := float64(float64(a*x) * x)
		return float64(b + 1)
	}
	if x < 2 {
		a := float64(float64(-0.5*x) + 2.5)
		b := float64(float64(a*x) - 4)
		return float64(float64(b*x) + 2)
	}
	return 0
}

func resize(i Image, dst []float32, w, h, stride int) {
	sx, sy := float64(i.Width)/float64(w), float64(i.Height)/float64(h)
	fx, fy, rx, ry := 1.0, 1.0, 2.0, 2.0
	if sx >= 1 {
		fx, rx = 1/sx, 2*sx
	}
	if sy >= 1 {
		fy, ry = 1/sy, 2*sy
	}
	for y := 0; y < h; y++ {
		// A conversion fixes the center rounding before the later subtraction.
		// Without it Go may fuse that subtraction with the center product.
		cy := float64(sy * (float64(y) + 0.5))
		y0, y1 := max(0, int(cy-ry+0.5)), min(i.Height, int(cy+ry+0.5))
		for x := 0; x < w; x++ {
			cx := float64(sx * (float64(x) + 0.5))
			x0, x1 := max(0, int(cx-rx+0.5)), min(i.Width, int(cx+rx+0.5))
			var sum [3]float64
			ws := 0.0
			for yy := y0; yy < y1; yy++ {
				wy := cubic((float64(yy) + 0.5 - cy) * fy)
				for xx := x0; xx < x1; xx++ {
					weight := cubic((float64(xx)+0.5-cx)*fx) * wy
					for c := 0; c < 3; c++ {
						// Match the C product-before-accumulation rounding.
						product := float64(float64(i.RGB[(yy*i.Width+xx)*3+c]) * weight)
						sum[c] = float64(sum[c] + product)
					}
					ws += weight
				}
			}
			for c := 0; c < 3; c++ {
				dst[(y*stride+x)*3+c] = float32(min(255.0, max(0.0, math.Round(sum[c]/ws))))
			}
		}
	}
}

// TokenType identifies synthetic tokens in an image block.
type TokenType uint8

const (
	Start TokenType = iota
	Pad
	Pixel
	Newline
	End
)

// Layout returns synthetic tokens and the permutation of row-major image rows.
// Grid dimensions are the compressed LLM grid, not the 14-pixel patch grid.
func Layout(height, width, startPos int) ([]TokenType, []int, error) {
	if height < 1 || width < 1 || height > 384 || width > 384 || startPos < 0 {
		return nil, nil, fmt.Errorf("invalid image token grid")
	}
	rows := height + height%2
	rowlen := width + 1
	pad := 3 - startPos%4
	last := (rows / 2 * rowlen % 2) * 2
	count := pad + 1 + rows*rowlen + last + 1
	if count > 384 {
		return nil, nil, fmt.Errorf("image token layout exceeds model limit")
	}
	types := make([]TokenType, 0, count)
	perm := make([]int, 0, height*width)
	for j := 0; j < pad; j++ {
		types = append(types, Pad)
	}
	types = append(types, Start)
	for pair := 0; pair < rows/2; pair++ {
		for col := 0; col < rowlen; col++ {
			for r := 0; r < 2; r++ {
				row := pair*2 + r
				if row >= height {
					types = append(types, Pad)
				} else if col >= width {
					types = append(types, Newline)
				} else {
					types = append(types, Pixel)
					perm = append(perm, row*width+col)
				}
			}
		}
	}
	for j := 0; j < last; j++ {
		types = append(types, Pad)
	}
	types = append(types, End)
	return types, perm, nil
}
