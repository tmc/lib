package dsv4vision

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
)

// Decode reads a bounded PNG or JPEG and applies JPEG EXIF orientation.
// JPEG pixel rounding follows Go's decoder, not the upstream Iris decoder.
func Decode(r io.Reader) (Image, error) {
	if r == nil {
		return Image{}, fmt.Errorf("nil image reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, MaxEncodedBytes+1))
	if err != nil {
		return Image{}, fmt.Errorf("read image: %w", err)
	}
	if len(b) == 0 || len(b) > MaxEncodedBytes {
		return Image{}, fmt.Errorf("encoded image exceeds limits")
	}
	jpeg := len(b) >= 2 && b[0] == 255 && b[1] == 216
	if !jpeg && !bytes.HasPrefix(b, []byte("\x89PNG\r\n\x1a\n")) {
		return Image{}, fmt.Errorf("image must be JPEG or PNG")
	}
	c, _, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return Image{}, fmt.Errorf("decode image dimensions: %w", err)
	}
	if c.Width < 1 || c.Height < 1 || c.Width > MaxDimension || c.Height > MaxDimension || int64(c.Width)*int64(c.Height) > MaxPixels {
		return Image{}, fmt.Errorf("decoded image exceeds limits")
	}
	src, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return Image{}, fmt.Errorf("decode image: %w", err)
	}
	o := 1
	if jpeg {
		o = orientation(b)
	}
	w, h := c.Width, c.Height
	if o >= 5 {
		w, h = h, w
	}
	rgb := make([]byte, w*h*3)
	for y := 0; y < c.Height; y++ {
		for x := 0; x < c.Width; x++ {
			ox, oy := x, y
			switch o {
			case 2:
				ox = c.Width - 1 - x
			case 3:
				ox, oy = c.Width-1-x, c.Height-1-y
			case 4:
				oy = c.Height - 1 - y
			case 5:
				ox, oy = y, x
			case 6:
				ox, oy = c.Height-1-y, x
			case 7:
				ox, oy = c.Height-1-y, c.Width-1-x
			case 8:
				ox, oy = y, c.Width-1-x
			}
			v := color.NRGBAModel.Convert(src.At(x+src.Bounds().Min.X, y+src.Bounds().Min.Y)).(color.NRGBA)
			j := (oy*w + ox) * 3
			rgb[j], rgb[j+1], rgb[j+2] = v.R, v.G, v.B
		}
	}
	return NewImage(w, h, rgb)
}

func orientation(b []byte) int {
	for pos := 2; pos+4 <= len(b) && b[pos] == 255; {
		marker := b[pos+1]
		pos += 2
		if marker == 216 || marker >= 208 && marker <= 217 {
			continue
		}
		if pos+2 > len(b) {
			break
		}
		n := int(binary.BigEndian.Uint16(b[pos:]))
		if n < 2 || n > len(b)-pos {
			break
		}
		if marker == 225 && n >= 14 && bytes.Equal(b[pos+2:pos+8], []byte("Exif\x00\x00")) {
			t := b[pos+8 : pos+n]
			if len(t) < 8 {
				break
			}
			var order binary.ByteOrder
			if bytes.Equal(t[:2], []byte("II")) {
				order = binary.LittleEndian
			} else if bytes.Equal(t[:2], []byte("MM")) {
				order = binary.BigEndian
			} else {
				break
			}
			if order.Uint16(t[2:]) != 42 {
				break
			}
			off := uint64(order.Uint32(t[4:]))
			if off+2 > uint64(len(t)) {
				break
			}
			count := uint64(order.Uint16(t[off:]))
			if off+2+count*12 > uint64(len(t)) {
				break
			}
			for j := uint64(0); j < count; j++ {
				e := t[off+2+j*12:]
				if order.Uint16(e) == 274 && order.Uint16(e[2:]) == 3 && order.Uint32(e[4:]) == 1 {
					o := int(order.Uint16(e[8:]))
					if o >= 1 && o <= 8 {
						return o
					}
					return 1
				}
			}
		}
		if marker == 218 {
			break
		}
		pos += n
	}
	return 1
}
