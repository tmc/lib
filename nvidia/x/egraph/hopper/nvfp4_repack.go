package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
)

// IsPositiveNormalE4M3 reports whether b is a positive normal float8_e4m3fn
// encoding (sign 0, exponent 1..15, not 0x7f NaN). For these 119 encodings,
// [ScaleE4M3ToBF16Fast] is bit-exact.
func IsPositiveNormalE4M3(b uint8) bool {
	return b&0x80 == 0 && (b>>3)&15 != 0 && b != 0x7f
}

// AllPositiveNormalE4M3 reports whether every byte in scales satisfies
// [IsPositiveNormalE4M3].
func AllPositiveNormalE4M3(scales []uint8) bool {
	for _, b := range scales {
		if !IsPositiveNormalE4M3(b) {
			return false
		}
	}
	return true
}

// ScaleE4M3ToBF16Fast returns the bfloat16 bits of FromE4M3Bits(b) * 2^118 for
// a positive normal E4M3 byte b, computed by shifting b's exponent and
// mantissa into bfloat16's fields and rebiasing by 118 + 127 - 7 = 238
// (0x7700). Multiplication of an unscaled E2M1 subnormal (code * 2^-126) by
// this scale gives code * scale * 2^-8, exact in bfloat16.
func ScaleE4M3ToBF16Fast(b uint8) uint16 {
	return uint16(b)*16 + 0x7700
}

// ScaleE4M3ToBF16 returns the bfloat16 bits of FromE4M3Bits(b) * 2^118 for any
// E4M3 byte b, including ±0, subnormals, negatives, and 0x7f/0xff NaN (which
// map to canonical bfloat16 NaN 0x7fc0). On positive normal E4M3 bytes it
// equals [ScaleE4M3ToBF16Fast].
func ScaleE4M3ToBF16(b uint8) uint16 {
	if b&0x7f == 0x7f {
		return 0x7fc0
	}
	v := math.Ldexp(float64(FromE4M3Bits(b)), 118)
	return BF16Bits(float32(v))
}

// ScaleE4M3ToBF16x2 packs [ScaleE4M3ToBF16](b) into both halves of a uint32.
func ScaleE4M3ToBF16x2(b uint8, fast bool) uint32 {
	var h uint16
	if fast {
		h = ScaleE4M3ToBF16Fast(b)
	} else {
		h = ScaleE4M3ToBF16(b)
	}
	return uint32(h) | uint32(h)<<16
}

// dequant9Src maps (register r in 0..3, code bit 4*half + bit in 0..7) to its
// bit position in the repacked 32-bit word consumed by [Dequant9].
var dequant9Src = [4][8]uint8{
	{6, 7, 8, 15, 22, 23, 24, 31},
	{2, 3, 4, 11, 18, 19, 20, 27},
	{28, 29, 30, 5, 12, 13, 14, 21},
	{0, 1, 10, 9, 16, 17, 26, 25},
}

// PackDequant9Word packs eight 4-bit E2M1 codes (code 2*r+h at bits
// 4*(2*r+h)..4*(2*r+h)+3 of codes, for register r in 0..3 and half h in 0..1)
// into one 32-bit word for [Dequant9].
func PackDequant9Word(codes uint32) uint32 {
	var q uint32
	for r := range 4 {
		for h := range 2 {
			c := (codes >> (4 * (2*r + h))) & 15
			for j := range 4 {
				q |= ((c >> j) & 1) << dequant9Src[r][4*h+j]
			}
		}
	}
	return q
}

// UnpackDequant9Word is the inverse of [PackDequant9Word]: it recovers the
// eight 4-bit E2M1 codes (code 2*r+h in nibble 2*r+h) from a [Dequant9] word.
func UnpackDequant9Word(q uint32) uint32 {
	var codes uint32
	for r := range 4 {
		for h := range 2 {
			var c uint32
			for j := range 4 {
				c |= ((q >> dequant9Src[r][4*h+j]) & 1) << j
			}
			codes |= c << (4 * (2*r + h))
		}
	}
	return codes
}

// Dequant9 decodes a repacked 32-bit word (built by [PackDequant9Word]) into
// four bf16x2 registers holding unscaled E2M1 values (FromE2M1(c) * 2^-126 as
// bfloat16 subnormals, with magnitude at bits 6..8 and sign at bit 15 of each
// 16-bit half) in 9 integer instructions:
//
//	a[0] = q & 0x81c081c0
//	a[1] = (q << 4) & 0x81c081c0
//	a[2] = rotl32(q, 10) & 0x81c081c0
//	a[3] = ((q >> 2) & 0x01000100) | ((q << 6) & 0x80c080c0)
func Dequant9(q uint32) [4]uint32 {
	const (
		m  = 0x81c081c0
		c1 = 0x01000100
		c2 = 0x80c080c0
	)
	rot10 := (q << 10) | (q >> 22)
	return [4]uint32{
		q & m,
		(q << 4) & m,
		rot10 & m,
		((q >> 2) & c1) | ((q << 6) & c2),
	}
}

// DecodeWGMMAWord decodes register j (0..3) of a WGMMA fragment word packed by
// [RepackNVFP4WGMMA]: low element's E2M1 code at nibble j, high element's at
// nibble j+4, placed as unscaled bfloat16 subnormals (FromE2M1(c) * 2^-126) in
// the two 16-bit halves.
func DecodeWGMMAWord(q uint32, j int) uint32 {
	x := q >> (4 * j)
	return ((x << 6) & 0x01c001c0) | ((x << 12) & 0x80008000)
}

// MulBF16x2 multiplies the two bfloat16 halves of a and b with round-to-nearest-even,
// matching PTX mul.rn.bf16x2.
func MulBF16x2(a, b uint32) uint32 {
	lo := BF16Bits(FromBF16Bits(uint16(a)) * FromBF16Bits(uint16(b)))
	hi := BF16Bits(FromBF16Bits(uint16(a>>16)) * FromBF16Bits(uint16(b>>16)))
	return uint32(lo) | uint32(hi)<<16
}

// RepackNVFP4Dequant9 losslessly repacks an [n, k] NVFP4 weight matrix
// (packed [n, k/2] E2M1 bytes with element 2i in the low nibble, scales
// [n, k/16] E4M3 bytes) into per-unit streams for the mma.sync dequant9 mainloop.
//
// Dimensions must satisfy n%32 == 0 and k%256 == 0. When glu is false, unit u
// (0 .. n/32-1) covers rows 32u..32u+15 (tile 0) and 32u+16..32u+31 (tile 1).
// When glu is true, unit u covers gate rows 16u..16u+15 (tile 0) and up rows
// n/2+16u .. n/2+16u+15 (tile 1).
//
// Each unit has k/256 stages of 4608 bytes: 512 bytes of E4M3 scales
// ([2 tiles][32 lanes][8 bytes], where byte 2*cc+h of lane (g, t) is row
// g+8*h, block 4*(4*hs+cc)+t) followed by 4096 bytes of [Dequant9] words
// ([4 chunks][2 tiles][32 lanes][4 words], where word s of lane (g, t) holds
// the 8 codes of k = 256*hs + 64*cc + 16*t + 4*s + 2*dp + half for rows g+8*h).
func RepackNVFP4Dequant9(packed, scales []uint8, n, k int, glu bool) ([]uint8, error) {
	if n <= 0 || n%32 != 0 || k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("repack nvfp4 dequant9: shape [%d, %d] must be a multiple of [32, 256]", n, k)
	}
	if len(packed) != n*k/2 || len(scales) != n*k/16 {
		return nil, fmt.Errorf("repack nvfp4 dequant9: [%d, %d] needs %d packed bytes and %d scales, have %d and %d",
			n, k, n*k/2, n*k/16, len(packed), len(scales))
	}
	units := n / 32
	stages := k / 256
	out := make([]uint8, units*stages*4608)
	tileRow := func(u, tile, r int) int {
		if glu {
			return tile*(n/2) + u*16 + r
		}
		return (2*u+tile)*16 + r
	}
	codeAt := func(row, col int) uint32 {
		b := packed[row*(k/2)+col/2]
		return uint32(b>>(4*(col&1))) & 15
	}
	for u := range units {
		for hs := range stages {
			base := (u*stages + hs) * 4608
			// Scales: [2 tiles][32 lanes][8 bytes: cc in 0..3, h in 0..1].
			for tile := range 2 {
				for lane := range 32 {
					g, t := lane>>2, lane&3
					sbase := base + (tile*32+lane)*8
					for cc := range 4 {
						blk := hs*16 + cc*4 + t
						for h := range 2 {
							row := tileRow(u, tile, g+8*h)
							out[sbase+2*cc+h] = scales[row*(k/16)+blk]
						}
					}
				}
			}
			// Code words: 512 + [4 cc][2 tiles][32 lanes][4 s] uint32.
			for cc := range 4 {
				for tile := range 2 {
					for lane := range 32 {
						g, t := lane>>2, lane&3
						wbase := base + 512 + ((cc*2+tile)*32+lane)*16
						for s := range 4 {
							var codes uint32
							for r := range 4 {
								h, dp := r&1, r>>1
								row := tileRow(u, tile, g+8*h)
								for half := range 2 {
									col := hs*256 + cc*64 + t*16 + s*4 + 2*dp + half
									codes |= codeAt(row, col) << (4 * (2*r + half))
								}
							}
							q := PackDequant9Word(codes)
							binary.LittleEndian.PutUint32(out[wbase+4*s:], q)
						}
					}
				}
			}
		}
	}
	return out, nil
}

// UnpackNVFP4Dequant9 reverses [RepackNVFP4Dequant9], recovering the original
// [n, k/2] packed E2M1 bytes and [n, k/16] E4M3 block scales bit-for-bit.
func UnpackNVFP4Dequant9(stream []uint8, n, k int, glu bool) (packed, scales []uint8, err error) {
	if n <= 0 || n%32 != 0 || k <= 0 || k%256 != 0 {
		return nil, nil, fmt.Errorf("unpack nvfp4 dequant9: shape [%d, %d] must be a multiple of [32, 256]", n, k)
	}
	units := n / 32
	stages := k / 256
	if len(stream) != units*stages*4608 {
		return nil, nil, fmt.Errorf("unpack nvfp4 dequant9: have %d stream bytes, want %d", len(stream), units*stages*4608)
	}
	packed = make([]uint8, n*k/2)
	scales = make([]uint8, n*k/16)
	tileRow := func(u, tile, r int) int {
		if glu {
			return tile*(n/2) + u*16 + r
		}
		return (2*u+tile)*16 + r
	}
	setCode := func(row, col int, c uint32) {
		idx := row*(k/2) + col/2
		shift := 4 * (col & 1)
		packed[idx] = (packed[idx] &^ (0xf << shift)) | uint8((c&15)<<shift)
	}
	for u := range units {
		for hs := range stages {
			base := (u*stages + hs) * 4608
			for tile := range 2 {
				for lane := range 32 {
					g, t := lane>>2, lane&3
					sbase := base + (tile*32+lane)*8
					for cc := range 4 {
						blk := hs*16 + cc*4 + t
						for h := range 2 {
							row := tileRow(u, tile, g+8*h)
							scales[row*(k/16)+blk] = stream[sbase+2*cc+h]
						}
					}
				}
			}
			for cc := range 4 {
				for tile := range 2 {
					for lane := range 32 {
						g, t := lane>>2, lane&3
						wbase := base + 512 + ((cc*2+tile)*32+lane)*16
						for s := range 4 {
							q := binary.LittleEndian.Uint32(stream[wbase+4*s:])
							codes := UnpackDequant9Word(q)
							for r := range 4 {
								h, dp := r&1, r>>1
								row := tileRow(u, tile, g+8*h)
								for half := range 2 {
									col := hs*256 + cc*64 + t*16 + s*4 + 2*dp + half
									c := (codes >> (4 * (2*r + half))) & 15
									setCode(row, col, c)
								}
							}
						}
					}
				}
			}
		}
	}
	return packed, scales, nil
}

// wgmmaTileRow returns the source row in [0, n) for tile nt (0 .. n/128-1) and
// in-tile row r (0..127) under [RepackNVFP4WGMMA]. When glu is true, rows
// 16b..16b+7 are gate rows nt*64 + 8b + 0..7 and rows 16b+8..16b+15 are the
// matching up rows n/2 + nt*64 + 8b + 0..7.
func wgmmaTileRow(n, nt, r int, glu bool) int {
	if !glu {
		return nt*128 + r
	}
	col := nt*64 + (r/16)*8 + r%8
	if (r % 16) < 8 {
		return col
	}
	return n/2 + col
}

// RepackNVFP4WGMMA losslessly repacks an [n, k] NVFP4 weight matrix (packed
// [n, k/2] E2M1 bytes, scales [n, k/16] E4M3 bytes) into per-thread WGMMA
// A-fragment order for a 256-thread consumer pair (two warpgroups of 64 rows
// each):
//   - wq has length (n/128) * (k/64) * 256 * 4 uint32 words
//     ([n/128][k/64][256 threads][4 ks]), where nibble j of word ks holds the
//     low element's E2M1 code of A-register j and nibble j+4 the high element's;
//   - ws has length (n/128) * (k/64) * 256 * 8 E4M3 bytes
//     ([n/128][k/64][256 threads][8: 2*ks + h]), holding the E4M3 scale of
//     rows rg + 8*h at k-block 4*kb + ks.
//
// Dimensions must satisfy n%128 == 0 and k%64 == 0.
func RepackNVFP4WGMMA(packed, scales []uint8, n, k int, glu bool) (wq []uint32, ws []uint8, err error) {
	if n <= 0 || n%128 != 0 || k <= 0 || k%64 != 0 {
		return nil, nil, fmt.Errorf("repack nvfp4 wgmma: shape [%d, %d] must be a multiple of [128, 64]", n, k)
	}
	if len(packed) != n*k/2 || len(scales) != n*k/16 {
		return nil, nil, fmt.Errorf("repack nvfp4 wgmma: [%d, %d] needs %d packed bytes and %d scales, have %d and %d",
			n, k, n*k/2, n*k/16, len(packed), len(scales))
	}
	ntiles := n / 128
	kblocks := k / 64
	wq = make([]uint32, ntiles*kblocks*256*4)
	ws = make([]uint8, ntiles*kblocks*256*8)
	codeAt := func(row, col int) uint32 {
		b := packed[row*(k/2)+col/2]
		return uint32(b>>(4*(col&1))) & 15
	}
	koff := [4]int{0, 0, 8, 8}
	for nt := range ntiles {
		for kb := range kblocks {
			for th := range 256 {
				cw, warp, lane := th/128, (th%128)/32, th%32
				g, tq := lane/4, lane%4
				rg := 64*cw + 16*warp + g
				r0 := wgmmaTileRow(n, nt, rg, glu)
				r1 := wgmmaTileRow(n, nt, rg+8, glu)
				trows := [4]int{r0, r1, r0, r1}
				wIdx := ((nt*kblocks+kb)*256 + th) * 4
				sIdx := ((nt*kblocks+kb)*256 + th) * 8
				for ks := range 4 {
					var word uint32
					for j := range 4 {
						row := trows[j]
						col := kb*64 + ks*16 + koff[j] + 2*tq
						lo := codeAt(row, col)
						hi := codeAt(row, col+1)
						word |= (lo << (4 * j)) | (hi << (16 + 4*j))
					}
					wq[wIdx+ks] = word
					grp := kb*4 + ks
					ws[sIdx+2*ks] = scales[r0*(k/16)+grp]
					ws[sIdx+2*ks+1] = scales[r1*(k/16)+grp]
				}
			}
		}
	}
	return wq, ws, nil
}

// UnpackNVFP4WGMMA reverses [RepackNVFP4WGMMA], recovering the original
// [n, k/2] packed E2M1 bytes and [n, k/16] E4M3 block scales bit-for-bit.
func UnpackNVFP4WGMMA(wq []uint32, ws []uint8, n, k int, glu bool) (packed, scales []uint8, err error) {
	if n <= 0 || n%128 != 0 || k <= 0 || k%64 != 0 {
		return nil, nil, fmt.Errorf("unpack nvfp4 wgmma: shape [%d, %d] must be a multiple of [128, 64]", n, k)
	}
	ntiles := n / 128
	kblocks := k / 64
	if len(wq) != ntiles*kblocks*256*4 || len(ws) != ntiles*kblocks*256*8 {
		return nil, nil, fmt.Errorf("unpack nvfp4 wgmma: have %d words and %d scale bytes, want %d and %d",
			len(wq), len(ws), ntiles*kblocks*256*4, ntiles*kblocks*256*8)
	}
	packed = make([]uint8, n*k/2)
	scales = make([]uint8, n*k/16)
	setCode := func(row, col int, c uint32) {
		idx := row*(k/2) + col/2
		shift := 4 * (col & 1)
		packed[idx] = (packed[idx] &^ (0xf << shift)) | uint8((c&15)<<shift)
	}
	koff := [4]int{0, 0, 8, 8}
	for nt := range ntiles {
		for kb := range kblocks {
			for th := range 256 {
				cw, warp, lane := th/128, (th%128)/32, th%32
				g, tq := lane/4, lane%4
				rg := 64*cw + 16*warp + g
				r0 := wgmmaTileRow(n, nt, rg, glu)
				r1 := wgmmaTileRow(n, nt, rg+8, glu)
				trows := [4]int{r0, r1, r0, r1}
				wIdx := ((nt*kblocks+kb)*256 + th) * 4
				sIdx := ((nt*kblocks+kb)*256 + th) * 8
				for ks := range 4 {
					word := wq[wIdx+ks]
					for j := range 4 {
						row := trows[j]
						col := kb*64 + ks*16 + koff[j] + 2*tq
						lo := (word >> (4 * j)) & 15
						hi := (word >> (16 + 4*j)) & 15
						setCode(row, col, lo)
						setCode(row, col+1, hi)
					}
					grp := kb*4 + ks
					if tq == 0 {
						scales[r0*(k/16)+grp] = ws[sIdx+2*ks]
						scales[r1*(k/16)+grp] = ws[sIdx+2*ks+1]
					}
				}
			}
		}
	}
	return packed, scales, nil
}
