package egraph

import (
	"encoding/binary"
	"fmt"
	"math"
)

// IQ2XXSBlockSize is the number of quantized weights in an IQ2_XXS super-block.
const IQ2XXSBlockSize = 256

// IQ2XXSBlockBytes is the byte length of a packed IQ2_XXS block.
// Format: 2-byte float16 scale + 64 bytes (grid indices, signs, sub-block scales).
const IQ2XXSBlockBytes = 66

// kgrid2bit256 is the standard 256-entry 2-bit 8-element codebook grid table for IQ2_XXS.
// Each uint16 packs 8 2-bit values: entry j in {0, 1, 2, 3}, producing grid value 2*j + 1.
var kgrid2bit256 = [256]uint16{
	0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97,
	100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
	1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288,
	1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113,
	2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240,
	4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400,
	5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260,
	8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872,
	10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
	16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
	17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
	20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
	22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
	25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
	33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
	37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
}

// ksignsIQ2XS is the standard 128-entry 7-bit sign table with parity bit.
var ksignsIQ2XS = [128]uint8{
	0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
	144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
	160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
	48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
	192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
	80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
	96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
	240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
}

// halfToFloat32 decodes an IEEE 754 binary16 half-precision float into float32.
func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int32(h >> 10 & 0x1f)
	frac := uint32(h & 0x3ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		// Subnormal: normalise into a float32 normal.
		e := int32(-1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return math.Float32frombits(sign | uint32(127+e-14)<<23 | frac<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(127-15+exp)<<23 | frac<<13)
	}
}

// DequantizeIQ2XXS unpacks packed IQ2_XXS blocks into dst in float32.
// It unpacks len(block)/IQ2XXSBlockBytes blocks (each 256 weights).
//
// This is a zero-allocation fast path matching the standard GGML / DeepSeek-V4
// IQ2_XXS codebook specification.
func DequantizeIQ2XXS(dst []float32, block []byte) error {
	if len(block) == 0 || len(block)%IQ2XXSBlockBytes != 0 {
		return fmt.Errorf("egraph: invalid IQ2_XXS block length %d (must be a positive multiple of %d)", len(block), IQ2XXSBlockBytes)
	}
	nblocks := len(block) / IQ2XXSBlockBytes
	needed := nblocks * IQ2XXSBlockSize
	if len(dst) < needed {
		return fmt.Errorf("egraph: destination buffer too small: got %d, want at least %d", len(dst), needed)
	}

	for b := 0; b < nblocks; b++ {
		blk := block[b*IQ2XXSBlockBytes : (b+1)*IQ2XXSBlockBytes]
		dHalf := binary.LittleEndian.Uint16(blk[0:2])
		d := halfToFloat32(dHalf)

		qs := blk[2:66]
		outBase := b * IQ2XXSBlockSize

		for ib32 := 0; ib32 < 8; ib32++ {
			aux0 := binary.LittleEndian.Uint32(qs[8*ib32 : 8*ib32+4])
			aux1 := binary.LittleEndian.Uint32(qs[8*ib32+4 : 8*ib32+8])

			db := d * (0.5 + float32(aux1>>28)) * 0.25

			for l := 0; l < 4; l++ {
				gridIdx := byte(aux0 >> (8 * l))
				signIdx := byte((aux1 >> (7 * l)) & 0x7f)
				signs := ksignsIQ2XS[signIdx]
				gridEntry := kgrid2bit256[gridIdx]

				offset := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					lVal := (gridEntry >> (2 * j)) & 0x03
					gridVal := float32(2*lVal + 1)
					if signs&(1<<j) != 0 {
						gridVal = -gridVal
					}
					dst[offset+j] = db * gridVal
				}
			}
		}
	}
	return nil
}
