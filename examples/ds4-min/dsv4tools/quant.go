package dsv4tools

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const q81Width = 32

const (
	Q4KBlockSize   = 144
	Q4KBlockValues = 256
	MXFP4BlockSize = 17
	MXFP4Values    = 32
)

// DequantizeQ4K decodes one Q4_K block for host-side reference tests. It does
// not dispatch a backend or imply that a serving backend accepts Q4_K.
func DequantizeQ4K(block []byte) ([]float32, error) {
	if len(block) != Q4KBlockSize {
		return nil, fmt.Errorf("Q4_K block has %d bytes, want %d", len(block), Q4KBlockSize)
	}
	d := halfToFloat(binary.LittleEndian.Uint16(block))
	dmin := halfToFloat(binary.LittleEndian.Uint16(block[2:]))
	scales, qs := block[4:16], block[16:]
	out := make([]float32, Q4KBlockValues)
	for group := 0; group < Q4KBlockValues/32; group++ {
		scale, min := q4KScaleMin(group, scales)
		for i := 0; i < 32; i++ {
			q := (qs[(group/2)*32+i] >> uint((group&1)*4)) & 0xf
			out[group*32+i] = d*float32(scale)*float32(q) - dmin*float32(min)
		}
	}
	return out, nil
}

// ReferenceQ4K computes a host reference dot product over whole Q4_K blocks.
// It is an oracle for serialized fixtures only; physical CUDA/Metal dispatch
// remains UNMEASURED here.
func ReferenceQ4K(weights []byte, input []float32) (float32, error) {
	if len(weights) == 0 || len(weights)%Q4KBlockSize != 0 || len(input) != len(weights)/Q4KBlockSize*Q4KBlockValues {
		return 0, errors.New("Q4_K input does not contain whole blocks")
	}
	var sum float32
	for block := 0; block < len(weights)/Q4KBlockSize; block++ {
		start := block * Q4KBlockSize
		values, err := DequantizeQ4K(weights[start : start+Q4KBlockSize])
		if err != nil {
			return 0, fmt.Errorf("Q4_K block %d: %w", block, err)
		}
		for i, value := range values {
			sum += value * input[block*Q4KBlockValues+i]
		}
	}
	return sum, nil
}

// DequantizeMXFP4 decodes one 17-byte MXFP4 block. Nibbles in the low half
// precede nibbles in the high half, matching the upstream file layout.
func DequantizeMXFP4(block []byte) ([]float32, error) {
	if len(block) != MXFP4BlockSize {
		return nil, fmt.Errorf("MXFP4 block has %d bytes, want %d", len(block), MXFP4BlockSize)
	}
	scale := e8m0ToFloat(block[0])
	values := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6, math.Float32frombits(0x80000000), -.5, -1, -1.5, -2, -3, -4, -6}
	out := make([]float32, MXFP4Values)
	for i, packed := range block[1:] {
		out[i] = scale * values[packed&0xf]
		out[i+16] = scale * values[packed>>4]
	}
	return out, nil
}

// ReferenceMXFP4 computes a host reference dot product over whole MXFP4
// blocks. It is an oracle only; physical backend qualification is UNMEASURED.
func ReferenceMXFP4(weights []byte, input []float32) (float32, error) {
	if len(weights) == 0 || len(weights)%MXFP4BlockSize != 0 || len(input) != len(weights)/MXFP4BlockSize*MXFP4Values {
		return 0, errors.New("MXFP4 input does not contain whole blocks")
	}
	var sum float32
	for block := 0; block < len(weights)/MXFP4BlockSize; block++ {
		start := block * MXFP4BlockSize
		values, err := DequantizeMXFP4(weights[start : start+MXFP4BlockSize])
		if err != nil {
			return 0, fmt.Errorf("MXFP4 block %d: %w", block, err)
		}
		for i, value := range values {
			sum += value * input[block*MXFP4Values+i]
		}
	}
	return sum, nil
}

func q4KScaleMin(group int, scales []byte) (scale, min uint8) {
	if group < 4 {
		return scales[group] & 63, scales[group+4] & 63
	}
	return (scales[group+4] & 15) | ((scales[group-4] >> 6) << 4),
		(scales[group+4] >> 4) | ((scales[group] >> 6) << 4)
}

func e8m0ToFloat(exponent byte) float32 {
	bits := uint32(exponent) << 23
	if exponent == 0 {
		bits = 0x00400000
	}
	return math.Float32frombits(bits)
}

func halfToFloat(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := (bits >> 10) & 0x1f
	fraction := uint32(bits & 0x3ff)
	switch exponent {
	case 0:
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		value := float32(math.Ldexp(float64(fraction), -24))
		if sign != 0 {
			return -value
		}
		return value
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	default:
		return math.Float32frombits(sign | uint32(exponent+112)<<23 | fraction<<13)
	}
}

// Q81Block is the host representation of one Q8_1 activation block. The
// scale and sum are IEEE binary16 bits; quants are signed int8 values.
type Q81Block struct {
	Scale  uint16
	Sum    uint16
	Quants [q81Width]int8
}

// ReferenceQ81 computes the declared Q8_1 transform using float64 arithmetic.
// It is a host oracle only; it does not dispatch or qualify CUDA kernels.
func ReferenceQ81(values []float32) ([]Q81Block, error) {
	if len(values) == 0 || len(values)%q81Width != 0 {
		return nil, errors.New("Q8_1 input length is not a positive multiple of 32")
	}
	blocks := make([]Q81Block, len(values)/q81Width)
	for b := range blocks {
		row := values[b*q81Width : (b+1)*q81Width]
		var amax, sum float64
		for _, value := range row {
			v := float64(value)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("Q8_1 input block %d is non-finite", b)
			}
			amax = math.Max(amax, math.Abs(v))
			sum += v
		}
		d := amax / 127
		blocks[b].Scale = float32ToHalf(float32(d))
		blocks[b].Sum = float32ToHalf(float32(sum))
		if d != 0 {
			for i, value := range row {
				q := math.Round(float64(value) / d)
				if q < -127 {
					q = -127
				}
				if q > 127 {
					q = 127
				}
				blocks[b].Quants[i] = int8(q)
			}
		}
	}
	return blocks, nil
}

// ValidateQ81 compares a claimed transform with the float64 host reference.
// It is intended for serialized fixtures and mutation tests, not device work.
func ValidateQ81(values []float32, got []Q81Block) error {
	want, err := ReferenceQ81(values)
	if err != nil {
		return err
	}
	if len(got) != len(want) {
		return fmt.Errorf("Q8_1 block count %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("Q8_1 block %d differs from reference", i)
		}
	}
	return nil
}

// ExpertLayoutRecord describes one routed-expert layer without tying the
// host validator to a GGUF reader or a backend implementation.
type ExpertLayoutRecord struct {
	Layer    int
	GateType string
	UpType   string
	DownType string
	GateDims []uint64
	UpDims   []uint64
	DownDims []uint64
}

// ValidateExpertLayout validates the 43-layer DSV4 routed-expert contract.
// Only the three declared layouts are accepted; this function performs no
// kernel dispatch and does not imply support for a quantized profile.
func ValidateExpertLayout(records []ExpertLayoutRecord) error {
	if len(records) != 43 {
		return fmt.Errorf("expert layout has %d layers, want 43", len(records))
	}
	for i, record := range records {
		if record.Layer != i {
			return fmt.Errorf("expert layer %d has ordinal %d", i, record.Layer)
		}
		if !validExpertPair(record.GateType, record.UpType, record.DownType) {
			return fmt.Errorf("expert layer %d has unsupported types %s/%s/%s", i, record.GateType, record.UpType, record.DownType)
		}
		if !sameDims(record.GateDims, 4096, 2048, 256) || !sameDims(record.UpDims, 4096, 2048, 256) || !sameDims(record.DownDims, 2048, 4096, 256) {
			return fmt.Errorf("expert layer %d has invalid dimensions", i)
		}
	}
	return nil
}

func validExpertPair(gate, up, down string) bool {
	return gate == up && ((gate == "IQ2_XXS" && down == "Q2_K") || (gate == "Q4_K" && down == "Q4_K") || (gate == "MXFP4" && down == "MXFP4"))
}

func sameDims(got []uint64, want ...uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func float32ToHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits >> 16 & 0x8000)
	rawExp := int(bits >> 23 & 0xff)
	mant := bits & 0x7fffff
	if rawExp == 0xff {
		return sign | 0x7c00
	}
	exp := rawExp - 127 + 15
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		cut := uint32(1 << (shift - 1))
		value := mant >> shift
		remainder := mant & uint32((1<<shift)-1)
		if remainder > cut || remainder == cut && value&1 != 0 {
			value++
		}
		return sign | uint16(value)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	rounded := mant >> 13
	remainder := mant & 0x1fff
	if remainder > 0x1000 || remainder == 0x1000 && rounded&1 != 0 {
		rounded++
		if rounded == 0x400 {
			exp++
			rounded = 0
			if exp >= 31 {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exp<<10) | uint16(rounded)
}

// EncodeQ81 is a compact fixture format: two little-endian half fields and
// 32 signed quant codes per block. It is useful for host-side receipts only.
func EncodeQ81(blocks []Q81Block) []byte {
	out := make([]byte, len(blocks)*36)
	for i, block := range blocks {
		p := out[i*36:]
		binary.LittleEndian.PutUint16(p, block.Scale)
		binary.LittleEndian.PutUint16(p[2:], block.Sum)
		for j, q := range block.Quants {
			p[4+j] = byte(q)
		}
	}
	return out
}
