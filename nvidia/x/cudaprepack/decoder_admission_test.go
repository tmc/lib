package cudaprepack

import (
	"bytes"
	"fmt"
	"math"
	"slices"
	"testing"
)

type decoderFixture struct {
	name   string
	p      *PrepackedTensor
	want   []float32
	raw    []byte
	decode func(*PrepackedTensor, []float32) ([]byte, error)
}

func decoderFixtures(t *testing.T) []decoderFixture {
	t.Helper()
	var out []decoderFixture
	for _, dt := range []DataType{TypeFP8_E4M3, TypeFP8_E5M2} {
		want := make([]float32, 2*17)
		for i := range want {
			want[i] = float32(i%9-4) / 2
		}
		p, err := QuantizeFP8(want, 2, 17, dt)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, decoderFixture{string(dt), p, want, nil, func(p *PrepackedTensor, dst []float32) ([]byte, error) { return nil, DequantizeFP8(p, dst) }})
	}
	want := make([]float32, 2*64)
	values := []float32{-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
	for i := range want {
		want[i] = values[i%len(values)]
	}
	p, err := PackMXFP4(want, 2, 64, 32)
	if err != nil {
		t.Fatal(err)
	}
	out = append(out, decoderFixture{"MXFP4", p, want, nil, func(p *PrepackedTensor, dst []float32) ([]byte, error) { return nil, DequantizeMXFP4(p, 32, dst) }})
	raw := make([]byte, 2*2*66)
	for i := range raw {
		raw[i] = byte(i*37 + 11)
	}
	p, err = RepackIQ2_XXS(raw, 2, 512)
	if err != nil {
		t.Fatal(err)
	}
	out = append(out, decoderFixture{"IQ2_XXS", p, nil, raw, func(p *PrepackedTensor, _ []float32) ([]byte, error) { return UnpackIQ2_XXS(p) }})
	return out
}

func cloneDecoderTensor(p *PrepackedTensor) *PrepackedTensor {
	copy := *p
	copy.Payload = slices.Clone(p.Payload)
	copy.Scales = slices.Clone(p.Scales)
	copy.Metadata = slices.Clone(p.Metadata)
	return &copy
}

func decoderRefuses(t *testing.T, f decoderFixture, p *PrepackedTensor) {
	t.Helper()
	dst := make([]float32, 4096)
	for i := range dst {
		dst[i] = 123.25
	}
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("decoder panicked: %v", r)
		}
	}()
	got, err := f.decode(p, dst)
	if err == nil || got != nil {
		t.Errorf("malformed storage accepted: bytes=%d err=%v", len(got), err)
	}
	for i, v := range dst {
		if v != 123.25 {
			t.Fatalf("refused decoder changed destination at %d", i)
		}
	}
}

func TestDecoderAdmission(t *testing.T) {
	for _, f := range decoderFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			for _, tt := range []struct {
				name string
				edit func(*PrepackedTensor)
			}{
				{"type", func(p *PrepackedTensor) { p.Type = TypeFloat32 }},
				{"descriptor_type", func(p *PrepackedTensor) { p.BoxDesc.DataType = TypeFloat32 }},
				{"rank", func(p *PrepackedTensor) { p.BoxDesc.Rank = 3 }},
				{"zero_cols", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[0] = 0 }},
				{"zero_rows", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[1] = 0 }},
				{"trailing_dimension", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[2] = 2 }},
				{"cols_conversion", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[0] = uint64(math.MaxInt) + 1 }},
				{"rows_conversion", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[1] = uint64(math.MaxInt) + 1 }},
				{"output_overflow", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[1] = uint64(math.MaxInt) }},
				{"zero_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = 0 }},
				{"negative_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = -16 }},
				{"unaligned_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0]++ }},
				{"short_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] -= 16 }},
				{"payload_overflow", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = int64(math.MaxInt &^ 15) }},
				{"extra_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[1] = 16 }},
				{"last_stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[2] = 16 }},
				{"element_stride", func(p *PrepackedTensor) { p.BoxDesc.ElementStrides[0] = 2 }},
				{"swizzle", func(p *PrepackedTensor) { p.BoxDesc.Swizzle = Swizzle32B }},
				{"interleave", func(p *PrepackedTensor) { p.BoxDesc.Interleave = Interleave16B }},
				{"sparse", func(p *PrepackedTensor) { p.Metadata = []byte{1} }},
			} {
				t.Run(tt.name, func(t *testing.T) { p := cloneDecoderTensor(f.p); tt.edit(p); decoderRefuses(t, f, p) })
			}
		})
	}
}

func TestDecoderNilAndTruncated(t *testing.T) {
	for _, f := range decoderFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			t.Run("nil", func(t *testing.T) { decoderRefuses(t, f, nil) })
			t.Run("payload", func(t *testing.T) {
				p := cloneDecoderTensor(f.p)
				p.Payload = p.Payload[:len(p.Payload)-1]
				decoderRefuses(t, f, p)
			})
			if len(f.p.Scales) > 0 {
				t.Run("scales", func(t *testing.T) {
					p := cloneDecoderTensor(f.p)
					p.Scales = p.Scales[:len(p.Scales)-1]
					decoderRefuses(t, f, p)
				})
			}
			if len(f.want) > 0 {
				t.Run("destination", func(t *testing.T) {
					dst := make([]float32, len(f.want)-1)
					for i := range dst {
						dst[i] = 123.25
					}
					if _, err := f.decode(f.p, dst); err == nil {
						t.Fatal("short destination accepted")
					}
					for _, v := range dst {
						if v != 123.25 {
							t.Fatal("short destination changed")
						}
					}
				})
			}
		})
	}
}

func TestDecoderBlockAndGroupAdmission(t *testing.T) {
	for _, f := range decoderFixtures(t) {
		if f.name == "IQ2_XXS" || f.name == "MXFP4" {
			t.Run(f.name, func(t *testing.T) { p := cloneDecoderTensor(f.p); p.BoxDesc.GlobalDims[0]--; decoderRefuses(t, f, p) })
		}
		if f.name != "MXFP4" {
			continue
		}
		for _, group := range []int{0, -32, 1, 31, 33, 96, math.MaxInt} {
			t.Run(fmt.Sprint(group), func(t *testing.T) {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("group size %d panicked: %v", group, r)
					}
				}()
				dst := make([]float32, len(f.want))
				if err := DequantizeMXFP4(f.p, group, dst); err == nil {
					t.Fatalf("group size %d accepted", group)
				}
			})
		}
	}
}

func TestDecoderValidPaddedStorage(t *testing.T) {
	for _, f := range decoderFixtures(t) {
		for _, extra := range []bool{false, true} {
			t.Run(f.name+map[bool]string{false: "/encoded", true: "/extra_padding"}[extra], func(t *testing.T) {
				p := cloneDecoderTensor(f.p)
				if extra {
					rows, oldStride := int(p.BoxDesc.GlobalDims[1]), int(p.BoxDesc.GlobalStrides[0])
					stride := oldStride + 16
					payload := bytes.Repeat([]byte{0xcc}, rows*stride+9)
					for row := 0; row < rows; row++ {
						copy(payload[row*stride:], p.Payload[row*oldStride:(row+1)*oldStride])
					}
					p.Payload = payload
					p.Scales = append(p.Scales, 0xab, 0xcd)
					p.BoxDesc.GlobalStrides[0] = int64(stride)
					p.Arch = "host-only"
					p.BoxDesc.GlobalAddress = math.MaxUint64
					p.BoxDesc.BoxDims = [3]uint32{}
				}
				dst := make([]float32, len(f.want)+1)
				dst[len(f.want)] = 123.25
				got, err := f.decode(p, dst)
				if err != nil || !bytes.Equal(got, f.raw) {
					t.Fatalf("bytes=%x err=%v", got, err)
				}
				for i, want := range f.want {
					if math.Float32bits(dst[i]) != math.Float32bits(want) {
						t.Fatalf("element %d = %v, want %v", i, dst[i], want)
					}
				}
				if dst[len(f.want)] != 123.25 {
					t.Fatal("destination tail changed")
				}
			})
		}
	}
}

func TestDecoderEncodedFloatValues(t *testing.T) {
	for _, tt := range []struct {
		dt    DataType
		codes []byte
		bits  []uint32
	}{
		{TypeFP8_E4M3, []byte{0, 0x80, 1, 8, 0x38, 0x7e, 0xfe}, []uint32{0, 0x80000000, 0x3b000000, 0x3c800000, 0x3f800000, 0x43e00000, 0xc3e00000}},
		{TypeFP8_E5M2, []byte{0, 0x80, 1, 4, 0x3c, 0x7b, 0x7c, 0xfc}, []uint32{0, 0x80000000, 0x37800000, 0x38800000, 0x3f800000, 0x47600000, 0x7f800000, 0xff800000}},
	} {
		t.Run(string(tt.dt), func(t *testing.T) {
			p, err := QuantizeFP8(make([]float32, len(tt.codes)), 1, len(tt.codes), tt.dt)
			if err != nil {
				t.Fatal(err)
			}
			copy(p.Payload, tt.codes)
			dst := make([]float32, len(tt.codes))
			if err := DequantizeFP8(p, dst); err != nil {
				t.Fatal(err)
			}
			for i, bits := range tt.bits {
				if got := math.Float32bits(dst[i]); got != bits {
					t.Fatalf("code %02x = %08x, want %08x", tt.codes[i], got, bits)
				}
			}
			p.Payload[0] = 0x7f
			if err := DequantizeFP8(p, dst); err != nil || !math.IsNaN(float64(dst[0])) {
				t.Fatalf("nan encoding: %v %v", dst[0], err)
			}
		})
	}
	p, err := PackMXFP4(make([]float32, 32), 1, 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	p.Scales[0], p.Scales[1] = 0, 0x40 // binary16 scale 2
	for i := range p.Payload {
		p.Payload[i] = byte((2*i)%16 | ((2*i+1)%16)<<4)
	}
	dst := make([]float32, 32)
	if err := DequantizeMXFP4(p, 32, dst); err != nil {
		t.Fatal(err)
	}
	// The existing E2M1 table represents nibble 8 as positive zero.
	want := []float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
	for i, got := range dst {
		if math.Float32bits(got) != math.Float32bits(want[i%16]) {
			t.Fatalf("nibble %d = %v, want %v", i%16, got, want[i%16])
		}
	}
}
