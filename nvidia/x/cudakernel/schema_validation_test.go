package cudakernel_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestSchemaValidationConstructor(t *testing.T) {
	tests := []struct {
		name   string
		params []cudakernel.Param
	}{
		{"u64 narrow", []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 1, Alignment: 1}}},
		{"f32 narrow", []cudakernel.Param{{Name: "n", Type: "f32", ByteWidth: 1, Alignment: 1}}},
		{"pointer narrow", []cudakernel.Param{{Name: "n", Type: "f32", IsPointer: true, ByteWidth: 1, Alignment: 1}}},
		{"negative width", []cudakernel.Param{{Name: "n", Type: "u32", ByteWidth: -1}}},
		{"negative alignment", []cudakernel.Param{{Name: "n", Type: "u32", Alignment: -1}}},
		{"negative offset", []cudakernel.Param{{Name: "n", Type: "u32", Offset: -1}}},
		{"nonpower alignment", []cudakernel.Param{{Name: "n", Type: "u32", Alignment: 3}}},
		{"large alignment", []cudakernel.Param{{Name: "n", Type: "u32", Alignment: 32}}},
		{"empty name", []cudakernel.Param{{Type: "u32"}}},
		{"nul name", []cudakernel.Param{{Name: "n\x00", Type: "u32"}}},
		{"empty scalar type", []cudakernel.Param{{Name: "n", ByteWidth: 4}}},
		{"nul type", []cudakernel.Param{{Name: "n", Type: "C\x00", ByteWidth: 4}}},
		{"ptr without flag", []cudakernel.Param{{Name: "n", Type: "ptr"}}},
		{"opaque without width", []cudakernel.Param{{Name: "n", Type: "Point"}}},
		{"duplicate names", []cudakernel.Param{{Name: "n", Type: "u32"}, {Name: "n", Type: "u64"}}},
		{"total padding overflow", []cudakernel.Param{{Name: "n", Type: "Point", ByteWidth: math.MaxInt, Alignment: 1}}},
		{"parameter addition overflow", []cudakernel.Param{{Name: "a", Type: "Point", ByteWidth: math.MaxInt - 7, Alignment: 1}, {Name: "b", Type: "Point", ByteWidth: 8, Alignment: 1}}},
		{"parameter padding overflow", []cudakernel.Param{{Name: "a", Type: "Point", ByteWidth: math.MaxInt - 6, Alignment: 1}, {Name: "b", Type: "Point", ByteWidth: 1, Alignment: 8}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := cudakernel.NewSchema(tt.params...)
			if err == nil || s.Params != nil || s.TotalParamBytes != 0 {
				t.Fatalf("invalid constructor = %+v, %v", s, err)
			}
		})
	}
}

func TestSchemaValidationRaw(t *testing.T) {
	tests := []struct {
		name   string
		schema cudakernel.Schema
	}{
		{"original u64 narrow", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 1, Alignment: 1}}, TotalParamBytes: 8}},
		{"pointer narrow", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "ptr", IsPointer: true, ByteWidth: 1, Alignment: 1}}, TotalParamBytes: 8}},
		{"negative total", cudakernel.Schema{TotalParamBytes: -1}},
		{"negative offset", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 8, Alignment: 8, Offset: -8}}, TotalParamBytes: 8}},
		{"bad alignment", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 8, Alignment: 32}}, TotalParamBytes: 32}},
		{"misaligned offset", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 8, Alignment: 8, Offset: 1}}, TotalParamBytes: 16}},
		{"overlap", cudakernel.Schema{Params: []cudakernel.Param{{Name: "a", Type: "u64", ByteWidth: 8, Alignment: 8}, {Name: "b", Type: "u32", ByteWidth: 4, Alignment: 4, Offset: 4}}, TotalParamBytes: 16}},
		{"offset addition overflow", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 8, Alignment: 1, Offset: math.MaxInt - 3}}, TotalParamBytes: math.MaxInt}},
		{"total padding overflow", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u8", ByteWidth: 1, Alignment: 1, Offset: math.MaxInt - 1}}, TotalParamBytes: math.MaxInt}},
		{"short total", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u64", ByteWidth: 8, Alignment: 8}}, TotalParamBytes: 0}},
		{"unpadded total", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u8", ByteWidth: 1, Alignment: 1}}, TotalParamBytes: 1}},
		{"duplicate name", cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u8", ByteWidth: 1, Alignment: 1}, {Name: "n", Type: "u8", ByteWidth: 1, Alignment: 1, Offset: 1}}, TotalParamBytes: 8}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.schema.Validate(); err == nil {
				t.Fatal("invalid schema validated")
			}
			args := make([]any, len(tt.schema.Params))
			for i := range args {
				args[i] = uint64(1)
			}
			if got, err := tt.schema.Pack(args...); err == nil || got != nil {
				t.Fatalf("invalid pack = %v, %v", got, err)
			}
			if got, err := tt.schema.PackPointers(args...); err == nil || got != nil {
				t.Fatalf("invalid pointers = %v, %v", got, err)
			}
			if a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n"), "entry", tt.schema, "sm_80"); err == nil || a != nil {
				t.Fatalf("invalid artifact = %v, %v", a, err)
			}
		})
	}
}

func TestSchemaValidationScalarPacking(t *testing.T) {
	tests := []struct {
		typ  string
		arg  any
		want []byte
	}{
		{"bool", true, []byte{1}}, {"u8", uint8(0xab), []byte{0xab}}, {"b8", uint8(0xab), []byte{0xab}}, {"s8", int8(-2), []byte{0xfe}},
		{"u16", uint16(0x1234), []byte{0x34, 0x12}}, {"b16", uint16(0x1234), []byte{0x34, 0x12}}, {"s16", int16(-2), []byte{0xfe, 0xff}},
		{"u32", uint32(0x12345678), []byte{0x78, 0x56, 0x34, 0x12}}, {"b32", uint32(0x12345678), []byte{0x78, 0x56, 0x34, 0x12}}, {"s32", int32(-2), []byte{0xfe, 0xff, 0xff, 0xff}},
		{"u64", uint64(0x123456789abcdef0), []byte{0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12}}, {"b64", uint64(0x123456789abcdef0), []byte{0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12}}, {"s64", int64(-2), []byte{0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
		{"f32", float32(1), []byte{0, 0, 0x80, 0x3f}}, {"f64", float64(1), []byte{0, 0, 0, 0, 0, 0, 0xf0, 0x3f}},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			s, err := cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: tt.typ})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
			got, err := s.Pack(tt.arg)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 8 || !bytes.Equal(got[:len(tt.want)], tt.want) || !bytes.Equal(got[len(tt.want):], make([]byte, 8-len(tt.want))) {
				t.Fatalf("packed %x, want %x plus zero padding", got, tt.want)
			}
		})
	}
	for _, typ := range []string{"ptr", "u64", "f32", "float *", ""} {
		t.Run("pointer "+typ, func(t *testing.T) {
			s, err := cudakernel.NewSchema(cudakernel.Param{Name: "p", Type: typ, IsPointer: true, Alignment: 16})
			if err != nil {
				t.Fatal(err)
			}
			got, err := s.Pack(uint64(0x1000))
			if err != nil || len(got) != 16 || binary.LittleEndian.Uint64(got) != 0x1000 {
				t.Fatalf("pointer %x %v", got, err)
			}
		})
	}
	s, err := cudakernel.NewSchema(cudakernel.Param{Name: "small", Type: "u8"}, cudakernel.Param{Name: "wide", Type: "u64"})
	if err != nil {
		t.Fatal(err)
	}
	if s.Params[1].Offset != 8 || s.TotalParamBytes != 16 {
		t.Fatalf("layout %+v", s)
	}
	empty, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	if got, err := empty.Pack(); err != nil || len(got) != 0 {
		t.Fatalf("empty %v %v", got, err)
	}
}

func TestSchemaValidationOpaqueMetadata(t *testing.T) {
	tests := []struct {
		typ          string
		width, align int
	}{{"Point", 16, 8}, {"Pair", 32, 16}, {"Nested", 20, 4}, {"float *", 8, 8}, {"int", 4, 4}, {"f16", 2, 2}, {"bf16", 2, 2}, {"tf32", 4, 4}, {"b128", 16, 16}}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			s, err := cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: tt.typ, ByteWidth: tt.width, Alignment: tt.align})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
			if got, err := s.Pack([]byte{1}); err == nil || got != nil || !strings.Contains(err.Error(), "unsupported parameter type") {
				t.Fatalf("unsupported pack %v %v", got, err)
			}
			if a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n"), "entry", s, "sm_80"); err != nil || a == nil {
				t.Fatalf("opaque artifact metadata %v %v", a, err)
			}
		})
	}
	// Supported layouts may deliberately retain aligned gaps and extra padding.
	s := cudakernel.Schema{Params: []cudakernel.Param{{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4, Offset: 8}}, TotalParamBytes: 24}
	if got, err := s.Pack(uint32(7)); err != nil || len(got) != 24 || binary.LittleEndian.Uint32(got[8:]) != 7 {
		t.Fatalf("explicit padded layout %v %v", got, err)
	}
}

func TestSchemaValidationArtifactRoundTrip(t *testing.T) {
	s, err := cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: "u64"})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n"), "entry", s, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var got cudakernel.Artifact
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	packed, err := got.Schema.Pack(uint64(1))
	if err != nil || binary.LittleEndian.Uint64(packed) != 1 {
		t.Fatalf("roundtrip pack %v %v", packed, err)
	}
	got.Schema.Params[0].ByteWidth = 1
	if err := got.Validate(); err == nil {
		t.Fatal("artifact mutation admitted")
	}
	bad, err := json.Marshal(&got)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(bad, &got); err == nil {
		t.Fatal("artifact decoder admitted narrow scalar")
	}
}

func TestSchemaValidationDeviceAddress(t *testing.T) {
	tests := []struct {
		name, typ string
		arg       any
		want      uint64
	}{
		{"FP32 input", "f32", driver.DevicePtr(0x1004), 0x1004},
		{"FP32 output", "f32", uint64(0x2004), 0x2004},
		{"FP32 scratch", "f32", uintptr(0x3004), 0x3004},
		{"unknown byte pointee", "ptr", uint64(0x8003), 0x8003},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := cudakernel.NewSchema(cudakernel.Param{Name: "prefix", Type: "u8"}, cudakernel.Param{Name: "address", Type: tt.typ, IsPointer: true})
			if err != nil {
				t.Fatal(err)
			}
			if s.Params[1].Offset != 8 || s.Params[1].ByteWidth != 8 || s.Params[1].Alignment != 8 || s.TotalParamBytes != 16 {
				t.Fatalf("pointer bank layout %+v", s)
			}
			got, err := s.Pack(uint8(7), tt.arg)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 16 || got[0] != 7 || binary.LittleEndian.Uint64(got[8:16]) != tt.want || !bytes.Equal(got[1:8], make([]byte, 7)) {
				t.Fatalf("numeric address bytes %x, want 0x%x at aligned bank offset8", got, tt.want)
			}
		})
	}
}
