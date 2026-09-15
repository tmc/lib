package tileir

import (
	"bytes"
	"testing"
)

// ver134 is the file version every byte sequence below was written at.
var ver134 = BytecodeVersion{Major: 13, Minor: 4}

// dyn is MLIR's ShapedType::kDynamic as it appears in the wire format.
const dyn = DynamicDim

// vendorTypeCase pairs a type payload NVIDIA's own writer produced with the
// Go value it denotes and the type pool its indices resolve against.
//
// Every want byte string below was read out of a file written by NVIDIA's
// toolchain, not by this package. The two sources are:
//
//	capture: an nvcc-produced CUDA 13.4 module, 4671 bytes, type section at
//	  file offset 2516, 18 types. Cited per case by its type index.
//	f32_array: NVIDIA/cuda-tile
//	  test/Bytecode/encoding_stability/goldens/f32_array.golden.tileirbc,
//	  type section at file offset 76, 3 types.
//
// The layout they establish, and that encodeSingleType must reproduce:
//
//	type        =: typeTag[varint] optionalFlags[varint]? parameters
//	ptr         =: tag(12) flags elementTypeIndex[varint]
//	tile        =: tag(13) elementTypeIndex[varint] shape[LEVarSize]
//	tensor_view =: tag(14) flags elementTypeIndex[varint] shape[LEVarSize] strides[LEVarSize]
//	function    =: tag(16) numInputs[varint] in[varint]* numResults[varint] res[varint]*
//
// optionalFlags is the unified presence bitfield, present only for types that
// carry an optional parameter and only once the file version reaches both 13.3
// (when the bitfield format was introduced) and the version that added the
// type's first optional parameter. It precedes every parameter. tile has no
// optional parameter and so carries no bitfield; ptr and tensor_view gained
// theirs (ptrAttr) at 13.4 and carry one in a 13.4 file. LEVarSize is an
// element count as a varint followed by that many little-endian int64s.
//
// tile is the one type whose parameters are written in reverse declaration
// order, which is why elementTypeIndex precedes shape.
type vendorTypeCase struct {
	name string
	src  string // which vendor file, and the type index within it
	pool []Type // pool the payload's indices resolve against
	typ  Type   // the type the payload denotes
	want []byte // tag + payload, exactly as the vendor wrote it
}

func le64(vals ...int64) []byte {
	var b []byte
	return appendLEVarSize(b, vals)
}

func vendorTypeCases() []vendorTypeCase {
	f32 := Float32
	ptrF32 := NewPointer(f32)
	tilePtrF32 := NewTile(nil, ptrF32)
	capturePool := []Type{f32, ptrF32, tilePtrF32}

	fnEmpty := NewFunction(nil, nil)
	tile4F32 := NewTile([]int64{4}, f32)
	goldenPool := []Type{fnEmpty, f32, tile4F32}

	tv := &TensorViewType{
		ElementType: f32,
		Shape:       []int64{dyn, dyn},
		Strides:     []int64{dyn, 1},
	}

	return []vendorTypeCase{
		{
			name: "f32",
			src:  "capture type[0]",
			pool: capturePool,
			typ:  f32,
			want: []byte{7},
		},
		{
			// The bug this test exists for: the leading 0x00 is the
			// optional-parameter bitfield. Dropping it made the assembler
			// read the pointee index off the end of the payload and report
			// "error at offset 1: failed to get pointeeType type".
			name: "ptr<f32>",
			src:  "capture type[1]",
			pool: capturePool,
			typ:  ptrF32,
			want: []byte{12, 0x00, 0x00},
		},
		{
			name: "tile<ptr<f32>>",
			src:  "capture type[2]",
			pool: capturePool,
			typ:  tilePtrF32,
			want: []byte{13, 0x01, 0x00},
		},
		{
			name: "(tile,tile,tile)->()",
			src:  "capture type[3]",
			pool: capturePool,
			typ:  NewFunction([]Type{tilePtrF32, tilePtrF32, tilePtrF32}, nil),
			want: []byte{16, 0x03, 0x02, 0x02, 0x02, 0x00},
		},
		{
			name: "tile<1x1xf32>",
			src:  "capture type[12]",
			pool: capturePool,
			typ:  NewTile([]int64{1, 1}, f32),
			want: append([]byte{13, 0x00}, le64(1, 1)...),
		},
		{
			name: "tile<16x16xf32>",
			src:  "capture type[13]",
			pool: capturePool,
			typ:  NewTile([]int64{16, 16}, f32),
			want: append([]byte{13, 0x00}, le64(16, 16)...),
		},
		{
			name: "tensor_view<?x?xf32,strides=[?,1]>",
			src:  "capture type[16]",
			pool: capturePool,
			typ:  tv,
			want: append(append([]byte{14, 0x00, 0x00}, le64(dyn, dyn)...), le64(dyn, 1)...),
		},
		{
			name: "()->()",
			src:  "f32_array type[0]",
			pool: goldenPool,
			typ:  fnEmpty,
			want: []byte{16, 0x00, 0x00},
		},
		{
			name: "tile<4xf32>",
			src:  "f32_array type[2]",
			pool: goldenPool,
			typ:  tile4F32,
			want: append([]byte{13, 0x01}, le64(4)...),
		},
	}
}

// TestEncodeSingleTypeVendorBytes requires encodeSingleType to reproduce, byte
// for byte, type encodings NVIDIA's writer produced. Unlike a round-trip
// through our own decoder, this cannot be satisfied by an encoder and decoder
// that agree with each other but not with the vendor.
func TestEncodeSingleTypeVendorBytes(t *testing.T) {
	for _, tc := range vendorTypeCases() {
		t.Run(tc.name, func(t *testing.T) {
			got := encodeSingleType(tc.typ, tc.pool, ver134)
			if !bytes.Equal(got, tc.want) {
				t.Errorf("encodeSingleType(%s) = % x, want % x (from %s)", tc.typ, got, tc.want, tc.src)
			}
		})
	}
}

// TestDecodeTypeVendorBytes requires decodeType to read the vendor's bytes back
// as the type they denote. Paired with TestEncodeSingleTypeVendorBytes it pins
// both directions to the vendor's layout: an encoder and decoder that swapped
// two fields consistently would still emit the vendor's bytes for a payload
// whose fields happen to be equal, but would not decode to the right type.
func TestDecodeTypeVendorBytes(t *testing.T) {
	for _, tc := range vendorTypeCases() {
		t.Run(tc.name, func(t *testing.T) {
			tag, n := readVarInt(tc.want)
			got := decodeType(tag, tc.want[n:], tc.pool, ver134)
			if !typeEqual(got, tc.typ) {
				t.Errorf("decodeType(%s) = %s, want %s", tc.src, got, tc.typ)
			}
		})
	}
}

// TestOptionalParamFlagsVersionGating pins when the presence bitfield is
// written. ptr gained its optional ptrAttr parameter at 13.4, and the unified
// bitfield format itself arrived at 13.3, so a 13.4 file carries the byte and
// a 13.2 file must not.
func TestOptionalParamFlagsVersionGating(t *testing.T) {
	pool := []Type{Float32, NewPointer(Float32)}
	tests := []struct {
		ver  BytecodeVersion
		want []byte
	}{
		{BytecodeVersion{Major: 13, Minor: 1}, []byte{12, 0x00}},
		{BytecodeVersion{Major: 13, Minor: 2}, []byte{12, 0x00}},
		{BytecodeVersion{Major: 13, Minor: 4}, []byte{12, 0x00, 0x00}},
		{BytecodeVersion{Major: 14, Minor: 0}, []byte{12, 0x00, 0x00}},
	}
	for _, tt := range tests {
		t.Run(tt.ver.String(), func(t *testing.T) {
			got := encodeSingleType(pool[1], pool, tt.ver)
			if !bytes.Equal(got, tt.want) {
				t.Errorf("encodeSingleType(ptr<f32>) at %s = % x, want % x", tt.ver, got, tt.want)
			}
		})
	}
}
