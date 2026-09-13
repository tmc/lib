// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestBoxDescriptorAlignment(t *testing.T) {
	tests := []struct {
		name    string
		desc    BoxDescriptor
		arch    string
		wantErr string
	}{
		{
			name: "valid sm_121a 2d box",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat16,
				GlobalAddress: 0x1000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{128, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1}, // 64 * 2 bytes = 128 bytes
				ElementStrides: [3]uint32{1, 1, 1},
				Swizzle:       SwizzleNone,
			},
			arch:    ArchSM121a,
			wantErr: "",
		},
		{
			name: "unaligned global address",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat16,
				GlobalAddress: 0x1008, // Not 16-byte aligned (offset 8)
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{128, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1},
				ElementStrides: [3]uint32{1, 1, 1},
			},
			arch:    ArchSM121a,
			wantErr: "not 16-byte aligned",
		},
		{
			name: "box width exceeds sm_121a 128-byte limit",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat32,
				GlobalAddress: 0x2000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{256, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1}, // 64 * 4 bytes = 256 bytes > 128 bytes
				ElementStrides: [3]uint32{1, 1, 1},
			},
			arch:    ArchSM121a,
			wantErr: "exceeds sm_121a limit 128 bytes",
		},
		{
			name: "box width 256 bytes permitted on sm_100a unswizzled",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat32,
				GlobalAddress: 0x2000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{256, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1}, // 256 bytes
				ElementStrides: [3]uint32{1, 1, 1},
				Swizzle:       SwizzleNone,
			},
			arch:    ArchSM100a,
			wantErr: "",
		},
		{
			name: "sm_121a rejects tcgen05 atom swizzle",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat16,
				GlobalAddress: 0x1000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{128, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1},
				ElementStrides: [3]uint32{1, 1, 1},
				Swizzle:       Swizzle128BAtom32B,
			},
			arch:    ArchSM121a,
			wantErr: "rejects tcgen05 swizzle mode",
		},
		{
			name: "sm_100a accepts tcgen05 atom swizzle",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat16,
				GlobalAddress: 0x1000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{128, 0, 0},
				BoxDims:       [3]uint32{64, 32, 1},
				ElementStrides: [3]uint32{1, 1, 1},
				Swizzle:       Swizzle128BAtom32B,
			},
			arch:    ArchSM100a,
			wantErr: "",
		},
		{
			name: "unaligned row stride",
			desc: BoxDescriptor{
				Rank:          2,
				DataType:      TypeFloat16,
				GlobalAddress: 0x1000,
				GlobalDims:    [3]uint64{64, 64, 1},
				GlobalStrides: [3]int64{130, 0, 0}, // 130 not divisible by 16
				BoxDims:       [3]uint32{64, 32, 1},
				ElementStrides: [3]uint32{1, 1, 1},
			},
			arch:    ArchSM121a,
			wantErr: "row stride 130 must be multiple of 16 bytes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.desc.Validate(tt.arch)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("got error %v, want substring %q", err, tt.wantErr)
				}
			}
		})
	}
}

func TestBoxStrideCalculations(t *testing.T) {
	// 2D Transformer test
	t.Run("Transform2D Padding and Strides", func(t *testing.T) {
		lt, err := NewLayoutTransformer(ArchSM121a, TypeFP8_E4M3, WithAlignment(16))
		if err != nil {
			t.Fatalf("NewLayoutTransformer failed: %v", err)
		}

		rows, cols := 32, 100 // 100 bytes raw row -> padded to 112 bytes (multiple of 16)
		raw := make([]byte, rows*cols)
		for i := range raw {
			raw[i] = byte(i & 0xFF)
		}

		p, err := lt.Transform2D(raw, rows, cols)
		if err != nil {
			t.Fatalf("Transform2D failed: %v", err)
		}

		if p.BoxDesc.GlobalStrides[0] != 112 {
			t.Errorf("expected padded row stride 112, got %d", p.BoxDesc.GlobalStrides[0])
		}
		if len(p.Payload) != rows*112 {
			t.Errorf("expected payload length %d, got %d", rows*112, len(p.Payload))
		}

		// Verify row contents preserved
		for r := 0; r < rows; r++ {
			gotRow := p.Payload[r*112 : r*112+cols]
			wantRow := raw[r*cols : (r+1)*cols]
			if !bytes.Equal(gotRow, wantRow) {
				t.Fatalf("row %d content mismatch", r)
			}
		}

		// Verify BoxSizeBytes calculation
		boxBytes := p.BoxDesc.BoxSizeBytes()
		if boxBytes != int(p.BoxDesc.BoxDims[0])*int(p.BoxDesc.BoxDims[1]) {
			t.Errorf("unexpected box size bytes: %d", boxBytes)
		}
	})

	// 3D Transformer test
	t.Run("Transform3D Strides", func(t *testing.T) {
		lt, err := NewLayoutTransformer(ArchSM121a, TypeFloat16, WithAlignment(16))
		if err != nil {
			t.Fatalf("NewLayoutTransformer failed: %v", err)
		}

		depth, rows, cols := 4, 16, 20 // 20 halfs = 40 bytes -> padded to 48 bytes
		raw := make([]byte, depth*rows*cols*2)
		for i := range raw {
			raw[i] = byte(i & 0xFF)
		}

		p, err := lt.Transform3D(raw, depth, rows, cols)
		if err != nil {
			t.Fatalf("Transform3D failed: %v", err)
		}

		if p.BoxDesc.GlobalStrides[0] != 48 {
			t.Errorf("expected row stride 48, got %d", p.BoxDesc.GlobalStrides[0])
		}
		expectedPlaneStride := int64(rows * 48)
		if p.BoxDesc.GlobalStrides[1] != expectedPlaneStride {
			t.Errorf("expected plane stride %d, got %d", expectedPlaneStride, p.BoxDesc.GlobalStrides[1])
		}
		if p.BoxDesc.Rank != 3 {
			t.Errorf("expected rank 3, got %d", p.BoxDesc.Rank)
		}
	})
}

func TestRepackQ8_0(t *testing.T) {
	rows, cols := 4, 64 // 64 elements = 2 blocks of 32
	numBlocks := rows * (cols / 32)
	rawBlocks := make([]byte, numBlocks*34)

	for b := 0; b < numBlocks; b++ {
		offset := b * 34
		scale := uint16(0x3C00 + b) // ~1.0
		rawBlocks[offset] = byte(scale)
		rawBlocks[offset+1] = byte(scale >> 8)
		for i := 0; i < 32; i++ {
			rawBlocks[offset+2+i] = byte(int8(i - 16))
		}
	}

	prepacked, err := RepackQ8_0(rawBlocks, rows, cols)
	if err != nil {
		t.Fatalf("RepackQ8_0 failed: %v", err)
	}

	// Stride check: 2 blocks of 32B = 64 bytes -> aligned to 64 bytes
	if prepacked.BoxDesc.GlobalStrides[0] != 64 {
		t.Errorf("expected stride 64, got %d", prepacked.BoxDesc.GlobalStrides[0])
	}
	if len(prepacked.Payload) != rows*64 {
		t.Errorf("expected payload length %d, got %d", rows*64, len(prepacked.Payload))
	}
	if len(prepacked.Scales) != numBlocks*2 {
		t.Errorf("expected scales length %d, got %d", numBlocks*2, len(prepacked.Scales))
	}

	// Test round-trip unpack
	unpacked, err := UnpackQ8_0(prepacked)
	if err != nil {
		t.Fatalf("UnpackQ8_0 failed: %v", err)
	}
	if !bytes.Equal(unpacked, rawBlocks) {
		t.Fatalf("Q8_0 round-trip unpacked bytes do not match original raw blocks")
	}
}

func TestRepackIQ2_XXS(t *testing.T) {
	rows, cols := 2, 512 // 512 elements = 2 super-blocks of 256
	numBlocks := rows * (cols / 256)
	rawBlocks := make([]byte, numBlocks*66)

	for b := 0; b < numBlocks; b++ {
		offset := b * 66
		d := uint16(0x3800 + b*10)
		rawBlocks[offset] = byte(d)
		rawBlocks[offset+1] = byte(d >> 8)
		for i := 0; i < 64; i++ {
			rawBlocks[offset+2+i] = byte((b*7 + i*3) & 0xFF)
		}
	}

	prepacked, err := RepackIQ2_XXS(rawBlocks, rows, cols)
	if err != nil {
		t.Fatalf("RepackIQ2_XXS failed: %v", err)
	}

	// 2 super-blocks = 128 bytes of qs per row -> exactly 128 bytes stride
	if prepacked.BoxDesc.GlobalStrides[0] != 128 {
		t.Errorf("expected row stride 128, got %d", prepacked.BoxDesc.GlobalStrides[0])
	}
	if len(prepacked.Payload) != rows*128 {
		t.Errorf("expected payload %d, got %d", rows*128, len(prepacked.Payload))
	}

	unpacked, err := UnpackIQ2_XXS(prepacked)
	if err != nil {
		t.Fatalf("UnpackIQ2_XXS failed: %v", err)
	}
	if !bytes.Equal(unpacked, rawBlocks) {
		t.Fatalf("IQ2_XXS round-trip unpacked bytes do not match original raw blocks")
	}
}

func TestRepackMXFP4(t *testing.T) {
	rows, cols := 4, 64 // 64 elements = 2 groups of 32
	src := make([]float32, rows*cols)

	// Populate with known values
	for i := range src {
		src[i] = float32((i%13) - 6)
	}

	prepacked, err := PackMXFP4(src, rows, cols, 32)
	if err != nil {
		t.Fatalf("PackMXFP4 failed: %v", err)
	}

	// 2 groups of 16B = 32 bytes row -> multiple of 16
	if prepacked.BoxDesc.GlobalStrides[0] != 32 {
		t.Errorf("expected stride 32, got %d", prepacked.BoxDesc.GlobalStrides[0])
	}

	dst := make([]float32, rows*cols)
	if err := DequantizeMXFP4(prepacked, 32, dst); err != nil {
		t.Fatalf("DequantizeMXFP4 failed: %v", err)
	}

	// Verify reasonable numerical accuracy for 4-bit float
	for i := range src {
		diff := math.Abs(float64(src[i] - dst[i]))
		if diff > 1.5 {
			t.Errorf("MXFP4 dequantize index %d: original %f, reconstructed %f, diff %f", i, src[i], dst[i], diff)
		}
	}
}

func TestFP8ConversionsAndRepack(t *testing.T) {
	// Test float conversion round-trip for representative numbers
	vals := []float32{0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0}
	for _, v := range vals {
		e4 := Float32ToE4M3(v)
		rec4 := E4M3ToFloat32(e4)
		if math.Abs(float64(v-rec4)) > 0.1*math.Abs(float64(v))+0.01 {
			t.Errorf("E4M3 conversion mismatch: orig %f, reconstructed %f", v, rec4)
		}

		e5 := Float32ToE5M2(v)
		rec5 := E5M2ToFloat32(e5)
		if math.Abs(float64(v-rec5)) > 0.1*math.Abs(float64(v))+0.01 {
			t.Errorf("E5M2 conversion mismatch: orig %f, reconstructed %f", v, rec5)
		}
	}

	// Repack test
	rows, cols := 8, 128
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i % 16)
	}

	pE4, err := QuantizeFP8(src, rows, cols, TypeFP8_E4M3)
	if err != nil {
		t.Fatalf("QuantizeFP8 E4M3 failed: %v", err)
	}
	if pE4.BoxDesc.GlobalStrides[0] != 128 {
		t.Errorf("expected stride 128, got %d", pE4.BoxDesc.GlobalStrides[0])
	}

	dst := make([]float32, rows*cols)
	if err := DequantizeFP8(pE4, dst); err != nil {
		t.Fatalf("DequantizeFP8 failed: %v", err)
	}
	for i := range src {
		if math.Abs(float64(src[i]-dst[i])) > 0.5 {
			t.Fatalf("FP8 reconstructed index %d too large diff: %f vs %f", i, src[i], dst[i])
		}
	}
}

func TestSparsity2to4(t *testing.T) {
	rows, cols := 2, 8
	// Row 0: group 0 has values at indices 0, 2; group 1 at indices 1, 3
	dense := []float32{
		10.0, 0.0, 20.0, 0.0, // top 2 are idx 0 and 2
		0.0, 30.0, 0.0, 40.0, // top 2 are idx 1 and 3
		5.0, 6.0, 0.0, 0.0,   // top 2 are idx 0 and 1
		0.0, 0.0, 7.0, 8.0,   // top 2 are idx 2 and 3
	}

	sparse, meta, err := Compress2to4(dense, rows, cols)
	if err != nil {
		t.Fatalf("Compress2to4 failed: %v", err)
	}

	if len(sparse) != rows*(cols/2) {
		t.Fatalf("expected sparse length %d, got %d", rows*(cols/2), len(sparse))
	}
	if len(meta) != rows {
		t.Fatalf("expected metadata length %d, got %d", rows, len(meta))
	}

	// Verify group 0 metadata:
	// indices 0 and 2 -> nibble is (0 | (2 << 2)) = 0x8
	// indices 1 and 3 -> nibble is (1 | (3 << 2)) = 0xD
	// Byte 0: 0x8 | (0xD << 4) = 0xD8
	if meta[0] != 0xD8 {
		t.Errorf("expected meta[0] = 0xD8, got 0x%02X", meta[0])
	}

	// Row 1 metadata:
	// indices 0 and 1 -> nibble is (0 | (1 << 2)) = 0x4
	// indices 2 and 3 -> nibble is (2 | (3 << 2)) = 0xE
	// Byte 1: 0x4 | (0xE << 4) = 0xE4
	if meta[1] != 0xE4 {
		t.Errorf("expected meta[1] = 0xE4, got 0x%02X", meta[1])
	}

	// Test decompress
	reconstructed, err := Decompress2to4(sparse, meta, rows, cols)
	if err != nil {
		t.Fatalf("Decompress2to4 failed: %v", err)
	}
	for i := range dense {
		if dense[i] != reconstructed[i] {
			t.Errorf("sparsity mismatch at %d: want %f, got %f", i, dense[i], reconstructed[i])
		}
	}

	// Test byte-level sparsity
	denseBytes := make([]byte, len(dense))
	for i := range dense {
		denseBytes[i] = byte(dense[i])
	}
	spBytes, metaBytes, err := Compress2to4Bytes(denseBytes, rows, cols)
	if err != nil {
		t.Fatalf("Compress2to4Bytes failed: %v", err)
	}
	recBytes, err := Decompress2to4Bytes(spBytes, metaBytes, rows, cols)
	if err != nil {
		t.Fatalf("Decompress2to4Bytes failed: %v", err)
	}
	if !bytes.Equal(denseBytes, recBytes) {
		t.Errorf("Compress2to4Bytes round-trip mismatch")
	}
}

func TestScaleFactorSwizzling(t *testing.T) {
	rows, cols := 16, 128
	scales := make([]byte, rows*cols)
	for i := range scales {
		scales[i] = byte(i & 0xFF)
	}

	modes := []SwizzleMode{Swizzle32B, Swizzle64B, Swizzle128B, Swizzle128BAtom32B}
	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			swizzled, err := SwizzleScales(scales, rows, cols, mode)
			if err != nil {
				t.Fatalf("SwizzleScales failed: %v", err)
			}

			unswizzled, err := UnswizzleScales(swizzled, rows, cols, mode)
			if err != nil {
				t.Fatalf("UnswizzleScales failed: %v", err)
			}

			if !bytes.Equal(scales, unswizzled) {
				t.Fatalf("swizzle %s did not restore original scale bytes", mode)
			}
		})
	}
}

func TestBinarySerializationAndZeroCopy(t *testing.T) {
	p := &PrepackedTensor{
		Arch: ArchSM121a,
		Type: TypeQ8_0,
		BoxDesc: BoxDescriptor{
			Rank:          2,
			DataType:      TypeQ8_0,
			GlobalAddress: 0x4000,
			GlobalDims:    [3]uint64{128, 64, 1},
			GlobalStrides: [3]int64{128, 0, 0},
			BoxDims:       [3]uint32{128, 32, 1},
			ElementStrides: [3]uint32{1, 1, 1},
		},
		Payload:  make([]byte, 64*128),
		Scales:   make([]byte, 64*4*2),
		Metadata: make([]byte, 64*16),
	}
	for i := range p.Payload {
		p.Payload[i] = byte(i & 0xFF)
	}
	for i := range p.Scales {
		p.Scales[i] = byte((i * 3) & 0xFF)
	}
	for i := range p.Metadata {
		p.Metadata[i] = byte((i * 7) & 0xFF)
	}

	// Test Marshal/Unmarshal
	data, err := p.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}

	readTensor, err := Read(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if readTensor.Arch != p.Arch || readTensor.Type != p.Type {
		t.Errorf("metadata mismatch: arch=%s, type=%s", readTensor.Arch, readTensor.Type)
	}
	if !bytes.Equal(readTensor.Payload, p.Payload) {
		t.Errorf("payload content mismatch")
	}
	if !bytes.Equal(readTensor.Scales, p.Scales) {
		t.Errorf("scales content mismatch")
	}
	if !bytes.Equal(readTensor.Metadata, p.Metadata) {
		t.Errorf("metadata content mismatch")
	}

	// Test ParseZeroCopy
	zc, err := ParseZeroCopy(data)
	if err != nil {
		t.Fatalf("ParseZeroCopy failed: %v", err)
	}
	if !zc.ZeroCopy() {
		t.Errorf("expected ZeroCopy() to be true")
	}
	if !bytes.Equal(zc.Payload, p.Payload) {
		t.Errorf("zero copy payload mismatch")
	}

	// Verify it slices the exact underlying slice
	origByte := zc.Payload[0]
	zc.Payload[0] ^= 0xFF
	if data[128] != zc.Payload[0] {
		t.Errorf("expected modification in zero-copy slice to mutate backing data")
	}
	zc.Payload[0] = origByte
}

func TestEncodeTensorMap(t *testing.T) {
	desc := BoxDescriptor{
		Rank:          2,
		DataType:      TypeFloat16,
		GlobalAddress: 0x8000,
		GlobalDims:    [3]uint64{256, 128, 1},
		GlobalStrides: [3]int64{512, 0, 0},
		BoxDims:       [3]uint32{64, 32, 1},
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:       Swizzle128B,
		Interleave:    InterleaveNone,
	}

	encoded, err := desc.EncodeTensorMap()
	if err != nil {
		t.Fatalf("EncodeTensorMap failed: %v", err)
	}
	if encoded[0] != 0x8000 {
		t.Errorf("expected address 0x8000, got 0x%x", encoded[0])
	}
}
