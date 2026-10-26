package tcgen05

import "testing"

func TestInstDescRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		desc InstDesc
	}{
		{
			name: "bf16_cta1_128x128",
			desc: InstDesc{
				DType:    F32,
				AType:    BF16,
				BType:    BF16,
				M:        128,
				N:        128,
				ScaleVec: ScaleVecNone,
			},
		},
		{
			name: "nvfp4_scale_vec_4x_sm103a",
			desc: InstDesc{
				Sparsity:   1,
				Saturate:   true,
				DType:      F32,
				AType:      E2M1,
				BType:      E2M1,
				NegateA:    true,
				NegateB:    false,
				TransposeA: false,
				TransposeB: true,
				M:          128,
				N:          256,
				ScaleVec:   ScaleVec4X,
			},
		},
		{
			name: "e4m3_cta2_256x512_scale_vec_2x",
			desc: InstDesc{
				Sparsity:   2,
				Saturate:   false,
				DType:      F32,
				AType:      E4M3,
				BType:      E4M3,
				NegateA:    false,
				NegateB:    true,
				TransposeA: true,
				TransposeB: false,
				M:          256,
				N:          512,
				ScaleVec:   ScaleVec2X,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc := tc.desc.Encode()
			got := DecodeInstDesc(enc)
			if got != tc.desc {
				t.Fatalf("DecodeInstDesc(Encode()) = %+v, want %+v (encoded 0x%08x)", got, tc.desc, enc)
			}
		})
	}
}

func TestSharedDescRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		desc SharedDesc
	}{
		{
			name: "no_swizzle",
			desc: SharedDesc{
				BaseAddr:          0x0100,
				LeadingByteOffset: 128,
				StrideByteOffset:  1024,
				Swizzle:           SwizzleNone,
			},
		},
		{
			name: "swizzle_32b",
			desc: SharedDesc{
				BaseAddr:          0x0400,
				LeadingByteOffset: 32,
				StrideByteOffset:  256,
				Swizzle:           Swizzle32B,
			},
		},
		{
			name: "swizzle_64b",
			desc: SharedDesc{
				BaseAddr:          0x2000,
				LeadingByteOffset: 64,
				StrideByteOffset:  512,
				Swizzle:           Swizzle64B,
			},
		},
		{
			name: "swizzle_128b",
			desc: SharedDesc{
				BaseAddr:          0x8000,
				LeadingByteOffset: 128,
				StrideByteOffset:  2048,
				Swizzle:           Swizzle128B,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc := tc.desc.Encode()
			got := DecodeSharedDesc(enc)
			if got != tc.desc {
				t.Fatalf("DecodeSharedDesc(Encode()) = %+v, want %+v (encoded 0x%016x)", got, tc.desc, enc)
			}
		})
	}
}

func TestTMEMAddrRoundTrip(t *testing.T) {
	for _, addr := range []TMEMAddr{
		{Row: 0, Col: 0},
		{Row: 32, Col: 64},
		{Row: 96, Col: 128},
		{Row: 127, Col: 511},
	} {
		enc := addr.Encode()
		if want := (uint32(addr.Row) << 16) | uint32(addr.Col); enc != want {
			t.Fatalf("TMEMAddr%+v.Encode() = 0x%08x, want 0x%08x", addr, enc, want)
		}
		got := DecodeTMEMAddr(enc)
		if got != addr {
			t.Fatalf("DecodeTMEMAddr(0x%08x) = %+v, want %+v", enc, got, addr)
		}
	}
}
