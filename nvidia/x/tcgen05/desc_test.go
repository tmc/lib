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
		{
			name: "rubin_sm107a_sparse_v1_scale_vec_4x",
			desc: InstDesc{
				Sparsity:        1,
				SparsityVersion: 1,
				Saturate:        true,
				DType:           F32,
				AType:           E2M1,
				BType:           E2M1,
				NegateA:         false,
				NegateB:         false,
				TransposeA:      false,
				TransposeB:      true,
				M:               128,
				N:               128,
				ScaleVec:        ScaleVec4X,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc := tc.desc.Encode()
			if gotBit12 := uint8((enc >> 12) & 1); gotBit12 != tc.desc.SparsityVersion {
				t.Fatalf("encoded 0x%08x bit 12 = %d, want SparsityVersion %d", enc, gotBit12, tc.desc.SparsityVersion)
			}
			got := DecodeInstDesc(enc)
			if got != tc.desc {
				t.Fatalf("DecodeInstDesc(Encode()) = %+v, want %+v (encoded 0x%08x)", got, tc.desc, enc)
			}
		})
	}
}

func TestInstDescValidateForTarget(t *testing.T) {
	denseDesc := InstDesc{
		DType: F32,
		AType: BF16,
		BType: BF16,
		M:     128,
		N:     128,
	}
	for _, target := range []string{"sm_100a", "sm_103a", "sm_107a", "sm_100f", "sm_103f", "sm_107f"} {
		if err := denseDesc.ValidateForTarget(target); err != nil {
			t.Fatalf("denseDesc.ValidateForTarget(%q) unexpected error: %v", target, err)
		}
	}

	sparseV0 := InstDesc{
		Sparsity:        1,
		SparsityVersion: 0,
		DType:           F32,
		AType:           E2M1,
		BType:           E2M1,
		M:               128,
		N:               128,
		ScaleVec:        ScaleVec4X,
	}
	for _, target := range []string{"sm_100a", "sm_103a", "sm_100f", "sm_103f"} {
		if err := sparseV0.ValidateForTarget(target); err != nil {
			t.Fatalf("sparseV0.ValidateForTarget(%q) unexpected error: %v", target, err)
		}
	}
	for _, target := range []string{"sm_107a", "sm_107f"} {
		if err := sparseV0.ValidateForTarget(target); err == nil {
			t.Fatalf("sparseV0.ValidateForTarget(%q) succeeded, want error for SparsityVersion 0 on Rubin", target)
		}
	}

	sparseV1 := sparseV0
	sparseV1.SparsityVersion = 1
	for _, target := range []string{"sm_107a", "sm_107f"} {
		if err := sparseV1.ValidateForTarget(target); err != nil {
			t.Fatalf("sparseV1.ValidateForTarget(%q) unexpected error: %v", target, err)
		}
	}
	for _, target := range []string{"sm_100a", "sm_103a", "sm_100f", "sm_103f"} {
		if err := sparseV1.ValidateForTarget(target); err == nil {
			t.Fatalf("sparseV1.ValidateForTarget(%q) succeeded, want error for SparsityVersion 1 on %s", target, target)
		}
	}

	if err := sparseV1.ValidateForTarget("sm_90a"); err == nil {
		t.Fatal("ValidateForTarget(sm_90a) succeeded, want unsupported target error")
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
