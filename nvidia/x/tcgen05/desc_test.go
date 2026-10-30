package tcgen05

import (
	"strings"
	"testing"
)

// The want words below are computed by hand from PTX ISA 9.4 section
// 9.7.18.4.2, Tables 51 (tf32, f16, f8f6f4, i8, ti16), 52 (mxf8f6f4), and
// 53 (mxf4, mxf4nvf4).
func TestInstDescEncode(t *testing.T) {
	tests := []struct {
		name string
		desc InstDesc
		want uint32
	}{
		{
			// dtype F32 1<<4 = 0x10, atype TF32 2<<7 = 0x100, btype 2<<10 = 0x800,
			// N>>3 = 16 <<17 = 0x200000, M>>4 = 8 <<24 = 0x08000000.
			name: "tf32_m128_n128",
			desc: InstDesc{Kind: KindTF32, DType: ElemF32, AType: ElemTF32, BType: ElemTF32, M: 128, N: 128, K: 8},
			want: 0x08200910,
		},
		{
			// tf32_m128_n128 | sparse 1<<2 = 0x4 | selector 2.
			name: "tf32_sparse_selector2",
			desc: InstDesc{Kind: KindTF32, Sparse: true, SparsitySelector: 2, DType: ElemF32, AType: ElemTF32, BType: ElemTF32, M: 128, N: 128, K: 16},
			want: 0x08200916,
		},
		{
			// dtype F32 0x10, atype BF16 1<<7 = 0x80, btype BF16 1<<10 = 0x400,
			// N>>3 = 32 <<17 = 0x400000, M>>4 = 8 <<24 = 0x08000000.
			name: "f16_bf16_m128_n256",
			desc: InstDesc{Kind: KindF16, DType: ElemF32, AType: ElemBF16, BType: ElemBF16, M: 128, N: 256, K: 16},
			want: 0x08400490,
		},
		{
			// dtype F16 0, atype/btype F16 0, negate A 1<<13 = 0x2000,
			// transpose B 1<<16 = 0x10000, N>>3 = 8 <<17 = 0x100000,
			// M>>4 = 4 <<24 = 0x04000000, max shift 32 → 3<<30 = 0xc0000000.
			name: "f16_negA_transB_shift32",
			desc: InstDesc{Kind: KindF16, DType: ElemF16, AType: ElemF16, BType: ElemF16, NegateA: true, TransposeB: true, M: 64, N: 64, K: 16, MaxShift: 32},
			want: 0xc4112000,
		},
		{
			// dtype F32 0x10, atype E4M3 0, btype E2M1 5<<10 = 0x1400,
			// N 0x200000, M 0x08000000, K=64 dense 1<<29 = 0x20000000.
			name: "f8f6f4_e4m3_e2m1_k64",
			desc: InstDesc{Kind: KindF8F6F4, DType: ElemF32, AType: ElemE4M3, BType: ElemE2M1, M: 128, N: 128, K: 64},
			want: 0x28201410,
		},
		{
			// saturate 1<<3 = 0x8, dtype S32 2<<4 = 0x20, atype S8 1<<7 = 0x80,
			// btype S8 1<<10 = 0x400, N>>3 = 8 <<17 = 0x100000, M 0x08000000.
			name: "i8_s8_saturate",
			desc: InstDesc{Kind: KindI8, Saturate: true, DType: ElemS32, AType: ElemS8, BType: ElemS8, M: 128, N: 64},
			want: 0x081004a8,
		},
		{
			// dtype S32 0x20, atype S1Z4M11 3<<7 = 0x180, btype 3<<10 = 0xc00,
			// N 0x200000, M 0x08000000.
			name: "ti16_m128_n128",
			desc: InstDesc{Kind: KindTI16, DType: ElemS32, AType: ElemS1Z4M11, BType: ElemS1Z4M11, M: 128, N: 128},
			want: 0x08200da0,
		},
		{
			// atype/btype E4M3 0, N 0x200000, scale UE8M0 1<<23 = 0x800000,
			// M>>7 = 1 <<27 = 0x08000000, K=32 dense → bit 31 clear.
			name: "mxf8f6f4_e4m3_ue8m0",
			desc: InstDesc{Kind: KindMXF8F6F4, AType: ElemE4M3, BType: ElemE4M3, M: 128, N: 128, K: 32, ScaleType: ScaleUE8M0},
			want: 0x08a00000,
		},
		{
			// sparse 0x4, atype E5M2 1<<7 = 0x80, btype E2M3 3<<10 = 0xc00,
			// N 0x200000, scale 0x800000, A SF 128 lane 1<<26 = 0x04000000,
			// M 0x08000000, sparse K=128 → bit 31 = 0x80000000.
			name: "mxf8f6f4_sparse_k128_128lane",
			desc: InstDesc{Kind: KindMXF8F6F4, Sparse: true, AType: ElemE5M2, BType: ElemE2M3, M: 128, N: 128, K: 128, ScaleType: ScaleUE8M0, ScaleA128Lane: true},
			want: 0x8ca00c84,
		},
		{
			// atype E2M1 1<<7 = 0x80, btype E2M1 1<<10 = 0x400, N 0x200000,
			// scale UE4M3 0, M>>7 = 1 <<27 = 0x08000000, K=64 → code 0.
			name: "mxf4nvf4_ue4m3_k64",
			desc: InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 64, ScaleType: ScaleUE4M3},
			want: 0x08200480,
		},
		{
			// K=96 → code 1: bit 3 (upper) clear, bit 31 (lower) set.
			name: "mxf4nvf4_ue4m3_k96",
			desc: InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 96, ScaleType: ScaleUE4M3},
			want: 0x88200480,
		},
		{
			// K=128 → code 2: bit 3 (upper) set, bit 31 (lower) clear.
			name: "mxf4nvf4_ue4m3_k128",
			desc: InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 128, ScaleType: ScaleUE4M3},
			want: 0x08200488,
		},
		{
			// sparse 0x4, atype 0x80, btype 0x400, sparsity version v1 1<<12 = 0x1000,
			// N 0x200000, scale UE4M3 0, M 0x08000000, sparse K=128 → code 0.
			name: "mxf4nvf4_sparse_v1",
			desc: InstDesc{Kind: KindMXF4NVF4, Sparse: true, SparsityVersion: 1, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 128, ScaleType: ScaleUE4M3},
			want: 0x08201484,
		},
		{
			// B SF ID 2<<4 = 0x20, atype 0x80, btype 0x400, N>>3 = 32 <<17 = 0x400000,
			// scale UE8M0 1<<23 = 0x800000, M>>7 = 2 <<27 = 0x10000000,
			// A SF ID 2<<29 = 0x40000000.
			name: "mxf4_m256_n256_sfid2",
			desc: InstDesc{Kind: KindMXF4, AType: ElemE2M1, BType: ElemE2M1, M: 256, N: 256, K: 64, ScaleType: ScaleUE8M0, ScaleAID: 2, ScaleBID: 2},
			want: 0x50c004a0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.desc.Encode()
			if err != nil {
				t.Fatalf("Encode() error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("Encode() = 0x%08x, want 0x%08x", got, tt.want)
			}
			dec, err := DecodeInstDesc(tt.desc.Kind, got)
			if err != nil {
				t.Fatalf("DecodeInstDesc(%v, 0x%08x) error: %v", tt.desc.Kind, got, err)
			}
			if dec != tt.desc {
				t.Fatalf("DecodeInstDesc(%v, 0x%08x) = %+v, want %+v", tt.desc.Kind, got, dec, tt.desc)
			}
		})
	}
}

func TestInstDescEncodeRejects(t *testing.T) {
	tf32 := InstDesc{Kind: KindTF32, DType: ElemF32, AType: ElemTF32, BType: ElemTF32, M: 128, N: 128, K: 8}
	nvf4 := InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 64, ScaleType: ScaleUE4M3}
	i8 := InstDesc{Kind: KindI8, DType: ElemS32, AType: ElemS8, BType: ElemS8, M: 128, N: 128}
	tests := []struct {
		name string
		edit func(d *InstDesc)
		base InstDesc
		want string
	}{
		{"no kind", func(d *InstDesc) { d.Kind = 0 }, tf32, "invalid kind"},
		{"f16 atype e4m3", func(d *InstDesc) { d.Kind, d.K, d.AType = KindF16, 16, ElemE4M3 }, tf32, "atype e4m3"},
		{"tf32 dtype f16", func(d *InstDesc) { d.DType = ElemF16 }, tf32, "dtype f16"},
		{"tf32 M 8", func(d *InstDesc) { d.M = 8 }, tf32, "M 8"},
		{"tf32 M 512", func(d *InstDesc) { d.M = 512 }, tf32, "M 512"},
		{"N 4", func(d *InstDesc) { d.N = 4 }, tf32, "N 4"},
		{"N 512", func(d *InstDesc) { d.N = 512 }, tf32, "N 512"},
		{"tf32 dense K 16", func(d *InstDesc) { d.K = 16 }, tf32, "K 16"},
		{"f8f6f4 sparse K 128", func(d *InstDesc) {
			d.Kind, d.AType, d.BType, d.Sparse, d.K = KindF8F6F4, ElemE4M3, ElemE4M3, true, 128
		}, tf32, "K 128"},
		{"dense selector", func(d *InstDesc) { d.SparsitySelector = 1 }, tf32, "dense"},
		{"tf32 sparsity version", func(d *InstDesc) { d.Sparse, d.K, d.SparsityVersion = true, 16, 1 }, tf32, "sparsity version"},
		{"tf32 saturate", func(d *InstDesc) { d.Saturate = true }, tf32, "saturate"},
		{"tf32 max shift 4", func(d *InstDesc) { d.MaxShift = 4 }, tf32, "max shift"},
		{"tf32 scale type", func(d *InstDesc) { d.ScaleType = ScaleUE8M0 }, tf32, "scale"},
		{"i8 K", func(d *InstDesc) { d.K = 32 }, i8, "no K field"},
		{"i8 negate", func(d *InstDesc) { d.NegateB = true }, i8, "negate"},
		{"ti16 transpose", func(d *InstDesc) {
			d.Kind, d.AType, d.BType, d.TransposeA = KindTI16, ElemS1Z4M11, ElemS1Z4M11, true
		}, i8, "transpose"},
		{"mxf4nvf4 M 64", func(d *InstDesc) { d.M = 64 }, nvf4, "M 64"},
		{"mxf4nvf4 dtype", func(d *InstDesc) { d.DType = ElemF32 }, nvf4, "dtype"},
		{"mxf4nvf4 atype e4m3", func(d *InstDesc) { d.AType = ElemE4M3 }, nvf4, "atype e4m3"},
		{"mxf4nvf4 transpose A", func(d *InstDesc) { d.TransposeA = true }, nvf4, "transpose A"},
		{"mxf4nvf4 no scale type", func(d *InstDesc) { d.ScaleType = 0 }, nvf4, "scale type"},
		{"mxf4nvf4 scale id 1", func(d *InstDesc) { d.ScaleAID = 1 }, nvf4, "0 or 2"},
		{"mxf4nvf4 sparse K 256", func(d *InstDesc) { d.Sparse, d.K = true, 256 }, nvf4, "K 256"},
		{"mxf4 ue4m3", func(d *InstDesc) { d.Kind = KindMXF4 }, nvf4, "scale type ue4m3"},
		{"mxf8f6f4 sparsity version", func(d *InstDesc) {
			d.Kind, d.AType, d.BType, d.K, d.ScaleType, d.Sparse, d.SparsityVersion = KindMXF8F6F4, ElemE4M3, ElemE4M3, 64, ScaleUE8M0, true, 1
		}, nvf4, "sparsity version"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := tt.base
			tt.edit(&d)
			w, err := d.Encode()
			if err == nil {
				t.Fatalf("Encode(%+v) = 0x%08x, want error", d, w)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Encode(%+v) error %q, want it to mention %q", d, err, tt.want)
			}
			if d.Validate() == nil {
				t.Fatalf("Validate(%+v) = nil, want error", d)
			}
		})
	}
}

func TestDecodeInstDescRejects(t *testing.T) {
	tests := []struct {
		name string
		kind Kind
		w    uint32
	}{
		{"tf32 reserved bit 6", KindTF32, 0x08200910 | 1<<6},
		{"tf32 reserved bit 23", KindTF32, 0x08200910 | 1<<23},
		{"f16 dtype code 3", KindF16, 0x08400490 | 3<<4},
		{"tf32 atype f16", KindTF32, 0x08200810},
		{"mxf8f6f4 reserved bit 24", KindMXF8F6F4, 0x08a00000 | 1<<24},
		{"mxf4nvf4 scale code 3", KindMXF4NVF4, 0x08200480 | 3<<23},
		{"mxf4nvf4 K code 3", KindMXF4NVF4, 0x08200480 | 1<<3 | 1<<31},
		{"invalid kind", 0, 0x08200910},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if d, err := DecodeInstDesc(tt.kind, tt.w); err == nil {
				t.Fatalf("DecodeInstDesc(%v, 0x%08x) = %+v, want error", tt.kind, tt.w, d)
			}
		})
	}
}

func TestInstDescValidateForTarget(t *testing.T) {
	f8k64 := InstDesc{Kind: KindF8F6F4, DType: ElemF32, AType: ElemE4M3, BType: ElemE4M3, M: 128, N: 128, K: 64}
	nvf4 := InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: 128, N: 128, K: 64, ScaleType: ScaleUE4M3}
	sparseV0 := nvf4
	sparseV0.Sparse, sparseV0.K = true, 128
	sparseV1 := sparseV0
	sparseV1.SparsityVersion = 1
	ue5m3 := nvf4
	ue5m3.ScaleType = ScaleUE5M3
	k128 := nvf4
	k128.K = 128
	lane128 := InstDesc{Kind: KindMXF8F6F4, AType: ElemE4M3, BType: ElemE4M3, M: 128, N: 128, K: 32, ScaleType: ScaleUE8M0, ScaleA128Lane: true}
	tf32Sparse := InstDesc{Kind: KindTF32, Sparse: true, DType: ElemF32, AType: ElemTF32, BType: ElemTF32, M: 128, N: 128, K: 16}

	tests := []struct {
		name   string
		desc   InstDesc
		target string
		ok     bool
	}{
		{"nvf4 dense sm_100a", nvf4, "sm_100a", true},
		{"f8f6f4 K64 sm_100a", f8k64, "sm_100a", false},
		{"f8f6f4 K64 sm_107a", f8k64, "sm_107a", true},
		{"mxf4nvf4 K128 sm_103a", k128, "sm_103a", false},
		{"mxf4nvf4 K128 sm_107f", k128, "sm_107f", true},
		{"mxf8f6f4 128 lane sm_103a", lane128, "sm_103a", false},
		{"mxf8f6f4 128 lane sm_107a", lane128, "sm_107a", true},
		{"ue5m3 sm_103a", ue5m3, "sm_103a", false},
		{"ue5m3 sm_107a", ue5m3, "sm_107a", true},
		{"sparse v0 sm_100a", sparseV0, "sm_100a", true},
		{"sparse v0 sm_110a", sparseV0, "sm_110a", true},
		{"sparse v0 sm_107a", sparseV0, "sm_107a", false},
		{"sparse v1 sm_107a", sparseV1, "sm_107a", true},
		{"sparse v1 sm_103a", sparseV1, "sm_103a", false},
		{"tf32 sparse sm_107a", tf32Sparse, "sm_107a", true},
		{"unknown target", nvf4, "sm_90a", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.desc.ValidateForTarget(tt.target)
			if (err == nil) != tt.ok {
				t.Fatalf("ValidateForTarget(%s) = %v, want ok=%v", tt.target, err, tt.ok)
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
