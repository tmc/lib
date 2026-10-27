package cudatma

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestL2CachePolicySpec(t *testing.T) {
	tests := []struct {
		name      string
		spec      L2CachePolicySpec
		dst, src  string
		wantPTX   string
		wantError bool
	}{
		{
			name: "persistent weights evict_last 1.0",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictLast,
				Fraction:      1.0,
			},
			dst:     "%0",
			src:     "%1",
			wantPTX: "createpolicy.fractional.L2::evict_last.b64 %0, %1;",
		},
		{
			name: "streaming activations evict_first with evict_unchanged secondary",
			spec: L2CachePolicySpec{
				PrimaryPolicy:   L2EvictFirst,
				SecondaryPolicy: L2EvictUnchanged,
				Fraction:        0.25,
			},
			dst:     "%0",
			src:     "%1",
			wantPTX: "createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, %1;",
		},
		{
			name: "normal policy with evict_normal secondary",
			spec: L2CachePolicySpec{
				PrimaryPolicy:   L2EvictNormal,
				SecondaryPolicy: L2EvictNormal,
				Fraction:        0.5,
			},
			dst:     "policy_reg",
			src:     "frac_reg",
			wantPTX: "createpolicy.fractional.L2::evict_normal.L2::evict_normal.b64 policy_reg, frac_reg;",
		},
		{
			name: "empty primary rejected",
			spec: L2CachePolicySpec{
				Fraction: 1.0,
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "evict_unchanged primary rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictUnchanged,
				Fraction:      0.5,
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "unknown secondary rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy:   L2EvictFirst,
				SecondaryPolicy: "evict_bogus",
				Fraction:        0.5,
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "zero fraction rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictLast,
				Fraction:      0.0,
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "fraction above 1 rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictLast,
				Fraction:      1.25,
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "NaN fraction rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictLast,
				Fraction:      float32(math.NaN()),
			},
			dst:       "%0",
			src:       "%1",
			wantError: true,
		},
		{
			name: "empty register operand rejected",
			spec: L2CachePolicySpec{
				PrimaryPolicy: L2EvictLast,
				Fraction:      1.0,
			},
			dst:       "",
			src:       "%1",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ptx, err := tt.spec.FormatPTX(tt.dst, tt.src)
			if tt.wantError {
				if err == nil {
					t.Fatalf("FormatPTX(%q, %q) = %q, nil; want error", tt.dst, tt.src, ptx)
				}
				return
			}
			if err != nil {
				t.Fatalf("FormatPTX(%q, %q) unexpected error: %v", tt.dst, tt.src, err)
			}
			if ptx != tt.wantPTX {
				t.Fatalf("FormatPTX = %q, want %q", ptx, tt.wantPTX)
			}
		})
	}
}

func TestTMAGather4Spec(t *testing.T) {
	evictFirst := &L2CachePolicySpec{
		PrimaryPolicy:   L2EvictFirst,
		SecondaryPolicy: L2EvictUnchanged,
		Fraction:        0.25,
	}

	for _, target := range []string{TargetSM100a, TargetSM103a, TargetSM107a, TargetSM100f, TargetSM103f, TargetSM107f, TargetSM120a, TargetSM121a} {
		spec := TMAGather4Spec{
			Target:     target,
			Mode:       TMAGather4Load,
			ColDim:     32,
			ElemBytes:  4,
			RowIndices: [4]int32{3, 19, 41, 107},
			L2Policy:   evictFirst,
		}
		if err := spec.Validate(); err != nil {
			t.Fatalf("target %s Validate() failed: %v", target, err)
		}
		if got, want := spec.TransferBytes(), 4*32*4; got != want {
			t.Fatalf("target %s TransferBytes() = %d, want %d", target, got, want)
		}
		ptx, err := spec.FormatPTX()
		if err != nil {
			t.Fatalf("target %s FormatPTX() failed: %v", target, err)
		}
		for _, want := range []string{
			"cp.async.bulk.tensor.2d.shared::cta.global.tile.gather4.mbarrier::complete_tx::bytes",
			".L2::cache_hint",
		} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("target %s FormatPTX() = %q, missing %q", target, ptx, want)
			}
		}
	}

	scatterSpec := TMAGather4Spec{
		Target:     TargetSM100a,
		Mode:       TMAScatter4Store,
		ColDim:     64,
		ElemBytes:  2,
		RowIndices: [4]int32{7, 12, 88, 250},
	}
	scatterPTX, err := scatterSpec.FormatPTX()
	if err != nil {
		t.Fatalf("scatter4 FormatPTX() failed: %v", err)
	}
	if !strings.Contains(scatterPTX, "cp.async.bulk.tensor.2d.global.shared::cta.tile.scatter4") {
		t.Fatalf("scatter4 FormatPTX() = %q, missing scatter4 opcode", scatterPTX)
	}
	if strings.Contains(scatterPTX, ".L2::cache_hint") {
		t.Fatalf("scatter4 FormatPTX() without L2Policy unexpectedly emitted .L2::cache_hint: %q", scatterPTX)
	}
	if got, want := scatterSpec.TransferBytes(), 4*64*2; got != want {
		t.Fatalf("scatter4 TransferBytes() = %d, want %d", got, want)
	}

	invalidTests := []struct {
		name string
		spec TMAGather4Spec
	}{
		{
			name: "sm_90a rejected for gather4",
			spec: TMAGather4Spec{Target: TargetSM90a, Mode: TMAGather4Load, ColDim: 32, ElemBytes: 4, RowIndices: [4]int32{0, 1, 2, 3}},
		},
		{
			name: "invalid mode",
			spec: TMAGather4Spec{Target: TargetSM100a, Mode: "gather8", ColDim: 32, ElemBytes: 4, RowIndices: [4]int32{0, 1, 2, 3}},
		},
		{
			name: "unaligned row width",
			spec: TMAGather4Spec{Target: TargetSM100a, Mode: TMAGather4Load, ColDim: 6, ElemBytes: 2, RowIndices: [4]int32{0, 1, 2, 3}},
		},
		{
			name: "unsupported element size",
			spec: TMAGather4Spec{Target: TargetSM100a, Mode: TMAGather4Load, ColDim: 16, ElemBytes: 8, RowIndices: [4]int32{0, 1, 2, 3}},
		},
		{
			name: "negative row index",
			spec: TMAGather4Spec{Target: TargetSM100a, Mode: TMAGather4Load, ColDim: 32, ElemBytes: 4, RowIndices: [4]int32{0, -1, 2, 3}},
		},
		{
			name: "invalid L2 policy",
			spec: TMAGather4Spec{
				Target:     TargetSM100a,
				Mode:       TMAGather4Load,
				ColDim:     32,
				ElemBytes:  4,
				RowIndices: [4]int32{0, 1, 2, 3},
				L2Policy:   &L2CachePolicySpec{PrimaryPolicy: L2EvictLast, Fraction: 0},
			},
		},
	}
	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.spec.FormatPTX(); err == nil {
				t.Fatalf("FormatPTX() = nil error, want rejection")
			}
		})
	}
}

func TestTMAMulticastSpec(t *testing.T) {
	evictLast := &L2CachePolicySpec{
		PrimaryPolicy: L2EvictLast,
		Fraction:      1.0,
	}

	validTests := []struct {
		name           string
		spec           TMAMulticastSpec
		wantRecipients int
		wantHint       bool
	}{
		{
			name: "sm_100a 2-CTA pair with evict_last hint",
			spec: TMAMulticastSpec{
				Target:      TargetSM100a,
				ClusterSize: 2,
				CTAMask:     0x3,
				TileBytes:   16384,
				L2Policy:    evictLast,
			},
			wantRecipients: 2,
			wantHint:       true,
		},
		{
			name: "sm_103a 8-CTA cluster subset without L2 hint",
			spec: TMAMulticastSpec{
				Target:      TargetSM103a,
				ClusterSize: 8,
				CTAMask:     0x55,
				TileBytes:   8192,
			},
			wantRecipients: 4,
			wantHint:       false,
		},
		{
			name: "sm_90a 16-CTA full broadcast",
			spec: TMAMulticastSpec{
				Target:      TargetSM90a,
				ClusterSize: 16,
				CTAMask:     0xffff,
				TileBytes:   4096,
			},
			wantRecipients: 16,
			wantHint:       false,
		},
		{
			name: "sm_100f 4-CTA cluster",
			spec: TMAMulticastSpec{
				Target:      TargetSM100f,
				ClusterSize: 4,
				CTAMask:     0xf,
				TileBytes:   4096,
				L2Policy:    evictLast,
			},
			wantRecipients: 4,
			wantHint:       true,
		},
		{
			name: "sm_107a 8-CTA Rubin cluster",
			spec: TMAMulticastSpec{
				Target:      TargetSM107a,
				ClusterSize: 8,
				CTAMask:     0xff,
				TileBytes:   8192,
				L2Policy:    evictLast,
			},
			wantRecipients: 8,
			wantHint:       true,
		},
		{
			name: "sm_107f 4-CTA Rubin family cluster",
			spec: TMAMulticastSpec{
				Target:      TargetSM107f,
				ClusterSize: 4,
				CTAMask:     0xf,
				TileBytes:   4096,
			},
			wantRecipients: 4,
			wantHint:       false,
		},
		{
			name: "sm_121a 2-CTA cluster allowed",
			spec: TMAMulticastSpec{
				Target:      TargetSM121a,
				ClusterSize: 2,
				CTAMask:     0x3,
				TileBytes:   4096,
			},
			wantRecipients: 2,
			wantHint:       false,
		},
	}

	for _, tt := range validTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.spec.Validate(); err != nil {
				t.Fatalf("Validate() unexpected error: %v", err)
			}
			if got := tt.spec.RecipientCount(); got != tt.wantRecipients {
				t.Fatalf("RecipientCount() = %d, want %d", got, tt.wantRecipients)
			}
			ptx, err := tt.spec.FormatPTX()
			if err != nil {
				t.Fatalf("FormatPTX() unexpected error: %v", err)
			}
			if !strings.Contains(ptx, "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster") {
				t.Fatalf("FormatPTX() = %q, missing multicast::cluster opcode", ptx)
			}
			if strings.Contains(ptx, ".L2::cache_hint") != tt.wantHint {
				t.Fatalf("FormatPTX() = %q, wantHint=%v", ptx, tt.wantHint)
			}
		})
	}

	invalidTests := []struct {
		name string
		spec TMAMulticastSpec
	}{
		{
			name: "sm_121a rejects cluster size > 2",
			spec: TMAMulticastSpec{Target: TargetSM121a, ClusterSize: 4, CTAMask: 0xf, TileBytes: 4096},
		},
		{
			name: "invalid cluster size 3",
			spec: TMAMulticastSpec{Target: TargetSM100a, ClusterSize: 3, CTAMask: 0x3, TileBytes: 4096},
		},
		{
			name: "zero CTA mask",
			spec: TMAMulticastSpec{Target: TargetSM100a, ClusterSize: 2, CTAMask: 0, TileBytes: 4096},
		},
		{
			name: "CTA mask exceeds cluster size",
			spec: TMAMulticastSpec{Target: TargetSM100a, ClusterSize: 2, CTAMask: 0x7, TileBytes: 4096},
		},
		{
			name: "tile bytes not multiple of 16",
			spec: TMAMulticastSpec{Target: TargetSM100a, ClusterSize: 2, CTAMask: 0x3, TileBytes: 100},
		},
		{
			name: "unknown target",
			spec: TMAMulticastSpec{Target: "sm_80", ClusterSize: 2, CTAMask: 0x3, TileBytes: 4096},
		},
	}
	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.spec.Validate(); err == nil {
				t.Fatalf("Validate() = nil, want error")
			}
		})
	}
}

func ExampleTMAGather4Spec() {
	policy := L2CachePolicySpec{
		PrimaryPolicy:   L2EvictFirst,
		SecondaryPolicy: L2EvictUnchanged,
		Fraction:        0.25,
	}
	policyPTX, _ := policy.FormatPTX("%0", "%1")

	gather := TMAGather4Spec{
		Target:     TargetSM100a,
		Mode:       TMAGather4Load,
		ColDim:     32,
		ElemBytes:  4,
		RowIndices: [4]int32{5, 18, 73, 211},
		L2Policy:   &policy,
	}
	gatherPTX, _ := gather.FormatPTX()

	fmt.Println(policyPTX)
	fmt.Println(gather.TransferBytes())
	fmt.Println(gatherPTX)
	// Output:
	// createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, %1;
	// 512
	// cp.async.bulk.tensor.2d.shared::cta.global.tile.gather4.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;
}
