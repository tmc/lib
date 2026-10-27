package tcgen05

import (
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestCLCPersistentTileGEMMSource(t *testing.T) {
	g := CLCPersistentTileGEMM{
		Name:       "clc_gemm_bf16",
		N:          128,
		K:          64,
		BM:         32,
		BN:         32,
		BK:         32,
		ActiveCTAs: 4,
		Weight:     WeightBF16,
		Out:        F32,
	}
	src, err := g.Source()
	if err != nil {
		t.Fatalf("Source: %v", err)
	}
	for _, want := range []string{
		"clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];",
		"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, %0;",
		"clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, %1;",
		"eg_emu_clc_try_cancel_async",
		"eg_emu_clc_query_is_canceled",
		"eg_emu_clc_query_first_ctaid_x",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("Source() missing %q", want)
		}
	}
}

func TestCLCPersistentTileGEMMEmulation(t *testing.T) {
	r := rand.New(rand.NewPCG(601, 602))

	tests := []struct {
		name string
		m    int
		g    CLCPersistentTileGEMM
	}{
		{
			name: "bf16_f32_16tiles",
			m:    128,
			g: CLCPersistentTileGEMM{
				Name:       "clc_bf16_f32",
				N:          128,
				K:          64,
				BM:         32,
				BN:         32,
				BK:         32,
				ActiveCTAs: 4,
				Weight:     WeightBF16,
				Out:        F32,
			},
		},
		{
			name: "bf16_bf16_relu_16tiles",
			m:    64,
			g: CLCPersistentTileGEMM{
				Name:       "clc_bf16_relu",
				N:          128,
				K:          64,
				BM:         16,
				BN:         32,
				BK:         32,
				ActiveCTAs: 4,
				Weight:     WeightBF16,
				Out:        BF16,
				Epilogue:   MustLambda("(max $0 0)"),
			},
		},
		{
			name: "nvfp4_f32_16tiles",
			m:    128,
			g: CLCPersistentTileGEMM{
				Name:       "clc_nvfp4_f32",
				N:          128,
				K:          64,
				BM:         32,
				BN:         32,
				BK:         32,
				ActiveCTAs: 4,
				Weight:     WeightNVFP4,
				Out:        F32,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			totalTiles := tt.g.TotalTiles(tt.m)
			if totalTiles != 16 {
				t.Fatalf("TotalTiles(%d) = %d, want 16", tt.m, totalTiles)
			}

			src, err := tt.g.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}

			x := make([]uint16, tt.m*tt.g.K)
			for i := range x {
				if tt.g.Weight == WeightE4M3 {
					x[i] = uint16(E4M3Bits(float32(r.NormFloat64() * 0.5)))
				} else {
					x[i] = BF16Bits(float32(r.NormFloat64()))
				}
			}

			var w GEMMWeights
			switch tt.g.Weight {
			case WeightNVFP4:
				w = randNVFP4(r, tt.g.N, tt.g.K)
				w.GlobalScal = 1.0 / 64
			case WeightE4M3:
				w.E4M3 = make([]uint8, tt.g.N*tt.g.K)
				for i := range w.E4M3 {
					w.E4M3[i] = E4M3Bits(float32(r.NormFloat64() * 0.25))
				}
			default:
				w.BF16 = make([]uint16, tt.g.N*tt.g.K)
				for i := range w.BF16 {
					w.BF16[i] = BF16Bits(float32(r.NormFloat64() * 0.25))
				}
			}

			ref, err := tt.g.Reference(tt.m, x, w)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			var args []cudahostemu.Arg
			args = append(args, cudahostemu.IntArg(tt.m))
			if tt.g.Weight == WeightE4M3 {
				xb := make([]byte, len(x))
				for i, v := range x {
					xb[i] = uint8(v)
				}
				args = append(args, cudahostemu.BufferArg("x", "uint8_t", xb))
			} else {
				args = append(args, cudahostemu.BufferArg("x", "uint16_t", u16Bytes(x)))
			}

			switch tt.g.Weight {
			case WeightNVFP4:
				args = append(args,
					cudahostemu.BufferArg("w", "uint8_t", w.Packed),
					cudahostemu.BufferArg("w_sf", "uint8_t", w.Scales),
					cudahostemu.BufferArg("w_gs", "float", f32Bytes([]float32{w.GlobalScal})),
				)
			case WeightE4M3:
				args = append(args, cudahostemu.BufferArg("w", "uint8_t", w.E4M3))
			default:
				args = append(args, cudahostemu.BufferArg("w", "uint16_t", u16Bytes(w.BF16)))
			}

			outBytes := poison(tt.m * tt.g.N * tt.g.Out.Bits() / 8)
			args = append(args, cudahostemu.BufferArg("out", tt.g.Out.cType(), outBytes))

			err = cudahostemu.RunWithCLC(
				context.Background(),
				src,
				tt.g.ActiveCTAs,
				totalTiles,
				cudahostemu.Dim3{X: 64},
				tt.g.SharedBytes(),
				args...,
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunWithCLC: %v", err)
			}

			var wantBytes []byte
			switch tt.g.Out {
			case BF16:
				wantU16 := make([]uint16, len(ref))
				for i, v := range ref {
					wantU16[i] = BF16Bits(v)
				}
				wantBytes = u16Bytes(wantU16)
			default:
				wantBytes = f32Bytes(ref)
			}

			if !bytes.Equal(outBytes, wantBytes) {
				t.Fatalf("output mismatch against float64 reference at %s", firstDiff(tt.g.Out, outBytes, wantBytes))
			}
		})
	}
}
