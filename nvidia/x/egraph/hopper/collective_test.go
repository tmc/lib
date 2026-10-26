package hopper

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestAllReduceResidualRMSNorm(t *testing.T) {
	const (
		ranks = 4
		rows  = 4
		cols  = 128
	)

	for _, dtype := range []DType{F32, BF16} {
		t.Run(dtype.String(), func(t *testing.T) {
			k := AllReduceResidualRMSNorm{
				Name:    "allreduce_res_rmsnorm_" + dtype.String(),
				Cols:    cols,
				Ranks:   ranks,
				Eps:     1e-5,
				DType:   dtype,
				Threads: 64,
			}
			src, err := k.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}
			for _, wantSub := range []string{
				"multimem.ld_reduce.global.add.f32",
				"bf16x2",
				"eg_emu_multimem_ld_reduce_add_f32",
				"eg_emu_multimem_ld_reduce_add_bf16",
			} {
				if !strings.Contains(src, wantSub) {
					t.Fatalf("Source missing %q", wantSub)
				}
			}

			elemsPerRank := rows * cols
			symInF64 := make([]float64, ranks*elemsPerRank)
			resInF64 := make([]float64, ranks*elemsPerRank)
			weightF64 := make([]float64, cols)

			// Use exact dyadic rationals so float64 and device FP32 sums have
			// zero associative rounding drift and can be checked bit-accurately.
			for r := 0; r < ranks; r++ {
				for row := 0; row < rows; row++ {
					for col := 0; col < cols; col++ {
						idx := r*elemsPerRank + row*cols + col
						v := float32(((r+1)*(row+1)+col)%7-3) * 0.25
						symInF64[idx] = float64(dtype.Round(v))
						rv := float32((r+row*3+col*5)%9-4) * 0.125
						resInF64[idx] = float64(dtype.Round(rv))
					}
				}
			}
			for col := 0; col < cols; col++ {
				wv := float32(1.0) + float32((col%5)-2)*0.125
				weightF64[col] = float64(dtype.Round(wv))
			}

			wantRes, wantNorm, err := k.Reference(rows, symInF64, resInF64, weightF64)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			encode := func(vals []float64) []byte {
				if dtype == F32 {
					f32 := make([]float32, len(vals))
					for i, v := range vals {
						f32[i] = float32(v)
					}
					return f32Bytes(f32)
				}
				u16 := make([]uint16, len(vals))
				for i, v := range vals {
					u16[i] = BF16Bits(float32(v))
				}
				return u16Bytes(u16)
			}

			elemBytes := dtype.Bits() / 8
			symInBytes := encode(symInF64)
			resInBytes := encode(resInF64)
			weightBytes := encode(weightF64)
			resOutBytes := cudahostemu.Poison(ranks * elemsPerRank * elemBytes)
			normOutBytes := cudahostemu.Poison(ranks * elemsPerRank * elemBytes)

			ct := dtype.cType()
			grid, block := k.Grid(rows)
			err = cudahostemu.RunMultiRank(context.Background(), src, ranks, cudahostemu.Launch{
				Kernel: k.Name,
				Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
				Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			},
				cudahostemu.IntArg(rows),
				cudahostemu.BufferArg("sym_in", ct, symInBytes),
				cudahostemu.BufferArg("residual", ct, resInBytes),
				cudahostemu.BufferArg("weight", ct, weightBytes),
				cudahostemu.BufferArg("residual_out", ct, resOutBytes),
				cudahostemu.BufferArg("norm_out", ct, normOutBytes),
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank: %v", err)
			}

			if dtype == F32 {
				gotRes := bytesF32(resOutBytes)
				gotNorm := bytesF32(normOutBytes)
				for i := range wantRes {
					wResBits := math.Float32bits(float32(wantRes[i]))
					gResBits := math.Float32bits(gotRes[i])
					if gResBits != wResBits {
						t.Fatalf("residual_out[%d] = %#08x (%v), want %#08x (%v)",
							i, gResBits, gotRes[i], wResBits, float32(wantRes[i]))
					}
					wNormBits := math.Float32bits(float32(wantNorm[i]))
					gNormBits := math.Float32bits(gotNorm[i])
					if gNormBits != wNormBits {
						t.Fatalf("norm_out[%d] = %#08x (%v), want %#08x (%v)",
							i, gNormBits, gotNorm[i], wNormBits, float32(wantNorm[i]))
					}
				}
			} else {
				gotRes := bytesU16(resOutBytes)
				gotNorm := bytesU16(normOutBytes)
				for i := range wantRes {
					wResBits := BF16Bits(float32(wantRes[i]))
					if gotRes[i] != wResBits {
						t.Fatalf("residual_out[%d] = %#04x (%v), want %#04x (%v)",
							i, gotRes[i], FromBF16Bits(gotRes[i]), wResBits, float32(wantRes[i]))
					}
					wNormBits := BF16Bits(float32(wantNorm[i]))
					if gotNorm[i] != wNormBits {
						t.Fatalf("norm_out[%d] = %#04x (%v), want %#04x (%v)",
							i, gotNorm[i], FromBF16Bits(gotNorm[i]), wNormBits, float32(wantNorm[i]))
					}
				}
			}
		})
	}
}

func TestNVLinkMoECombine(t *testing.T) {
	const (
		ranks          = 4
		topK           = 2
		expertsPerRank = 2
		tokens         = 6
		cols           = 64
	)

	for _, dtype := range []DType{F32, BF16} {
		t.Run(dtype.String(), func(t *testing.T) {
			k := NVLinkMoECombine{
				Name:    "nvlink_moe_combine_" + dtype.String(),
				Cols:    cols,
				TopK:    topK,
				Ranks:   ranks,
				DType:   dtype,
				Threads: 64,
			}
			src, err := k.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}
			if !strings.Contains(src, "eg_emu_peer_ptr") {
				t.Fatalf("Source missing eg_emu_peer_ptr")
			}

			totalExperts := ranks * expertsPerRank
			sliceElems := expertsPerRank * tokens * cols
			symExpertF64 := make([]float64, ranks*sliceElems)
			for i := range symExpertF64 {
				v := float32((i%13)-6) * 0.25
				symExpertF64[i] = float64(dtype.Round(v))
			}

			routesPerRank := tokens * topK
			expertIDs := make([]int32, ranks*routesPerRank)
			routerWeightsF64 := make([]float64, ranks*routesPerRank)
			routerWeightsF32 := make([]float32, ranks*routesPerRank)
			for r := 0; r < ranks; r++ {
				for m := 0; m < tokens; m++ {
					// Route k=0 and k=1 to distinct experts across local and remote GPUs.
					e0 := (r*expertsPerRank + m) % totalExperts
					e1 := (e0 + 3) % totalExperts
					idx0 := r*routesPerRank + m*topK + 0
					idx1 := r*routesPerRank + m*topK + 1
					expertIDs[idx0] = int32(e0)
					expertIDs[idx1] = int32(e1)
					w0 := float32(0.625)
					w1 := float32(0.375)
					routerWeightsF32[idx0] = w0
					routerWeightsF32[idx1] = w1
					routerWeightsF64[idx0] = float64(w0)
					routerWeightsF64[idx1] = float64(w1)
				}
			}

			wantOut, err := k.Reference(tokens, expertsPerRank, symExpertF64, expertIDs, routerWeightsF64)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			encode := func(vals []float64) []byte {
				if dtype == F32 {
					f32 := make([]float32, len(vals))
					for i, v := range vals {
						f32[i] = float32(v)
					}
					return f32Bytes(f32)
				}
				u16 := make([]uint16, len(vals))
				for i, v := range vals {
					u16[i] = BF16Bits(float32(v))
				}
				return u16Bytes(u16)
			}

			elemBytes := dtype.Bits() / 8
			symExpertBytes := encode(symExpertF64)
			idsBytes := i32Bytes(expertIDs)
			weightsBytes := f32Bytes(routerWeightsF32)
			outBytes := cudahostemu.Poison(ranks * tokens * cols * elemBytes)

			ct := dtype.cType()
			grid, block := k.Grid(tokens)
			err = cudahostemu.RunMultiRank(context.Background(), src, ranks, cudahostemu.Launch{
				Kernel: k.Name,
				Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
				Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			},
				cudahostemu.IntArg(tokens),
				cudahostemu.IntArg(expertsPerRank),
				cudahostemu.BufferArg("sym_expert_out", ct, symExpertBytes),
				cudahostemu.BufferArg("expert_ids", "int32_t", idsBytes),
				cudahostemu.BufferArg("router_weights", "float", weightsBytes),
				cudahostemu.BufferArg("out", ct, outBytes),
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank: %v", err)
			}

			if dtype == F32 {
				got := bytesF32(outBytes)
				for i, w := range wantOut {
					if got[i] != float32(w) {
						t.Fatalf("out[%d] = %v, want %v", i, got[i], float32(w))
					}
				}
			} else {
				got := bytesU16(outBytes)
				for i, w := range wantOut {
					wantBits := BF16Bits(float32(w))
					if got[i] != wantBits {
						t.Fatalf("out[%d] = %#04x (%v), want %#04x (%v)",
							i, got[i], FromBF16Bits(got[i]), wantBits, float32(w))
					}
				}
			}
		})
	}
}
