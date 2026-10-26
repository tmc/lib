package cudahostemu

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// TestFragmentLayoutsMatchEmulator verifies that MMAM16N8K16A, MMAM16N8K16B,
// MMAM16N8C, MMAM16N8K32A, MMAM16N8K32B, LDMatrixM8N8X4, WGMMADesc, and
// SwizzleOffset agree with cudahostemu.h for every lane 0..31 and register.
func TestFragmentLayoutsMatchEmulator(t *testing.T) {
	const src = `
__global__ void probe_layouts(
    int* a16_coords,
    int* b16_coords,
    int* c16_coords,
    int* a32_coords,
    int* b32_coords,
    uint32_t* ldm_out,
    uint64_t* desc_out,
    uint32_t* swz_out
) {
    const int lane = threadIdx.x;
    const uint16_t one_bf16 = 0x3f80u;  // 1.0f in bf16
    const uint8_t one_e4m3 = 0x38u;     // 1.0f in e4m3

    // 1. Probe mma.m16n8k16 A and C coordinates for every (tlane, treg, thalf).
    // Set B[k, col] = (k + 1) + (col + 1) * 100 in bf16 so a single 1.0 at A[row, k]
    // produces D[row, col] = (k + 1) + (col + 1) * 100 at the lane/c_reg owning (row, col).
    uint32_t b16_id[2];
    {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 2; ++i) {
            int k0 = 2 * t + 8 * i;
            uint16_t lo = eg_emu_f32_to_bf16((float)((k0 + 1) + (g + 1) * 16));
            uint16_t hi = eg_emu_f32_to_bf16((float)((k0 + 2) + (g + 1) * 16));
            b16_id[i] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
    }
    for (int tlane = 0; tlane < 32; ++tlane) {
        for (int treg = 0; treg < 4; ++treg) {
            for (int thalf = 0; thalf < 2; ++thalf) {
                uint32_t a[4] = {0, 0, 0, 0};
                if (lane == tlane) {
                    a[treg] = (uint32_t)one_bf16 << (16 * thalf);
                }
                float z[4] = {0, 0, 0, 0}, d[4];
                eg_emu_mma_bf16_16816(d, a, b16_id, z);
                // When col == 0 (lanes with (lane % 4) == 0 and c_reg in {0, 2}),
                // d[c_reg] == (k + 1) + 16 when this lane's c_reg owns A's row.
                for (int creg = 0; creg < 4; ++creg) {
                    if (d[creg] != 0.0f) {
                        int val = (int)(d[creg] + 0.5f);
                        int col = (val - 1) / 16 - 1;
                        int k = (val - 1) % 16;
                        int row = (lane / 4) + 8 * (creg / 2);
                        if (col == 0) {
                            int idx = (tlane * 4 + treg) * 2 + thalf;
                            a16_coords[idx * 2 + 0] = row;
                            a16_coords[idx * 2 + 1] = k;
                        }
                        if (tlane == 0 && treg == 0 && thalf == 0) {
                            // Also record C(lane, creg) for the rows active here.
                            c16_coords[(lane * 4 + creg) * 2 + 0] = row;
                            c16_coords[(lane * 4 + creg) * 2 + 1] = col;
                        }
                    }
                }
            }
        }
    }
    // Complete C(lane, creg) for all rows by running with all-ones A.
    {
        uint32_t a_ones[4];
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) {
            int row = g + 8 * (i % 2), k0 = 2 * t + 8 * (i / 2);
            uint16_t v = (k0 == 0) ? eg_emu_f32_to_bf16((float)(row + 1)) : 0;
            a_ones[i] = v;
        }
        uint32_t b_col[2] = {0, 0};
        if (t == 0) {
            b_col[0] = eg_emu_f32_to_bf16((float)(g + 1));
        }
        float z[4] = {0, 0, 0, 0}, d[4];
        eg_emu_mma_bf16_16816(d, a_ones, b_col, z);
        // Each d[creg] = (row + 1) * (col + 1); probe row and col separately:
        uint32_t b_one[2] = {t == 0 ? (uint32_t)one_bf16 : 0u, 0u};
        eg_emu_mma_bf16_16816(d, a_ones, b_one, z);
        for (int creg = 0; creg < 4; ++creg) {
            c16_coords[(lane * 4 + creg) * 2 + 0] = (int)(d[creg] + 0.5f) - 1;
        }
        uint32_t a_one[4] = {t == 0 ? (uint32_t)one_bf16 : 0u, t == 0 ? (uint32_t)one_bf16 : 0u, 0u, 0u};
        eg_emu_mma_bf16_16816(d, a_one, b_col, z);
        for (int creg = 0; creg < 4; ++creg) {
            c16_coords[(lane * 4 + creg) * 2 + 1] = (int)(d[creg] + 0.5f) - 1;
        }
    }

    // 2. Probe mma.m16n8k16 B coordinates for every (tlane, treg, thalf).
    // Set A[0, k] = k + 1 in bf16 so a single 1.0 at B[k, col] gives D[0, col] = k + 1.
    uint32_t a16_row0[4] = {0, 0, 0, 0};
    if (lane / 4 == 0) {
        int t = lane % 4;
        for (int i = 0; i < 2; ++i) {
            int k0 = 2 * t + 8 * i;
            uint16_t lo = eg_emu_f32_to_bf16((float)(k0 + 1));
            uint16_t hi = eg_emu_f32_to_bf16((float)(k0 + 2));
            a16_row0[2 * i] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
    }
    for (int tlane = 0; tlane < 32; ++tlane) {
        for (int treg = 0; treg < 2; ++treg) {
            for (int thalf = 0; thalf < 2; ++thalf) {
                uint32_t b[2] = {0, 0};
                if (lane == tlane) {
                    b[treg] = (uint32_t)one_bf16 << (16 * thalf);
                }
                float z[4] = {0, 0, 0, 0}, d[4];
                eg_emu_mma_bf16_16816(d, a16_row0, b, z);
                if (lane / 4 == 0) {
                    for (int creg = 0; creg < 2; ++creg) {
                        if (d[creg] != 0.0f) {
                            int k = (int)(d[creg] + 0.5f) - 1;
                            int col = 2 * (lane % 4) + creg;
                            int idx = (tlane * 2 + treg) * 2 + thalf;
                            b16_coords[idx * 2 + 0] = k;
                            b16_coords[idx * 2 + 1] = col;
                        }
                    }
                }
            }
        }
    }

    // 3. Probe mma.m16n8k32 A and B coordinates for every (tlane, treg, tbyte).
    // Split k into low/high halves (each 1..16, which is powers-of-two/exact in e4m3
    // when encoded as (k%8)+1 in col 0 and (k/8)+1 in col 1, since 1..8 are exact in e4m3).
    uint32_t b32_probe[2] = {0, 0};
    {
        int g = lane / 4, t = lane % 4;
        if (g == 0 || g == 1) {
            for (int i = 0; i < 2; ++i) {
                uint32_t w = 0;
                for (int b = 0; b < 4; ++b) {
                    int k = 4 * t + 16 * i + b;
                    int digit = (g == 0) ? ((k % 8) + 1) : ((k / 8) + 1);
                    w |= (uint32_t)eg_emu_f32_to_e4m3((float)digit) << (8 * b);
                }
                b32_probe[i] = w;
            }
        }
    }
    for (int tlane = 0; tlane < 32; ++tlane) {
        for (int treg = 0; treg < 4; ++treg) {
            for (int tb = 0; tb < 4; ++tb) {
                uint32_t a[4] = {0, 0, 0, 0};
                if (lane == tlane) {
                    a[treg] = (uint32_t)one_e4m3 << (8 * tb);
                }
                float z[4] = {0, 0, 0, 0}, d[4];
                eg_emu_mma_e4m3_16832(d, a, b32_probe, z);
                if (lane % 4 == 0) {
                    for (int half_r = 0; half_r < 2; ++half_r) {
                        float lo = d[2 * half_r + 0];
                        float hi = d[2 * half_r + 1];
                        if (lo != 0.0f || hi != 0.0f) {
                            int row = (lane / 4) + 8 * half_r;
                            int k = ((int)(hi + 0.5f) - 1) * 8 + ((int)(lo + 0.5f) - 1);
                            int idx = (tlane * 4 + treg) * 4 + tb;
                            a32_coords[idx * 2 + 0] = row;
                            a32_coords[idx * 2 + 1] = k;
                        }
                    }
                }
            }
        }
    }

    uint32_t a32_probe[4] = {0, 0, 0, 0};
    {
        int g = lane / 4, t = lane % 4;
        if (g == 0 || g == 1) {
            for (int i = 0; i < 2; ++i) {
                uint32_t w = 0;
                for (int b = 0; b < 4; ++b) {
                    int k = 4 * t + 16 * i + b;
                    int digit = (g == 0) ? ((k % 8) + 1) : ((k / 8) + 1);
                    w |= (uint32_t)eg_emu_f32_to_e4m3((float)digit) << (8 * b);
                }
                a32_probe[2 * i] = w;
            }
        }
    }
    for (int tlane = 0; tlane < 32; ++tlane) {
        for (int treg = 0; treg < 2; ++treg) {
            for (int tb = 0; tb < 4; ++tb) {
                uint32_t b[2] = {0, 0};
                if (lane == tlane) {
                    b[treg] = (uint32_t)one_e4m3 << (8 * tb);
                }
                float z[4] = {0, 0, 0, 0}, d[4];
                eg_emu_mma_e4m3_16832(d, a32_probe, b, z);
                // Lane 0..3 hold row 0 in d[0..1], lane 4..7 hold row 1 in d[0..1].
                // Send row 1's d[0..1] to lane 0..3 via shuffle:
                for (int creg = 0; creg < 2; ++creg) {
                    float hi = __shfl_xor_sync(0xffffffffu, d[creg], 4);
                    if (lane / 4 == 0 && d[creg] != 0.0f) {
                        int k = ((int)(hi + 0.5f) - 1) * 8 + ((int)(d[creg] + 0.5f) - 1);
                        int col = 2 * (lane % 4) + creg;
                        int idx = (tlane * 2 + treg) * 4 + tb;
                        b32_coords[idx * 2 + 0] = k;
                        b32_coords[idx * 2 + 1] = col;
                    }
                }
            }
        }
    }

    // 4. Probe ldmatrix.m8n8.x4.b16 for every lane 0..31.
    __shared__ uint16_t smat[4][8][8];
    for (int idx = lane; idx < 4 * 8 * 8; idx += 32) {
        int m = idx / 64, r = (idx / 8) % 8, c = idx % 8;
        smat[m][r][c] = (uint16_t)((m << 8) | (r << 4) | (2 * c));
    }
    __syncthreads();
    uint32_t ldm_regs[4];
    eg_emu_ldmatrix_x4(ldm_regs, &smat[lane / 8][lane % 8][0], false);
    for (int j = 0; j < 4; ++j) {
        ldm_out[lane * 4 + j] = ldm_regs[j];
    }

    // 5. Probe WGMMADesc and SwizzleOffset.
    if (lane < 4) {
        desc_out[lane] = eg_emu_wgmma_desc(256 * (lane + 1), 128 * (lane + 1), 1024 * (lane + 1), lane);
    }
    for (int swz = 0; swz < 4; ++swz) {
        uint64_t desc = eg_emu_wgmma_desc(0, 16, 128, swz);
        for (int off = lane * 16; off < 4096; off += 32 * 16) {
            swz_out[swz * 256 + off / 16] = eg_emu_wgmma_swizzle((uint32_t)off, desc);
        }
    }
}
`
	res := run(t, Program{
		Source: src,
		Launch: "eg_emu_launch(dim3(1), dim3(32), 0, probe_layouts, a16, b16, c16, a32, b32, ldm, desc, swz)",
		Buffers: []Buffer{
			{"a16", "int", Poison(32 * 4 * 2 * 2 * 4)},
			{"b16", "int", Poison(32 * 2 * 2 * 2 * 4)},
			{"c16", "int", Poison(32 * 4 * 2 * 4)},
			{"a32", "int", Poison(32 * 4 * 4 * 2 * 4)},
			{"b32", "int", Poison(32 * 2 * 4 * 2 * 4)},
			{"ldm", "uint32_t", Poison(32 * 4 * 4)},
			{"desc", "uint64_t", Poison(4 * 8)},
			{"swz", "uint32_t", Poison(4 * 256 * 4)},
		},
	})

	a16 := u32s(res["a16"])
	for lane := range 32 {
		for reg := range 4 {
			wantRow, wantColPair := MMAM16N8K16A(lane, reg)
			for half := range 2 {
				idx := (lane*4+reg)*2 + half
				gotRow, gotCol := int(a16[idx*2+0]), int(a16[idx*2+1])
				if gotRow != wantRow || gotCol != wantColPair+half {
					t.Errorf("MMAM16N8K16A(lane=%d, reg=%d) half=%d: emu=(%d,%d), want=(%d,%d)",
						lane, reg, half, gotRow, gotCol, wantRow, wantColPair+half)
				}
			}
		}
	}

	b16 := u32s(res["b16"])
	for lane := range 32 {
		for reg := range 2 {
			wantRowPair, wantCol := MMAM16N8K16B(lane, reg)
			for half := range 2 {
				idx := (lane*2+reg)*2 + half
				gotRow, gotCol := int(b16[idx*2+0]), int(b16[idx*2+1])
				if gotRow != wantRowPair+half || gotCol != wantCol {
					t.Errorf("MMAM16N8K16B(lane=%d, reg=%d) half=%d: emu=(%d,%d), want=(%d,%d)",
						lane, reg, half, gotRow, gotCol, wantRowPair+half, wantCol)
				}
			}
		}
	}

	c16 := u32s(res["c16"])
	for lane := range 32 {
		for reg := range 4 {
			wantRow, wantCol := MMAM16N8C(lane, reg)
			idx := lane*4 + reg
			gotRow, gotCol := int(c16[idx*2+0]), int(c16[idx*2+1])
			if gotRow != wantRow || gotCol != wantCol {
				t.Errorf("MMAM16N8C(lane=%d, reg=%d): emu=(%d,%d), want=(%d,%d)",
					lane, reg, gotRow, gotCol, wantRow, wantCol)
			}
		}
	}

	a32 := u32s(res["a32"])
	for lane := range 32 {
		for reg := range 4 {
			wantRow, wantColQuad := MMAM16N8K32A(lane, reg)
			for b := range 4 {
				idx := (lane*4+reg)*4 + b
				gotRow, gotCol := int(a32[idx*2+0]), int(a32[idx*2+1])
				if gotRow != wantRow || gotCol != wantColQuad+b {
					t.Errorf("MMAM16N8K32A(lane=%d, reg=%d) byte=%d: emu=(%d,%d), want=(%d,%d)",
						lane, reg, b, gotRow, gotCol, wantRow, wantColQuad+b)
				}
			}
		}
	}

	b32 := u32s(res["b32"])
	for lane := range 32 {
		for reg := range 2 {
			wantRowQuad, wantCol := MMAM16N8K32B(lane, reg)
			for b := range 4 {
				idx := (lane*2+reg)*4 + b
				gotRow, gotCol := int(b32[idx*2+0]), int(b32[idx*2+1])
				if gotRow != wantRowQuad+b || gotCol != wantCol {
					t.Errorf("MMAM16N8K32B(lane=%d, reg=%d) byte=%d: emu=(%d,%d), want=(%d,%d)",
						lane, reg, b, gotRow, gotCol, wantRowQuad+b, wantCol)
				}
			}
		}
	}

	ldm := u32s(res["ldm"])
	for lane := range 32 {
		mat, row, colBytes := LDMatrixM8N8X4(lane)
		if mat != lane/8 || row != lane%8 {
			t.Errorf("LDMatrixM8N8X4(%d) matrix,row = (%d,%d), want (%d,%d)", lane, mat, row, lane/8, lane%8)
		}
		for j := range 4 {
			got := ldm[lane*4+j]
			rRow := lane / 4
			lo := uint32((j << 8) | (rRow << 4) | colBytes)
			hi := uint32((j << 8) | (rRow << 4) | (colBytes + 2))
			want := lo | (hi << 16)
			if got != want {
				t.Errorf("LDMatrixM8N8X4(%d) reg %d = %#08x, want %#08x", lane, j, got, want)
			}
		}
	}

	for mode := range 4 {
		got := binary.LittleEndian.Uint64(res["desc"][8*mode:])
		want := WGMMADesc(256*(mode+1), 128*(mode+1), 1024*(mode+1), SwizzleMode(mode))
		if got != want {
			t.Errorf("WGMMADesc(mode=%d) = %#x, want %#x", mode, got, want)
		}
	}

	swz := u32s(res["swz"])
	for _, mode := range []SwizzleMode{SwizzleNone, Swizzle128B, Swizzle64B, Swizzle32B} {
		for step := range 256 {
			off := step * 16
			got := int(swz[int(mode)*256+step])
			want := SwizzleOffset(off, mode)
			if got != want {
				t.Errorf("SwizzleOffset(%d, %d) = %d, want %d", off, mode, got, want)
			}
		}
	}
}

// TestFragSimPackedMatmul ports frag_sim.py: it packs a [32, 256] weight matrix
// into per-lane m16n8k16 fragments and multiplies by an [8, 256] input matrix
// using MMAM16N8K16A, MMAM16N8K16B, and MMAM16N8C, verifying exact agreement
// with a direct float64 matmul.
func TestFragSimPackedMatmul(t *testing.T) {
	const N, K, M = 32, 256, 8
	r := rand.New(rand.NewPCG(10, 20))
	w := make([]float64, N*K)
	x := make([]float64, M*K)
	for i := range w {
		w[i] = r.NormFloat64()
	}
	for i := range x {
		x[i] = r.NormFloat64()
	}

	// pack(w): view(N/16, 2, 8, K/64, 4, 2, 8) -> permute(0, 3, 1, 5, 2, 4, 6)
	// resulting in shape [N/16, K/64, 4, 32, 8] (nt, kb, q, lane, e8).
	packAt := func(nt, kb, q, lane, e8 int) float64 {
		rh := q >> 1
		eh := q & 1
		g := lane >> 2
		t := lane & 3
		n := ((nt*2+rh)*8 + g)
		k := (((kb*4+t)*2+eh)*8 + e8)
		return w[n*K+k]
	}

	mmaStep := func(c *[32][4]float64, a [32][4][2]float64, b [32][2][2]float64) {
		var matA [16][16]float64
		var matB [16][8]float64
		for lane := range 32 {
			for reg := range 4 {
				row, colPair := MMAM16N8K16A(lane, reg)
				matA[row][colPair] = a[lane][reg][0]
				matA[row][colPair+1] = a[lane][reg][1]
			}
			for reg := range 2 {
				rowPair, col := MMAM16N8K16B(lane, reg)
				matB[rowPair][col] = b[lane][reg][0]
				matB[rowPair+1][col] = b[lane][reg][1]
			}
		}
		var matC [16][8]float64
		for m := range 16 {
			for n := range 8 {
				for k := range 16 {
					matC[m][n] += matA[m][k] * matB[k][n]
				}
			}
		}
		for lane := range 32 {
			for reg := range 4 {
				row, col := MMAM16N8C(lane, reg)
				c[lane][reg] += matC[row][col]
			}
		}
	}

	var out [M][N]float64
	for nt := range N / 16 {
		var acc [32][4]float64
		for kb := range K / 64 {
			var rg, rg8, xw [32][16]float64
			for l := range 32 {
				for e := range 8 {
					rg[l][e] = packAt(nt, kb, 0, l, e)
					rg[l][8+e] = packAt(nt, kb, 1, l, e)
					rg8[l][e] = packAt(nt, kb, 2, l, e)
					rg8[l][8+e] = packAt(nt, kb, 3, l, e)
				}
				row := l >> 2
				k0 := kb*64 + (l&3)*16
				for e := range 16 {
					xw[l][e] = x[row*K+k0+e]
				}
			}
			for s := range 4 {
				var a [32][4][2]float64
				var b [32][2][2]float64
				for l := range 32 {
					i0, i1 := 2*(2*s), 2*(2*s+1)
					a[l][0] = [2]float64{rg[l][i0], rg[l][i0+1]}
					a[l][1] = [2]float64{rg8[l][i0], rg8[l][i0+1]}
					a[l][2] = [2]float64{rg[l][i1], rg[l][i1+1]}
					a[l][3] = [2]float64{rg8[l][i1], rg8[l][i1+1]}
					b[l][0] = [2]float64{xw[l][i0], xw[l][i0+1]}
					b[l][1] = [2]float64{xw[l][i1], xw[l][i1+1]}
				}
				mmaStep(&acc, a, b)
			}
		}
		for l := range 32 {
			for reg := range 4 {
				row, col := MMAM16N8C(l, reg)
				out[col][nt*16+row] = acc[l][reg]
			}
		}
	}

	for m := range M {
		for n := range N {
			var want float64
			for k := range K {
				want += x[m*K+k] * w[n*K+k]
			}
			if diff := math.Abs(out[m][n] - want); diff > 1e-12 {
				t.Fatalf("out[%d][%d] = %v, want %v (diff %v)", m, n, out[m][n], want, diff)
			}
		}
	}
}
