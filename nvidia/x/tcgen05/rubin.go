package tcgen05

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// RubinSpec describes the hardware capabilities of an NVIDIA Vera Rubin R100 or
// Rubin Ultra R200 GPU (PTX architecture target sm_107a, family target sm_107f).
type RubinSpec struct {
	Name                      string  // GPU model name ("Vera Rubin R100" / "Rubin Ultra R200")
	Target                    string  // PTX architecture-accelerated target ("sm_107a")
	FamilyTarget              string  // PTX family-conditional target ("sm_107f")
	HBM4Bytes                 int64   // HBM4 capacity in bytes (288 GiB for R100, 576 GiB for R200 Ultra)
	HBM4BandwidthGBps         float64 // HBM4 memory bandwidth in GB/s (13,000 for R100, 22,000 for R200 Ultra)
	NVLink6BandwidthGBps      float64 // NVLink 6 bidirectional bandwidth in GB/s per die (3,600 GB/s)
	MaxClusterCTAs            int     // maximum CTAs per cluster via .multicast::cluster::32b (32)
	SupportsLUTDecompressB    bool    // true: supports .decompress::lut::b 3-bit weight LUT decompression in tcgen05.mma
	SupportsCollectorB        bool    // true: supports .collector::b::{fill,use,lastuse,discard} reuse buffer control
	SupportsSyncRestrictReadA bool    // true: supports .sync_restrict::shared::read::mma::a early-release commit
	SupportsTI16              bool    // true: supports 16-bit Tensor Index (TI16) addressing modes
}

// DefaultRubinSpec returns the canonical hardware specification for an NVIDIA
// Vera Rubin R100 (sm_107a / sm_107f) GPU.
func DefaultRubinSpec() RubinSpec {
	return RubinSpec{
		Name:                      "Vera Rubin R100",
		Target:                    "sm_107a",
		FamilyTarget:              "sm_107f",
		HBM4Bytes:                 288 << 30,
		HBM4BandwidthGBps:         13000,
		NVLink6BandwidthGBps:      3600,
		MaxClusterCTAs:            32,
		SupportsLUTDecompressB:    true,
		SupportsCollectorB:        true,
		SupportsSyncRestrictReadA: true,
		SupportsTI16:              true,
	}
}

// DefaultRubinUltraSpec returns the canonical hardware specification for an
// NVIDIA Rubin Ultra R200 (sm_107a / sm_107f) GPU.
func DefaultRubinUltraSpec() RubinSpec {
	return RubinSpec{
		Name:                      "Rubin Ultra R200",
		Target:                    "sm_107a",
		FamilyTarget:              "sm_107f",
		HBM4Bytes:                 576 << 30,
		HBM4BandwidthGBps:         22000,
		NVLink6BandwidthGBps:      3600,
		MaxClusterCTAs:            32,
		SupportsLUTDecompressB:    true,
		SupportsCollectorB:        true,
		SupportsSyncRestrictReadA: true,
		SupportsTI16:              true,
	}
}

// CollectorOp specifies the operand collector reuse-buffer operation for
// .collector::a::<op> and .collector::b::<op> on Rubin sm_107a/sm_107f
// tcgen05.mma instructions (PTX ISA 9.4 section 9.7.18.10.10.1).
type CollectorOp string

const (
	CollectorNone    CollectorOp = ""
	CollectorFill    CollectorOp = "fill"
	CollectorUse     CollectorOp = "use"
	CollectorLastUse CollectorOp = "lastuse"
	CollectorDiscard CollectorOp = "discard"
)

func (op CollectorOp) valid() bool {
	switch op {
	case CollectorNone, CollectorFill, CollectorUse, CollectorLastUse, CollectorDiscard:
		return true
	default:
		return false
	}
}

// FromUE8M0Bits decodes an 8-bit OCP UE8M0 biased exponent scale code to float32.
func FromUE8M0Bits(code uint8) float32 {
	return DecodeUE8M0(code)
}

// UE8M0Bits encodes a positive float32 scale factor into an 8-bit OCP UE8M0 code.
func UE8M0Bits(scale float32) uint8 {
	return EncodeUE8M0(scale)
}

// Quantize3BitLUTB compresses a weight matrix B [K, N] (where K is a positive
// multiple of 32 and N is a positive multiple of 8) for PTX ISA 9.4
// (§9.7.18.10.10.1 decompress::lut::b) 3-bit look-up-table weight decompression:
//   - bScales [N * (K/32)] holds one UE8M0 per-32-element block scale for each
//     column n and 32-row block kb (indexed at n*(K/32) + kb),
//   - lutE4M3 [(N/8) * 8] holds an 8-entry E4M3 lookup table of centroids for
//     each 8-column slice n0..n0+7 (indexed at (n/8)*8 + idx),
//   - indices [K * N] holds the 3-bit centroid index in [0..7] for each B[k, n], and
//   - dequantized [K * N] holds the reconstructed float32 weights
//     FromE4M3Bits(lut[idx]) * FromUE8M0Bits(scale).
func Quantize3BitLUTB(b []float32, K, N int) (indices []uint8, lutE4M3 []uint8, bScales []uint8, dequantized []float32, err error) {
	if K <= 0 || K%MXBlock != 0 {
		return nil, nil, nil, nil, fmt.Errorf("quantize 3-bit lut b: K=%d must be a positive multiple of 32", K)
	}
	if N <= 0 || N%8 != 0 {
		return nil, nil, nil, nil, fmt.Errorf("quantize 3-bit lut b: N=%d must be a positive multiple of 8", N)
	}
	if len(b) != K*N {
		return nil, nil, nil, nil, fmt.Errorf("quantize 3-bit lut b: len(b)=%d, want %d (K=%d, N=%d)", len(b), K*N, K, N)
	}

	kBlocks := K / MXBlock
	nGroups := N / 8
	indices = make([]uint8, K*N)
	lutE4M3 = make([]uint8, nGroups*8)
	bScales = make([]uint8, N*kBlocks)
	dequantized = make([]float32, K*N)

	// 1. Compute per-column, per-32-row-block UE8M0 scales and normalized values.
	normVals := make([]float32, K*N)
	for n := 0; n < N; n++ {
		for kb := 0; kb < kBlocks; kb++ {
			var amax float32
			for j := 0; j < MXBlock; j++ {
				k := kb*MXBlock + j
				ax := float32(math.Abs(float64(b[k*N+n])))
				if ax > amax {
					amax = ax
				}
			}
			sc := uint8(UE8M0Bias)
			if amax > 0 {
				// Normalize block into [-4, 4] so E4M3 centroids have fine resolution.
				sc = EncodeUE8M0(amax / 4.0)
			}
			bScales[n*kBlocks+kb] = sc
			scale := FromUE8M0Bits(sc)
			for j := 0; j < MXBlock; j++ {
				k := kb*MXBlock + j
				normVals[k*N+n] = b[k*N+n] / scale
			}
		}
	}

	// 2. For each 8-column slice g (columns 8*g .. 8*g+7), fit 8 deterministic
	//    E4M3 centroids and quantize each element to its nearest centroid index 0..7.
	sliceVals := make([]float32, K*8)
	for g := 0; g < nGroups; g++ {
		n0 := g * 8
		idx := 0
		for k := 0; k < K; k++ {
			for c := 0; c < 8; c++ {
				sliceVals[idx] = normVals[k*N+n0+c]
				idx++
			}
		}
		sorted := slices.Clone(sliceVals)
		slices.Sort(sorted)

		// Collect unique E4M3-quantized values in sorted order.
		var uniq []uint8
		for _, v := range sorted {
			eb := E4M3Bits(v)
			if len(uniq) == 0 || uniq[len(uniq)-1] != eb {
				uniq = append(uniq, eb)
			}
		}

		var centroids [8]uint8
		if len(uniq) <= 8 {
			for m := 0; m < 8; m++ {
				if m < len(uniq) {
					centroids[m] = uniq[m]
				} else if len(uniq) > 0 {
					centroids[m] = uniq[len(uniq)-1]
				}
			}
		} else {
			// Initialize 8 centroids from empirical quantiles across sorted values,
			// followed by 4 deterministic Lloyd-Max refinement steps in E4M3 space.
			for m := 0; m < 8; m++ {
				qPos := (m*(len(sorted)-1) + 3) / 7
				centroids[m] = E4M3Bits(sorted[qPos])
			}
			for range 4 {
				var sums [8]float64
				var counts [8]int
				for _, v := range sliceVals {
					bestM := 0
					bestDist := math.Abs(float64(v - FromE4M3Bits(centroids[0])))
					for m := 1; m < 8; m++ {
						d := math.Abs(float64(v - FromE4M3Bits(centroids[m])))
						if d < bestDist {
							bestDist = d
							bestM = m
						}
					}
					sums[bestM] += float64(v)
					counts[bestM]++
				}
				for m := 0; m < 8; m++ {
					if counts[m] > 0 {
						centroids[m] = E4M3Bits(float32(sums[m] / float64(counts[m])))
					}
				}
			}
		}

		for m := 0; m < 8; m++ {
			lutE4M3[g*8+m] = centroids[m]
		}

		for k := 0; k < K; k++ {
			kb := k / MXBlock
			for c := 0; c < 8; c++ {
				n := n0 + c
				v := normVals[k*N+n]
				bestM := uint8(0)
				bestDist := math.Abs(float64(v - FromE4M3Bits(centroids[0])))
				for m := uint8(1); m < 8; m++ {
					d := math.Abs(float64(v - FromE4M3Bits(centroids[m])))
					if d < bestDist {
						bestDist = d
						bestM = m
					}
				}
				indices[k*N+n] = bestM
				scale := FromUE8M0Bits(bScales[n*kBlocks+kb])
				dequantized[k*N+n] = FromE4M3Bits(centroids[bestM]) * scale
			}
		}
	}

	return indices, lutE4M3, bScales, dequantized, nil
}

// TCGen05LUTDecompressGEMM configures a Rubin sm_107a/sm_107f Tensor Memory GEMM
// kernel that decompresses 3-bit indexed weights B on the fly via
// tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32
// and controls the operand B collector reuse buffer with .collector::b::<op>
// (PTX ISA 9.4 section 9.7.18.10.10.1 #7).
type TCGen05LUTDecompressGEMM struct {
	Name       string
	M, N, K    int         // M=128, N=128, K multiple of 64 in 64..256 (default 64)
	Target     string      // "sm_107a" or "sm_107f" (defaults to "sm_107f")
	CollectorA CollectorOp // optional .collector::a::<op> qualifier
	CollectorB CollectorOp // .collector::b::<op> qualifier (defaults to CollectorFill)
}

func (k TCGen05LUTDecompressGEMM) norm() (TCGen05LUTDecompressGEMM, error) {
	if k.Name == "" {
		k.Name = "tcgen05_lut_decompress_gemm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 lut decompress gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 64
	}
	if k.Target == "" {
		k.Target = "sm_107f"
	}
	if k.Target != "sm_107a" && k.Target != "sm_107f" {
		return k, fmt.Errorf("tcgen05 lut decompress gemm %s: unsupported target %q (want sm_107a or sm_107f)", k.Name, k.Target)
	}
	if k.CollectorB == "" {
		k.CollectorB = CollectorFill
	}
	if !k.CollectorA.valid() {
		return k, fmt.Errorf("tcgen05 lut decompress gemm %s: invalid CollectorA %q", k.Name, k.CollectorA)
	}
	if !k.CollectorB.valid() {
		return k, fmt.Errorf("tcgen05 lut decompress gemm %s: invalid CollectorB %q", k.Name, k.CollectorB)
	}
	if k.M != 128 || k.N != 128 || k.K < 64 || k.K > 256 || k.K%64 != 0 {
		return k, fmt.Errorf("tcgen05 lut decompress gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N=128, K in 64..256 multiple of 64)", k.Name, k.M, k.N, k.K)
	}
	return k, nil
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05LUTDecompressGEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// InstDescriptor packs the 32-bit tcgen05.mma instruction descriptor (idesc)
// for kind::mxf8f6f4.block_scale.decompress::lut::b.block32.
func (k TCGen05LUTDecompressGEMM) InstDescriptor() (uint32, error) {
	k, err := k.norm()
	if err != nil {
		return 0, err
	}
	return InstDesc{
		DType:    F32,
		AType:    E4M3,
		BType:    E4M3,
		M:        uint16(k.M),
		N:        uint16(k.N),
		ScaleVec: ScaleVec1X,
	}.Encode(), nil
}

func (k TCGen05LUTDecompressGEMM) collectorQualifiers() string {
	var q strings.Builder
	if k.CollectorA != CollectorNone {
		q.WriteString(".collector::a::")
		q.WriteString(string(k.CollectorA))
	}
	if k.CollectorB != CollectorNone {
		q.WriteString(".collector::b::")
		q.WriteString(string(k.CollectorB))
	}
	return q.String()
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05LUTDecompressGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idesc, err := k.InstDescriptor()
	if err != nil {
		return "", err
	}
	collectorQuals := k.collectorQualifiers()

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(`
EG_DEV float eg_rubin_ue8m0_to_f32(uint8_t code) {
    if (code == 0xffu) return __uint_as_float(0x7fc00000u);
    if (code == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float((uint32_t)code << 23);
}

// eg_tcgen05_cp_lut_b_32x128b stages the 8-entry E4M3 LUT per 8-column slice
// into 2-column-aligned TMEM [lut_taddr] via tcgen05.cp.cta_group::1.32x128b.warpx4.
EG_DEV void eg_tcgen05_cp_lut_b_32x128b(
    float* tmem, uint32_t lut_taddr,
    const uint8_t* smem_b_lut, int n_cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)n_cols;
    const uint64_t desc = eg_wgmma_desc(smem_b_lut, 16, 128);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(lut_taddr), "l"(desc) : "memory");
#else
    for (int n = 0; n < n_cols; ++n) {
        const int g = n / 8;
        for (int idx = 0; idx < 8; ++idx) {
            tmem[n * 512 + lut_taddr + idx] = eg_e4m3_to_f32(smem_b_lut[g * 8 + idx]);
        }
    }
#endif
}

// eg_tcgen05_cp_rubin_scales_32x128b stages 32-element UE8M0 block scales for A
// and B into TMEM [scale_taddr] via tcgen05.cp.cta_group::1.32x128b.warpx4.
EG_DEV void eg_tcgen05_cp_rubin_scales_32x128b(
    float* tmem, uint32_t scale_taddr,
    const uint8_t* smem_a_scales, const uint8_t* smem_b_scales,
    int m_rows, int n_cols, int k_blocks) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)smem_b_scales;
    (void)m_rows;
    (void)n_cols;
    (void)k_blocks;
    const uint64_t desc = eg_wgmma_desc(smem_a_scales, 16, 128);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(scale_taddr), "l"(desc) : "memory");
#else
    for (int r = 0; r < m_rows; ++r) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[r * 512 + scale_taddr + kb] = eg_rubin_ue8m0_to_f32(smem_a_scales[r * k_blocks + kb]);
        }
    }
    for (int n = 0; n < n_cols; ++n) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[n * 512 + scale_taddr + 16 + kb] = eg_rubin_ue8m0_to_f32(smem_b_scales[n * k_blocks + kb]);
        }
    }
#endif
}
`)

	fmt.Fprintf(&b, "// %s: Rubin %s 3-bit LUT-decompressed GEMM (M=%d, N=%d, K=%d, collector=%s).\n",
		k.Name, k.Target, k.M, k.N, k.K, collectorQuals)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const uint8_t* __restrict__ a_e4m3,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_indices,
    const uint8_t* __restrict__ b_lut_e4m3,
    const uint8_t* __restrict__ b_scales,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr int K_BLOCKS = K / 32, LUT_BYTES = (N / 8) * 8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(uint8_t, smem_a, [M * K]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_idx, [K * N]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_lut, [LUT_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_a_scales, [M * K_BLOCKS]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_scales, [N * K_BLOCKS]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_lut, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_scale, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate TMEM slices: d_taddr (offset 0, 128 cols), lut_taddr (offset 128,
    //    16 cols, 2-column aligned), and scale_taddr (offset 144, 32 cols).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, 128u);
        eg_tcgen05_alloc_cta1(tmem, &slot_lut[0], 128u, 16u);
        eg_tcgen05_alloc_cta1(tmem, &slot_scale[0], 144u, 32u);
    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];
    const uint32_t lut_taddr = slot_lut[0];
    const uint32_t scale_taddr = slot_scale[0];

    // 2. Stage E4M3 activations, 3-bit B indices, 8-entry E4M3 LUTs, and UE8M0 scales.
    for (int i = tid; i < M * K; i += 128) smem_a[i] = a_e4m3[i];
    for (int i = tid; i < K * N; i += 128) smem_b_idx[i] = b_indices[i];
    for (int i = tid; i < LUT_BYTES; i += 128) smem_b_lut[i] = b_lut_e4m3[i];
    for (int i = tid; i < M * K_BLOCKS; i += 128) smem_a_scales[i] = a_scales[i];
    for (int i = tid; i < N * K_BLOCKS; i += 128) smem_b_scales[i] = b_scales[i];
    __syncthreads();

    // 3. Stage LUT metadata into [lut_taddr] and UE8M0 scales into [scale_taddr] via
    //    tcgen05.cp.cta_group::1.32x128b.warpx4, then issue decompress::lut::b.block32 MMA.
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_cp_lut_b_32x128b(tmem, lut_taddr, smem_b_lut, N);
        eg_tcgen05_cp_rubin_scales_32x128b(tmem, scale_taddr, smem_a_scales, smem_b_scales, M, N, K_BLOCKS);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
        const uint64_t desc_a = eg_wgmma_desc(smem_a, 32, 256);
        const uint64_t desc_b = eg_wgmma_desc(smem_b_idx, 32, 256);
        const uint32_t sfa_taddr = scale_taddr;
        const uint32_t sfb_taddr = scale_taddr + 16u;
        const uint32_t idesc = 0x%08xu;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32%s "
            "[%%0], %%1, %%2, [%%3], %%4, [%%5], [%%6], p;\n"
            "}"
            :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(lut_taddr), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr)
            : "memory");
#else
        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = tmem[r * 512 + d_taddr + n];
                for (int kb = 0; kb < K_BLOCKS; ++kb) {
                    const float sa = tmem[r * 512 + scale_taddr + kb];
                    const float sb = tmem[n * 512 + scale_taddr + 16 + kb];
                    float block_dot = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const int kk = kb * 32 + j;
                        const float av = eg_e4m3_to_f32(smem_a[r * K + kk]);
                        const uint8_t idx = smem_b_idx[kk * N + n] & 0x7u;
                        const float bv = tmem[n * 512 + lut_taddr + idx];
                        block_dot += av * bv;
                    }
                    acc += block_dot * (sa * sb);
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
#endif
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 4. 4-warp 32-row partitioned tcgen05.ld epilogue.
    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Deallocate all 3 TMEM slices.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, 128u);
        eg_tcgen05_dealloc_cta1(lut_taddr, 16u);
        eg_tcgen05_dealloc_cta1(scale_taddr, 32u);
    }
}
`, k.Name, k.M, k.N, k.K, idesc, collectorQuals)

	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05LUTDecompressGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes the host reference output C [M, N] for k given E4M3
// activations aE4M3 [M, K], UE8M0 activation scales aScales [M, K/32],
// 3-bit weight indices bIndices [K, N], 8-entry E4M3 LUTs bLUTE4M3 [(N/8)*8],
// and UE8M0 weight scales bScales [N, K/32].
func (k TCGen05LUTDecompressGEMM) Reference(aE4M3, aScales, bIndices, bLUTE4M3, bScales []uint8) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	kBlocks := k.K / MXBlock
	lutBytes := (k.N / 8) * 8
	if len(aE4M3) != k.M*k.K {
		return nil, fmt.Errorf("tcgen05 lut decompress gemm %s reference: len(aE4M3)=%d, want %d", k.Name, len(aE4M3), k.M*k.K)
	}
	if len(aScales) != k.M*kBlocks {
		return nil, fmt.Errorf("tcgen05 lut decompress gemm %s reference: len(aScales)=%d, want %d", k.Name, len(aScales), k.M*kBlocks)
	}
	if len(bIndices) != k.K*k.N {
		return nil, fmt.Errorf("tcgen05 lut decompress gemm %s reference: len(bIndices)=%d, want %d", k.Name, len(bIndices), k.K*k.N)
	}
	if len(bLUTE4M3) != lutBytes {
		return nil, fmt.Errorf("tcgen05 lut decompress gemm %s reference: len(bLUTE4M3)=%d, want %d", k.Name, len(bLUTE4M3), lutBytes)
	}
	if len(bScales) != k.N*kBlocks {
		return nil, fmt.Errorf("tcgen05 lut decompress gemm %s reference: len(bScales)=%d, want %d", k.Name, len(bScales), k.N*kBlocks)
	}

	out := make([]float32, k.M*k.N)
	for r := 0; r < k.M; r++ {
		for n := 0; n < k.N; n++ {
			var acc float32
			g := n / 8
			for kb := 0; kb < kBlocks; kb++ {
				sa := FromUE8M0Bits(aScales[r*kBlocks+kb])
				sb := FromUE8M0Bits(bScales[n*kBlocks+kb])
				var blockDot float32
				for j := 0; j < MXBlock; j++ {
					kk := kb*MXBlock + j
					av := FromE4M3Bits(aE4M3[r*k.K+kk])
					idx := bIndices[kk*k.N+n] & 0x7
					bv := FromE4M3Bits(bLUTE4M3[g*8+int(idx)])
					blockDot += av * bv
				}
				acc += blockDot * (sa * sb)
			}
			out[r*k.N+n] = acc
		}
	}
	return out, nil
}

// TCGen05EarlyReleasePipelineGEMM configures a Rubin sm_107a/sm_107f multi-stage
// pipelined GEMM kernel that uses:
//   - tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64
//     (PTX ISA 9.4 §9.7.18.10.12) to signal mbar_read_a as soon as operand A is
//     read from shared memory so the next K-tile stage of A can be overwritten
//     before the MMA finishes, and
//   - tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64
//     with a 32-bit ctaMask for up to 32-CTA cluster completion.
type TCGen05EarlyReleasePipelineGEMM struct {
	Name        string
	M, N, K     int    // M=128, N=128, K multiple of 64 (default 128)
	Stages      int    // shared-memory pipeline stages (default 2)
	ClusterCTAs int    // CTAs in cluster in 1..32 (default 4)
	Target      string // "sm_107a" or "sm_107f" (defaults to "sm_107f")
}

func (k TCGen05EarlyReleasePipelineGEMM) norm() (TCGen05EarlyReleasePipelineGEMM, error) {
	if k.Name == "" {
		k.Name = "tcgen05_early_release_pipeline_gemm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 early release pipeline gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 128
	}
	if k.Stages == 0 {
		k.Stages = 2
	}
	if k.ClusterCTAs == 0 {
		k.ClusterCTAs = 4
	}
	if k.Target == "" {
		k.Target = "sm_107f"
	}
	if k.Target != "sm_107a" && k.Target != "sm_107f" {
		return k, fmt.Errorf("tcgen05 early release pipeline gemm %s: unsupported target %q (want sm_107a or sm_107f)", k.Name, k.Target)
	}
	if k.M != 128 || k.N != 128 || k.K < 64 || k.K%64 != 0 {
		return k, fmt.Errorf("tcgen05 early release pipeline gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N=128, K=64*m)", k.Name, k.M, k.N, k.K)
	}
	if k.Stages < 1 || k.Stages > 8 {
		return k, fmt.Errorf("tcgen05 early release pipeline gemm %s: Stages=%d must be in 1..8", k.Name, k.Stages)
	}
	if k.ClusterCTAs < 1 || k.ClusterCTAs > 32 {
		return k, fmt.Errorf("tcgen05 early release pipeline gemm %s: ClusterCTAs=%d must be in 1..32", k.Name, k.ClusterCTAs)
	}
	return k, nil
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05EarlyReleasePipelineGEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

func (k TCGen05EarlyReleasePipelineGEMM) ctaMask() uint32 {
	if k.ClusterCTAs >= 32 {
		return 0xffffffff
	}
	return (uint32(1) << uint(k.ClusterCTAs)) - 1
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05EarlyReleasePipelineGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ctaMask := k.ctaMask()

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(`
// eg_tcgen05_commit_sync_restrict_read_a signals mbar_read_a as soon as the MMA
// engine has consumed operand A from shared memory (PTX ISA 9.4 §9.7.18.10.12),
// allowing the producer warp to overwrite the shared-memory A buffer for the
// next K-tile stage before the MMA finishes computing.
EG_DEV void eg_tcgen05_commit_sync_restrict_read_a(uint64_t* mbar_read_a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar_read_a);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [%0];"
                 :: "r"(s) : "memory");
#else
    if (mbar_read_a != nullptr) {
        *mbar_read_a += 1ULL;
    }
#endif
}

// eg_tcgen05_commit_multicast_32b signals final MMA completion across up to 32
// CTAs in a Rubin cluster using a 32-bit CTA mask (.multicast::cluster::32b).
EG_DEV void eg_tcgen05_commit_multicast_32b(uint64_t* mbar_done, uint32_t cta_mask) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar_done);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [%0], %1;"
                 :: "r"(s), "r"(cta_mask) : "memory");
#else
    if (mbar_done != nullptr) {
        *mbar_done = (uint64_t)cta_mask;
    }
#endif
}
`)

	fmt.Fprintf(&b, "// %s: Rubin %s early-release A pipeline GEMM (M=%d, N=%d, K=%d, stages=%d, cluster_ctas=%d).\n",
		k.Name, k.Target, k.M, k.N, k.K, k.Stages, k.ClusterCTAs)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d, BK = 64;
    constexpr int KT = K / BK, STAGES = %d;
    constexpr uint32_t CTA_MASK = 0x%08xu;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(float, smem_a, [STAGES * M * BK]);
    EG_TCGEN05_SHARED(float, smem_b, [STAGES * BK * N]);
    EG_TCGEN05_SHARED(uint64_t, mbar_read_a, [STAGES]);
    EG_TCGEN05_SHARED(uint64_t, mbar_mma_done, [1]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);

    if (tid < STAGES) mbar_read_a[tid] = 0ULL;
    if (tid == 0) mbar_mma_done[0] = 0ULL;

    // 1. Allocate 128 accumulator columns at TMEM offset 0.
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, 128u);
    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];

    // 2. Multi-stage K-tile pipeline with early shared-memory A release via
    //    .sync_restrict::shared::read::mma::a.
    for (int kt = 0; kt < KT; ++kt) {
        const int stage = kt %% STAGES;
        float* stage_a = smem_a + stage * M * BK;
        float* stage_b = smem_b + stage * BK * N;

        for (int i = tid; i < M * BK; i += 128) {
            const int r = i / BK, kk = i %% BK;
            stage_a[i] = a[r * K + kt * BK + kk];
        }
        for (int i = tid; i < BK * N; i += 128) {
            const int kk = i / N, n = i %% N;
            stage_b[i] = b[(kt * BK + kk) * N + n];
        }
        __syncthreads();

        if (warp == 0 && eg_elect_one_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
            const uint64_t desc_a = eg_wgmma_desc(stage_a, 128, 1024);
            const uint64_t desc_b = eg_wgmma_desc(stage_b, 128, 1024);
            const uint32_t idesc = 0u;
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, 1, 0;\n"
                "tcgen05.mma.cta_group::1.kind::tf32 [%%0], %%1, %%2, %%3, p;\n"
                "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
#else
            for (int r = 0; r < M; ++r) {
                for (int n = 0; n < N; ++n) {
                    float acc = tmem[r * 512 + d_taddr + n];
                    for (int kk = 0; kk < BK; ++kk) {
                        acc += stage_a[r * BK + kk] * stage_b[kk * N + n];
                    }
                    tmem[r * 512 + d_taddr + n] = acc;
                }
            }
#endif
            eg_tcgen05_commit_sync_restrict_read_a(&mbar_read_a[stage]);
        }
        __syncthreads();
    }

    // 3. Final 32-CTA cluster multicast completion commit.
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_commit_multicast_32b(&mbar_mma_done[0], CTA_MASK);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 4. 4-warp 32-row partitioned tcgen05.ld epilogue.
    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Deallocate TMEM accumulator columns.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, 128u);
    }
}
`, k.Name, k.M, k.N, k.K, k.Stages, ctaMask)

	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05EarlyReleasePipelineGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes the host reference output C [M, N] for k given row-major
// float32 matrices a [M, K] and b [K, N].
func (k TCGen05EarlyReleasePipelineGEMM) Reference(a, b []float32) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	if len(a) != k.M*k.K {
		return nil, fmt.Errorf("tcgen05 early release pipeline gemm %s reference: len(a)=%d, want %d", k.Name, len(a), k.M*k.K)
	}
	if len(b) != k.K*k.N {
		return nil, fmt.Errorf("tcgen05 early release pipeline gemm %s reference: len(b)=%d, want %d", k.Name, len(b), k.K*k.N)
	}
	const bk = 64
	ktTotal := k.K / bk
	out := make([]float32, k.M*k.N)
	for kt := 0; kt < ktTotal; kt++ {
		for r := 0; r < k.M; r++ {
			for n := 0; n < k.N; n++ {
				acc := out[r*k.N+n]
				for kk := 0; kk < bk; kk++ {
					acc += a[r*k.K+kt*bk+kk] * b[(kt*bk+kk)*k.N+n]
				}
				out[r*k.N+n] = acc
			}
		}
	}
	return out, nil
}
