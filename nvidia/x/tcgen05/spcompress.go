package tcgen05

import (
	"fmt"
	"math"
	"strings"
)

// SpCompressOp specifies the comparison operation used by PTX ISA 9.4
// §9.7.10.30 spcompress (bits 0..1 of the 32-bit spdesc descriptor, Table 36).
type SpCompressOp uint8

const (
	SpOpMax    SpCompressOp = 0 // MAX: select the 2 elements with largest signed value
	SpOpMaxAbs SpCompressOp = 1 // MAXABS: select the 2 elements with largest absolute value
	SpOpMin    SpCompressOp = 2 // MIN: select the 2 elements with smallest signed value
	SpOpMinAbs SpCompressOp = 3 // MINABS: select the 2 elements with smallest absolute value
)

// String returns the PTX ISA 9.4 Table 36 mnemonic for op.
func (op SpCompressOp) String() string {
	switch op {
	case SpOpMax:
		return "MAX"
	case SpOpMaxAbs:
		return "MAXABS"
	case SpOpMin:
		return "MIN"
	case SpOpMinAbs:
		return "MINABS"
	default:
		return fmt.Sprintf("SpCompressOp(%d)", uint8(op))
	}
}

func (op SpCompressOp) valid() bool {
	return op <= SpOpMinAbs
}

// SpElemType specifies the element data type encoded in bits 2..4 of the
// 32-bit spdesc descriptor for PTX ISA 9.4 §9.7.10.30 spcompress (Table 36).
type SpElemType uint8

const (
	SpTypeF16  SpElemType = 0 // .f16 (16-bit IEEE 754 binary16)
	SpTypeBF16 SpElemType = 1 // .bf16 (16-bit bfloat16)
	SpTypeE5M2 SpElemType = 2 // .e5m2 (8-bit OCP FP8 E5M2)
	SpTypeE4M3 SpElemType = 3 // .e4m3 (8-bit OCP FP8 E4M3)
	SpTypeE3M2 SpElemType = 4 // .e3m2 (8-bit-padded OCP FP6 E3M2)
	SpTypeE2M3 SpElemType = 5 // .e2m3 (8-bit-padded OCP FP6 E2M3)
	SpTypeU8   SpElemType = 6 // .u8 (8-bit unsigned integer)
	SpTypeS8   SpElemType = 7 // .s8 (8-bit signed integer)
)

// String returns the PTX type qualifier for t.
func (t SpElemType) String() string {
	switch t {
	case SpTypeF16:
		return ".f16"
	case SpTypeBF16:
		return ".bf16"
	case SpTypeE5M2:
		return ".e5m2"
	case SpTypeE4M3:
		return ".e4m3"
	case SpTypeE3M2:
		return ".e3m2"
	case SpTypeE2M3:
		return ".e2m3"
	case SpTypeU8:
		return ".u8"
	case SpTypeS8:
		return ".s8"
	default:
		return fmt.Sprintf("SpElemType(%d)", uint8(t))
	}
}

// BitWidth returns the storage bit width (16 for .f16/.bf16, 8 for 8-bit types)
// of t in PTX spcompress (.b16 vs .b8).
func (t SpElemType) BitWidth() int {
	switch t {
	case SpTypeF16, SpTypeBF16:
		return 16
	default:
		return 8
	}
}

func (t SpElemType) valid() bool {
	return t <= SpTypeS8
}

// SpCompressDesc models the 32-bit sparsity compression descriptor (spdesc)
// consumed by PTX ISA 9.4 §9.7.10.30.1 Table 36:
//   - bits 0..1: Op (MAX=0, MAXABS=1, MIN=2, MINABS=3)
//   - bits 2..4: ElemType (.f16=0, .bf16=1, .e5m2=2, .e4m3=3, .e3m2=4, .e2m3=5, .u8=6, .s8=7)
//   - bits 5..31: reserved (must be 0)
type SpCompressDesc struct {
	Op       SpCompressOp
	ElemType SpElemType
}

// Validate checks that d.Op and d.ElemType are within their PTX ISA 9.4 Table 36 ranges.
func (d SpCompressDesc) Validate() error {
	if !d.Op.valid() {
		return fmt.Errorf("tcgen05 spcompress desc: invalid Op %d (want 0..3)", uint8(d.Op))
	}
	if !d.ElemType.valid() {
		return fmt.Errorf("tcgen05 spcompress desc: invalid ElemType %d (want 0..7)", uint8(d.ElemType))
	}
	return nil
}

// Encode packs d into a 32-bit spdesc register value with bits 5..31 cleared to 0.
func (d SpCompressDesc) Encode() uint32 {
	return (uint32(d.Op) & 0x3) | ((uint32(d.ElemType) & 0x7) << 2)
}

// DecodeSpCompressDesc unpacks a 32-bit spdesc register value into an
// SpCompressDesc, returning an error if any reserved bit in bits 5..31 is non-zero.
func DecodeSpCompressDesc(v uint32) (SpCompressDesc, error) {
	if (v >> 5) != 0 {
		return SpCompressDesc{}, fmt.Errorf("tcgen05 spcompress desc: non-zero reserved bits in 0x%08x (bits 5..31 = 0x%07x)", v, v>>5)
	}
	return SpCompressDesc{
		Op:       SpCompressOp(v & 0x3),
		ElemType: SpElemType((v >> 2) & 0x7),
	}, nil
}

// Structured sparsity ratio qualifiers (.sp::<ratio>) for PTX ISA 9.4
// §9.7.10.30 spcompress and §9.7.10.31 spdecompress.
const (
	SpRatio1to2  = "1:2"
	SpRatio2to4  = "2:4"
	SpRatio4to8  = "4:8"
	SpRatio1to4  = "1:4"
	SpRatio2to8  = "2:8"
	SpRatio4to16 = "4:16"
)

func validSpRatio(r string) bool {
	switch r {
	case SpRatio1to2, SpRatio2to4, SpRatio4to8, SpRatio1to4, SpRatio2to8, SpRatio4to16:
		return true
	default:
		return false
	}
}

func spCompareKey(v float32, op SpCompressOp) float64 {
	x := float64(v)
	if op == SpOpMaxAbs || op == SpOpMinAbs {
		return math.Abs(x)
	}
	return x
}

func spBetterIndex(i, j int, vals [4]float32, op SpCompressOp) bool {
	ki := spCompareKey(vals[i], op)
	kj := spCompareKey(vals[j], op)
	switch op {
	case SpOpMax, SpOpMaxAbs:
		if ki != kj {
			return ki > kj
		}
	case SpOpMin, SpOpMinAbs:
		if ki != kj {
			return ki < kj
		}
	}
	return i < j
}

func spSelect2Of4(vals [4]float32, op SpCompressOp) (idx0, idx1 int) {
	order := [4]int{0, 1, 2, 3}
	for a := 0; a < 4; a++ {
		for b := a + 1; b < 4; b++ {
			if spBetterIndex(order[b], order[a], vals, op) {
				order[a], order[b] = order[b], order[a]
			}
		}
	}
	idx0, idx1 = order[0], order[1]
	if idx0 > idx1 {
		idx0, idx1 = idx1, idx0
	}
	return idx0, idx1
}

// SpCompress2to4F32 compresses a dense slice in (whose length must be a
// positive multiple of 32) using 2:4 structured sparsity according to op
// (MAX, MAXABS, MIN, or MINABS).
//
// For each contiguous 4-element group in[4*g .. 4*g+3], it selects the 2
// elements ordered by op (breaking ties deterministically by lower index
// 0 < 1 < 2 < 3), places their values into cdata[2*g .. 2*g+1] in ascending
// index order (idx0 < idx1), and packs the 4-bit metadata nibble
// (idx0 | (idx1 << 2)) into bits 4*(g%8)..4*(g%8)+3 of mdata[g/8].
func SpCompress2to4F32(in []float32, op SpCompressOp) (cdata []float32, mdata []uint32, err error) {
	if !op.valid() {
		return nil, nil, fmt.Errorf("tcgen05 spcompress 2:4 f32: invalid op %d (want 0..3)", uint8(op))
	}
	n := len(in)
	if n <= 0 || n%32 != 0 {
		return nil, nil, fmt.Errorf("tcgen05 spcompress 2:4 f32: len(in)=%d must be a positive multiple of 32", n)
	}
	numGroups := n / 4
	cdata = make([]float32, n/2)
	mdata = make([]uint32, n/32)
	for g := 0; g < numGroups; g++ {
		base := 4 * g
		vals := [4]float32{in[base+0], in[base+1], in[base+2], in[base+3]}
		idx0, idx1 := spSelect2Of4(vals, op)
		cdata[2*g+0] = vals[idx0]
		cdata[2*g+1] = vals[idx1]
		nib := uint32(idx0&0x3) | (uint32(idx1&0x3) << 2)
		mdata[g>>3] |= nib << (4 * (g & 7))
	}
	return cdata, mdata, nil
}

// SpDecompress2to4F32 reconstructs a dense slice of totalElems float32 values
// from 2:4 compressed data cdata (length totalElems/2) and packed 2-bit index
// metadata mdata (length totalElems/32), scattering each compressed pair back
// to (idx0, idx1) and zero-filling the remaining 2 slots per 4-element group.
func SpDecompress2to4F32(cdata []float32, mdata []uint32, totalElems int) ([]float32, error) {
	if totalElems <= 0 || totalElems%32 != 0 {
		return nil, fmt.Errorf("tcgen05 spdecompress 2:4 f32: totalElems=%d must be a positive multiple of 32", totalElems)
	}
	if len(cdata) != totalElems/2 {
		return nil, fmt.Errorf("tcgen05 spdecompress 2:4 f32: len(cdata)=%d, want %d", len(cdata), totalElems/2)
	}
	if len(mdata) != totalElems/32 {
		return nil, fmt.Errorf("tcgen05 spdecompress 2:4 f32: len(mdata)=%d, want %d", len(mdata), totalElems/32)
	}
	dense := make([]float32, totalElems)
	numGroups := totalElems / 4
	for g := 0; g < numGroups; g++ {
		nib := (mdata[g>>3] >> (4 * (g & 7))) & 0xf
		idx0 := int(nib & 0x3)
		idx1 := int((nib >> 2) & 0x3)
		if idx0 >= idx1 {
			return nil, fmt.Errorf("tcgen05 spdecompress 2:4 f32: group %d invalid metadata indices (%d, %d)", g, idx0, idx1)
		}
		dense[4*g+idx0] = cdata[2*g+0]
		dense[4*g+idx1] = cdata[2*g+1]
	}
	return dense, nil
}

// RubinSpCompressDecompressKernel configures a Rubin sm_107a/sm_107f
// register-to-register structured sparsity compression and decompression kernel
// demonstrating PTX ISA 9.4 §9.7.10.30 (spcompress) and §9.7.10.31 (spdecompress).
type RubinSpCompressDecompressKernel struct {
	NumElements int            // number of elements; defaults to 256 (must be a positive multiple of 32)
	Desc        SpCompressDesc // 32-bit spdesc configuration; defaults to {Op: SpOpMaxAbs, ElemType: SpTypeE4M3}
	Ratio       string         // sparsity ratio; defaults to "2:4"
	Target      string         // PTX target ("sm_107a" or "sm_107f"; defaults to "sm_107a")
}

func (k RubinSpCompressDecompressKernel) norm() (RubinSpCompressDecompressKernel, error) {
	if k.NumElements == 0 {
		k.NumElements = 256
	}
	if k.Desc == (SpCompressDesc{}) {
		k.Desc = SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpTypeE4M3}
	}
	if k.Ratio == "" {
		k.Ratio = SpRatio2to4
	}
	if k.Target == "" {
		k.Target = "sm_107a"
	}
	if k.NumElements < 32 || k.NumElements%32 != 0 {
		return k, fmt.Errorf("tcgen05 rubin spcompress kernel: NumElements=%d must be a positive multiple of 32", k.NumElements)
	}
	if err := k.Desc.Validate(); err != nil {
		return k, err
	}
	if !validSpRatio(k.Ratio) {
		return k, fmt.Errorf("tcgen05 rubin spcompress kernel: unsupported Ratio %q", k.Ratio)
	}
	if k.Target != "sm_107a" && k.Target != "sm_107f" {
		return k, fmt.Errorf("tcgen05 rubin spcompress kernel: unsupported Target %q (want sm_107a or sm_107f)", k.Target)
	}
	return k, nil
}

// Validate checks that k has valid element count, descriptor, sparsity ratio,
// and Rubin target ("sm_107a" or "sm_107f").
func (k RubinSpCompressDecompressKernel) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 64-thread block dimensions for k.
func (k RubinSpCompressDecompressKernel) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{64, 1, 1}
}

// CUDA emits the CUDA C++ kernel source (without Prelude) for
// eg_rubin_spcompress_roundtrip, containing inline PTX blocks for
// spcompress.b8.b2.sp::2:4.x4 and spdecompress.b8.b2.sp::2:4.x32 guarded by
// __CUDA_ARCH__ == 1070 plus exact C++ execution of 2:4 compression and
// decompression driven by spdesc.
func (k RubinSpCompressDecompressKernel) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	spdesc := k.Desc.Encode()

	var b strings.Builder
	fmt.Fprintf(&b, "// eg_rubin_spcompress_roundtrip: Rubin %s register-to-register spcompress & spdecompress (n=%d, ratio=%s, op=%s, elem=%s, spdesc=0x%08x).\n",
		k.Target, k.NumElements, k.Ratio, k.Desc.Op, k.Desc.ElemType, spdesc)
	b.WriteString(`EG_DEV bool eg_spcompress_better(int i, int j, const float vals[4], unsigned int op) {
    const float vi = vals[i];
    const float vj = vals[j];
    const float ki = (op == 1u || op == 3u) ? fabsf(vi) : vi;
    const float kj = (op == 1u || op == 3u) ? fabsf(vj) : vj;
    if (op == 0u || op == 1u) {
        if (ki != kj) return ki > kj;
    } else {
        if (ki != kj) return ki < kj;
    }
    return i < j;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070 && !defined(EG_HOST_EMU) && !defined(EG_EMU)
// eg_ptx_spcompress_b8_2to4_x4 issues PTX ISA 9.4 §9.7.10.30 register-to-register
// spcompress.b8.b2.sp::2:4.x4 on sm_107a.
EG_DEV void eg_ptx_spcompress_b8_2to4_x4(
    uint32_t& mdata_out, uint32_t& cdata0, uint32_t& cdata1,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t spdesc) {
    asm volatile(
        "spcompress.b8.b2.sp::2:4.x4 {%0}, {%1, %2}, {%3, %4, %5, %6}, %7;"
        : "=r"(mdata_out), "=r"(cdata0), "=r"(cdata1)
        : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(spdesc));
}

// eg_ptx_spdecompress_b8_2to4_x32 issues PTX ISA 9.4 §9.7.10.31 register-to-register
// spdecompress.b8.b2.sp::2:4.x32 on sm_107a.
EG_DEV void eg_ptx_spdecompress_b8_2to4_x32(
    uint32_t d[16], uint32_t mdata, const uint32_t c[8]) {
    asm volatile(
        "spdecompress.b8.b2.sp::2:4.x32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, "
        "{%17, %18, %19, %20, %21, %22, %23, %24};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]),
          "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
          "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]),
          "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
        : "r"(mdata),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]),
          "r"(c[4]), "r"(c[5]), "r"(c[6]), "r"(c[7]));
}
#endif

__global__ void eg_rubin_spcompress_roundtrip(
    const float* __restrict__ in_dense,
    float* __restrict__ out_cdata,
    unsigned int* __restrict__ out_mdata,
    float* __restrict__ out_reconstructed,
    int n,
    unsigned int spdesc) {
    const unsigned int op = spdesc & 0x3u;
    const int num_blocks32 = n / 32;
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(gridDim.x * blockDim.x);

    for (int blk = tid; blk < num_blocks32; blk += stride) {
        unsigned int mdata_word = 0u;
        float blk_cdata[16];

        // 1. Register-to-register 2:4 compression across 8 groups of 4 elements (32 elements).
        for (int g = 0; g < 8; ++g) {
            const int base = blk * 32 + g * 4;
            const float vals[4] = {
                in_dense[base + 0],
                in_dense[base + 1],
                in_dense[base + 2],
                in_dense[base + 3],
            };
            int order[4] = {0, 1, 2, 3};
            for (int a = 0; a < 4; ++a) {
                for (int b = a + 1; b < 4; ++b) {
                    if (eg_spcompress_better(order[b], order[a], vals, op)) {
                        const int tmp = order[a];
                        order[a] = order[b];
                        order[b] = tmp;
                    }
                }
            }
            int idx0 = order[0];
            int idx1 = order[1];
            if (idx0 > idx1) {
                const int tmp = idx0;
                idx0 = idx1;
                idx1 = tmp;
            }
            const float c0 = vals[idx0];
            const float c1 = vals[idx1];
            blk_cdata[2 * g + 0] = c0;
            blk_cdata[2 * g + 1] = c1;
            out_cdata[blk * 16 + 2 * g + 0] = c0;
            out_cdata[blk * 16 + 2 * g + 1] = c1;

            const unsigned int nib = ((unsigned int)(idx0 & 0x3)) | (((unsigned int)(idx1 & 0x3)) << 2);
            mdata_word |= (nib << (4 * g));
        }
        out_mdata[blk] = mdata_word;

        // 2. Register-to-register 2:4 decompression of 16 compressed elements + 32-bit mdata_word
        //    back to 32 dense elements with zero fill.
        for (int g = 0; g < 8; ++g) {
            const int base = blk * 32 + g * 4;
            out_reconstructed[base + 0] = 0.0f;
            out_reconstructed[base + 1] = 0.0f;
            out_reconstructed[base + 2] = 0.0f;
            out_reconstructed[base + 3] = 0.0f;
            const unsigned int nib = (mdata_word >> (4 * g)) & 0xfu;
            const int idx0 = (int)(nib & 0x3u);
            const int idx1 = (int)((nib >> 2) & 0x3u);
            out_reconstructed[base + idx0] = blk_cdata[2 * g + 0];
            out_reconstructed[base + idx1] = blk_cdata[2 * g + 1];
        }
    }
}
`)
	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k RubinSpCompressDecompressKernel) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference executes the host reference 2:4 compression and decompression
// round-trip on in (length k.NumElements, or defaulted to 256) using k.Desc.Op.
func (k RubinSpCompressDecompressKernel) Reference(in []float32) (cdata []float32, mdata []uint32, reconstructed []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(in) != k.NumElements {
		return nil, nil, nil, fmt.Errorf("tcgen05 rubin spcompress kernel: len(in)=%d, want %d", len(in), k.NumElements)
	}
	cdata, mdata, err = SpCompress2to4F32(in, k.Desc.Op)
	if err != nil {
		return nil, nil, nil, err
	}
	reconstructed, err = SpDecompress2to4F32(cdata, mdata, k.NumElements)
	if err != nil {
		return nil, nil, nil, err
	}
	return cdata, mdata, reconstructed, nil
}
