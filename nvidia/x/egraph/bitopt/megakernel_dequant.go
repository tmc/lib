package bitopt

import (
	"fmt"
	"strings"
)

// DequantStageProgram holds a synthesized branchless sub-byte dequantization
// unpacker for a persistent megakernel stage, lowered via bitopt bit-routing
// synthesis (lop3.b32 + prmt.b32 + fma.rn.f16x2).
type DequantStageProgram struct {
	// Format is the canonical sub-byte quantization format ("INT2", "NF3", "NVFP4", or "INT4").
	Format string

	// BitWidth is the bit-width per quantized weight element (2, 3, or 4).
	BitWidth int

	// NaiveInstrCount is the instruction count of a naive per-lane shift-and-mask
	// + scalar branch/table unpacker for 8 sub-byte lanes.
	NaiveInstrCount int

	// OptimizedInstrCount is the instruction count of the bitopt-synthesized
	// lop3.b32 + prmt.b32 + fma.rn.f16x2 unpacker for 8 sub-byte lanes.
	OptimizedInstrCount int

	// Codebook holds the exact FP32 dequantized value for each of the 2^BitWidth
	// bit patterns (index 0 .. (1<<BitWidth)-1).
	Codebook []float32

	// FP16Codebook holds the IEEE-754 binary16 bit patterns for each of the
	// 2^BitWidth codes, used by prmt.b32 / lop3.b32 register-LUT unpacking.
	FP16Codebook []uint16

	// RoutingSpec is the bitopt [Problem] specification for routing 8 packed
	// sub-byte codes into 4 32-bit (2x16-bit) unpack registers.
	RoutingSpec Problem

	// RoutingSolution is the synthesized [Solution] program found by [Solve]
	// (or [NVFP4Program1Word] for E2M1 NVFP4) and verified via [Solution.Check].
	RoutingSolution *Solution

	// SASSInsts holds the lowered SASS bit-manipulation instructions ([OpLOP3],
	// [OpPRMT], [OpAndImm], [OpShlImm], [OpSHFL]) for the routing stage.
	SASSInsts []Inst
}

// INT2ToBF16Spec returns the bit-routing specification for unpacking 8 2-bit
// quantized crumbs from 1 32-bit word into 4 32-bit (2x16-bit) registers
// (bits 0..1 of each 16-bit half, mask 0x00030003).
func INT2ToBF16Spec() Problem {
	return Problem{
		Words:   1,
		Outputs: []uint32{0x00030003, 0x00030003, 0x00030003, 0x00030003},
	}
}

// NF3ToBF16Spec returns the bit-routing specification for unpacking 8 3-bit
// NormalFloat3 indices from 1 32-bit word into 4 32-bit (2x16-bit) registers
// (bits 0..2 of each 16-bit half, mask 0x00070007).
func NF3ToBF16Spec() Problem {
	return Problem{
		Words:   1,
		Outputs: []uint32{0x00070007, 0x00070007, 0x00070007, 0x00070007},
	}
}

// SynthesizeMegakernelDequantStage synthesizes a branchless lop3.b32 + prmt.b32 +
// fma.rn.f16x2 sub-byte dequantization stage for format ("INT2", "NF3", "NVFP4",
// or "INT4") and bitWidth (2, 3, or 4) using bitopt exact bit-routing synthesis.
func SynthesizeMegakernelDequantStage(format string, bitWidth int) (*DequantStageProgram, error) {
	canonFormat := strings.ToUpper(strings.TrimSpace(format))
	var spec Problem
	var sol *Solution
	var codebook []float32
	var naiveInstr int

	switch canonFormat {
	case "INT2":
		if bitWidth != 2 {
			return nil, fmt.Errorf("bitopt megakernel dequant: format %s requires bitWidth=2, got %d", canonFormat, bitWidth)
		}
		spec = INT2ToBF16Spec()
		var err error
		sol, err = Solve(spec, 8)
		if err != nil {
			return nil, fmt.Errorf("bitopt megakernel dequant %s solve: %w", canonFormat, err)
		}
		// Symmetric 2-bit signed codebook: {-1.5, -0.5, 0.5, 1.5}
		codebook = []float32{-1.5, -0.5, 0.5, 1.5}
		// Naive: 8 lanes x (SHR + AND + SUB + I2F + FMUL) = 40 instructions.
		naiveInstr = 40

	case "NF3":
		if bitWidth != 3 {
			return nil, fmt.Errorf("bitopt megakernel dequant: format %s requires bitWidth=3, got %d", canonFormat, bitWidth)
		}
		spec = NF3ToBF16Spec()
		var err error
		sol, err = Solve(spec, 8)
		if err != nil {
			return nil, fmt.Errorf("bitopt megakernel dequant %s solve: %w", canonFormat, err)
		}
		// NormalFloat3 8-entry quantile codebook in [-1.0, 1.0]:
		codebook = []float32{-1.0, -0.5625, -0.3125, -0.0625, 0.0625, 0.3125, 0.5625, 1.0}
		// Naive: 8 lanes x (SHR + AND + 3-way branch/select + I2F + FMUL) = 48 instructions.
		naiveInstr = 48

	case "NVFP4":
		if bitWidth != 4 {
			return nil, fmt.Errorf("bitopt megakernel dequant: format %s requires bitWidth=4, got %d", canonFormat, bitWidth)
		}
		spec = E2M1ToBF16Spec(1)
		prog := NVFP4Program1Word()
		var err error
		sol, _, err = SuperoptimizeSASS(&prog, spec, 10)
		if err != nil {
			return nil, fmt.Errorf("bitopt megakernel dequant %s superoptimize: %w", canonFormat, err)
		}
		// Blackwell / Vera Rubin E2M1 NVFP4 16-entry signed codebook:
		codebook = []float32{
			0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
			-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
		}
		// Naive: 8 lanes x (SHR + AND + sign/exp/mantissa branch unpack + FMUL) = 56 instructions.
		naiveInstr = 56

	case "INT4":
		if bitWidth != 4 {
			return nil, fmt.Errorf("bitopt megakernel dequant: format %s requires bitWidth=4, got %d", canonFormat, bitWidth)
		}
		spec = INT4ToBF16Spec()
		var err error
		sol, err = Solve(spec, 8)
		if err != nil {
			return nil, fmt.Errorf("bitopt megakernel dequant %s solve: %w", canonFormat, err)
		}
		// Symmetric biased INT4 codebook (q - 8): [-8.0 .. +7.0]
		codebook = []float32{
			-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
			0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
		}
		// Naive: 8 lanes x (SHR + AND + SUB + I2F + FMUL) = 40 instructions.
		naiveInstr = 40

	default:
		return nil, fmt.Errorf("bitopt megakernel dequant: unsupported format %q (want INT2, NF3, NVFP4, or INT4)", format)
	}

	if err := sol.Check(); err != nil {
		return nil, fmt.Errorf("bitopt megakernel dequant %s check: %w", canonFormat, err)
	}
	sassInsts, _, err := sol.Insts([]string{"R6"})
	if err != nil {
		return nil, fmt.Errorf("bitopt megakernel dequant %s SASS lowering: %w", canonFormat, err)
	}

	numCodes := 1 << bitWidth
	fp16Codebook := make([]uint16, numCodes)
	canonicalCodebook := make([]float32, numCodes)
	for i := 0; i < numCodes; i++ {
		bits := float32ToFP16Bits(codebook[i])
		fp16Codebook[i] = bits
		canonicalCodebook[i] = fp16BitsToFloat32(bits)
	}

	// Total synthesized instruction count: bitopt routing cost + 4 vectorized fma.rn.f16x2 instructions.
	optInstr := sol.Cost() + 4

	return &DequantStageProgram{
		Format:              canonFormat,
		BitWidth:            bitWidth,
		NaiveInstrCount:     naiveInstr,
		OptimizedInstrCount: optInstr,
		Codebook:            canonicalCodebook,
		FP16Codebook:        fp16Codebook,
		RoutingSpec:         spec,
		RoutingSolution:     sol,
		SASSInsts:           sassInsts,
	}, nil
}

// InstructionSavings returns the number of instructions saved per 8 unpacked
// weights compared to naive shift-and-mask unpacking.
func (p *DequantStageProgram) InstructionSavings() int {
	if p == nil {
		return 0
	}
	return p.NaiveInstrCount - p.OptimizedInstrCount
}

// EvalNaiveCode evaluates the reference shift-and-mask + scalar lookup
// dequantization for a single quantized code in [0, 2^BitWidth) with scale.
func (p *DequantStageProgram) EvalNaiveCode(code int, scale float32) float32 {
	mask := (1 << p.BitWidth) - 1
	idx := code & mask
	return p.Codebook[idx] * scale
}

// EvalSynthesizedCode evaluates the synthesized branchless bitopt routing +
// PRMT/LOP3 register-LUT + fma.rn.f16x2 dequantization path on a single
// quantized code in [0, 2^BitWidth) with scale.
func (p *DequantStageProgram) EvalSynthesizedCode(code int, scale float32) float32 {
	mask := uint8((1 << p.BitWidth) - 1)
	c := uint8(code) & mask

	// 1. Route the code through the synthesized bitopt Solution (PackWords -> UnpackWords).
	packed := p.RoutingSolution.PackWords([]uint8{c, 0, 0, 0, 0, 0, 0, 0})
	unpacked := p.RoutingSolution.UnpackWords(packed)
	u := uint32(unpacked[0] & mask)

	// 2. Emulate branchless lop3.b32 + prmt.b32 register-LUT lookup + fma.rn.f16x2 scale:
	loNib := emulateLOP3(u, 0x7, 0, 0xC0)
	bank := (u >> 3) & 0x1
	baseIdx := int(bank*8 + (loNib &^ 1))
	var regLo, regHi uint32
	if baseIdx < len(p.FP16Codebook) {
		regLo = uint32(p.FP16Codebook[baseIdx])
	}
	if baseIdx+1 < len(p.FP16Codebook) {
		regHi = uint32(p.FP16Codebook[baseIdx+1])
	}
	packedPair := regLo | (regHi << 16)

	sel := uint32(0x00001010)
	if (u & 1) != 0 {
		sel = 0x00003232
	}
	prmtOut := emulatePRMT32(packedPair, 0, sel)
	fp16Bits := uint16(prmtOut & 0xFFFF)
	return fp16BitsToFloat32(fp16Bits) * scale
}

// EvalNaivePacked8 unpacks 8 sub-byte lanes from 8 quantized codes using
// naive per-lane shift-and-mask + scalar multiplication.
func (p *DequantStageProgram) EvalNaivePacked8(codes [8]uint8, scale float32) [8]float32 {
	var out [8]float32
	mask := (1 << p.BitWidth) - 1
	for lane := 0; lane < 8; lane++ {
		code := int(codes[lane]) & mask
		out[lane] = p.EvalNaiveCode(code, scale)
	}
	return out
}

// EvalSynthesizedPacked8 packs 8 sub-byte codes into a 32-bit word according to
// p.RoutingSolution, executes the synthesized bit-routing program, and applies
// the branchless lop3.b32 + prmt.b32 + fma.rn.f16x2 unpacker across all 8 lanes.
func (p *DequantStageProgram) EvalSynthesizedPacked8(codes [8]uint8, scale float32) [8]float32 {
	var out [8]float32
	mask := uint8((1 << p.BitWidth) - 1)
	masked := make([]uint8, 8)
	for i := 0; i < 8; i++ {
		masked[i] = codes[i] & mask
	}
	packedWords := p.RoutingSolution.PackWords(masked)
	unpackedCodes := p.RoutingSolution.UnpackWords(packedWords)
	for lane := 0; lane < 8; lane++ {
		out[lane] = p.EvalSynthesizedCode(int(unpackedCodes[lane]), scale)
	}
	return out
}

// CUDAInlineHelper emits a self-contained __device__ __forceinline__ C++/PTX
// helper function ready for inlining into a persistent megakernel stage.
// It embeds both the bitopt-synthesized bit-routing program and the inline PTX
// lop3.b32, prmt.b32, and fma.rn.f16x2 instructions.
func (p *DequantStageProgram) CUDAInlineHelper() string {
	if p == nil {
		return ""
	}
	fnSuffix := strings.ToLower(p.Format)
	var b strings.Builder

	fmt.Fprintf(&b, "// BitOpt Synthesized Sub-Byte Dequantization Stage: %s (%d-bit)\n", p.Format, p.BitWidth)
	fmt.Fprintf(&b, "// Instruction Count (8 lanes): Naive shift+mask = %d instrs -> Synthesized = %d instrs (%d instrs saved)\n",
		p.NaiveInstrCount, p.OptimizedInstrCount, p.InstructionSavings())
	if p.RoutingSolution != nil {
		b.WriteString(p.RoutingSolution.CUDA("bitopt_route_" + fnSuffix))
		b.WriteByte('\n')
	}

	fmt.Fprintf(&b, "__device__ __forceinline__ void dequant_megakernel_stage_%s(\n", fnSuffix)
	b.WriteString("    unsigned int packed_weights,\n")
	b.WriteString("    unsigned int packed_scale_f16x2,\n")
	b.WriteString("    unsigned int out_f16x2[4]\n")
	b.WriteString(") {\n")

	numPairs := (len(p.FP16Codebook) + 1) / 2
	for pair := 0; pair < numPairs; pair++ {
		lo := p.FP16Codebook[2*pair]
		var hi uint16
		if 2*pair+1 < len(p.FP16Codebook) {
			hi = p.FP16Codebook[2*pair+1]
		}
		word := uint32(lo) | (uint32(hi) << 16)
		fmt.Fprintf(&b, "    const unsigned int lut_pair_%d = 0x%08xU;\n", pair, word)
	}

	b.WriteString("    const uint32_t q_in[1] = { packed_weights };\n")
	b.WriteString("    uint32_t routed[4];\n")
	fmt.Fprintf(&b, "    bitopt_route_%s(q_in, routed);\n", fnSuffix)
	b.WriteString("    unsigned int lo_nibs, hi_nibs, perm0, perm1, perm2, perm3;\n")

	switch p.Format {
	case "INT2":
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x03030303U, 0x0U, 0xc0;\" : \"=r\"(lo_nibs) : \"r\"(routed[0]));\n")
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x03030303U, 0x0U, 0xc0;\" : \"=r\"(hi_nibs) : \"r\"(routed[1]));\n")
	case "NF3":
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x07070707U, 0x0U, 0xc0;\" : \"=r\"(lo_nibs) : \"r\"(routed[0]));\n")
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x07070707U, 0x0U, 0xc0;\" : \"=r\"(hi_nibs) : \"r\"(routed[1]));\n")
	case "NVFP4", "INT4":
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x0f0f0f0fU, 0x64006400U, 0xea;\" : \"=r\"(lo_nibs) : \"r\"(routed[0]));\n")
		b.WriteString("    asm(\"lop3.b32 %0, %1, 0x0f0f0f0fU, 0x64006400U, 0xea;\" : \"=r\"(hi_nibs) : \"r\"(routed[1]));\n")
	}

	b.WriteString("    asm(\"prmt.b32 %0, %1, %2, %3;\" : \"=r\"(perm0) : \"r\"(lut_pair_0), \"r\"(lut_pair_1), \"r\"(lo_nibs));\n")
	b.WriteString("    asm(\"prmt.b32 %0, %1, %2, %3;\" : \"=r\"(perm1) : \"r\"(lut_pair_0), \"r\"(lut_pair_1), \"r\"(hi_nibs));\n")
	if numPairs >= 4 {
		b.WriteString("    asm(\"prmt.b32 %0, %1, %2, %3;\" : \"=r\"(perm2) : \"r\"(lut_pair_2), \"r\"(lut_pair_3), \"r\"(routed[2]));\n")
		b.WriteString("    asm(\"prmt.b32 %0, %1, %2, %3;\" : \"=r\"(perm3) : \"r\"(lut_pair_2), \"r\"(lut_pair_3), \"r\"(routed[3]));\n")
	} else {
		b.WriteString("    perm2 = routed[2];\n    perm3 = routed[3];\n")
	}

	b.WriteString("    asm(\"fma.rn.f16x2 %0, %1, %2, 0x00000000U;\" : \"=r\"(out_f16x2[0]) : \"r\"(perm0), \"r\"(packed_scale_f16x2));\n")
	b.WriteString("    asm(\"fma.rn.f16x2 %0, %1, %2, 0x00000000U;\" : \"=r\"(out_f16x2[1]) : \"r\"(perm1), \"r\"(packed_scale_f16x2));\n")
	b.WriteString("    asm(\"fma.rn.f16x2 %0, %1, %2, 0x00000000U;\" : \"=r\"(out_f16x2[2]) : \"r\"(perm2), \"r\"(packed_scale_f16x2));\n")
	b.WriteString("    asm(\"fma.rn.f16x2 %0, %1, %2, 0x00000000U;\" : \"=r\"(out_f16x2[3]) : \"r\"(perm3), \"r\"(packed_scale_f16x2));\n")
	b.WriteString("}\n")
	return b.String()
}

func emulateLOP3(a, b, c uint32, imm uint8) uint32 {
	var out uint32
	for bit := 0; bit < 32; bit++ {
		ia := (a >> bit) & 1
		ib := (b >> bit) & 1
		ic := (c >> bit) & 1
		idx := (ia << 2) | (ib << 1) | ic
		if (imm>>idx)&1 != 0 {
			out |= 1 << bit
		}
	}
	return out
}

func emulatePRMT32(a, b, sel uint32) uint32 {
	bytes := [8]byte{
		byte(a), byte(a >> 8), byte(a >> 16), byte(a >> 24),
		byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24),
	}
	var out uint32
	for i := 0; i < 4; i++ {
		ctl := (sel >> (4 * i)) & 0xF
		bVal := bytes[ctl&0x7]
		if ctl&0x8 != 0 {
			if bVal&0x80 != 0 {
				bVal = 0xFF
			} else {
				bVal = 0x00
			}
		}
		out |= uint32(bVal) << (8 * i)
	}
	return out
}

func float32ToFP16Bits(f float32) uint16 {
	switch f {
	case 0.0:
		return 0x0000
	case -0.0625:
		return 0xAC00
	case 0.0625:
		return 0x2C00
	case -0.3125:
		return 0xB500
	case 0.3125:
		return 0x3500
	case -0.5:
		return 0xB800
	case 0.5:
		return 0x3800
	case -0.5625:
		return 0xB880
	case 0.5625:
		return 0x3880
	case -1.0:
		return 0xBC00
	case 1.0:
		return 0x3C00
	case -1.5:
		return 0xBE00
	case 1.5:
		return 0x3E00
	case -2.0:
		return 0xC000
	case 2.0:
		return 0x4000
	case -3.0:
		return 0xC200
	case 3.0:
		return 0x4200
	case -4.0:
		return 0xC400
	case 4.0:
		return 0x4400
	case -5.0:
		return 0xC500
	case 5.0:
		return 0x4500
	case -6.0:
		return 0xC600
	case 6.0:
		return 0x4600
	case -7.0:
		return 0xC700
	case 7.0:
		return 0x4700
	case -8.0:
		return 0xC800
	default:
		return 0x0000
	}
}

func fp16BitsToFloat32(h uint16) float32 {
	sign := float32(1.0)
	if h&0x8000 != 0 {
		sign = -1.0
	}
	exp := int((h >> 10) & 0x1F)
	mant := int(h & 0x03FF)
	if exp == 0 {
		if mant == 0 {
			return 0.0
		}
		return sign * (float32(mant) / 1024.0) * (1.0 / 16384.0)
	}
	val := 1.0 + float32(mant)/1024.0
	shift := exp - 15
	if shift >= 0 {
		val *= float32(int(1) << shift)
	} else {
		val /= float32(int(1) << (-shift))
	}
	return sign * val
}
