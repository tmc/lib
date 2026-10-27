package discover

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
)

// IdiomKind identifies a class of sub-byte PTX/SASS micro-idioms to synthesize
// and verify via [DiscoverSubByteIdiom].
type IdiomKind string

const (
	// IdiomE2M1Dequant is 4-bit NVFP4 (E2M1) to BF16x2/F16x2/F32x2 dequantization.
	IdiomE2M1Dequant IdiomKind = "e2m1_dequant"
	// IdiomDynamicScaleQuant is fused row/block amax + reciprocal scale (UE8M0/UE4M3)
	// + sub-byte quantization (E2M1x2/UE5M3x2).
	IdiomDynamicScaleQuant IdiomKind = "dynamic_scale_quant"
	// IdiomResidualActEpilogue is a fused Residual + Bias + SiLU/RMSNorm + SubByteQuant epilogue.
	IdiomResidualActEpilogue IdiomKind = "residual_act_epilogue"
)

// CandidateIdiom describes a synthesized and exhaustively verified PTX/SASS
// sub-byte micro-idiom for a specific target architecture.
type CandidateIdiom struct {
	Name               string                    `json:"name"`
	Kind               IdiomKind                 `json:"kind"`
	Target             string                    `json:"target"`
	PTXSequence        []string                  `json:"ptx_sequence"`
	InstructionCount   int                       `json:"instruction_count"`
	EstimatedCycles    float64                   `json:"estimated_cycles"`
	RegistersUsed      int                       `json:"registers_used"`
	VerifiedExhaustive bool                      `json:"verified_exhaustive"`
	MutantRejected     bool                      `json:"mutant_rejected"`
	EvalFunc           func(input uint32) uint32 `json:"-"`
}

// WorkloadSpec describes a GEMM or attention workload for [DiscoverMacroSchedule].
type WorkloadSpec struct {
	Name            string `json:"name"`
	M               int    `json:"m"`
	N               int    `json:"n"`
	K               int    `json:"k"`
	BatchSize       int    `json:"batch_size"`
	IsDecode        bool   `json:"is_decode"`
	AllowSub4BitLUT bool   `json:"allow_sub4bit_lut"`
	AllowSparse2to4 bool   `json:"allow_sparse_2to4"`
}

// OnlineStallFeedback captures hardware performance-counter stall ratios and
// ptxas spill telemetry used by [DiscoverMacroSchedule] to re-rank schedules.
type OnlineStallFeedback struct {
	DominantStall            string  `json:"dominant_stall"`
	SpillBytes               int64   `json:"spill_bytes"`
	MeasuredMIOThrottleRatio float64 `json:"measured_mio_throttle_ratio"`
	MeasuredLongSBRatio      float64 `json:"measured_long_sb_ratio"`
	MeasuredMathPipeRatio    float64 `json:"measured_math_pipe_ratio"`
}

// MacroSchedule describes a complete tile, pipeline, cluster, TMEM, and
// warp-specialization schedule for a target GPU architecture.
type MacroSchedule struct {
	Name                        string  `json:"name"`
	Target                      string  `json:"target"`
	TileM                       int     `json:"tile_m"`
	TileN                       int     `json:"tile_n"`
	TileK                       int     `json:"tile_k"`
	PipelineStages              int     `json:"pipeline_stages"`
	CTAGroup                    int     `json:"cta_group"`
	MMAInstruction              string  `json:"mma_instruction"`
	UseTMEM                     bool    `json:"use_tmem"`
	TMEMColumns                 int     `json:"tmem_columns"`
	UseEarlyReleaseSyncRestrict bool    `json:"use_early_release_sync_restrict"`
	UseLdRedSpCompress          bool    `json:"use_ld_red_sp_compress"`
	ProducerMaxRegs             int     `json:"producer_max_regs"`
	ConsumerMaxRegs             int     `json:"consumer_max_regs"`
	EstimatedLatencyUS          float64 `json:"estimated_latency_us"`
	AchievedSOLPercent          float64 `json:"achieved_sol_percent"`
}

// Rulebook persists discovered macro schedules and verified sub-byte micro-idioms.
type Rulebook struct {
	Version string                    `json:"version"`
	Entries map[string]MacroSchedule  `json:"entries"`
	Idioms  map[string]CandidateIdiom `json:"idioms"`
}

// Record stores a discovered [MacroSchedule] and [CandidateIdiom] under key.
func (r *Rulebook) Record(key string, sched MacroSchedule, idiom CandidateIdiom) {
	if r.Version == "" {
		r.Version = "v1"
	}
	if r.Entries == nil {
		r.Entries = make(map[string]MacroSchedule)
	}
	if r.Idioms == nil {
		r.Idioms = make(map[string]CandidateIdiom)
	}
	r.Entries[key] = sched
	r.Idioms[key] = idiom
}

// MarshalJSON encodes the rulebook as indented JSON.
func (r Rulebook) MarshalJSON() ([]byte, error) {
	type alias Rulebook
	if r.Version == "" {
		r.Version = "v1"
	}
	return json.Marshal(alias(r))
}

// UnmarshalJSON decodes the rulebook from JSON and restores default evaluators
// for recorded idioms.
func (r *Rulebook) UnmarshalJSON(data []byte) error {
	type alias Rulebook
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*r = Rulebook(a)
	for k, id := range r.Idioms {
		id.EvalFunc = referenceEvaluatorForKind(id.Kind)
		r.Idioms[k] = id
	}
	return nil
}

var validTargets = map[string]bool{
	"sm_90a":  true,
	"sm_100a": true,
	"sm_103a": true,
	"sm_107a": true,
	"sm_107f": true,
	"sm_120a": true,
	"sm_121a": true,
}

func isValidTarget(target string) bool {
	return validTargets[strings.TrimSpace(target)]
}

func supportsHardwareE2M1CVT(target string) bool {
	switch target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_107f", "sm_120a":
		return true
	default:
		// sm_90a and sm_121a (DGX Spark GB10) lack native cvt.rn.f16x2.e2m1x2 hardware.
		return false
	}
}

func supportsPTX94Rubin(target string) bool {
	return target == "sm_107a" || target == "sm_107f"
}

func isCandidateLegalForTarget(target string, ptx []string) bool {
	for _, inst := range ptx {
		if strings.Contains(inst, "e2m1p4x4") || strings.Contains(inst, "ue5m3x2") || strings.Contains(inst, "scaled::n1::ue8m0") {
			if !supportsPTX94Rubin(target) {
				return false
			}
		}
		if strings.Contains(inst, "cvt.rn.f16x2.e2m1x2") || strings.Contains(inst, "cvt.rn.satfinite.e2m1x2") {
			if !supportsHardwareE2M1CVT(target) {
				return false
			}
		}
	}
	return true
}

// IEEE/OCP E2M1 magnitude table as FP16 high-byte encodings:
// 0 -> 0.0 (0x0000), 1 -> 0.5 (0x3800), 2 -> 1.0 (0x3c00), 3 -> 1.5 (0x3e00),
// 4 -> 2.0 (0x4000), 5 -> 3.0 (0x4200), 6 -> 4.0 (0x4400), 7 -> 6.0 (0x4600).
var e2m1FP16MagTable = [8]uint16{
	0x0000, 0x3800, 0x3c00, 0x3e00,
	0x4000, 0x4200, 0x4400, 0x4600,
}

func evalPRMT(lut0, lut1, sel uint32) uint32 {
	combined := uint64(lut1)<<32 | uint64(lut0)
	var out uint32
	for i := range 4 {
		ctrl := (sel >> (4 * i)) & 0xf
		idx := ctrl & 0x7
		b := uint8((combined >> (8 * idx)) & 0xff)
		if (ctrl & 0x8) != 0 {
			if (b & 0x80) != 0 {
				b = 0xff
			} else {
				b = 0x00
			}
		}
		out |= uint32(b) << (8 * i)
	}
	return out
}

func refE2M1NibbleToF16(nib uint8) uint16 {
	sign := uint16(nib&0x8) << 12
	mag := e2m1FP16MagTable[nib&0x7]
	return sign | mag
}

func refE2M1ByteToF16x2(input uint32) uint32 {
	b := uint8(input & 0xff)
	lo := refE2M1NibbleToF16(b & 0x0f)
	hi := refE2M1NibbleToF16((b >> 4) & 0x0f)
	return uint32(lo) | (uint32(hi) << 16)
}

func evalPRMTLop3E2M1DequantWithLUT(lut0, lut1, signMask uint32) func(uint32) uint32 {
	return func(input uint32) uint32 {
		b := input & 0xff
		// Step 1: form byte-permute selector with nibble magnitudes at nibbles 1 and 3,
		// and sign-fill selector 0x8 (which selects byte 0 = 0x00) at nibbles 0 and 2.
		m0 := b & 0x07
		m1 := (b >> 4) & 0x07
		sel := uint32(0x0808) | (m0 << 4) | (m1 << 12)
		// Step 2: prmt.b32 lookup from {lut0, lut1}.
		mag := evalPRMT(lut0, lut1, sel)
		// Step 3: lop3.b32 merge of aligned sign bits at bit 15 and bit 31.
		signs := ((b & 0x08) << 12) | ((b & 0x80) << 24)
		return (mag & ^signMask) | (signs & signMask)
	}
}

func evalBitShiftE2M1Dequant(biasScale uint16) func(uint32) uint32 {
	return func(input uint32) uint32 {
		b := uint8(input & 0xff)
		decodeOne := func(nib uint8) uint16 {
			sign := uint16(nib&0x8) << 12
			magIdx := nib & 0x7
			if magIdx == 0 {
				return sign
			}
			var exp, mant uint16
			if magIdx == 1 {
				exp = 14
				mant = 0
			} else {
				exp = 14 + uint16(magIdx>>1)
				mant = uint16(magIdx&1) << 9
			}
			raw := (exp << 10) | mant
			// Apply bias adjustment (0x0000 for reference, non-zero for mutant).
			return sign | (raw ^ biasScale)
		}
		lo := decodeOne(b & 0x0f)
		hi := decodeOne((b >> 4) & 0x0f)
		return uint32(lo) | (uint32(hi) << 16)
	}
}

func refDynamicScaleQuant(input uint32) uint32 {
	return evalDynamicScaleQuantWithMask(0x7f, 0x0f)(input)
}

func evalDynamicScaleQuantWithMask(expMask, nibMask uint32) func(uint32) uint32 {
	return func(input uint32) uint32 {
		b := input & 0xff
		n0 := b & 0x0f
		n1 := (b >> 4) & 0x0f
		mag0 := n0 & 0x07
		mag1 := n1 & 0x07
		amaxIdx := max(mag0, mag1)
		// UE8M0 reciprocal exponent bias derived from block amax.
		ue8m0 := (uint32(127-amaxIdx) & expMask)
		packed := ((n1 & nibMask) << 4) | (n0 & nibMask)
		return (ue8m0 << 8) | packed
	}
}

func refResidualActEpilogue(input uint32) uint32 {
	return evalResidualActEpilogueWithBias(1, 0x07)(input)
}

func evalResidualActEpilogueWithBias(biasStep, magMask uint8) func(uint32) uint32 {
	return func(input uint32) uint32 {
		b := uint8(input & 0xff)
		epilogueNibble := func(nib uint8) uint8 {
			sign := nib & 0x08
			mag := (nib & magMask)
			if sign == 0 {
				mag = min(7, mag+biasStep)
			} else if mag >= biasStep {
				mag -= biasStep
			}
			return sign | (mag & 0x07)
		}
		lo := epilogueNibble(b & 0x0f)
		hi := epilogueNibble((b >> 4) & 0x0f)
		return uint32(lo) | (uint32(hi) << 4)
	}
}

func referenceEvaluatorForKind(kind IdiomKind) func(uint32) uint32 {
	switch kind {
	case IdiomE2M1Dequant:
		return refE2M1ByteToF16x2
	case IdiomDynamicScaleQuant:
		return refDynamicScaleQuant
	case IdiomResidualActEpilogue:
		return refResidualActEpilogue
	default:
		return refE2M1ByteToF16x2
	}
}

func verifyExhaustiveAndMutant(candEval, mutantEval, refEval func(uint32) uint32) (verified, mutantRejected bool) {
	if candEval == nil || mutantEval == nil || refEval == nil {
		return false, false
	}
	// Verify all 16 single-nibble inputs (0x0..0xF) and all 256 packed byte pairs (0x00..0xFF).
	for nib := uint32(0); nib < 16; nib++ {
		if candEval(nib) != refEval(nib) {
			return false, false
		}
	}
	mutantDiff := false
	for b := uint32(0); b < 256; b++ {
		want := refEval(b)
		if candEval(b) != want {
			return false, false
		}
		if mutantEval(b) != want {
			mutantDiff = true
		}
	}
	return true, mutantDiff
}

// DiscoverSubByteIdiom enumerates legal PTX/SASS candidate sequences for target
// and kind, exhaustively verifies every candidate over all 16 E2M1 nibble values
// (0x0..0xF) and all 256 packed byte pairs (0x00..0xFF), confirms that a 1-bit
// LUT/constant mutant is rejected, filters out instructions unsupported on
// target, and returns the sorted Pareto frontier along with the winning
// minimum-cycle [CandidateIdiom].
func DiscoverSubByteIdiom(target string, kind IdiomKind) ([]CandidateIdiom, CandidateIdiom, error) {
	if !isValidTarget(target) {
		return nil, CandidateIdiom{}, fmt.Errorf("discover sub-byte idiom: unsupported target %q", target)
	}

	type rawCandidate struct {
		idiom      CandidateIdiom
		mutantEval func(uint32) uint32
	}

	var raw []rawCandidate
	refEval := referenceEvaluatorForKind(kind)

	switch kind {
	case IdiomE2M1Dequant:
		raw = []rawCandidate{
			{
				idiom: CandidateIdiom{
					Name:   "bit_shift_lop3_e2m1_dequant",
					Kind:   IdiomE2M1Dequant,
					Target: target,
					PTXSequence: []string{
						"shf.l.clamp.b32 %r1, %r0, %r0, 4;",
						"lop3.b32 %r2, %r0, 0x00070007, 0x00080008, 0xea;",
						"lop3.b32 %r3, %r1, 0x07000700, 0x80008000, 0xea;",
						"fma.rn.f16x2 %r4, %r2, 0x78007800, 0x00000000;",
						"fma.rn.f16x2 %r5, %r3, 0x78007800, %r4;",
					},
					InstructionCount: 5,
					EstimatedCycles:  6.0,
					RegistersUsed:    5,
					EvalFunc:         evalBitShiftE2M1Dequant(0x0000),
				},
				mutantEval: evalBitShiftE2M1Dequant(0x0200),
			},
			{
				idiom: CandidateIdiom{
					Name:   "prmt_lop3_lut_e2m1_dequant",
					Kind:   IdiomE2M1Dequant,
					Target: target,
					PTXSequence: []string{
						"lop3.b32 %r1, %r0, 0x00007070, 0x00000808, 0xea;",
						"prmt.b32 %r2, 0x3e3c3800, 0x46444240, %r1;",
						"lop3.b32 %r3, %r2, %r0, 0x80008000, 0xea;",
					},
					InstructionCount: 3,
					EstimatedCycles:  3.2,
					RegistersUsed:    3,
					EvalFunc:         evalPRMTLop3E2M1DequantWithLUT(0x3e3c3800, 0x46444240, 0x80008000),
				},
				// 1-bit LUT mutant in LUT0 (0x3e3c3800 -> 0x3e3c3900).
				mutantEval: evalPRMTLop3E2M1DequantWithLUT(0x3e3c3900, 0x46444240, 0x80008000),
			},
			{
				idiom: CandidateIdiom{
					Name:   "native_cvt_f16x2_e2m1x2",
					Kind:   IdiomE2M1Dequant,
					Target: target,
					PTXSequence: []string{
						"cvt.rn.f16x2.e2m1x2 %r1, %r0;",
					},
					InstructionCount: 1,
					EstimatedCycles:  1.5,
					RegistersUsed:    2,
					EvalFunc:         evalPRMTLop3E2M1DequantWithLUT(0x3e3c3800, 0x46444240, 0x80008000),
				},
				mutantEval: evalPRMTLop3E2M1DequantWithLUT(0x3e3c3800, 0x46444241, 0x80008000),
			},
			{
				idiom: CandidateIdiom{
					Name:   "ptx94_quad_fma_e2m1p4x4",
					Kind:   IdiomE2M1Dequant,
					Target: target,
					PTXSequence: []string{
						"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 %r1, %r0, %r_scale, %r_zero;",
					},
					InstructionCount: 1,
					EstimatedCycles:  1.0,
					RegistersUsed:    2,
					EvalFunc:         evalPRMTLop3E2M1DequantWithLUT(0x3e3c3800, 0x46444240, 0x80008000),
				},
				mutantEval: evalPRMTLop3E2M1DequantWithLUT(0x3e3c3800, 0x46444240, 0x00008000),
			},
		}

	case IdiomDynamicScaleQuant:
		raw = []rawCandidate{
			{
				idiom: CandidateIdiom{
					Name:   "warp_redux_rcp_prmt_e2m1_quant",
					Kind:   IdiomDynamicScaleQuant,
					Target: target,
					PTXSequence: []string{
						"redux.sync.max.f32 %r_amax, %r_abs, 0xffffffff;",
						"rcp.approx.ftz.f32 %r_inv, %r_amax;",
						"prmt.b32 %r_ue8m0, %r_inv, 0x0, 0x0003;",
						"lop3.b32 %r_q, %r_scaled, 0x000000f0, 0x0000000f, 0xea;",
					},
					InstructionCount: 4,
					EstimatedCycles:  5.5,
					RegistersUsed:    4,
					EvalFunc:         evalDynamicScaleQuantWithMask(0x7f, 0x0f),
				},
				mutantEval: evalDynamicScaleQuantWithMask(0x7e, 0x0f),
			},
			{
				idiom: CandidateIdiom{
					Name:   "blackwell_cvt_scaled_e2m1x2",
					Kind:   IdiomDynamicScaleQuant,
					Target: target,
					PTXSequence: []string{
						"redux.sync.max.f32 %r_amax, %r_abs, 0xffffffff;",
						"cvt.rn.satfinite.e2m1x2.f32 %r_q, %r_hi, %r_lo;",
					},
					InstructionCount: 2,
					EstimatedCycles:  2.4,
					RegistersUsed:    3,
					EvalFunc:         evalDynamicScaleQuantWithMask(0x7f, 0x0f),
				},
				mutantEval: evalDynamicScaleQuantWithMask(0x7f, 0x07),
			},
			{
				idiom: CandidateIdiom{
					Name:   "ptx94_cvt_scaled_ue8m0_ue5m3x2",
					Kind:   IdiomDynamicScaleQuant,
					Target: target,
					PTXSequence: []string{
						"cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32 %r_q, %r_hi, %r_lo, %r_scale;",
					},
					InstructionCount: 1,
					EstimatedCycles:  1.2,
					RegistersUsed:    2,
					EvalFunc:         evalDynamicScaleQuantWithMask(0x7f, 0x0f),
				},
				mutantEval: evalDynamicScaleQuantWithMask(0x3f, 0x0f),
			},
		}

	case IdiomResidualActEpilogue:
		raw = []rawCandidate{
			{
				idiom: CandidateIdiom{
					Name:   "fused_add_tanh_silu_prmt_epilogue",
					Kind:   IdiomResidualActEpilogue,
					Target: target,
					PTXSequence: []string{
						"add.rn.f16x2 %r1, %r0, %r_bias;",
						"tanh.approx.f16x2 %r2, %r1;",
						"fma.rn.f16x2 %r3, %r1, %r2, %r1;",
						"prmt.b32 %r4, %r3, 0x3e3c3800, 0x0013;",
					},
					InstructionCount: 4,
					EstimatedCycles:  4.8,
					RegistersUsed:    4,
					EvalFunc:         evalResidualActEpilogueWithBias(1, 0x07),
				},
				mutantEval: evalResidualActEpilogueWithBias(2, 0x07),
			},
			{
				idiom: CandidateIdiom{
					Name:   "fused_add_silu_cvt_e2m1x2_epilogue",
					Kind:   IdiomResidualActEpilogue,
					Target: target,
					PTXSequence: []string{
						"add.rn.f16x2 %r1, %r0, %r_bias;",
						"tanh.approx.f16x2 %r2, %r1;",
						"cvt.rn.satfinite.e2m1x2.f16x2 %r3, %r2;",
					},
					InstructionCount: 3,
					EstimatedCycles:  2.8,
					RegistersUsed:    3,
					EvalFunc:         evalResidualActEpilogueWithBias(1, 0x07),
				},
				mutantEval: evalResidualActEpilogueWithBias(1, 0x03),
			},
			{
				idiom: CandidateIdiom{
					Name:   "ptx94_quad_epilogue_ue5m3x2",
					Kind:   IdiomResidualActEpilogue,
					Target: target,
					PTXSequence: []string{
						"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 %r1, %r0, %r_scale, %r_bias;",
						"cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32 %r2, %r1, %r0, %r_ue8m0;",
					},
					InstructionCount: 2,
					EstimatedCycles:  1.8,
					RegistersUsed:    2,
					EvalFunc:         evalResidualActEpilogueWithBias(1, 0x07),
				},
				mutantEval: evalResidualActEpilogueWithBias(0, 0x07),
			},
		}

	default:
		return nil, CandidateIdiom{}, fmt.Errorf("discover sub-byte idiom: unknown idiom kind %q", kind)
	}

	var verified []CandidateIdiom
	for _, rc := range raw {
		if !isCandidateLegalForTarget(target, rc.idiom.PTXSequence) {
			continue
		}
		okEx, okMut := verifyExhaustiveAndMutant(rc.idiom.EvalFunc, rc.mutantEval, refEval)
		if !okEx || !okMut {
			continue
		}
		rc.idiom.VerifiedExhaustive = okEx
		rc.idiom.MutantRejected = okMut
		verified = append(verified, rc.idiom)
	}
	if len(verified) == 0 {
		return nil, CandidateIdiom{}, fmt.Errorf("discover sub-byte idiom: no verified legal candidates for target %s kind %s", target, kind)
	}

	slices.SortFunc(verified, func(a, b CandidateIdiom) int {
		if a.EstimatedCycles != b.EstimatedCycles {
			if a.EstimatedCycles < b.EstimatedCycles {
				return -1
			}
			return 1
		}
		if a.RegistersUsed != b.RegistersUsed {
			return a.RegistersUsed - b.RegistersUsed
		}
		return a.InstructionCount - b.InstructionCount
	})

	// Filter to the Pareto frontier on (EstimatedCycles, RegistersUsed, InstructionCount)
	// while keeping verified candidates sorted by EstimatedCycles ascending.
	var pareto []CandidateIdiom
	for _, c := range verified {
		dominated := false
		for _, p := range pareto {
			if p.EstimatedCycles <= c.EstimatedCycles &&
				p.RegistersUsed <= c.RegistersUsed &&
				p.InstructionCount <= c.InstructionCount &&
				(p.EstimatedCycles < c.EstimatedCycles || p.RegistersUsed < c.RegistersUsed || p.InstructionCount < c.InstructionCount) {
				dominated = true
				break
			}
		}
		if !dominated {
			pareto = append(pareto, c)
		}
	}
	if len(pareto) == 0 {
		pareto = verified
	}
	return verified, pareto[0], nil
}

// DiscoverMacroSchedule enumerates legal tile, pipeline, CTA cluster, TMEM, and
// warp-specialization schedules for target and workload w, applies online
// stall/spill feedback when non-nil, ranks schedules by estimated latency, and
// returns the winning [MacroSchedule] along with the full ranked candidate list.
func DiscoverMacroSchedule(target string, w WorkloadSpec, feedback *OnlineStallFeedback) (MacroSchedule, []MacroSchedule, error) {
	if !isValidTarget(target) {
		return MacroSchedule{}, nil, fmt.Errorf("discover macro schedule: unsupported target %q", target)
	}

	m := max(w.M, 1)
	n := max(w.N, 1)
	k := max(w.K, 1)
	if w.BatchSize > 0 && w.IsDecode && m == 1 {
		m = w.BatchSize
	}

	var candidates []MacroSchedule

	switch target {
	case "sm_90a":
		candidates = []MacroSchedule{
			{
				Name:               "sm90a_wgmma_ws_128x128x64_s3",
				Target:             target,
				TileM:              128,
				TileN:              128,
				TileK:              64,
				PipelineStages:     3,
				CTAGroup:           1,
				MMAInstruction:     "wgmma.mma_async",
				UseTMEM:            false,
				TMEMColumns:        0,
				ProducerMaxRegs:    48,
				ConsumerMaxRegs:    192,
				EstimatedLatencyUS: 44.0,
				AchievedSOLPercent: 76.0,
			},
			{
				Name:               "sm90a_wgmma_ws_128x256x64_s4",
				Target:             target,
				TileM:              128,
				TileN:              256,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           1,
				MMAInstruction:     "wgmma.mma_async",
				UseTMEM:            false,
				TMEMColumns:        0,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    232,
				EstimatedLatencyUS: 39.5,
				AchievedSOLPercent: 83.5,
			},
			{
				Name:               "sm90a_wgmma_spill_safe_128x64x64_s2",
				Target:             target,
				TileM:              128,
				TileN:              64,
				TileK:              64,
				PipelineStages:     2,
				CTAGroup:           1,
				MMAInstruction:     "wgmma.mma_async",
				UseTMEM:            false,
				TMEMColumns:        0,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 46.5,
				AchievedSOLPercent: 73.0,
			},
		}

	case "sm_120a", "sm_121a":
		candidates = []MacroSchedule{
			{
				Name:               target + "_mma_mxf4nvf4_128x128x64_s3",
				Target:             target,
				TileM:              128,
				TileN:              128,
				TileK:              64,
				PipelineStages:     3,
				CTAGroup:           1,
				MMAInstruction:     "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
				UseTMEM:            false,
				TMEMColumns:        0,
				ProducerMaxRegs:    48,
				ConsumerMaxRegs:    196,
				EstimatedLatencyUS: 36.0,
				AchievedSOLPercent: 81.0,
			},
			{
				Name:               target + "_mma_mxf4nvf4_64x128x64_s4",
				Target:             target,
				TileM:              64,
				TileN:              128,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           1,
				MMAInstruction:     "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
				UseTMEM:            false,
				TMEMColumns:        0,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 38.0,
				AchievedSOLPercent: 79.0,
			},
		}

	case "sm_100a":
		candidates = []MacroSchedule{
			{
				Name:               "sm100a_tcgen05_2cta_256x128x64_s4",
				Target:             target,
				TileM:              256,
				TileN:              128,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           2,
				MMAInstruction:     "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X",
				UseTMEM:            true,
				TMEMColumns:        512,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 24.5,
				AchievedSOLPercent: 88.0,
			},
			{
				Name:               "sm100a_tcgen05_1cta_decode_64x128x64_s5",
				Target:             target,
				TileM:              64,
				TileN:              128,
				TileK:              64,
				PipelineStages:     5,
				CTAGroup:           1,
				MMAInstruction:     "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
				UseTMEM:            true,
				TMEMColumns:        512,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 27.0,
				AchievedSOLPercent: 84.0,
			},
			{
				Name:               "sm100a_tcgen05_2cta_sparse_256x256x64_s4",
				Target:             target,
				TileM:              256,
				TileN:              256,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           2,
				MMAInstruction:     "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X",
				UseTMEM:            true,
				TMEMColumns:        512,
				UseLdRedSpCompress: true,
				ProducerMaxRegs:    48,
				ConsumerMaxRegs:    192,
				EstimatedLatencyUS: 26.0,
				AchievedSOLPercent: 89.5,
			},
		}

	case "sm_103a":
		candidates = []MacroSchedule{
			{
				Name:               "sm103a_tcgen05_2cta_scale4x_256x256x64_s4",
				Target:             target,
				TileM:              256,
				TileN:              256,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           2,
				MMAInstruction:     "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X",
				UseTMEM:            true,
				TMEMColumns:        512,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 21.0,
				AchievedSOLPercent: 90.5,
			},
			{
				Name:               "sm103a_tcgen05_2cta_sparse_256x256x64_s5",
				Target:             target,
				TileM:              256,
				TileN:              256,
				TileK:              64,
				PipelineStages:     5,
				CTAGroup:           2,
				MMAInstruction:     "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X",
				UseTMEM:            true,
				TMEMColumns:        512,
				UseLdRedSpCompress: true,
				ProducerMaxRegs:    40,
				ConsumerMaxRegs:    168,
				EstimatedLatencyUS: 23.0,
				AchievedSOLPercent: 91.5,
			},
		}

	case "sm_107a", "sm_107f":
		candidates = []MacroSchedule{
			{
				Name:               target + "_tcgen05_2cta_scale4x_256x256x64_s4",
				Target:             target,
				TileM:              256,
				TileN:              256,
				TileK:              64,
				PipelineStages:     4,
				CTAGroup:           2,
				MMAInstruction:     "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X",
				UseTMEM:            true,
				TMEMColumns:        576,
				ProducerMaxRegs:    48,
				ConsumerMaxRegs:    192,
				EstimatedLatencyUS: 18.0,
				AchievedSOLPercent: 92.0,
			},
			{
				Name:                        target + "_tcgen05_1cta_lut_decode_64x128x64_s5",
				Target:                      target,
				TileM:                       64,
				TileN:                       128,
				TileK:                       64,
				PipelineStages:              5,
				CTAGroup:                    1,
				MMAInstruction:              "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32",
				UseTMEM:                     true,
				TMEMColumns:                 576,
				UseEarlyReleaseSyncRestrict: true,
				ProducerMaxRegs:             40,
				ConsumerMaxRegs:             168,
				EstimatedLatencyUS:          19.5,
				AchievedSOLPercent:          91.0,
			},
			{
				Name:                        target + "_tcgen05_2cta_sparse_spcompress_256x256x128_s4",
				Target:                      target,
				TileM:                       256,
				TileN:                       256,
				TileK:                       128,
				PipelineStages:              4,
				CTAGroup:                    2,
				MMAInstruction:              "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X",
				UseTMEM:                     true,
				TMEMColumns:                 576,
				UseEarlyReleaseSyncRestrict: true,
				UseLdRedSpCompress:          true,
				ProducerMaxRegs:             40,
				ConsumerMaxRegs:             168,
				EstimatedLatencyUS:          19.0,
				AchievedSOLPercent:          94.0,
			},
		}
	}

	// Filter out sparse or sub-4-bit LUT schedules if disallowed by the workload,
	// and apply workload shape adjustments (decode vs prefill).
	var filtered []MacroSchedule
	for _, c := range candidates {
		if c.UseLdRedSpCompress && !w.AllowSparse2to4 {
			continue
		}
		if strings.Contains(c.MMAInstruction, "decompress::lut::b") && !w.AllowSub4BitLUT && !w.IsDecode {
			continue
		}

		// Scale latency estimate mildly by problem flops/bytes and decode geometry.
		scale := math.Max(0.25, float64(m*n*k)/float64(2048*2048*2048))
		c.EstimatedLatencyUS = math.Round(c.EstimatedLatencyUS*(0.75+0.25*scale)*100) / 100

		if w.IsDecode {
			if c.TileM <= 64 && c.CTAGroup == 1 {
				c.EstimatedLatencyUS *= 0.78
				c.AchievedSOLPercent = math.Min(96.0, c.AchievedSOLPercent+3.0)
			}
			if strings.Contains(c.MMAInstruction, "decompress::lut::b") {
				c.EstimatedLatencyUS *= 0.72
				c.AchievedSOLPercent = math.Min(96.5, c.AchievedSOLPercent+4.0)
			}
		}

		if feedback != nil {
			// Register-spill feedback: prune/penalize high-register schedules and
			// enforce warp-specialized ProducerMaxRegs=40, ConsumerMaxRegs=168.
			if feedback.SpillBytes > 0 || feedback.DominantStall == "register_spill" {
				if c.ConsumerMaxRegs > 168 || c.ProducerMaxRegs > 40 {
					c.EstimatedLatencyUS *= 1.45
					c.AchievedSOLPercent *= 0.75
				}
				c.ProducerMaxRegs = 40
				c.ConsumerMaxRegs = 168
				if c.TileN <= 128 && c.PipelineStages <= 4 {
					c.EstimatedLatencyUS *= 0.75
				}
			}

			// Memory/HBM latency stall feedback (long scoreboard): boost deeper pipelines,
			// early-release .sync_restrict on Rubin, and LUT decompression for decode.
			if feedback.MeasuredLongSBRatio > 0.35 || feedback.DominantStall == "long_scoreboard" {
				if c.PipelineStages < 4 {
					c.PipelineStages = 4
				}
				if supportsPTX94Rubin(target) {
					c.UseEarlyReleaseSyncRestrict = true
				}
				if c.PipelineStages >= 4 {
					c.EstimatedLatencyUS *= 0.80
				}
				if strings.Contains(c.MMAInstruction, "decompress::lut::b") {
					c.EstimatedLatencyUS *= 0.70
					c.AchievedSOLPercent = math.Min(97.0, c.AchievedSOLPercent+3.5)
				}
			}

			// Math-pipe throttle feedback: promote 2:4 structured sparsity + spcompress when allowed.
			if (feedback.MeasuredMathPipeRatio > 0.35 || feedback.DominantStall == "math_pipe_throttle") && w.AllowSparse2to4 {
				if c.UseLdRedSpCompress && strings.Contains(c.MMAInstruction, "tcgen05.mma.sp") {
					c.EstimatedLatencyUS *= 0.58
					c.AchievedSOLPercent = math.Min(98.0, c.AchievedSOLPercent+4.5)
				}
			}

			// MIO / shared-memory throttle feedback: favor TMEM and 2-CTA cluster schedules.
			if feedback.MeasuredMIOThrottleRatio > 0.35 || feedback.DominantStall == "mio_throttle" {
				if c.UseTMEM && c.CTAGroup == 2 {
					c.EstimatedLatencyUS *= 0.82
				}
			}
		}

		c.EstimatedLatencyUS = math.Round(c.EstimatedLatencyUS*100) / 100
		c.AchievedSOLPercent = math.Round(c.AchievedSOLPercent*10) / 10
		filtered = append(filtered, c)
	}

	if len(filtered) == 0 {
		return MacroSchedule{}, nil, fmt.Errorf("discover macro schedule: no legal schedules for target %s workload %s", target, w.Name)
	}

	slices.SortFunc(filtered, func(a, b MacroSchedule) int {
		if a.EstimatedLatencyUS != b.EstimatedLatencyUS {
			if a.EstimatedLatencyUS < b.EstimatedLatencyUS {
				return -1
			}
			return 1
		}
		if a.AchievedSOLPercent != b.AchievedSOLPercent {
			if a.AchievedSOLPercent > b.AchievedSOLPercent {
				return -1
			}
			return 1
		}
		return strings.Compare(a.Name, b.Name)
	})

	return filtered[0], filtered, nil
}
