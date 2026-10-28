package cudasass

import (
	"fmt"
	"strconv"
	"strings"
)

// SM100Kind classifies a Blackwell sm_100a tensor-memory or cluster SASS instruction.
type SM100Kind string

const (
	SM100None      SM100Kind = ""
	SM100UTMMA     SM100Kind = "UTMMA"        // tcgen05.mma -> UTCHMMA (kind::f16), UTCOMMA (block-scaled kind::mxf4nvf4)
	SM100UTMALDG   SM100Kind = "UTMALDG"      // TMA global -> TMEM / shared load
	SM100UTMACP    SM100Kind = "UTMACP"       // tcgen05.cp shared -> TMEM copy (e.g. block scales) -> UTCCP
	SM100UTMAFlush SM100Kind = "UTMACMDFLUSH" // tcgen05.commit / fence -> UTCBAR
	SM100LDTM      SM100Kind = "LDTM"         // tcgen05.ld TMEM -> register load
	SM100STTM      SM100Kind = "STTM"         // tcgen05.st register -> TMEM store
	SM100ELECT     SM100Kind = "ELECT"        // elect.sync single-thread leader election
)

// SM100InstInfo holds decoded attributes of a Blackwell/Rubin sm_100a/sm_103a/sm_107a SASS instruction.
type SM100InstInfo struct {
	Kind        SM100Kind
	Mnemonic    string
	CTAGroup    int    // 1 or 2 (.CTA_GROUP_1 / .CTA_GROUP_2 / .2CTA)
	BlockScaled bool   // .MXF4NVF4, .MXF8F6F4, .BLOCK_SCALE
	ScaleVec    int    // 2 or 4 when .SCALE_VEC_2X / .SCALE_VEC_4X / .2X / .4X is present
	TMEMSourceA bool   // true if operand A is sourced from TMEM ([TMEM:...]) rather than shared memory
	DataType    string // e.g., "F16", "BF16", "E4M3", "MXF4NVF4"
	TMEMCols    int    // width/columns for LDTM/STTM (e.g., 32, 64, 128)
}

// SM100Summary holds instruction counts and attributes for Blackwell/Rubin sm_100a/sm_103a/sm_107a
// tensor-memory and cluster instructions in a disassembled function.
type SM100Summary struct {
	Total        int
	UTMMA        int
	UTMALDG      int
	UTMACP       int
	UTMACMDFLUSH int
	LDTM         int
	STTM         int
	ELECT        int
	CTAGroup1    int
	CTAGroup2    int
	BlockScaled  int
	ScaleVec2X   int
	ScaleVec4X   int
	TMEMSourceA  int
	ByKind       map[SM100Kind]int
}

// Count returns the number of instructions of the given SM100Kind.
func (s SM100Summary) Count(kind SM100Kind) int {
	switch kind {
	case SM100UTMMA:
		return s.UTMMA
	case SM100UTMALDG:
		return s.UTMALDG
	case SM100UTMACP:
		return s.UTMACP
	case SM100UTMAFlush:
		return s.UTMACMDFLUSH
	case SM100LDTM:
		return s.LDTM
	case SM100STTM:
		return s.STTM
	case SM100ELECT:
		return s.ELECT
	default:
		if s.ByKind != nil {
			return s.ByKind[kind]
		}
		return 0
	}
}

var sm100DataTypes = map[string]bool{
	"F16":      true,
	"BF16":     true,
	"TF32":     true,
	"F32":      true,
	"F64":      true,
	"E4M3":     true,
	"E5M2":     true,
	"FP8":      true,
	"FP4":      true,
	"MXF4NVF4": true,
	"MXF8F6F4": true,
	"NVF4":     true,
	"MXF4":     true,
	"MXF6":     true,
	"MXF8":     true,
	"S8":       true,
	"U8":       true,
	"I8":       true,
	"B8":       true,
}

var sm100BlockScaledTypes = map[string]bool{
	"MXF4NVF4":    true,
	"MXF8F6F4":    true,
	"NVF4":        true,
	"MXF4":        true,
	"MXF6":        true,
	"MXF8":        true,
	"BLOCK_SCALE": true,
	"BLOCKSCALE":  true,
}

// ClassifySM100Inst classifies a SASS instruction if it belongs to the
// Blackwell sm_100a tensor-memory (TMEM / tcgen05) or cluster execution family.
func ClassifySM100Inst(inst Instruction) (SM100InstInfo, bool) {
	op := strings.TrimSpace(inst.Opcode)
	if op == "" {
		return SM100InstInfo{}, false
	}
	upperOp := strings.ToUpper(op)
	parts := strings.Split(upperOp, ".")
	base := parts[0]

	var kind SM100Kind
	switch base {
	// ptxas 13.4 spells tcgen05.mma kind::f16 UTCHMMA, block-scaled
	// kind::mxf4nvf4 UTCOMMA.4X on sm_100a and UTCOMMA.BLOCK16 on sm_103a,
	// tcgen05.cp UTCCP and tcgen05.commit UTCBAR
	// (testdata/sm100a-umma-13.4.sass, testdata/sm103a-umma-13.4.sass).
	case "UTMMA", "UTCQMMA", "UTCHMMA", "UTCMMA", "UTCOMMA":
		kind = SM100UTMMA
	case "UTMALDG":
		kind = SM100UTMALDG
	case "UTMACP", "UTCCP":
		kind = SM100UTMACP
	case "UTMACMDFLUSH", "UTCBAR":
		kind = SM100UTMAFlush
	case "LDTM":
		kind = SM100LDTM
	case "STTM":
		kind = SM100STTM
	case "ELECT":
		kind = SM100ELECT
	default:
		return SM100InstInfo{}, false
	}

	info := SM100InstInfo{
		Kind:        kind,
		Mnemonic:    op,
		BlockScaled: base == "UTCOMMA",
	}

	for _, mod := range parts[1:] {
		switch mod {
		case "CTA_GROUP_2", "2CTA", "CTA2":
			info.CTAGroup = 2
		case "CTA_GROUP_1", "1CTA", "CTA1":
			info.CTAGroup = 1
		case "TS", "TMEM_A":
			info.TMEMSourceA = true
		case "SCALE_VEC_4X", "SCALE_VEC4X", "4X", "VEC4X":
			info.ScaleVec = 4
			info.BlockScaled = true
		case "SCALE_VEC_2X", "SCALE_VEC2X", "2X", "VEC2X":
			info.ScaleVec = 2
			info.BlockScaled = true
		case "BLOCK16": // 16-element scale blocks: scale_vec::4X for 4-bit operands
			info.ScaleVec = 4
			info.BlockScaled = true
		case "BLOCK32":
			info.ScaleVec = 2
			info.BlockScaled = true
		}
		if sm100BlockScaledTypes[mod] {
			info.BlockScaled = true
		}
		if info.DataType == "" && sm100DataTypes[mod] {
			info.DataType = mod
		}
		if kind == SM100LDTM || kind == SM100STTM {
			if cols, ok := parseSM100TMEMCols(mod); ok && info.TMEMCols == 0 {
				info.TMEMCols = cols
			}
		}
	}

	upperOps := strings.ToUpper(inst.Operands)
	if info.CTAGroup == 0 {
		switch {
		case strings.Contains(upperOps, "CTA_GROUP::2") || strings.Contains(upperOps, "CTA_GROUP_2") || strings.Contains(upperOps, ".2CTA"):
			info.CTAGroup = 2
		case strings.Contains(upperOps, "CTA_GROUP::1") || strings.Contains(upperOps, "CTA_GROUP_1") || strings.Contains(upperOps, ".1CTA"):
			info.CTAGroup = 1
		case kind == SM100UTMMA || kind == SM100UTMACP || kind == SM100UTMAFlush || kind == SM100UTMALDG:
			info.CTAGroup = 1
		}
	}

	if info.ScaleVec == 0 {
		switch {
		case strings.Contains(upperOps, "SCALE_VEC::4X") || strings.Contains(upperOps, "SCALE_VEC_4X") || strings.Contains(upperOps, ".4X"):
			info.ScaleVec = 4
			info.BlockScaled = true
		case strings.Contains(upperOps, "SCALE_VEC::2X") || strings.Contains(upperOps, "SCALE_VEC_2X") || strings.Contains(upperOps, ".2X"):
			info.ScaleVec = 2
			info.BlockScaled = true
		}
	}

	if !info.BlockScaled && (strings.Contains(upperOps, "BLOCK_SCALE") || strings.Contains(upperOps, "MXF4NVF4") || strings.Contains(upperOps, "MXF8F6F4") || strings.Contains(upperOps, "TMEM:SCALE") || strings.Contains(upperOps, "SCALE_DESC")) {
		info.BlockScaled = true
	}

	if kind == SM100UTMMA && !info.TMEMSourceA {
		info.TMEMSourceA = hasTMEMOperandA(upperOps)
	}

	return info, true
}

func parseSM100TMEMCols(mod string) (int, bool) {
	if n, err := strconv.Atoi(mod); err == nil && n > 0 {
		return n, true
	}
	if rest, ok := strings.CutPrefix(mod, "X"); ok {
		if n, err := strconv.Atoi(rest); err == nil && n > 0 {
			return n, true
		}
	}
	if rest, ok := strings.CutPrefix(mod, "COLS"); ok {
		rest = strings.TrimPrefix(rest, "_")
		if n, err := strconv.Atoi(rest); err == nil && n > 0 {
			return n, true
		}
	}
	return 0, false
}

func hasTMEMOperandA(upperOps string) bool {
	if upperOps == "" {
		return false
	}
	ops := splitOperands(upperOps)
	if len(ops) == 0 {
		return false
	}
	if len(ops) >= 3 {
		// In 3+ operand UTMMA (D, A, B, ...), operand[0] is typically accumulator D
		// in TMEM and operand[1] is matrix A (either [TMEM:...] or SMEM descriptor).
		// We also recognize explicit [TMEM:A...] or [TMEM_A...] in operand[0].
		if strings.Contains(ops[1], "TMEM") {
			return true
		}
		if strings.Contains(ops[0], "TMEM:A") || strings.Contains(ops[0], "TMEM_A") {
			return true
		}
		// If operand[0] is [TMEM:...] and operand[1] is a register/descriptor while
		// no other operand mentions SMEM and there are exactly 3 operands where
		// operand[2] is an immediate/predicate, do not mistake D for A unless
		// operand[0] is the only matrix operand.
		return false
	}
	if len(ops) == 2 {
		return strings.Contains(ops[0], "TMEM") || strings.Contains(ops[1], "TMEM")
	}
	return strings.Contains(ops[0], "TMEM")
}

// SummarizeSM100Function aggregates counts and attributes of Blackwell sm_100a
// tensor-memory and cluster SASS instructions in fn.
func SummarizeSM100Function(fn Function) SM100Summary {
	s := SM100Summary{
		ByKind: make(map[SM100Kind]int),
	}
	for _, inst := range fn.Instructions {
		info, ok := ClassifySM100Inst(inst)
		if !ok {
			continue
		}
		s.Total++
		s.ByKind[info.Kind]++
		switch info.Kind {
		case SM100UTMMA:
			s.UTMMA++
			switch info.CTAGroup {
			case 1:
				s.CTAGroup1++
			case 2:
				s.CTAGroup2++
			}
			if info.BlockScaled {
				s.BlockScaled++
			}
			switch info.ScaleVec {
			case 2:
				s.ScaleVec2X++
			case 4:
				s.ScaleVec4X++
			}
			if info.TMEMSourceA {
				s.TMEMSourceA++
			}
		case SM100UTMALDG:
			s.UTMALDG++
		case SM100UTMACP:
			s.UTMACP++
		case SM100UTMAFlush:
			s.UTMACMDFLUSH++
		case SM100LDTM:
			s.LDTM++
		case SM100STTM:
			s.STTM++
		case SM100ELECT:
			s.ELECT++
		}
	}
	return s
}

// AuditSM100Function checks static Blackwell sm_100a SASS invariants on fn and
// returns human-readable findings for any violations. An empty or nil slice
// means all invariants hold.
//
// Checked invariants:
//   - Every UTMMA instruction must be guarded by a leader predicate (@P0, @UP0, etc.)
//     or preceded by an ELECT leader-election instruction in the function.
//   - Every function containing UTMMA must also issue UTMACMDFLUSH (commit) and
//     LDTM (TMEM -> register readback).
//   - Every block-scaled UTMMA (.MXF4NVF4, .MXF8F6F4, .BLOCK_SCALE) must have
//     UTMACP scale staging into TMEM in the function.
func AuditSM100Function(fn Function) []string {
	var findings []string
	summary := SummarizeSM100Function(fn)
	if summary.UTMMA == 0 {
		return nil
	}

	seenElect := false
	hasBlockScaled := false
	for _, inst := range fn.Instructions {
		info, ok := ClassifySM100Inst(inst)
		if !ok {
			continue
		}
		if info.Kind == SM100ELECT {
			seenElect = true
			continue
		}
		if info.Kind == SM100UTMMA {
			if info.BlockScaled {
				hasBlockScaled = true
			}
			if !seenElect && !hasLeaderPredicate(inst.Predicate) {
				findings = append(findings, fmt.Sprintf(
					"%s: 0x%04x: %s issued without preceding ELECT or leader predicate (@UP0/@P0)",
					fn.Name, inst.Address, info.Mnemonic,
				))
			}
		}
	}

	if summary.UTMACMDFLUSH == 0 {
		findings = append(findings, fmt.Sprintf(
			"%s: %d UTMMA instruction(s) without corresponding UTMACMDFLUSH commit",
			fn.Name, summary.UTMMA,
		))
	}

	if summary.LDTM == 0 {
		findings = append(findings, fmt.Sprintf(
			"%s: %d UTMMA instruction(s) without corresponding LDTM accumulator readback",
			fn.Name, summary.UTMMA,
		))
	}

	if hasBlockScaled && summary.UTMACP == 0 {
		findings = append(findings, fmt.Sprintf(
			"%s: block-scaled UTMMA without UTMACP scale copy into TMEM",
			fn.Name,
		))
	}

	return findings
}

func hasLeaderPredicate(pred string) bool {
	pred = strings.ToUpper(strings.TrimSpace(pred))
	if pred == "" || pred == "@PT" || pred == "@UPT" || pred == "@!PT" || pred == "@!UPT" {
		return false
	}
	return strings.HasPrefix(pred, "@P") || strings.HasPrefix(pred, "@UP") || strings.HasPrefix(pred, "@!P") || strings.HasPrefix(pred, "@!UP")
}
