package cudainspect

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// PrecisionAttribution is the value of the precision_attribution report
// feature. Counts are static instruction counts in the selected function's
// disassembly, not executed instructions, FLOPs, cycle weights, or runtime
// pipe shares.
type PrecisionAttribution struct {
	Symbol      string                            `json:"symbol"`
	Artifact    Digest                            `json:"artifact"`
	SASS        Digest                            `json:"sass"`
	State       string                            `json:"state"`
	Precision   cudaroofline.Precision            `json:"precision"`
	Method      cudacensus.Method                 `json:"method"`
	Share       float64                           `json:"share"`
	Known       bool                              `json:"known"`
	Counted     uint64                            `json:"counted"`
	Excluded    uint64                            `json:"excluded"`
	Predicated  uint64                            `json:"predicated"`
	Counts      map[cudaroofline.Precision]uint64 `json:"counts"`
	Opcodes     map[string]uint64                 `json:"opcodes"`
	Limitations []string                          `json:"limitations"`
}

var sassTensorShape = regexp.MustCompile(`^(?:[0-9]+|M[0-9]+N[0-9]+K[0-9]+)$`)

// nonArithmeticMnemonics names instructions that do not execute on a
// floating-point or tensor pipe modeled by cudaroofline.Precision and are
// excluded from static precision attribution. Any mnemonic absent from both
// this table and the arithmetic classifiers is treated as ambiguous rather than
// silently ignored.
var nonArithmeticMnemonics = map[string]bool{
	// Memory loads, stores, constants, atomics, barriers, cache control.
	"LD": true, "LDG": true, "LDGDEPBAR": true, "LDGSTS": true, "LDSM": true,
	"LDS": true, "LDL": true, "LDC": true, "LDCU": true, "LDTM": true,
	"ST": true, "STG": true, "STS": true, "STL": true, "STSM": true, "STTM": true,
	"ATOM": true, "ATOMS": true, "ATOMG": true, "RED": true, "REDG": true, "REDS": true,
	"CCTL": true, "CCTLL": true, "CCTLT": true, "MEMBAR": true, "ERRBAR": true,
	"DEPBAR": true, "FENCE": true, "SYNCS": true, "ACQBULK": true, "CGAERRBAR": true,
	"UTMACCTL": true, "UTMALDG": true, "UTMASTG": true, "UTMAPF": true, "UTMACMDFLUSH": true,
	// Control flow, synchronization, warp orchestration.
	"BRA": true, "BRX": true, "BRXU": true, "JMP": true, "JMX": true, "JMXU": true,
	"CALL": true, "RET": true, "EXIT": true, "BREAK": true, "CONT": true,
	"BSSY": true, "BSYNC": true, "WARPSYNC": true, "YIELD": true, "NANOSLEEP": true,
	"BAR": true, "BPT": true, "ELECT": true, "ENDCOLLECTIVE": true, "RPCMOV": true,
	// Data movement, special registers, predicate and uniform manipulation, nop.
	"NOP": true, "MOV": true, "UMOV": true, "MOVM": true, "PRMT": true, "UPRMT": true,
	"SEL": true, "USEL": true, "SHFL": true, "VOTE": true, "UVOTE": true, "VOTEU": true,
	"S2R": true, "S2UR": true, "R2UR": true, "UR2UP": true, "CS2R": true, "B2R": true, "R2B": true,
	"P2R": true, "R2P": true, "PLOP3": true, "UPLOP3": true, "PSETP": true, "UPSETP": true,
	// Integer addressing, bitwise logic, shifts, comparisons, and format conversions.
	"IADD": true, "IADD3": true, "UIADD3": true, "ISUB": true, "IMAD": true, "UIMAD": true,
	"IMUL": true, "ISETP": true, "UISETP": true, "ISET": true, "UISET": true,
	"ICMP": true, "IABS": true, "IMNMX": true, "VIMNMX": true,
	"LEA": true, "ULEA": true, "LOP": true, "LOP3": true, "ULOP3": true,
	"SHF": true, "USHF": true, "SHR": true, "USHR": true, "SHL": true, "USHL": true,
	"BFE": true, "UBFE": true, "BFI": true, "UBFI": true, "BREV": true, "UBREV": true,
	"FLO": true, "UFLO": true, "POPC": true, "UPOPC": true, "SGXT": true, "USGXT": true,
	"I2I": true, "I2F": true, "I2FP": true, "F2I": true, "F2F": true, "F2FP": true,
	"FRND": true, "UI2I": true, "UI2F": true, "UI2FP": true, "UF2FP": true,
	"AL2P": true, "IPA": true,
}

var (
	fp32ScalarMnemonics = map[string]bool{
		"FADD": true, "FADD32I": true, "FMUL": true, "FMUL32I": true,
		"FFMA": true, "FFMA32I": true, "FMNMX": true, "FSEL": true,
		"FSET": true, "FSETP": true, "FSWZADD": true, "FCHK": true, "MUFU": true,
	}
	fp64ScalarMnemonics = map[string]bool{
		"DADD": true, "DMUL": true, "DFMA": true, "DMNMX": true, "DSET": true, "DSETP": true,
	}
	fp16ScalarMnemonics = map[string]bool{
		"HADD2": true, "HADD2_32I": true, "HMUL2": true, "HMUL2_32I": true,
		"HFMA2": true, "HFMA2_32I": true, "HMNMX2": true, "HSET2": true, "HSETP2": true,
	}
	tensorControlModifiers = map[string]bool{
		"SATFINITE": true, "SPARSE": true, "SP": true, "MMA_ASYNC": true,
		"SYNC": true, "ALIGNED": true, "SS": true, "RS": true,
		"TN": true, "NT": true, "TT": true, "NN": true,
		"TRANS": true, "NEG": true, "SCALE": true,
	}
	unsupportedTensorModifiers = map[string]bool{
		"E2M1": true, "FP4": true, "F4": true, "S4": true, "U4": true,
		"I4": true, "B1": true, "S16": true, "U16": true,
	}
	nonFP32FormatModifiers = map[string]bool{
		"F16": true, "FP16": true, "F16_V2": true, "BF16": true, "BF16_V2": true,
		"F64": true, "FP64": true, "TF32": true, "E4M3": true, "E5M2": true, "FP8": true,
		"E2M1": true, "FP4": true, "F4": true, "S8": true, "U8": true, "I8": true,
		"S4": true, "U4": true, "I4": true, "B1": true,
	}
	nonFP64FormatModifiers = map[string]bool{
		"F32": true, "FP32": true, "F16": true, "FP16": true, "F16_V2": true,
		"BF16": true, "BF16_V2": true, "TF32": true, "E4M3": true, "E5M2": true, "FP8": true,
		"E2M1": true, "FP4": true, "F4": true, "S8": true, "U8": true, "I8": true,
		"S4": true, "U4": true, "I4": true, "B1": true,
	}
	nonFP16FormatModifiers = map[string]bool{
		"F32": true, "FP32": true, "F64": true, "FP64": true, "TF32": true,
		"E4M3": true, "E5M2": true, "FP8": true, "E2M1": true, "FP4": true, "F4": true,
		"S8": true, "U8": true, "I8": true, "S4": true, "U4": true, "I4": true, "B1": true,
		"MMA": true,
	}
)

// classifyPrecisionOpcode classifies a full SASS opcode spelling into
// cudaroofline.Precision. When counted is false and reason is empty, the
// instruction is explicitly excluded as non-arithmetic. When reason is
// nonempty, the instruction has an unrecognized or ambiguous arithmetic/tensor
// format and prevents fabricating a dominant precision attribution.
func classifyPrecisionOpcode(opcode string) (prec cudaroofline.Precision, counted bool, reason string) {
	fields := strings.Split(opcode, ".")
	base := fields[0]
	if !sassMnemonic.MatchString(base) {
		return "", false, fmt.Sprintf("opcode %q is not a mnemonic", opcode)
	}
	mods := fields[1:]
	for _, m := range mods {
		if !sassModifier.MatchString(m) {
			return "", false, fmt.Sprintf("opcode %q has malformed modifier %q", opcode, m)
		}
	}

	switch {
	case fp32ScalarMnemonics[base]:
		for _, m := range mods {
			if nonFP32FormatModifiers[m] {
				return "", false, fmt.Sprintf("opcode %q has conflicting precision modifier %q for FP32 pipe", opcode, m)
			}
		}
		return cudaroofline.FP32, true, ""

	case fp64ScalarMnemonics[base]:
		for _, m := range mods {
			if nonFP64FormatModifiers[m] {
				return "", false, fmt.Sprintf("opcode %q has conflicting precision modifier %q for FP64 pipe", opcode, m)
			}
		}
		return cudaroofline.FP64, true, ""

	case fp16ScalarMnemonics[base]:
		for _, m := range mods {
			if m == "BF16" || m == "BF16_V2" {
				return "", false, fmt.Sprintf("opcode %q uses non-tensor BF16 modifier %q with no cudaroofline.Precision representation", opcode, m)
			}
			if nonFP16FormatModifiers[m] {
				return "", false, fmt.Sprintf("opcode %q has ambiguous modifier %q for FP16 pipe", opcode, m)
			}
		}
		return cudaroofline.FP16, true, ""

	case base == "DMMA":
		if len(mods) == 0 {
			return "", false, fmt.Sprintf("opcode %q states no DMMA shape or format modifier", opcode)
		}
		for _, m := range mods {
			if nonFP64FormatModifiers[m] {
				return "", false, fmt.Sprintf("opcode %q has conflicting modifier %q for DMMA", opcode, m)
			}
			if m != "F64" && m != "FP64" && !sassTensorShape.MatchString(m) && !tensorControlModifiers[m] {
				return "", false, fmt.Sprintf("opcode %q has unrecognized DMMA modifier %q", opcode, m)
			}
		}
		return cudaroofline.FP64, true, ""

	case base == "IMMA" || base == "UTCIMMA":
		var int8Inputs int
		for _, m := range mods {
			switch {
			case m == "S8" || m == "U8" || m == "I8":
				int8Inputs++
			case m == "S32" || m == "U32" || sassTensorShape.MatchString(m) || tensorControlModifiers[m]:
			case unsupportedTensorModifiers[m] || nonFP32FormatModifiers[m] || m == "F32" || m == "FP32":
				return "", false, fmt.Sprintf("opcode %q uses unsupported or ambiguous IMMA modifier %q", opcode, m)
			default:
				return "", false, fmt.Sprintf("opcode %q has unrecognized IMMA modifier %q", opcode, m)
			}
		}
		if int8Inputs == 0 {
			return "", false, fmt.Sprintf("opcode %q does not state an 8-bit integer tensor input format", opcode)
		}
		return cudaroofline.INT8Tensor, true, ""

	case base == "HMMA" || base == "QMMA" || base == "WGMMA" || base == "UTCMMA" || base == "UTCHMMA" || base == "UTCQMMA":
		return classifyTensorOpcode(opcode, base, mods)

	case base == "BMMA" || base == "OMMA" || base == "QDMA":
		return "", false, fmt.Sprintf("opcode %q uses a tensor format with no cudaroofline.Precision representation", opcode)

	case base == "IDP4A" || base == "IDP2A" || base == "IDP":
		return "", false, fmt.Sprintf("opcode %q uses a non-tensor integer dot-product pipe with no cudaroofline.Precision representation", opcode)

	case nonArithmeticMnemonics[base]:
		return "", false, ""

	default:
		return "", false, fmt.Sprintf("opcode %q has unrecognized arithmetic precision classification", opcode)
	}
}

func classifyTensorOpcode(opcode, base string, mods []string) (cudaroofline.Precision, bool, string) {
	var typeTokens []string
	for _, m := range mods {
		switch {
		case sassTensorShape.MatchString(m) || tensorControlModifiers[m]:
			continue
		case unsupportedTensorModifiers[m]:
			return "", false, fmt.Sprintf("opcode %q uses tensor format modifier %q with no cudaroofline.Precision representation", opcode, m)
		case m == "F32" || m == "FP32" || m == "S32" ||
			m == "F16" || m == "FP16" || m == "BF16" || m == "TF32" ||
			m == "E4M3" || m == "E5M2" || m == "FP8" ||
			m == "S8" || m == "U8" || m == "I8" ||
			m == "F64" || m == "FP64":
			typeTokens = append(typeTokens, m)
		default:
			return "", false, fmt.Sprintf("opcode %q has unrecognized tensor modifier %q", opcode, m)
		}
	}
	if len(typeTokens) == 0 {
		return "", false, fmt.Sprintf("opcode %q states no tensor input format", opcode)
	}

	inputTokens := typeTokens
	if first := typeTokens[0]; first == "F32" || first == "FP32" || first == "S32" {
		inputTokens = typeTokens[1:]
		if len(inputTokens) == 0 {
			return "", false, fmt.Sprintf("opcode %q names accumulator %s without an explicit tensor input format", opcode, first)
		}
	} else if (first == "F16" || first == "FP16") && len(typeTokens) > 1 && isFP8Token(typeTokens[1]) {
		// FP8 tensor instructions may accumulate into F16, e.g. QMMA.16832.F16.E4M3.E5M2.
		inputTokens = typeTokens[1:]
	}

	var want cudaroofline.Precision
	for _, tok := range inputTokens {
		p, ok := tensorInputPrecision(base, tok)
		if !ok {
			return "", false, fmt.Sprintf("opcode %q has invalid or accumulator-only tensor input modifier %q", opcode, tok)
		}
		if want == "" {
			want = p
		} else if p != want {
			return "", false, fmt.Sprintf("opcode %q mixes incompatible tensor input formats (%s and %s)", opcode, want, p)
		}
	}
	if (base == "QMMA" || base == "UTCQMMA") && want != cudaroofline.FP8Tensor {
		return "", false, fmt.Sprintf("opcode %q uses %s input on a quarter-precision tensor mnemonic", opcode, want)
	}
	return want, true, ""
}

func isFP8Token(tok string) bool {
	return tok == "E4M3" || tok == "E5M2" || tok == "FP8"
}

func tensorInputPrecision(base, tok string) (cudaroofline.Precision, bool) {
	switch tok {
	case "F16", "FP16":
		return cudaroofline.FP16Tensor, true
	case "BF16":
		return cudaroofline.BF16Tensor, true
	case "TF32":
		return cudaroofline.TF32Tensor, true
	case "E4M3", "E5M2", "FP8":
		return cudaroofline.FP8Tensor, true
	case "S8", "U8", "I8":
		if base == "WGMMA" || base == "UTCMMA" {
			return cudaroofline.INT8Tensor, true
		}
		return "", false
	case "F64", "FP64":
		if base == "WGMMA" || base == "UTCMMA" {
			return cudaroofline.FP64, true
		}
		return "", false
	default:
		return "", false
	}
}

func (r *Report) requested(name string) bool {
	for _, req := range r.Request.Required {
		if req == name {
			return true
		}
	}
	for _, opt := range r.Request.Optional {
		if opt == name {
			return true
		}
	}
	return false
}

// addPrecisionAttribution classifies the selected function's full SASS opcodes,
// aggregates checked uint64 counts per cudaroofline.Precision, and invokes
// cudacensus.AttributeFromMix.
func (r *Report) addPrecisionAttribution(fn *cudasass.Function, state, evidence string, quarantined uint64) {
	if quarantined > 0 {
		r.unavailable("precision_attribution", StateUnknown, fmt.Sprintf("%d quarantined instructions may be arithmetic operations", quarantined))
		return
	}
	if r.SASS == nil {
		r.unavailable("precision_attribution", StateUnknown, "no SASS supplied")
		return
	}

	counts := map[cudaroofline.Precision]uint64{}
	opcodes := map[string]uint64{}
	var counted, excluded, predicated, ambiguous uint64
	var overflow bool

	for _, inst := range fn.Instructions {
		prec, ok, reason := classifyPrecisionOpcode(inst.Opcode)
		if reason != "" {
			ambiguous++
			if ambiguous <= maxQuarantineDiagnostics {
				r.Diagnostics = append(r.Diagnostics, Diagnostic{
					Feature: "precision_attribution",
					Message: fmt.Sprintf("ambiguous instruction at %#x: %s", inst.Address, reason),
				})
			}
			continue
		}
		if !ok {
			if excluded == math.MaxUint64 {
				overflow = true
				break
			}
			excluded++
			continue
		}
		if counted == math.MaxUint64 || counts[prec] == math.MaxUint64 || opcodes[inst.Opcode] == math.MaxUint64 {
			overflow = true
			break
		}
		counted++
		counts[prec]++
		opcodes[inst.Opcode]++
		if inst.Predicate != "" {
			predicated++
		}
	}

	if extra := ambiguous - min(ambiguous, maxQuarantineDiagnostics); extra > 0 {
		r.Diagnostics = append(r.Diagnostics, Diagnostic{
			Feature: "precision_attribution",
			Message: fmt.Sprintf("%d more ambiguous instructions", extra),
		})
	}
	if overflow {
		r.unavailable("precision_attribution", StateUnknown, "cudacensus: instruction mix total overflows uint64")
		return
	}
	if ambiguous > 0 {
		r.unavailable("precision_attribution", StateUnknown, fmt.Sprintf("%d instructions have ambiguous or unsupported arithmetic precision", ambiguous))
		return
	}

	mix := cudacensus.InstructionMix{Counts: counts}
	attr, err := cudacensus.AttributeFromMix(mix)
	if err != nil || !attr.Known {
		reason := attr.Reason
		if reason == "" && err != nil {
			reason = err.Error()
		}
		if reason == "" {
			reason = "precision attribution unavailable"
		}
		r.unavailable("precision_attribution", StateUnknown, reason)
		return
	}

	limitations := []string{
		"static instruction counts from disassembly, not executed instructions, FLOPs, cycle weights, or runtime pipe activity",
		"predication, branch divergence, and loop trip counts are not resolved statically",
	}
	if state == StateInferred {
		limitations = append(limitations, "SASS listing is not bound to image .text bytes; attribution is inferred from supplied text, not observed on the executed artifact")
	}

	pa := PrecisionAttribution{
		Symbol:      r.Request.Symbol,
		Artifact:    r.Artifact,
		SASS:        *r.SASS,
		State:       state,
		Precision:   attr.Precision,
		Method:      attr.Method,
		Share:       attr.Share,
		Known:       attr.Known,
		Counted:     counted,
		Excluded:    excluded,
		Predicated:  predicated,
		Counts:      counts,
		Opcodes:     opcodes,
		Limitations: limitations,
	}
	r.set("precision_attribution", state, pa, "instructions", evidence)
}

func checkPrecisionAttribution(v json.RawMessage) error {
	var pa PrecisionAttribution
	if err := decodeValue(v, &pa); err != nil {
		return err
	}
	return pa.validate(nil, "")
}

func (pa *PrecisionAttribution) validate(r *Report, featureState string) error {
	if pa.Symbol == "" {
		return errors.New("empty symbol")
	}
	if pa.State != StateObserved && pa.State != StateInferred {
		return fmt.Errorf("state %q", pa.State)
	}
	if featureState != "" && pa.State != featureState {
		return fmt.Errorf("attribution state %q differs from feature state %q", pa.State, featureState)
	}
	if r != nil {
		if pa.Symbol != r.Request.Symbol {
			return fmt.Errorf("attribution symbol %q differs from request symbol %q", pa.Symbol, r.Request.Symbol)
		}
		if pa.Artifact != r.Artifact {
			return fmt.Errorf("attribution artifact %+v differs from report artifact %+v", pa.Artifact, r.Artifact)
		}
		if r.SASS == nil || pa.SASS != *r.SASS {
			return errors.New("attribution SASS digest differs from report SASS digest")
		}
	}
	if !pa.Known || pa.Method != cudacensus.FromInstructionMix {
		return fmt.Errorf("known=%v method=%q, want true %q", pa.Known, pa.Method, cudacensus.FromInstructionMix)
	}
	if len(pa.Counts) == 0 || len(pa.Opcodes) == 0 {
		return errors.New("empty counts or opcodes")
	}
	if len(pa.Limitations) == 0 {
		return errors.New("missing limitations")
	}
	for _, lim := range pa.Limitations {
		if strings.TrimSpace(lim) == "" {
			return errors.New("empty limitation")
		}
	}

	var countSum uint64
	for p, n := range pa.Counts {
		if !validRooflinePrecision(p) || n == 0 {
			return fmt.Errorf("precision %q count %d", p, n)
		}
		if n > math.MaxUint64-countSum {
			return errors.New("precision counts overflow")
		}
		countSum += n
	}
	if countSum != pa.Counted {
		return fmt.Errorf("precision counts sum to %d, counted %d", countSum, pa.Counted)
	}

	derivedCounts := map[cudaroofline.Precision]uint64{}
	var opcodeSum uint64
	for op, n := range pa.Opcodes {
		if n == 0 {
			return fmt.Errorf("opcode %q count 0", op)
		}
		p, ok, reason := classifyPrecisionOpcode(op)
		if reason != "" || !ok {
			return fmt.Errorf("opcode %q is not a countable arithmetic instruction (%s)", op, reason)
		}
		if n > math.MaxUint64-opcodeSum || n > math.MaxUint64-derivedCounts[p] {
			return errors.New("opcode counts overflow")
		}
		opcodeSum += n
		derivedCounts[p] += n
	}
	if opcodeSum != pa.Counted {
		return fmt.Errorf("opcode counts sum to %d, counted %d", opcodeSum, pa.Counted)
	}
	if len(derivedCounts) != len(pa.Counts) {
		return errors.New("opcode precision buckets differ from counts")
	}
	for p, n := range derivedCounts {
		if pa.Counts[p] != n {
			return fmt.Errorf("precision %q opcode sum %d differs from count %d", p, n, pa.Counts[p])
		}
	}
	if pa.Predicated > pa.Counted {
		return fmt.Errorf("predicated %d exceeds counted %d", pa.Predicated, pa.Counted)
	}

	attr, err := cudacensus.AttributeFromMix(cudacensus.InstructionMix{Counts: pa.Counts})
	if err != nil {
		return err
	}
	if pa.Precision != attr.Precision || pa.Method != attr.Method || pa.Share != attr.Share || pa.Known != attr.Known {
		return fmt.Errorf("attribution (%q, %q, %v, %v) inconsistent with mix (%q, %q, %v, %v)",
			pa.Precision, pa.Method, pa.Share, pa.Known,
			attr.Precision, attr.Method, attr.Share, attr.Known)
	}
	return nil
}

func validRooflinePrecision(p cudaroofline.Precision) bool {
	switch p {
	case cudaroofline.FP64, cudaroofline.FP32, cudaroofline.FP16,
		cudaroofline.TF32Tensor, cudaroofline.FP16Tensor, cudaroofline.BF16Tensor,
		cudaroofline.FP8Tensor, cudaroofline.INT8Tensor:
		return true
	default:
		return false
	}
}

// PrecisionAttribution decodes and validates the report's precision_attribution
// feature when available. It returns an error wrapping [ErrInvalidReport] if the
// report is invalid, or [ErrRequiredFeature] if precision_attribution is absent
// or unavailable.
func (r *Report) PrecisionAttribution() (*PrecisionAttribution, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	f := r.Feature("precision_attribution")
	if f == nil || !f.Known() {
		reason := "feature unavailable"
		for _, d := range r.Diagnostics {
			if d.Feature == "precision_attribution" && d.Message != "" {
				reason = d.Message
				break
			}
		}
		return nil, fmt.Errorf("%w: precision_attribution: %s", ErrRequiredFeature, reason)
	}
	var pa PrecisionAttribution
	if err := decodeValue(f.Value, &pa); err != nil {
		return nil, fmt.Errorf("%w: precision_attribution: %v", ErrInvalidReport, err)
	}
	if err := pa.validate(r, f.State); err != nil {
		return nil, fmt.Errorf("%w: precision_attribution: %v", ErrInvalidReport, err)
	}
	return &pa, nil
}

// InstructionMix returns a validated [cudacensus.InstructionMix] and its
// evidence state ([StateObserved] or [StateInferred]) from the report's
// precision_attribution feature. An inferred mix came from supplied SASS text
// whose encodings did not cover the image's .text section and must not be
// described as an observed executed artifact.
func (r *Report) InstructionMix() (cudacensus.InstructionMix, string, error) {
	pa, err := r.PrecisionAttribution()
	if err != nil {
		state := StateUnknown
		if r != nil {
			if f := r.Feature("precision_attribution"); f != nil && f.State != "" {
				state = f.State
			}
		}
		return cudacensus.InstructionMix{}, state, err
	}
	counts := make(map[cudaroofline.Precision]uint64, len(pa.Counts))
	for p, n := range pa.Counts {
		counts[p] = n
	}
	return cudacensus.InstructionMix{Counts: counts}, pa.State, nil
}
