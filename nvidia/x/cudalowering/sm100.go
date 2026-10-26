package cudalowering

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// SM100PTXSummary records counts of Blackwell sm_100a PTX instructions
// (tcgen05.*, elect.sync, clusterlaunchcontrol.try_cancel, and local memory traffic).
type SM100PTXSummary struct {
	Total            int
	Alloc            int // tcgen05.alloc
	Dealloc          int // tcgen05.dealloc
	Cp               int // tcgen05.cp
	MMA              int // tcgen05.mma
	MMACTAGroup1     int // tcgen05.mma.cta_group::1
	MMACTAGroup2     int // tcgen05.mma.cta_group::2
	MMABlockScaled   int // tcgen05.mma with kind::mxf4nvf4, kind::mxf8f6f4, or .block_scale
	MMATMEMSourceA   int // tcgen05.mma with TMEM-resident matrix A ([tmem])
	Commit           int // tcgen05.commit
	Wait             int // tcgen05.wait
	Ld               int // tcgen05.ld
	St               int // tcgen05.st
	Elect            int // elect.sync
	ClusterTryCancel int // clusterlaunchcontrol.try_cancel
	TMALoad          int // cp.async.bulk.tensor
	LocalLd          int // ld.local
	LocalSt          int // st.local
}

// SM100Correlation links a PTX sm_100a instruction to its lowered SASS instruction(s).
type SM100Correlation struct {
	PTXLine     int
	PTXOpcode   string
	SASSKind    cudasass.SM100Kind
	SASSOpcodes []string
	CTAGroup    int
	BlockScaled bool
	TMEMSourceA bool
	Matched     bool
}

// SM100LoweringReport holds the PTX-to-SASS correlation and audit results for a
// Blackwell sm_100a kernel.
type SM100LoweringReport struct {
	Kernel       string
	PTX          SM100PTXSummary
	SASS         cudasass.SM100Summary
	Correlations []SM100Correlation
	SpillLoads   int
	SpillStores  int
	AuditIssues  []string
	Divergences  []string
}

// OK reports whether all PTX-to-SASS correlations, SASS invariants, and
// zero-spill checks succeeded without divergence.
func (r *SM100LoweringReport) OK() bool {
	return r != nil && len(r.Divergences) == 0 && len(r.AuditIssues) == 0 && r.SpillLoads == 0 && r.SpillStores == 0
}

// Err returns nil if the report has no divergences or audit issues, or a single
// joined error describing all findings.
func (r *SM100LoweringReport) Err() error {
	if r == nil {
		return errors.New("cudalowering: nil SM100LoweringReport")
	}
	if len(r.Divergences) == 0 {
		return nil
	}
	errs := make([]error, len(r.Divergences))
	for i, d := range r.Divergences {
		errs[i] = errors.New(d)
	}
	return fmt.Errorf("cudalowering: %w", errors.Join(errs...))
}

// String formats a human-readable summary of the PTX-to-SASS correlation report.
func (r *SM100LoweringReport) String() string {
	if r == nil {
		return "<nil>"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s: ptx(mma=%d cp=%d commit=%d ld=%d st=%d elect=%d) sass(UTMMA=%d UTMACP=%d UTMACMDFLUSH=%d LDTM=%d STTM=%d ELECT=%d)",
		r.Kernel,
		r.PTX.MMA, r.PTX.Cp, r.PTX.Commit, r.PTX.Ld, r.PTX.St, r.PTX.Elect,
		r.SASS.UTMMA, r.SASS.UTMACP, r.SASS.UTMACMDFLUSH, r.SASS.LDTM, r.SASS.STTM, r.SASS.ELECT,
	)
	for _, d := range r.Divergences {
		b.WriteString("\n  divergence: ")
		b.WriteString(d)
	}
	return b.String()
}

// CompareSM100PTXAndSASS parses PTX and SASS sources for a Blackwell sm_100a
// kernel, correlates each PTX tcgen05.* / elect.sync operation to its lowered
// SASS instruction(s), and checks CTA group width, block-scaling, TMEM load/store
// counts, static SASS invariants, and register spills.
func CompareSM100PTXAndSASS(ptxSrc, sassSrc string) (*SM100LoweringReport, error) {
	if strings.TrimSpace(ptxSrc) == "" {
		return nil, errors.New("cudalowering: empty PTX source")
	}
	if strings.TrimSpace(sassSrc) == "" {
		return nil, errors.New("cudalowering: empty SASS source")
	}

	k, err := parseSM100PTXKernel(ptxSrc)
	if err != nil {
		return nil, err
	}
	fn, err := parseSM100SASSFunction(sassSrc, k.Name)
	if err != nil {
		return nil, err
	}

	report := &SM100LoweringReport{
		Kernel: k.Name,
		SASS:   cudasass.SummarizeSM100Function(*fn),
	}

	sassByKind := make(map[cudasass.SM100Kind][]cudasass.SM100InstInfo)
	sassConsumed := make(map[cudasass.SM100Kind]int)
	var sassLDL, sassSTL int
	for _, inst := range fn.Instructions {
		upper := strings.ToUpper(strings.TrimSpace(inst.Opcode))
		if upper == "LDL" || strings.HasPrefix(upper, "LDL.") {
			sassLDL++
		} else if upper == "STL" || strings.HasPrefix(upper, "STL.") {
			sassSTL++
		}
		if info, ok := cudasass.ClassifySM100Inst(inst); ok {
			sassByKind[info.Kind] = append(sassByKind[info.Kind], info)
		}
	}

	for _, inst := range k.Instructions {
		op := strings.ToLower(strings.TrimSpace(inst.Opcode))
		if op == "" {
			continue
		}
		report.PTX.Total++

		switch {
		case hasPTXPrefix(op, "tcgen05.alloc"):
			report.PTX.Alloc++

		case hasPTXPrefix(op, "tcgen05.dealloc"):
			report.PTX.Dealloc++

		case hasPTXPrefix(op, "tcgen05.cp"):
			report.PTX.Cp++
			corr := correlateNextSASS(inst, cudasass.SM100UTMACP, sassByKind, sassConsumed)
			corr.CTAGroup = parsePTXCTAGroup(op)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "tcgen05.mma"):
			report.PTX.MMA++
			ctaGroup := parsePTXCTAGroup(op)
			if ctaGroup == 2 {
				report.PTX.MMACTAGroup2++
			} else {
				ctaGroup = 1
				report.PTX.MMACTAGroup1++
			}
			blockScaled := isPTXBlockScaled(op)
			if blockScaled {
				report.PTX.MMABlockScaled++
			}
			tmemA := isPTXMMATMEMSourceA(inst.Operands)
			if tmemA {
				report.PTX.MMATMEMSourceA++
			}
			corr := correlateNextSASS(inst, cudasass.SM100UTMMA, sassByKind, sassConsumed)
			corr.CTAGroup = ctaGroup
			corr.BlockScaled = blockScaled
			corr.TMEMSourceA = tmemA
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "tcgen05.commit"):
			report.PTX.Commit++
			corr := correlateNextSASS(inst, cudasass.SM100UTMAFlush, sassByKind, sassConsumed)
			corr.CTAGroup = parsePTXCTAGroup(op)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "tcgen05.wait"):
			report.PTX.Wait++

		case hasPTXPrefix(op, "tcgen05.ld"):
			report.PTX.Ld++
			corr := correlateNextSASS(inst, cudasass.SM100LDTM, sassByKind, sassConsumed)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "tcgen05.st"):
			report.PTX.St++
			corr := correlateNextSASS(inst, cudasass.SM100STTM, sassByKind, sassConsumed)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "elect.sync") || op == "elect":
			report.PTX.Elect++
			corr := correlateNextSASS(inst, cudasass.SM100ELECT, sassByKind, sassConsumed)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "clusterlaunchcontrol.try_cancel"):
			report.PTX.ClusterTryCancel++

		case hasPTXPrefix(op, "cp.async.bulk.tensor"):
			report.PTX.TMALoad++
			corr := correlateNextSASS(inst, cudasass.SM100UTMALDG, sassByKind, sassConsumed)
			report.Correlations = append(report.Correlations, corr)

		case hasPTXPrefix(op, "ld.local"):
			report.PTX.LocalLd++

		case hasPTXPrefix(op, "st.local"):
			report.PTX.LocalSt++
		}
	}

	if report.PTX.Alloc > 0 && report.PTX.Dealloc == 0 {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: %d tcgen05.alloc without matching tcgen05.dealloc",
			k.Name, report.PTX.Alloc,
		))
	}

	if report.PTX.MMA != report.SASS.UTMMA {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX tcgen05.mma count (%d) != SASS UTMMA count (%d)",
			k.Name, report.PTX.MMA, report.SASS.UTMMA,
		))
	}
	if report.PTX.MMACTAGroup1 != report.SASS.CTAGroup1 {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX cta_group::1 MMA count (%d) != SASS CTA_GROUP_1 UTMMA count (%d)",
			k.Name, report.PTX.MMACTAGroup1, report.SASS.CTAGroup1,
		))
	}
	if report.PTX.MMACTAGroup2 != report.SASS.CTAGroup2 {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX cta_group::2 MMA count (%d) != SASS CTA_GROUP_2 UTMMA count (%d)",
			k.Name, report.PTX.MMACTAGroup2, report.SASS.CTAGroup2,
		))
	}
	if report.PTX.MMABlockScaled != report.SASS.BlockScaled {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX block-scaled MMA count (%d) != SASS block-scaled UTMMA count (%d)",
			k.Name, report.PTX.MMABlockScaled, report.SASS.BlockScaled,
		))
	}
	if report.PTX.MMATMEMSourceA != report.SASS.TMEMSourceA {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX TMEM-sourced operand A MMA count (%d) != SASS TMEM-sourced UTMMA count (%d)",
			k.Name, report.PTX.MMATMEMSourceA, report.SASS.TMEMSourceA,
		))
	}
	if report.PTX.Cp != report.SASS.UTMACP {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX tcgen05.cp count (%d) != SASS UTMACP count (%d)",
			k.Name, report.PTX.Cp, report.SASS.UTMACP,
		))
	}
	if report.PTX.Commit != report.SASS.UTMACMDFLUSH {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX tcgen05.commit count (%d) != SASS UTMACMDFLUSH count (%d)",
			k.Name, report.PTX.Commit, report.SASS.UTMACMDFLUSH,
		))
	}
	if report.PTX.Ld != report.SASS.LDTM {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX tcgen05.ld count (%d) != SASS LDTM count (%d)",
			k.Name, report.PTX.Ld, report.SASS.LDTM,
		))
	}
	if report.PTX.St != report.SASS.STTM {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX tcgen05.st count (%d) != SASS STTM count (%d)",
			k.Name, report.PTX.St, report.SASS.STTM,
		))
	}
	if report.PTX.Elect != report.SASS.ELECT {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX elect.sync count (%d) != SASS ELECT count (%d)",
			k.Name, report.PTX.Elect, report.SASS.ELECT,
		))
	}
	if report.PTX.TMALoad > 0 && report.PTX.TMALoad != report.SASS.UTMALDG {
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: PTX cp.async.bulk.tensor count (%d) != SASS UTMALDG count (%d)",
			k.Name, report.PTX.TMALoad, report.SASS.UTMALDG,
		))
	}

	if report.PTX.LocalLd == 0 && sassLDL > 0 {
		report.SpillLoads = sassLDL
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: unexpected register spill: %d LDL instruction(s) in SASS without ld.local in PTX",
			k.Name, sassLDL,
		))
	}
	if report.PTX.LocalSt == 0 && sassSTL > 0 {
		report.SpillStores = sassSTL
		report.Divergences = append(report.Divergences, fmt.Sprintf(
			"%s: unexpected register spill: %d STL instruction(s) in SASS without st.local in PTX",
			k.Name, sassSTL,
		))
	}

	audit := cudasass.AuditSM100Function(*fn)
	report.AuditIssues = append(report.AuditIssues, audit...)
	report.Divergences = append(report.Divergences, audit...)

	return report, nil
}

func correlateNextSASS(
	inst cudaptx.Instruction,
	kind cudasass.SM100Kind,
	byKind map[cudasass.SM100Kind][]cudasass.SM100InstInfo,
	consumed map[cudasass.SM100Kind]int,
) SM100Correlation {
	corr := SM100Correlation{
		PTXLine:   inst.Line,
		PTXOpcode: inst.Opcode,
		SASSKind:  kind,
	}
	list := byKind[kind]
	idx := consumed[kind]
	if idx < len(list) {
		corr.SASSOpcodes = []string{list[idx].Mnemonic}
		corr.Matched = true
		consumed[kind] = idx + 1
	}
	return corr
}

func hasPTXPrefix(op, prefix string) bool {
	return op == prefix || strings.HasPrefix(op, prefix+".") || strings.HasPrefix(op, prefix+"::")
}

func parsePTXCTAGroup(op string) int {
	switch {
	case strings.Contains(op, "cta_group::2") || strings.Contains(op, "cta_group_2") || strings.Contains(op, ".2cta"):
		return 2
	case strings.Contains(op, "cta_group::1") || strings.Contains(op, "cta_group_1") || strings.Contains(op, ".1cta"):
		return 1
	default:
		return 0
	}
}

func isPTXBlockScaled(op string) bool {
	return strings.Contains(op, "mxf4nvf4") ||
		strings.Contains(op, "mxf8f6f4") ||
		strings.Contains(op, "block_scale") ||
		strings.Contains(op, "nvf4") ||
		strings.Contains(op, "mxf4") ||
		strings.Contains(op, "mxf8")
}

func isPTXMMATMEMSourceA(operands string) bool {
	// In PTX tcgen05.mma:
	//   tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, p;
	//   tcgen05.mma.cta_group::1.kind::f16 [d-tmem], [a-tmem], b-desc, idesc, p;
	// Matrix A is the second operand; if it is bracketed ([...]), A resides in TMEM.
	ops := splitTopLevelComma(operands)
	if len(ops) < 2 {
		return false
	}
	opA := strings.TrimSpace(ops[1])
	return strings.HasPrefix(opA, "[") && strings.HasSuffix(opA, "]")
}

func splitTopLevelComma(s string) []string {
	var out []string
	depth := 0
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '{', '[', '(':
			depth++
		case '}', ']', ')':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				out = append(out, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	if tail := strings.TrimSpace(s[start:]); tail != "" {
		out = append(out, tail)
	}
	return out
}

func parseSM100PTXKernel(ptxSrc string) (*cudaptx.Kernel, error) {
	m, err := cudaptx.ParseString(ptxSrc)
	if err == nil && m != nil {
		for _, k := range m.Kernels {
			if k != nil && len(k.Instructions) > 0 {
				return k, nil
			}
		}
	}
	wrapped := ".version 8.8\n.target sm_100a\n.address_size 64\n.visible .entry sm100_kernel()\n{\n" + ptxSrc + "\n}\n"
	m, err = cudaptx.ParseString(wrapped)
	if err != nil {
		return nil, fmt.Errorf("cudalowering: parse PTX: %w", err)
	}
	for _, k := range m.Kernels {
		if k != nil && len(k.Instructions) > 0 {
			return k, nil
		}
	}
	return &cudaptx.Kernel{Name: "sm100_kernel"}, nil
}

func parseSM100SASSFunction(sassSrc, defaultName string) (*cudasass.Function, error) {
	if defaultName == "" {
		defaultName = "sm100_kernel"
	}
	d, err := cudasass.ParseString(sassSrc)
	if err == nil && d != nil {
		for _, fn := range d.Functions {
			if fn != nil && len(fn.Instructions) > 0 {
				return fn, nil
			}
		}
	}

	wrapped := fmt.Sprintf(".headerflags @\"EF_CUDA_SM100\"\n.section .text.%s,\"ax\",@progbits\n.text.%s:\n%s\n", defaultName, defaultName, sassSrc)
	d, err = cudasass.ParseString(wrapped)
	if err == nil && d != nil {
		for _, fn := range d.Functions {
			if fn != nil && len(fn.Instructions) > 0 {
				return fn, nil
			}
		}
	}

	// Fallback for bare SASS instruction lines without /*0000*/ address prefixes.
	fn := &cudasass.Function{Name: defaultName}
	var addr uint64
	for _, rawLine := range strings.Split(sassSrc, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "//") || strings.HasPrefix(line, ".") || strings.HasSuffix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "/*") {
			line = fmt.Sprintf("/*%04x*/ %s", addr, line)
		}
		if inst, ok := cudasass.ParseInstructionLine(line); ok {
			fn.Instructions = append(fn.Instructions, inst)
			addr += 16
		}
	}
	if len(fn.Instructions) == 0 {
		return nil, errors.New("cudalowering: no SASS instructions parsed")
	}
	return fn, nil
}
