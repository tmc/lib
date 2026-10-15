package cudalowering

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

// A Kind classifies a divergence between what the PTX asked for and what the
// SASS delivered.
type Kind int

const (
	// Erased is a PTX opcode with no SASS instruction it lowers to.
	Erased Kind = iota + 1
	// Unsourced is a SASS opcode that ptxas emits only for a PTX opcode the
	// kernel never used.
	Unsourced
	// Spill is an unsourced local memory access: the register allocator's, not
	// the kernel's.
	Spill
	// Padding is a kernel whose interior NOP fraction exceeds the reporting
	// threshold.
	Padding
	// MissingSASS is a PTX kernel with no function in the disassembly.
	MissingSASS
	// MissingPTX is a disassembled function with no kernel in the PTX.
	MissingPTX
	// WidthCollapse is a vector operation lowered to equivalent scalar traffic.
	WidthCollapse
	// SharedConversion is a shared-space address conversion present in a kernel
	// or slot that was expected to have direct shared access.
	SharedConversion
)

func (k Kind) String() string {
	switch k {
	case Erased:
		return "erased"
	case Unsourced:
		return "unsourced"
	case Spill:
		return "spill"
	case Padding:
		return "padding"
	case MissingSASS:
		return "missing sass"
	case MissingPTX:
		return "missing ptx"
	case WidthCollapse:
		return "width collapse"
	case SharedConversion:
		return "shared conversion"
	}
	return fmt.Sprintf("kind(%d)", int(k))
}

// A Divergence is one place where the SASS differs from what the PTX asked for.
type Divergence struct {
	Kind        Kind
	Slot        string // exact producer slot ID, empty for aggregate findings
	IRLine      int    // terminal IR use, zero for aggregate findings
	SASSAddress uint64 // conversion generic-access PC, when attributed
	Kernel      string // kernel name, as the level that has the kernel spells it
	PTXOpcode   string // PTX opcode prefix involved, if any
	SASSOpcode  string // SASS opcode prefix involved, if any
	PTXCount    int    // matching PTX instructions
	SASSCount   int    // matching SASS instructions
	Detail      string // supporting measurement, such as peak live register count
}

func (d Divergence) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s: ", d.Kernel, d.Kind)
	if d.Slot != "" {
		fmt.Fprintf(&b, "%s IR line %d: ", d.Slot, d.IRLine)
	}
	switch d.Kind {
	case Erased:
		fmt.Fprintf(&b, "%d %s, no %s", d.PTXCount, d.PTXOpcode, d.SASSOpcode)
	case Unsourced, Spill:
		fmt.Fprintf(&b, "%d %s, no %s", d.SASSCount, d.SASSOpcode, d.PTXOpcode)
	case Padding:
		fmt.Fprintf(&b, "%d NOP ahead of the last real instruction", d.SASSCount)
	case MissingSASS:
		b.WriteString("kernel is in the ptx and not in the disassembly")
	case MissingPTX:
		b.WriteString("function is in the disassembly and not in the ptx")
	case WidthCollapse:
		fmt.Fprintf(&b, "%d %s, %d %s in the kernel", d.PTXCount, d.PTXOpcode, d.SASSCount, d.SASSOpcode)
	case SharedConversion:
		if d.PTXCount > 0 && d.SASSCount > 0 {
			fmt.Fprintf(&b, "%d %s in PTX, %d %s in SASS", d.PTXCount, d.PTXOpcode, d.SASSCount, d.SASSOpcode)
		} else if d.PTXCount > 0 {
			fmt.Fprintf(&b, "%d %s in PTX", d.PTXCount, d.PTXOpcode)
		} else {
			fmt.Fprintf(&b, "%d %s in SASS", d.SASSCount, d.SASSOpcode)
		}
	}
	if d.Detail != "" {
		fmt.Fprintf(&b, " (%s)", d.Detail)
	}
	return b.String()
}

// Stats counts one matched kernel at both levels. It is reported whether or not
// the kernel diverged, so that a quiet comparison still shows what it read.
type Stats struct {
	Kernel           string
	PTXInstructions  int
	SASSInstructions int
	NOPs             int // NOP instructions, trailing alignment padding included
	InteriorNOPs     int // NOPs ahead of the kernel's last real instruction
	PeakGPR          int // 0 unless the disassembly carries live register counts
}

// Options configures [Compare]. The zero Options is usable.
type Options struct {
	// Lowerings is the PTX to SASS opcode table. Nil means [DefaultLowerings].
	Lowerings []Lowering

	// MaxNOPFraction is the interior NOP fraction of a kernel above which
	// [Padding] is reported. Zero means [DefaultMaxNOPFraction]; a negative
	// value disables the check.
	MaxNOPFraction float64

	// InferredSharedSlots lists slot or kernel names whose SASS must not contain
	// shared-space address conversions. Labels do not prove slot attribution.
	InferredSharedSlots []string

	// CheckSharedConversions, when true, verifies that no shared-space address
	// conversions (cvta.shared in PTX, or conversion instructions in SASS) exist.
	CheckSharedConversions bool
}

// DefaultMaxNOPFraction is the interior NOP fraction [Compare] reports above.
//
// Only NOPs ahead of a kernel's last real instruction are counted. Every kernel
// ends in NOPs, because ptxas pads the section to its alignment, and on a short
// kernel that padding is a third of the instructions; counting it would make
// the check fire everywhere and mean nothing. A NOP in the body is the other
// thing: ptxas issuing nothing to hold a pipe, which is what displaces the
// instruction mix a throughput prediction was made from.
const DefaultMaxNOPFraction = 0.10

// A Report is the result of comparing a PTX module with its SASS.
type Report struct {
	Stats       []Stats
	Slots       []SlotEvidence // checked per-slot coverage; nil for aggregate comparisons
	Divergences []Divergence
}

// Err returns an error naming every divergence, or nil if there are none.
func (r *Report) Err() error {
	if r == nil || len(r.Divergences) == 0 {
		return nil
	}
	msgs := make([]error, len(r.Divergences))
	for i, d := range r.Divergences {
		msgs[i] = errors.New(d.String())
	}
	return fmt.Errorf("cudalowering: %w", errors.Join(msgs...))
}

func (r *Report) String() string {
	var b strings.Builder
	for _, s := range r.Stats {
		fmt.Fprintf(&b, "%s: %d ptx, %d sass, %d nop (%d interior)",
			s.Kernel, s.PTXInstructions, s.SASSInstructions, s.NOPs, s.InteriorNOPs)
		if s.PeakGPR > 0 {
			fmt.Fprintf(&b, ", peak %d gpr", s.PeakGPR)
		}
		b.WriteString("\n")
	}
	for _, d := range r.Divergences {
		b.WriteString(d.String())
		b.WriteString("\n")
	}
	return b.String()
}

// Compare pairs the kernels of m with the functions of d and reports where the
// lowering diverged from the PTX. The error is non-nil only for unusable input;
// divergences are in the report, and [Report.Err] turns them into one error.
func Compare(m *cudaptx.Module, d *cudasass.Disassembly, opts *Options) (*Report, error) {
	if m == nil {
		return nil, errors.New("cudalowering: nil module")
	}
	if d == nil {
		return nil, errors.New("cudalowering: nil disassembly")
	}
	if opts == nil {
		opts = &Options{}
	}
	lowerings := opts.Lowerings
	if lowerings == nil {
		lowerings = DefaultLowerings
	}
	maxNOP := opts.MaxNOPFraction
	if maxNOP == 0 {
		maxNOP = DefaultMaxNOPFraction
	}

	match := Match(m, d)
	r := &Report{}
	if (opts.CheckSharedConversions || len(opts.InferredSharedSlots) > 0) && len(match.Pairs) == 0 {
		return r, errors.New("cudalowering: shared conversion analysis requires a matched executable kernel")
	}
	for _, p := range match.Pairs {
		ptx := p.PTX.Summary()
		sass := p.SASS.Summary()
		peak := peakGPR(p.SASS)
		nops, interior := countNOPs(p.SASS)
		r.Stats = append(r.Stats, Stats{
			Kernel:           p.PTX.Name,
			PTXInstructions:  ptx.Total,
			SASSInstructions: sass.Total,
			NOPs:             nops,
			InteriorNOPs:     interior,
			PeakGPR:          peak,
		})

		for _, l := range lowerings {
			ptxCount := ptx.Count(l.PTX)
			sassCount := 0
			for _, op := range l.SASS {
				sassCount += sass.Count(op)
			}
			switch {
			case len(l.ScalarSASS) > 0 && ptxCount > 0 && sassCount == 0 && countExact(sass, l.ScalarSASS) > 0:
				// A vector access the SASS does not contain, in a kernel that
				// does contain the scalar form: report which it is rather than
				// the bare Erased. The scalar count is the whole kernel's, so
				// it is not a count of the replacements; requiring it to reach
				// ptxCount*VectorWidth would miss a partial collapse and would
				// still be inflated by scalar accesses the kernel wrote itself.
				r.Divergences = append(r.Divergences, Divergence{
					Kind:       WidthCollapse,
					Kernel:     p.PTX.Name,
					PTXOpcode:  l.PTX,
					SASSOpcode: strings.Join(l.ScalarSASS, " or "),
					PTXCount:   ptxCount,
					SASSCount:  countExact(sass, l.ScalarSASS),
					Detail:     fmt.Sprintf("no %s, %d-wide", strings.Join(l.SASS, " or "), l.VectorWidth),
				})
			case ptxCount > 0 && sassCount == 0:
				r.Divergences = append(r.Divergences, Divergence{
					Kind:       Erased,
					Kernel:     p.PTX.Name,
					PTXOpcode:  l.PTX,
					SASSOpcode: strings.Join(l.SASS, " or "),
					PTXCount:   ptxCount,
				})
			case l.Exclusive && sassCount > 0 && ptxCount == 0:
				div := Divergence{
					Kind:       Unsourced,
					Kernel:     p.PTX.Name,
					PTXOpcode:  l.PTX,
					SASSOpcode: strings.Join(l.SASS, " or "),
					SASSCount:  sassCount,
				}
				if l.Spill {
					div.Kind = Spill
					if peak > 0 {
						div.Detail = fmt.Sprintf("peak %d live gpr", peak)
					}
				}
				r.Divergences = append(r.Divergences, div)
			}
		}

		if opts.CheckSharedConversions || len(opts.InferredSharedSlots) > 0 {
			if m.Target != "sm_121" || d.Target != "sm_121" {
				return r, fmt.Errorf("cudalowering: %s: shared conversion analysis unavailable for PTX target %q and SASS target %q (requires both exact sm_121)", p.PTX.Name, m.Target, d.Target)
			}
			cvtaCount := ptx.Count("cvta.shared") + ptx.Count("cvta.to.shared")
			scan := scanSharedConversions(p.SASS)
			sassConvCount := len(scan.chains)
			if cvtaCount > 0 || sassConvCount > 0 {
				detail := ""
				if len(opts.InferredSharedSlots) > 0 {
					detail = fmt.Sprintf("requested labels (not slot attribution): %s", strings.Join(opts.InferredSharedSlots, ", "))
				}
				for _, chain := range scan.chains {
					if detail != "" {
						detail += "; "
					}
					detail += "observed SM121 mechanism: " + chain.String()
				}
				r.Divergences = append(r.Divergences, Divergence{
					Kind:       SharedConversion,
					Kernel:     p.PTX.Name,
					PTXOpcode:  "cvta.shared",
					SASSOpcode: "conversion",
					PTXCount:   cvtaCount,
					SASSCount:  sassConvCount,
					Detail:     detail,
				})
			}
			if len(scan.refusals) != 0 {
				return r, fmt.Errorf("cudalowering: %s: shared conversion analysis unavailable: %s", p.PTX.Name, strings.Join(scan.refusals, "; "))
			}
		}

		if maxNOP > 0 && sass.Total > 0 && float64(interior)/float64(sass.Total) > maxNOP {
			r.Divergences = append(r.Divergences, Divergence{
				Kind:       Padding,
				Kernel:     p.PTX.Name,
				SASSOpcode: "NOP",
				SASSCount:  interior,
				Detail:     fmt.Sprintf("%d instructions in all", sass.Total),
			})
		}
	}

	for _, k := range match.UnmatchedPTX {
		r.Divergences = append(r.Divergences, Divergence{Kind: MissingSASS, Kernel: k.Name})
	}
	for _, fn := range match.UnmatchedSASS {
		r.Divergences = append(r.Divergences, Divergence{Kind: MissingPTX, Kernel: fn.Name})
	}
	return r, nil
}

// VerifyNoSharedConversions checks matched kernels for PTX shared conversions
// and the bounded observed SM121 SASS conversion mechanism. Unsupported SASS
// returns an error and a partial report; it cannot establish conversion absence.
// Slots are descriptive caller labels, not evidence of per-slot attribution.
// This textual check does not attest a machine-code encoding or other targets.
func VerifyNoSharedConversions(m *cudaptx.Module, d *cudasass.Disassembly, slots ...string) (*Report, error) {
	if len(slots) == 0 {
		return nil, errors.New("cudalowering: no inferred shared slots specified")
	}
	opts := &Options{
		CheckSharedConversions: true,
		InferredSharedSlots:    slots,
	}
	return Compare(m, d, opts)
}

func countSharedConversions(fn *cudasass.Function) int {
	return len(scanSharedConversions(fn).chains)
}

// countNOPs returns the number of NOP instructions in fn and the number of them
// that precede its last instruction that is not a NOP.
func countNOPs(fn *cudasass.Function) (total, interior int) {
	last := -1
	for i, inst := range fn.Instructions {
		if inst.Opcode == "NOP" {
			total++
		} else {
			last = i
		}
	}
	for i := 0; i < last; i++ {
		if fn.Instructions[i].Opcode == "NOP" {
			interior++
		}
	}
	return total, interior
}

// peakGPR returns the highest live general purpose register count recorded on
// the function's instructions, or 0 if the disassembly carries no live ranges.
func peakGPR(fn *cudasass.Function) int {
	var peak int
	for _, inst := range fn.Instructions {
		if inst.LiveGPR > peak {
			peak = inst.LiveGPR
		}
	}
	return peak
}

// countExact returns how many instructions in s have exactly one of the given
// opcodes. [cudasass.OpcodeSummary.Count] matches by prefix, which is right for
// a family such as LDG but wrong for a scalar form whose spelling is a prefix
// of the vector one.
func countExact(s *cudasass.OpcodeSummary, opcodes []string) int {
	if s == nil {
		return 0
	}
	var total int
	for op, count := range s.Counts {
		for _, want := range opcodes {
			if strings.EqualFold(op, want) {
				total += count
				break
			}
		}
	}
	return total
}

// SlotUseEvidence records observed text locations for an actual producer use.
type SlotUseEvidence struct {
	IRLine                   int
	Value                    string
	PTXLines                 []int
	SharedPCs, ConversionPCs []uint64
}

// SlotEvidence describes bounded textual coverage, not compiler or GPU attestation.
// Verified requires every use to have a direct shared witness and no conversion.
type SlotEvidence struct {
	Slot, OriginalSHA256, PreparedSHA256, PTXSHA256, SASSSHA256 string
	Verified                                                    bool
	Reason                                                      string
	Uses                                                        []SlotUseEvidence
}

// VerifyInferredSharedConversions derives slot attribution from actual inference
// origins and observed compiler locations in caller-paired PTX/SASS text. It
// validates original IR identity but does not attest that a compiler emitted the
// supplied texts. Missing, ambiguous, elided or unsupported coverage returns an
// error and a partial report. The legacy verifier remains kernel-aggregate.
func VerifyInferredSharedConversions(ir []byte, inference *nvvmx.InferenceReport, ptx, sass []byte) (*Report, error) {
	provenance, err := inference.Provenance()
	if err != nil {
		return nil, err
	}
	if provenance.OriginalSHA256 != fmt.Sprintf("%x", sha256.Sum256(ir)) {
		return nil, errors.New("cudalowering: original IR identity differs")
	}
	return verifySlotOrigins(*provenance, ptx, sass)
}

// VerifyCompiledSharedConversions uses the actual CompileWithOptions-retained
// input origins and checks its copied PTX digest. The first supported contract
// requires exactly one inferred input. SASS remains caller-paired text; native
// artifact association and debug-location qualification require a real receipt.
func VerifyCompiledSharedConversions(result *nvvmx.Result, sass []byte) (*Report, error) {
	provenance, err := result.AddressSpaceProvenance()
	if err != nil {
		return nil, err
	}
	if len(provenance.Inputs) != 1 {
		return nil, errors.New("cudalowering: slot attribution requires one inferred input")
	}
	return verifySlotOrigins(provenance.Inputs[0].Provenance, result.PTX, sass)
}

func verifySlotOrigins(provenance nvvmx.SlotProvenance, ptx, sass []byte) (*Report, error) {
	report := &Report{}
	if len(provenance.Slots) > 65536 || len(ptx) == 0 || len(sass) == 0 || len(ptx) > 4<<20 || len(sass) > 4<<20 || len(provenance.Slots) == 0 {
		return report, errors.New("cudalowering: missing or oversized slot evidence")
	}
	m, err := cudaptx.Parse(bytes.NewReader(ptx))
	if err != nil {
		return report, err
	}
	d, err := cudasass.Parse(bytes.NewReader(sass))
	if err != nil {
		return report, err
	}
	if m.Target != "sm_121" || len(m.TargetDeclarations) != 1 || d.Target != "sm_121" {
		return report, errors.New("cudalowering: slot attribution requires both exact sm_121")
	}
	pairs := make(map[string]Pair)
	for _, pair := range Match(m, d).Pairs {
		pairs[pair.PTX.Name] = pair
	}
	type useRef struct{ slot, use int }
	locations := make(map[nvvmx.SourceLocation][]useRef)
	functions := make(map[string]bool)
	ptxSHA, sassSHA := fmt.Sprintf("%x", sha256.Sum256(ptx)), fmt.Sprintf("%x", sha256.Sum256(sass))
	var unavailable []error
	fail := func(slot int, reason string) {
		report.Slots[slot].Reason = reason
		unavailable = append(unavailable, fmt.Errorf("cudalowering: %s: %s", report.Slots[slot].Slot, reason))
	}
	seenSlots := make(map[string]bool)
	for i, origin := range provenance.Slots {
		if seenSlots[origin.Slot] {
			return report, errors.New("cudalowering: duplicate producer slot identity")
		}
		seenSlots[origin.Slot] = true
		evidence := SlotEvidence{Slot: origin.Slot, OriginalSHA256: provenance.OriginalSHA256, PreparedSHA256: provenance.PreparedSHA256, PTXSHA256: ptxSHA, SASSSHA256: sassSHA}
		for _, use := range origin.Uses {
			evidence.Uses = append(evidence.Uses, SlotUseEvidence{IRLine: use.IRLine, Value: use.Value})
		}
		report.Slots = append(report.Slots, evidence)
		function, _, ok := strings.Cut(origin.Slot, "/")
		if !ok || !strings.HasPrefix(function, "@") || origin.Reason != "" || len(origin.Uses) == 0 {
			fail(i, "unsupported origin: "+origin.Reason)
			continue
		}
		function = strings.TrimPrefix(function, "@")
		functions[function] = true
		for j, use := range origin.Uses {
			loc := use.Location
			if loc.Function != function || loc.File == "" || loc.Line < 1 || loc.Column < 1 {
				fail(i, "terminal use lacks a direct resolved location")
				continue
			}
			locations[loc] = append(locations[loc], useRef{i, j})
		}
	}
	names := make([]string, 0, len(functions))
	for name := range functions {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		pair, ok := pairs[name]
		if !ok {
			for i, origin := range provenance.Slots {
				if strings.HasPrefix(origin.Slot, "@"+name+"/") {
					fail(i, "no unique matched executable kernel")
				}
			}
			continue
		}
		nops, interior := countNOPs(pair.SASS)
		report.Stats = append(report.Stats, Stats{Kernel: name, PTXInstructions: len(pair.PTX.Instructions), SASSInstructions: len(pair.SASS.Instructions), NOPs: nops, InteriorNOPs: interior, PeakGPR: peakGPR(pair.SASS)})
		scan := scanSharedConversions(pair.SASS)
		if len(scan.refusals) != 0 {
			unavailable = append(unavailable, fmt.Errorf("cudalowering: %s: %s", name, strings.Join(scan.refusals, "; ")))
		}
		if len(pair.PTX.Instructions) > 65536 {
			return report, errors.New("cudalowering: PTX instruction budget exceeded")
		}
		lines := make(map[int][]cudaptx.Instruction)
		for _, inst := range pair.PTX.Instructions {
			lines[inst.Line] = append(lines[inst.Line], inst)
		}
		pcs := make(map[uint64]cudasass.Instruction)
		for _, inst := range pair.SASS.Instructions {
			pcs[inst.Address] = inst
		}
		locate := func(pc uint64) (useRef, bool, error) {
			inst := pcs[pc]
			candidates := lines[inst.PTXLine]
			if inst.PTXLine < 1 || len(candidates) != 1 {
				return useRef{}, false, fmt.Errorf("0x%x lacks a unique physical PTX line", pc)
			}
			source := candidates[0]
			loc, direct := slotSourceLocation(name, source)
			if !direct || inst.InlinedFile != "" || inst.InlinedLine != 0 {
				return useRef{}, false, fmt.Errorf("0x%x lacks a direct PTX source location", pc)
			}
			if (inst.SourceFile != "" || inst.SourceLine != 0) && (inst.SourceFile != source.Loc.File || inst.SourceLine != source.Loc.Line) {
				return useRef{}, false, fmt.Errorf("0x%x conflicting source and PTX location", pc)
			}
			refs := locations[loc]
			if len(refs) == 0 {
				return useRef{}, false, nil
			}
			if len(refs) != 1 {
				return useRef{}, false, fmt.Errorf("0x%x ambiguous slot origin", pc)
			}
			ref := refs[0]
			use := provenance.Slots[ref.slot].Uses[ref.use]
			want := "ld"
			if use.Operation == "store" {
				want = "st"
			}
			if source.Mnemonic != want {
				return useRef{}, false, fmt.Errorf("0x%x memory operation conflicts with IR use", pc)
			}
			op := strings.ToUpper(inst.Opcode)
			if (want == "ld" && op != "LDS" && op != "LD.E") || (want == "st" && op != "STS" && op != "ST.E") {
				return useRef{}, false, fmt.Errorf("0x%x SASS memory operation conflicts with IR use", pc)
			}
			return ref, true, nil
		}
		for _, pc := range scan.sharedAccesses {
			ref, found, e := locate(pc)
			if e != nil {
				unavailable = append(unavailable, e)
				continue
			}
			if !found {
				continue
			}
			inst := lines[pcs[pc].PTXLine][0]
			if !strings.HasPrefix(inst.Opcode, inst.Mnemonic+".shared.") {
				fail(ref.slot, "direct SASS access lacks direct shared PTX")
				continue
			}
			use := &report.Slots[ref.slot].Uses[ref.use]
			use.SharedPCs = append(use.SharedPCs, pc)
			use.PTXLines = append(use.PTXLines, inst.Line)
		}
		for _, chain := range scan.chains {
			ref, found, e := locate(chain.genericAccess)
			if e != nil || !found {
				unavailable = append(unavailable, fmt.Errorf("cudalowering: unattributed conversion at 0x%x: %v", chain.genericAccess, e))
				continue
			}
			use := &report.Slots[ref.slot].Uses[ref.use]
			use.ConversionPCs = append(use.ConversionPCs, chain.genericAccess)
			use.PTXLines = append(use.PTXLines, pcs[chain.genericAccess].PTXLine)
			report.Divergences = append(report.Divergences, Divergence{Kind: SharedConversion, Kernel: name, Slot: report.Slots[ref.slot].Slot, IRLine: use.IRLine, SASSAddress: chain.genericAccess, SASSOpcode: "conversion", SASSCount: 1, Detail: chain.String()})
		}
		// PTX-only conversions also need a unique actual origin. Counts never stand
		// in for the SASS reaching-value witnesses or complete use coverage.
		for _, inst := range pair.PTX.Instructions {
			if !strings.HasPrefix(inst.Opcode, "cvta.shared.") && !strings.HasPrefix(inst.Opcode, "cvta.to.shared.") {
				continue
			}
			loc, direct := slotSourceLocation(name, inst)
			if !direct {
				unavailable = append(unavailable, fmt.Errorf("cudalowering: PTX conversion line %d lacks a direct source location", inst.Line))
				continue
			}
			refs := locations[loc]
			if len(refs) != 1 {
				unavailable = append(unavailable, fmt.Errorf("cudalowering: PTX conversion line %d has no unique slot", inst.Line))
				continue
			}
			ref := refs[0]
			use := &report.Slots[ref.slot].Uses[ref.use]
			report.Divergences = append(report.Divergences, Divergence{Kind: SharedConversion, Kernel: name, Slot: report.Slots[ref.slot].Slot, IRLine: use.IRLine, PTXOpcode: inst.Opcode, PTXCount: 1})
		}
	}
	for i := range report.Slots {
		evidence := &report.Slots[i]
		converted := false
		for _, use := range evidence.Uses {
			if len(use.SharedPCs) == 0 && len(use.ConversionPCs) == 0 {
				fail(i, "terminal use missing or elided in observed SASS")
			}
			converted = converted || len(use.ConversionPCs) > 0
		}
		for _, finding := range report.Divergences {
			if finding.Slot == evidence.Slot {
				converted = true
			}
		}
		evidence.Verified = evidence.Reason == "" && !converted && len(unavailable) == 0
	}
	if len(unavailable) != 0 {
		for i := range report.Slots {
			report.Slots[i].Verified = false
		}
	}
	return report, errors.Join(unavailable...)
}

// slotSourceLocation excludes unsupported inline and unresolved source forms for
// every instruction that participates in a slot attribution.
func slotSourceLocation(function string, inst cudaptx.Instruction) (nvvmx.SourceLocation, bool) {
	if len(inst.InlinedAt) != 0 || inst.Loc.FunctionName != "" || inst.Loc.File == "" || inst.Loc.Line < 1 || inst.Loc.Column < 1 {
		return nvvmx.SourceLocation{}, false
	}
	return nvvmx.SourceLocation{Function: function, File: inst.Loc.File, Line: inst.Loc.Line, Column: inst.Loc.Column}, true
}
