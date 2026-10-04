package cudalowering

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
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
	Kind       Kind
	Kernel     string // kernel name, as the level that has the kernel spells it
	PTXOpcode  string // PTX opcode prefix involved, if any
	SASSOpcode string // SASS opcode prefix involved, if any
	PTXCount   int    // matching PTX instructions
	SASSCount  int    // matching SASS instructions
	Detail     string // supporting measurement, such as peak live register count
}

func (d Divergence) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s: ", d.Kernel, d.Kind)
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
