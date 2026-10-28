package cudasass

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// RegisterFileSize is the number of 32-bit registers in one SM's register
// file on sm_90 through sm_121.
const RegisterFileSize = 65536

// A WarpGroupRegs is the register budget of one group of warps in a CTA,
// such as a producer or consumer warpgroup after setmaxnreg.
type WarpGroupRegs struct {
	Warps   int
	RegsPer int // registers per thread
}

// CheckRegisterFile reports whether the warp groups of one CTA fit the
// register file: Σ 32·Warps·round8(RegsPer) ≤ [RegisterFileSize], with
// registers counted at the per-thread allocation granularity of 8.
//
// It checks the arithmetic only. Whether more than one CTA fits an SM, and
// any per-warp allocation unit beyond the granularity of 8, are left to
// the occupancy calculator.
func CheckRegisterFile(groups []WarpGroupRegs) error {
	total := 0
	for _, g := range groups {
		if g.Warps <= 0 || g.RegsPer <= 0 || g.RegsPer > 255 {
			return fmt.Errorf("warp group %+v: want positive warps and 1..255 registers", g)
		}
		total += 32 * g.Warps * ((g.RegsPer + 7) &^ 7)
	}
	if total > RegisterFileSize {
		return fmt.Errorf("warp groups need %d registers, the register file holds %d", total, RegisterFileSize)
	}
	return nil
}

// A CubinAuditReport is what [ProveDisassembly] found for one kernel.
type CubinAuditReport struct {
	Kernel string
	Arch   string

	// UsedGPRs is the per-thread register count ptxas reported, or, when
	// GPRsFromSASS is set, one more than the highest GPR the SASS names,
	// which is a lower bound.
	UsedGPRs       int
	GPRsFromSASS   bool
	MaxAllowedGPRs int

	StackFrame  int // bytes per thread, from ptxas; -1 if not reported
	SpillStores int // bytes, from ptxas; -1 if not reported
	SpillLoads  int // bytes, from ptxas; -1 if not reported

	STL int // local-memory store instructions in the SASS
	LDL int // local-memory load instructions in the SASS

	// TensorMMA counts HMMA, IMMA, QMMA, OMMA, HGMMA (wgmma) and the
	// UTC*MMA forms (tcgen05.mma).
	TensorMMA int

	// RedundantWaits counts waits that cannot delay anything: a DEPBAR or
	// tensor-memory wait identical to the previous one in its basic block
	// with no memory, tensor or barrier-setting instruction between, and,
	// on sm_121a where control codes decode, a wait on a scoreboard that
	// no instruction in the block has set since the last wait on it.
	RedundantWaits int

	// PeakLiveGPRs is the largest live-GPR count nvdisasm's life-range
	// analysis reported, or 0 if the listing has none.
	PeakLiveGPRs int

	Violations []string
}

// OK reports whether the report has no violations.
func (r CubinAuditReport) OK() bool { return len(r.Violations) == 0 }

// A CubinConstraint is a pass/fail limit for [ProveDisassembly].
type CubinConstraint struct {
	MaxGPRs          int  // 0 means no limit
	ForbidSpills     bool // no STL, no LDL, no spill bytes and no stack frame
	RequireTensorMMA bool // at least one tensor-core MMA instruction

	// MaxRedundantWaits bounds RedundantWaits; a negative value means no
	// limit.
	MaxRedundantWaits int
}

// ErrConstraint is wrapped by the error [ProveDisassembly] and [ProveCubin]
// return when a kernel violates its constraint.
var ErrConstraint = errors.New("kernel violates its SASS constraint")

// ProveCubin disassembles cubin with nvdisasm and checks every kernel in it
// against c, taking register counts and spill sizes from ptxasLog, the
// output of ptxas -v for the same compilation (empty if unavailable). See
// [ProveDisassembly].
func ProveCubin(cubin []byte, ptxasLog string, c CubinConstraint) ([]CubinAuditReport, error) {
	d, err := DisassembleCUBIN(cubin)
	if err != nil {
		return nil, fmt.Errorf("disassemble cubin: %w", err)
	}
	var rep *ptxasinfo.Report
	if strings.TrimSpace(ptxasLog) != "" {
		rep, err = ptxasinfo.Parse(strings.NewReader(ptxasLog))
		if err != nil {
			return nil, fmt.Errorf("parse ptxas log: %w", err)
		}
	}
	return ProveDisassembly(d, rep, c)
}

// ProveDisassembly checks every function in d against c and returns one
// report per function, and an error wrapping [ErrConstraint] if any report
// has a violation.
//
// The checks are counts over the instruction stream, not a proof about
// executions: an STL in dead code is still an STL. A kernel ptxas reports
// spilling is a violation under ForbidSpills even if the SASS has no
// STL/LDL, and the reverse, since either source alone could be stale. rep
// may be nil; then spill sizes are unknown (-1), which ForbidSpills does not
// count as a spill, and register counts come from the SASS.
func ProveDisassembly(d *Disassembly, rep *ptxasinfo.Report, c CubinConstraint) ([]CubinAuditReport, error) {
	if d == nil {
		return nil, errors.New("no disassembly")
	}
	var out []CubinAuditReport
	var bad []string
	for _, fn := range d.Functions {
		r := auditFunction(fn, d.Target, rep)
		r.MaxAllowedGPRs = c.MaxGPRs
		r.Violations = violations(r, c)
		if !r.OK() {
			bad = append(bad, fn.Name)
		}
		out = append(out, r)
	}
	if len(bad) > 0 {
		return out, fmt.Errorf("%w: %s", ErrConstraint, strings.Join(bad, ", "))
	}
	return out, nil
}

func violations(r CubinAuditReport, c CubinConstraint) []string {
	var v []string
	if c.MaxGPRs > 0 && r.UsedGPRs > c.MaxGPRs {
		v = append(v, fmt.Sprintf("uses %d registers, limit %d", r.UsedGPRs, c.MaxGPRs))
	}
	if c.ForbidSpills {
		if r.STL > 0 || r.LDL > 0 {
			v = append(v, fmt.Sprintf("%d STL and %d LDL instructions", r.STL, r.LDL))
		}
		if r.SpillStores > 0 || r.SpillLoads > 0 {
			v = append(v, fmt.Sprintf("ptxas reports %d bytes spill stores, %d bytes spill loads", r.SpillStores, r.SpillLoads))
		}
		if r.StackFrame > 0 {
			v = append(v, fmt.Sprintf("%d bytes stack frame", r.StackFrame))
		}
	}
	if c.RequireTensorMMA && r.TensorMMA == 0 {
		v = append(v, "no tensor-core MMA instruction")
	}
	if c.MaxRedundantWaits >= 0 && r.RedundantWaits > c.MaxRedundantWaits {
		v = append(v, fmt.Sprintf("%d redundant waits, limit %d", r.RedundantWaits, c.MaxRedundantWaits))
	}
	return v
}

// tensorMMA reports whether a base opcode is a tensor-core MMA.
func tensorMMA(base string) bool {
	switch base {
	case "HMMA", "IMMA", "QMMA", "OMMA", "DMMA", "HGMMA", "IGMMA", "QGMMA", "BGMMA":
		return true
	}
	return strings.HasPrefix(base, "UTC") && strings.HasSuffix(base, "MMA")
}

// waitOpcode reports whether a base opcode only waits.
func waitOpcode(base string) bool {
	switch base {
	case "DEPBAR", "UTCBAR", "WARPGROUP":
		return true
	}
	return false
}

func auditFunction(fn *Function, target string, rep *ptxasinfo.Report) CubinAuditReport {
	r := CubinAuditReport{Kernel: fn.Name, Arch: target, StackFrame: -1, SpillStores: -1, SpillLoads: -1}
	var pf *ptxasinfo.Function
	switch {
	case rep == nil:
	case target != "":
		pf = rep.Lookup(fn.Name, target)
	default:
		// A listing that declares no target matches the function's only
		// report, and nothing if ptxas compiled it for several targets.
		for _, f := range rep.Functions() {
			if f.Name != fn.Name {
				continue
			}
			if pf != nil {
				pf = nil
				break
			}
			pf = f
		}
	}
	if pf != nil {
		r.UsedGPRs = pf.Registers
		r.StackFrame, r.SpillStores, r.SpillLoads = pf.StackFrame, pf.SpillStores, pf.SpillLoads
		if r.Arch == "" {
			r.Arch = pf.Arch
		}
	} else {
		r.GPRsFromSASS = true
	}
	maxReg := -1
	for _, inst := range fn.Instructions {
		base := BaseOpcode(inst.Opcode)
		switch {
		case base == "STL":
			r.STL++
		case base == "LDL":
			r.LDL++
		case tensorMMA(base):
			r.TensorMMA++
		}
		r.PeakLiveGPRs = max(r.PeakLiveGPRs, inst.LiveGPR)
		for _, tok := range splitOperands(inst.Operands) {
			for _, reg := range wideRegisters(tok, 1) {
				if reg.Class == GPR {
					maxReg = max(maxReg, reg.Num)
				}
			}
		}
	}
	if pf == nil {
		r.UsedGPRs = maxReg + 1
	}
	for _, b := range BasicBlocks(fn) {
		r.RedundantWaits += redundantWaits(b, Arch(target))
	}
	return r
}

// redundantWaits counts the waits in b that cannot delay anything; see
// [CubinAuditReport.RedundantWaits].
func redundantWaits(b BasicBlock, arch Arch) int {
	n := 0
	prev := ""
	// set[k] is true once barrier k may be pending: unknown at block entry,
	// so every barrier starts pending.
	set := [6]bool{true, true, true, true, true, true}
	for _, inst := range b.Instructions {
		base := BaseOpcode(inst.Opcode)
		if waitOpcode(base) {
			text := inst.Opcode + " " + inst.Operands
			if text == prev {
				n++
			}
			prev = text
		} else if c, ok := Classify(inst.Opcode); !ok || c.Loads || c.Stores || c.Fence || tensorMMA(base) {
			prev = ""
		}
		ctl, err := DecodeControl(inst.Encoding, arch)
		if err != nil {
			continue
		}
		for k := range 6 {
			if !ctl.Waits(k) {
				continue
			}
			if !set[k] {
				n++
			}
			set[k] = false
		}
		for _, k := range []uint8{ctl.WriteBarrier, ctl.ReadBarrier} {
			if k < 6 {
				set[k] = true
			}
		}
	}
	return n
}
