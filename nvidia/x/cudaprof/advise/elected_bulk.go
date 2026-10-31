package advise

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// electedBulkLoop is a static compiler pattern, not a measured stall or an
// executed-lane count. Keep this detector limited to the explicit target and
// straight-line loop shape witnessed in the retained sm90a compiler control.
type electedBulkLoop struct {
	copy       cudasass.Instruction
	start, end uint64
}

var electedBulkUniformRegister = regexp.MustCompile(`^UR[0-9]+$`)
var electedBulkAddresses = regexp.MustCompile(`\[(UR[0-9]+)\]`)

func electedBulkLoops(fn *cudasass.Function) []electedBulkLoop {
	cfg, err := cudasass.NewCFG(fn)
	if err != nil || cfg.IsDegenerate {
		return nil
	}
	var loops []electedBulkLoop
	seen := make(map[uint64]bool)
	for end, branch := range fn.Instructions {
		if branch.Opcode != "BRA.U.ANY" {
			continue
		}
		target, err := strconv.ParseUint(strings.TrimSpace(branch.Operands), 0, 64)
		if err != nil || target >= branch.Address || branch.Address-target > 512 {
			continue
		}
		start := end
		for start > 0 && fn.Instructions[start-1].Address >= target {
			start--
		}
		if fn.Instructions[start].Address != target {
			continue
		}
		var elected, complex bool
		lastWrite := make(map[string]string)
		var copies []cudasass.Instruction
		for i := start; i < end; i++ {
			inst := fn.Instructions[i]
			// Reject unknown and control-transfer operations rather than
			// guessing that an unfamiliar opcode falls through.
			switch inst.Opcode {
			case "ELECT", "R2UR", "PLOP3.LUT", "UBLKCP.S.G", "UMOV":
			default:
				complex = true
			}
			if complex {
				break
			}
			if inst.Opcode == "ELECT" {
				elected = true
			}
			if inst.Opcode == "UBLKCP.S.G" && elected {
				matched := 0
				used := make(map[string]bool)
				for _, address := range electedBulkAddresses.FindAllStringSubmatch(inst.Operands, -1) {
					used[address[1]] = true
				}
				for r := range used {
					if lastWrite[r] == "R2UR" {
						matched++
					}
				}
				if matched >= 2 {
					copies = append(copies, inst)
				}
			}
			dest, _, ok := strings.Cut(inst.Operands, ",")
			dest = strings.TrimSpace(dest)
			if ok && electedBulkUniformRegister.MatchString(dest) {
				lastWrite[dest] = inst.Opcode
			}
		}
		if complex || len(copies) != 1 {
			continue
		}
		copy := copies[0]
		if seen[copy.Address] {
			continue
		}
		seen[copy.Address] = true
		loops = append(loops, electedBulkLoop{copy, target, branch.Address})
	}
	return loops
}

func auditElectedBulkCopies(report *Report, disasm *cudasass.Disassembly, modules map[*cudasass.Function]string) {
	// No target inference from host device, kernel name, or opcode spelling.
	// Other targets need their own compiler/control validation before enabling.
	if report == nil || disasm == nil || disasm.Target != "sm_90a" {
		return
	}
	for _, fn := range disasm.Functions {
		loops := electedBulkLoops(fn)
		if len(loops) == 0 {
			continue
		}
		first := loops[0]
		report.Items = append(report.Items, Item{
			rule:                     "elected-bulk-copy",
			NoRuntimeAttribution:     true,
			Severity:                 SeverityContext,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceSuspected,
			Location:                 Location{Kernel: fn.Name, Module: modules[fn], PC: first.copy.Address, File: first.copy.SourceFile, Line: first.copy.SourceLine},
			Finding:                  fmt.Sprintf("Static sm_90a elected bulk-copy pattern in kernel %q: %d straight-line ELECT/R2UR/UBLKCP/backedge loops (first copy at 0x%x, loop 0x%x-0x%x). Register-to-uniform address gathers may turn lane-varying bulk-copy requests into repeated elected issuance; static code establishes neither executed lanes nor latency cost.", fn.Name, len(loops), first.copy.Address, first.start, first.end),
			ActionableRecommendation: "Inspect whether these requests describe a contiguous or affine tile. Compare a vectorized per-thread loader with tile-level TMA only where the layout permits, preserving barrier, lifetime, alignment and proxy-visibility contracts. Require output/graph agreement before keeping a loader change; this pattern alone does not establish a bottleneck.",
			NextMeasurement:          "Retain the exact target image and matched source, then correctness-gated unprofiled paired loader timings plus complete invocation-bound PC/issue/memory evidence. Kernel context time, if available, is not time caused by this pattern.",
		})
	}
}
