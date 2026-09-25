package advise

import (
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

var sassRegisterRE = regexp.MustCompile(`\bR[0-9]+\b`)

func auditUncoalescedMemory(cap *cudacapture.Capture, report *Report, disasm *cudasass.Disassembly) {
	if cap == nil {
		return
	}

	// A source-fusion failure zeroes every instruction's SourceFile and
	// SourceLine, which are exactly the fields the findings below cite. Left
	// undeclared, each finding reads as a location the analysis could not
	// determine rather than one the capture refused to bind.
	code, fusion, _ := capturefiles.LoadCode(cap)
	if fusion != nil {
		report.SourceFusion = fusion.Error()
	}

	reportedLocations := make(map[string]bool)

	// 1. Correlate SASS metrics / PC sampling stalls
	if cap.Profile != nil && len(cap.Profile.Sample) > 0 {
		var stallIndices []int
		var memThrottleIndices []int
		// memThrottleNames records which sample types the ratio is actually
		// computed from, because the finding below has to cite them by name.
		// The disjunction is historical: only "stall_mem" exists in a profile
		// this tree produces, and a finding that named one of the other two
		// would cite a counter no capture contains.
		var memThrottleNames []string
		var scoreboardIndices []int
		var branchIndices []int
		var waitIndices []int
		for i, st := range cap.Profile.SampleType {
			if strings.HasPrefix(st.Type, "stall_") {
				stallIndices = append(stallIndices, i)
			}
			tLower := strings.ToLower(st.Type)
			if tLower == "stall_memory_throttle" || tLower == "stall_mem_throttle" || tLower == "stall_mem" {
				memThrottleIndices = append(memThrottleIndices, i)
				memThrottleNames = append(memThrottleNames, st.Type)
			} else if tLower == "stall_scoreboard" {
				scoreboardIndices = append(scoreboardIndices, i)
			} else if tLower == "stall_branch" {
				branchIndices = append(branchIndices, i)
			} else if tLower == "stall_wait" || tLower == "stall_barrier" {
				waitIndices = append(waitIndices, i)
			}
		}
		memSource := strings.Join(memThrottleNames, "+")
		if memSource == "" {
			memSource = "sass.stall labels"
		}

		type pcStats struct {
			totalStalls      int64
			memStalls        int64
			scoreboardStalls int64
			branchStalls     int64
			waitStalls       int64
			opcode           string
			file             string
			line             int
		}
		byKernelPC := make(map[string]map[uint64]*pcStats)

		for _, s := range cap.Profile.Sample {
			var totalStall, memStall, scoreboardStall, branchStall, waitStall int64
			for _, idx := range stallIndices {
				totalStall += s.Value[idx]
			}
			for _, idx := range memThrottleIndices {
				memStall += s.Value[idx]
			}
			for _, idx := range scoreboardIndices {
				scoreboardStall += s.Value[idx]
			}
			for _, idx := range branchIndices {
				branchStall += s.Value[idx]
			}
			for _, idx := range waitIndices {
				waitStall += s.Value[idx]
			}

			if memStall == 0 && scoreboardStall == 0 && branchStall == 0 && waitStall == 0 && s.Label != nil {
				stallReason := ""
				if len(s.Label["sass.stall"]) > 0 {
					stallReason = s.Label["sass.stall"][0]
				} else if len(s.Label["stall"]) > 0 {
					stallReason = s.Label["stall"][0]
				}
				rLower := strings.ToLower(stallReason)
				if strings.Contains(rLower, "memory_throttle") || strings.Contains(rLower, "mem_throttle") || strings.Contains(rLower, "stall_mem") {
					if len(s.Value) > 0 {
						memStall = s.Value[0]
					}
				} else if strings.Contains(rLower, "scoreboard") {
					if len(s.Value) > 0 {
						scoreboardStall = s.Value[0]
					}
				} else if strings.Contains(rLower, "branch") {
					if len(s.Value) > 0 {
						branchStall = s.Value[0]
					}
				} else if strings.Contains(rLower, "wait") || strings.Contains(rLower, "barrier") {
					if len(s.Value) > 0 {
						waitStall = s.Value[0]
					}
				}
				if totalStall == 0 && (memStall > 0 || scoreboardStall > 0 || branchStall > 0 || waitStall > 0) && len(s.Value) > 0 {
					totalStall = s.Value[0]
				}
			}

			if totalStall == 0 && memStall == 0 && scoreboardStall == 0 && branchStall == 0 && waitStall == 0 {
				continue
			}

			var kName string
			var pc uint64
			var file string
			var line int
			if len(s.Location) > 0 {
				kName, pc = capturefiles.LocationPC(s.Location[0], s, code, disasm)
				if len(s.Location[0].Line) > 0 && s.Location[0].Line[0].Function != nil {
					file = s.Location[0].Line[0].Function.Filename
					line = int(s.Location[0].Line[0].Line)
				}
			}
			if kName == "" && s.Label != nil && len(s.Label["kernel"]) > 0 {
				kName = s.Label["kernel"][0]
			}
			if kName == "" && disasm != nil && len(disasm.Functions) == 1 {
				kName = disasm.Functions[0].Name
			}
			if kName == "" {
				continue
			}

			opcode := ""
			if s.Label != nil {
				if len(s.Label["sass.opcode"]) > 0 {
					opcode = s.Label["sass.opcode"][0]
				} else if len(s.Label["opcode"]) > 0 {
					opcode = s.Label["opcode"][0]
				}
			}

			if byKernelPC[kName] == nil {
				byKernelPC[kName] = make(map[uint64]*pcStats)
			}
			stat := byKernelPC[kName][pc]
			if stat == nil {
				stat = &pcStats{opcode: opcode, file: file, line: line}
				byKernelPC[kName][pc] = stat
			}
			stat.totalStalls += totalStall
			stat.memStalls += memStall
			stat.scoreboardStalls += scoreboardStall
			stat.branchStalls += branchStall
			stat.waitStalls += waitStall
			if stat.opcode == "" && opcode != "" {
				stat.opcode = opcode
			}
		}

		for kName, pcMap := range byKernelPC {
			var fn *cudasass.Function
			if disasm != nil {
				fn = disasm.Function(kName)
				if fn == nil && len(disasm.Functions) == 1 {
					fn = disasm.Functions[0]
				}
			}

			type stallCand struct {
				pc             uint64
				opcode         string
				file           string
				line           int
				branchPC       uint64
				branchOpcode   string
				branchFound    bool
				defPC          uint64
				usePC          uint64
				defUseDistance int
				defRegister    string
				ratio          float64
				stalls         int64
				sev            Severity
			}
			var memCands []stallCand
			var sbCands []stallCand
			var brCands []stallCand
			var wtCands []stallCand

			for pc, stat := range pcMap {
				if stat.totalStalls <= 0 {
					continue
				}

				opcode := stat.opcode
				file := stat.file
				line := stat.line
				if fn != nil {
					for _, in := range fn.Instructions {
						if in.Address == pc {
							opcode = in.Opcode
							if in.SourceFile != "" {
								file = in.SourceFile
								line = in.SourceLine
							}
							break
						}
					}
				}

				// Memory stalls
				if stat.memStalls > 0 {
					ratio := float64(stat.memStalls) / float64(stat.totalStalls)
					if ratio > 0.35 {
						opUpper := strings.ToUpper(opcode)
						if strings.HasPrefix(opUpper, "LDG.E") || strings.HasPrefix(opUpper, "LDG") || strings.Contains(opUpper, "LDG") {
							sev := SeverityCost
							if ratio >= 0.50 {
								sev = SeverityReadFirst
							}
							memCands = append(memCands, stallCand{
								pc: pc, opcode: opcode, file: file, line: line,
								ratio: ratio, stalls: stat.memStalls, sev: sev,
							})
						}
					}
				}

				// Scoreboard dependency stalls
				if stat.scoreboardStalls > 0 {
					sbRatio := float64(stat.scoreboardStalls) / float64(stat.totalStalls)
					if sbRatio >= 0.25 && stat.scoreboardStalls >= 50 {
						sev := SeverityCost
						if sbRatio >= 0.50 {
							sev = SeverityReadFirst
						}
						defPC, usePC, distance, register := nearestLoadUse(fn, pc)
						sbCands = append(sbCands, stallCand{
							pc: pc, opcode: opcode, file: file, line: line,
							defPC: defPC, usePC: usePC, defUseDistance: distance, defRegister: register,
							ratio: sbRatio, stalls: stat.scoreboardStalls, sev: sev,
						})
					}
				}

				// Branch divergence stalls
				if stat.branchStalls > 0 {
					brRatio := float64(stat.branchStalls) / float64(stat.totalStalls)
					if brRatio >= 0.20 && stat.branchStalls >= 50 {
						sev := SeverityCost
						if brRatio >= 0.50 {
							sev = SeverityReadFirst
						}
						branchPC, branchOpcode := precedingControlFlowBranch(fn, pc)
						brCands = append(brCands, stallCand{
							pc: pc, opcode: opcode, file: file, line: line,
							branchPC: branchPC, branchOpcode: branchOpcode,
							branchFound: branchOpcode != "",
							ratio:       brRatio, stalls: stat.branchStalls, sev: sev,
						})
					}
				}

				// Barrier wait stalls
				if stat.waitStalls > 0 {
					wtRatio := float64(stat.waitStalls) / float64(stat.totalStalls)
					if wtRatio >= 0.35 && stat.waitStalls >= 50 {
						wtCands = append(wtCands, stallCand{
							pc: pc, opcode: opcode, file: file, line: line,
							ratio: wtRatio, stalls: stat.waitStalls, sev: SeverityCost,
						})
					}
				}
			}

			// Report Memory Stalls (capped to 3 individual locations + 1 summary)
			sort.Slice(memCands, func(i, j int) bool { return memCands[i].stalls > memCands[j].stalls })
			limit := len(memCands)
			if limit > 3 {
				// Many adjacent PCs in an unrolled loop are one diagnosis, not
				// separate optimization opportunities. Keep the detailed rows only
				// when the sample set is small enough to remain interpretable.
				limit = 0
			}
			for _, c := range memCands[:limit] {
				recommendation := "Check whether adjacent lane IDs access contiguous bytes within a single 128B cache line; if they already do, the stall is memory latency or bandwidth rather than transaction count, and the fix is reuse or a wider access, not realignment."
				nextMeasurement := "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions"
				if reduction, ok := reductionLayoutAdvice(kName, fn); ok {
					recommendation = reduction
					nextMeasurement = "measure sectors per request and compare a transposed/SoA tile against shared-memory or Tensor Core MMA staging"
				}
				locKey := fmt.Sprintf("%s:0x%x", kName, c.pc)
				reportedLocations[locKey] = true
				report.Items = append(report.Items, Item{
					Severity:      c.sev,
					Category:      CategoryMemory,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: kName,
						PC:     c.pc,
						File:   c.file,
						Line:   c.line,
					},
					Finding:                  fmt.Sprintf("Memory stalls dominate this load: %s is %.1f%% of stalls (over 35%%) on %s at PC 0x%04x in kernel %q", memSource, c.ratio*100, instructionName(c.opcode), c.pc, kName),
					ActionableRecommendation: recommendation,
					NextMeasurement:          nextMeasurement,
				})
			}
			if len(memCands) > 3 {
				var remainingStalls int64
				var maxRatio float64
				var minPC, maxPC uint64
				for i, c := range memCands {
					remainingStalls += c.stalls
					if c.ratio > maxRatio {
						maxRatio = c.ratio
					}
					if i == 0 || c.pc < minPC {
						minPC = c.pc
					}
					if c.pc > maxPC {
						maxPC = c.pc
					}
				}
				report.Items = append(report.Items, Item{
					Severity:      SeverityCost,
					Category:      CategoryMemory,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: kName,
					},
					Finding:                  fmt.Sprintf("Loop/control-flow region PC 0x%04x-0x%04x: %d load instructions share memory stalls in kernel %q (up to %.1f%% of stalls, %d total stall cycles)", minPC, maxPC, len(memCands), kName, maxRatio*100, remainingStalls),
					ActionableRecommendation: memoryRegionRecommendation(kName, fn),
					NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
				})
			}

			// Report Scoreboard Dependency Stalls (capped to 3)
			sort.Slice(sbCands, func(i, j int) bool { return sbCands[i].stalls > sbCands[j].stalls })
			sbLimit := len(sbCands)
			if sbLimit > 3 {
				sbLimit = 0
			}
			for _, c := range sbCands[:sbLimit] {
				distance := ""
				if c.defUseDistance > 0 {
					distance = fmt.Sprintf("; %s loaded at PC 0x%04x is first consumed at PC 0x%04x (%d instructions later)", c.defRegister, c.defPC, c.usePC, c.defUseDistance)
				}
				report.Items = append(report.Items, Item{
					Severity:      c.sev,
					Category:      CategoryMemoryCompute,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: kName,
						PC:     c.pc,
						File:   c.file,
						Line:   c.line,
					},
					Finding:                  fmt.Sprintf("Register scoreboard latency stall: stall_scoreboard is %.1f%% of stalls on %s at PC 0x%04x in kernel %q (%d stall cycles)%s", c.ratio*100, instructionName(c.opcode), c.pc, kName, c.stalls, distance),
					ActionableRecommendation: "Instruction is stalled waiting for register data from a pending memory load or high-latency operation. Use the measured def-use distance to increase instruction-level parallelism (ILP), hoist memory loads further ahead of first use, or use double buffering / software pipelining. Treat the distance as a diagnostic, not a universal architecture threshold.",
					NextMeasurement:          "stall_scoreboard sample reduction or A/B wall-clock timing",
				})
			}
			if len(sbCands) > 3 {
				var total int64
				var minPC, maxPC uint64
				var maxRatio float64
				for i, c := range sbCands {
					total += c.stalls
					if i == 0 || c.pc < minPC {
						minPC = c.pc
					}
					if c.pc > maxPC {
						maxPC = c.pc
					}
					if c.ratio > maxRatio {
						maxRatio = c.ratio
					}
				}
				report.Items = append(report.Items, Item{
					Severity:                 SeverityCost,
					Category:                 CategoryMemoryCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 Location{Kernel: kName, PC: minPC},
					Finding:                  fmt.Sprintf("Loop/control-flow region PC 0x%04x-0x%04x: %d instructions share register scoreboard stalls in kernel %q (up to %.1f%% of stalls, %d total stall cycles)", minPC, maxPC, len(sbCands), kName, maxRatio*100, total),
					ActionableRecommendation: "Treat these repeated PCs as one dependency pattern: increase the distance between independent loads and uses, double-buffer the tile, or software-pipeline the loop before optimizing individual instructions.",
					NextMeasurement:          "compare grouped stall cycles and def-use distances before and after loop pipelining",
				})
			}

			// Report Branch Divergence Stalls (capped to 3)
			sort.Slice(brCands, func(i, j int) bool { return brCands[i].stalls > brCands[j].stalls })
			brLimit := len(brCands)
			if brLimit > 3 {
				brLimit = 0
			}
			for _, c := range brCands[:brLimit] {
				locationPC := c.pc
				attribution := fmt.Sprintf("sampled at %s", c.opcode)
				if c.branchFound {
					locationPC = c.branchPC
					attribution = fmt.Sprintf("created by preceding %s at PC 0x%04x; sampled at %s at PC 0x%04x, which is awaiting warp reconvergence", c.branchOpcode, c.branchPC, c.opcode, c.pc)
				}
				report.Items = append(report.Items, Item{
					Severity:      c.sev,
					Category:      CategoryArchitectureCompute,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: kName,
						PC:     locationPC,
						File:   c.file,
						Line:   c.line,
					},
					Finding:                  fmt.Sprintf("Warp branch divergence: stall_branch is %.1f%% of stalls in kernel %q (%d stall cycles), %s", c.ratio*100, kName, c.stalls, attribution),
					ActionableRecommendation: "Warp execution path divergence is attributed to the control-flow branch, not the sampled successor instruction. Replace the branch with branchless predicated selects (e.g. SELP / fmaxf) only when the condition is semantically safe, or make the branch condition uniform across each warp.",
					NextMeasurement:          "sm__branch_targets_threads_divergent.sum or A/B wall-clock timing",
				})
			}
			if len(brCands) > 3 {
				var total int64
				var minPC, maxPC uint64
				var maxRatio float64
				for i, c := range brCands {
					total += c.stalls
					if i == 0 || c.pc < minPC {
						minPC = c.pc
					}
					if c.pc > maxPC {
						maxPC = c.pc
					}
					if c.ratio > maxRatio {
						maxRatio = c.ratio
					}
				}
				report.Items = append(report.Items, Item{
					Severity:                 SeverityCost,
					Category:                 CategoryArchitectureCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 Location{Kernel: kName, PC: minPC},
					Finding:                  fmt.Sprintf("Loop/control-flow region PC 0x%04x-0x%04x: %d sampled successor instructions share reconvergence stalls in kernel %q (up to %.1f%% of stalls, %d total stall cycles)", minPC, maxPC, len(brCands), kName, maxRatio*100, total),
					ActionableRecommendation: "Treat these repeated successor samples as one control-flow diagnosis: inspect the corresponding branch targets for warp divergence, make predicates uniform where possible, or use branchless selects only when semantically safe.",
					NextMeasurement:          "sm__branch_targets_threads_divergent.sum and a control-flow A/B wall-clock comparison",
				})
			}

			// Report Barrier Synchronization Stalls (capped to 3)
			sort.Slice(wtCands, func(i, j int) bool { return wtCands[i].stalls > wtCands[j].stalls })
			wtLimit := len(wtCands)
			if wtLimit > 3 {
				wtLimit = 3
			}
			for _, c := range wtCands[:wtLimit] {
				report.Items = append(report.Items, Item{
					Severity:      c.sev,
					Category:      CategoryArchitectureDispatch,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: kName,
						PC:     c.pc,
						File:   c.file,
						Line:   c.line,
					},
					Finding:                  fmt.Sprintf("Barrier synchronization latency: stall_wait is %.1f%% of stalls on %s at PC 0x%04x in kernel %q (%d stall cycles)", c.ratio*100, instructionName(c.opcode), c.pc, kName, c.stalls),
					ActionableRecommendation: "Threads are stalled waiting at a block barrier (__syncthreads / BAR.SYNC). Check for workload imbalance across warps before the barrier, or reduce barrier synchronization frequency.",
					NextMeasurement:          "A/B wall-clock timing with reduced barrier synchronization",
				})
			}
		}
	}

	// 2. Address arithmetic exhibiting non-contiguous strides
	remarks, _, _ := capturefiles.Remarks(cap)
	for _, rmk := range remarks {
		rLower := strings.ToLower(rmk.Reason)
		remLower := strings.ToLower(rmk.Remark)
		nameLower := strings.ToLower(rmk.Name)
		if (strings.Contains(rLower, "non-contiguous") || strings.Contains(rLower, "stride") ||
			strings.Contains(remLower, "uncoalesced") || strings.Contains(nameLower, "uncoalesced")) &&
			!rmk.IsTMAFailure() {
			locKey := fmt.Sprintf("%s:%s:%d", rmk.Function, rmk.File, rmk.Line)
			if !reportedLocations[locKey] {
				reportedLocations[locKey] = true
				finding := "Threads within the warp generate strided or disparate 128-byte cache line requests, generating up to 32 cache transactions per instruction"
				if rmk.Reason != "" {
					finding += fmt.Sprintf(" (reason: %s)", rmk.Reason)
				}
				report.Items = append(report.Items, Item{
					Severity:      SeverityCost,
					Category:      CategoryMemory,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: rmk.Function,
						File:   rmk.File,
						Line:   rmk.Line,
					},
					Finding:                  finding,
					ActionableRecommendation: "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line.",
					NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
				})
			}
		}
	}

	if cap.Profile != nil {
		for _, s := range cap.Profile.Sample {
			kName := extractKernelNameFromSample(s)
			isUncoalesced := false
			strideVal := ""
			if s.Label != nil {
				if len(s.Label["uncoalesced"]) > 0 && s.Label["uncoalesced"][0] == "true" {
					isUncoalesced = true
				}
				if len(s.Label["stride"]) > 0 {
					strideVal = s.Label["stride"][0]
					if strideVal != "1" && strideVal != "" {
						isUncoalesced = true
					}
				}
			}
			if isUncoalesced && kName != "" {
				locKey := fmt.Sprintf("%s:uncoalesced", kName)
				if !reportedLocations[locKey] {
					reportedLocations[locKey] = true
					finding := fmt.Sprintf("Threads within the warp generate strided or disparate 128-byte cache line requests, generating up to 32 cache transactions per instruction (address arithmetic exhibits non-contiguous stride in kernel %q)", kName)
					if strideVal != "" {
						finding = fmt.Sprintf("Threads within the warp generate strided or disparate 128-byte cache line requests, generating up to 32 cache transactions per instruction (stride %s in kernel %q)", strideVal, kName)
					}
					report.Items = append(report.Items, Item{
						Severity:      SeverityCost,
						Category:      CategoryMemory,
						EvidenceLevel: EvidenceSuspected,
						Location: Location{
							Kernel: kName,
						},
						Finding:                  finding,
						ActionableRecommendation: "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line.",
						NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
					})
				}
			}
		}
	}
}

// precedingControlFlowBranch identifies the static branch that most likely
// created a sampled reconvergence stall. PC sampling reports the instruction
// being issued while the warp waits; that instruction is commonly the branch
// successor, not the branch itself. Keep the correlation conservative: only a
// nearby branch in the same function is used, otherwise the sample remains
// un-attributed rather than inventing a control-flow cause.
func precedingControlFlowBranch(fn *cudasass.Function, pc uint64) (uint64, string) {
	if fn == nil {
		return 0, ""
	}
	var best cudasass.Instruction
	found := false
	for _, inst := range fn.Instructions {
		if inst.Address >= pc {
			break
		}
		base := strings.ToUpper(cudasass.BaseOpcode(inst.Opcode))
		if base == "BRA" || base == "BRX" || base == "CBRA" || base == "WARPSYNC" {
			best = inst
			found = true
		}
	}
	if !found || pc-best.Address > 0x200 {
		return 0, ""
	}
	return best.Address, best.Opcode
}

// nearestLoadUse reports the first use of a register defined by a global load
// before the sampled PC. It is intentionally a small SASS def-use heuristic:
// predication, reuse flags, and memory-level parallelism can make a raw
// instruction distance insufficient to establish causality.
func nearestLoadUse(fn *cudasass.Function, pc uint64) (defPC, usePC uint64, distance int, register string) {
	if fn == nil {
		return 0, 0, 0, ""
	}
	for i, inst := range fn.Instructions {
		if inst.Address >= pc {
			break
		}
		base := strings.ToUpper(cudasass.BaseOpcode(inst.Opcode))
		if base != "LDG" && base != "LD" {
			continue
		}
		regs := sassRegisterRE.FindAllString(inst.Operands, -1)
		if len(regs) == 0 {
			continue
		}
		reg := regs[0]
		for j := i + 1; j < len(fn.Instructions); j++ {
			next := fn.Instructions[j]
			if next.Address > pc {
				break
			}
			if slices.Contains(sassRegisterRE.FindAllString(next.Operands, -1), reg) {
				return inst.Address, next.Address, j - i, reg
			}
		}
	}
	return 0, 0, 0, ""
}

func reductionLayoutAdvice(kernel string, fn *cudasass.Function) (string, bool) {
	name := strings.ToLower(kernel)
	if !strings.Contains(name, "gather") && !strings.Contains(name, "reduce") && !strings.Contains(name, "dot") {
		return "", false
	}
	if fn == nil {
		return "", false
	}
	loads, reductions := 0, 0
	for _, inst := range fn.Instructions {
		base := strings.ToUpper(cudasass.BaseOpcode(inst.Opcode))
		if base == "LDG" || base == "LD" {
			loads++
		}
		if base == "FADD" || base == "FFMA" || base == "FMUL" || base == "IADD3" {
			reductions++
		}
	}
	if loads < 2 || reductions < 2 {
		return "", false
	}
	return "This looks like an inner-product reduction with strided expert/channel loads, not proof of a coalescing bug. Compare transposed or SoA-repacked weights, direct-to-register tile layouts, shared-memory double buffering, and Tensor Core MMA before recommending lane realignment.", true
}

func memoryRegionRecommendation(kernel string, fn *cudasass.Function) string {
	if advice, ok := reductionLayoutAdvice(kernel, fn); ok {
		return advice
	}
	return "If memory access is contiguous and aligned, vectorize memory loads to 128-bit (uint4/v4), unroll loops, and pipeline global memory loads across iterations to overlap transfer latency."
}

// instructionName renders the SASS opcode a finding is about. A capture with
// no disassembly and no sass.opcode label cannot name the instruction at all;
// the name is then absent, not empty, and must read as such. Rendering the
// empty string turned "on LDG at PC" into "on  at PC" and quietly demoted a
// Critical finding into one nobody could act on.
func instructionName(opcode string) string {
	if strings.TrimSpace(opcode) == "" {
		return "an unnamed instruction"
	}
	return opcode
}
