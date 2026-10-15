package advise

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// Analyze synthesizes diagnostics across compiler remarks, register pressure curves,
// and PTX architecture audits.
func Analyze(cap *cudacapture.Capture) (*Report, error) {
	return analyze(cap, nil)
}

// AnalyzeWithEvidence analyzes cap with caller-supplied artifact, compiler and
// workload provenance. source must describe the exact kernel entry. Its locator
// identifies the caller's retained association, not a hardware attestation.
//
// Numeric shares require a complete, loss-free journal agreeing with the
// profile's kernel-time population. Missing or contradictory capture facts
// leave the share Unread. The capture can check recorded image, target and
// workload facts, but cannot authenticate the supplied compiler key or prove
// that supplied records came from a physical device. Shares are never Measured.
// The caller must not mutate cap or its profile during either analysis call.
func AnalyzeWithEvidence(cap *cudacapture.Capture, source cudaevidence.Source) (*Report, error) {
	if err := source.Identity.Validate(); err != nil {
		return nil, fmt.Errorf("analyze evidence: %w", err)
	}
	if strings.TrimSpace(source.Producer) == "" || strings.TrimSpace(source.Locator) == "" {
		return nil, errors.New("analyze evidence: missing producer or locator")
	}
	if cap != nil && cap.Profile != nil {
		if err := cap.Profile.CheckValid(); err != nil {
			return nil, fmt.Errorf("analyze evidence profile: %w", err)
		}
		for _, t := range cap.Profile.SampleType {
			if t == nil {
				return nil, errors.New("analyze evidence profile: nil sample type")
			}
		}
	}
	return analyze(cap, &source)
}

func analyze(cap *cudacapture.Capture, source *cudaevidence.Source) (*Report, error) {
	if cap == nil {
		return nil, errors.New("nil capture")
	}

	report := &Report{
		Items: make([]Item, 0),
	}

	// 1. Compiler Remarks (via x/tileiras)
	remarks, _, _ := capturefiles.Remarks(cap)
	for _, rmk := range remarks {
		if rmk.IsTMAFailure() {
			finding := "TMA instruction selection failed"
			if rmk.Remark != "" {
				finding = rmk.Remark
			}
			if rmk.Reason != "" {
				finding += fmt.Sprintf(" (reason: %s)", rmk.Reason)
			}
			if rmk.Shape != "" {
				finding += fmt.Sprintf(" [shape: %s]", rmk.Shape)
			}

			rec := "Ensure tensor tile dimensions, alignments, and strides satisfy hardware TMA requirements (16-byte alignment and contiguous inner strides)."
			rLower := strings.ToLower(rmk.Reason)
			if strings.Contains(rLower, "stride") || strings.Contains(rLower, "contiguous") {
				rec = "Modify tensor layout or reshape matrix so the innermost dimension has contiguous unit stride matching TMA hardware requirements."
			} else if strings.Contains(rLower, "align") {
				rec = "Align global tensor memory base address and row stride to 16-byte boundaries required by TMA descriptors."
			}

			report.Items = append(report.Items, Item{
				Severity:      SeverityCost,
				Category:      CategoryMemory,
				EvidenceLevel: EvidenceObserved,
				Location: Location{
					Kernel: rmk.Function,
					File:   rmk.File,
					Line:   rmk.Line,
				},
				Finding:                  finding,
				ActionableRecommendation: rec,
				NextMeasurement:          "verify tensor tile dimensions and 16-byte alignment or inspect SASS for cp.async.bulk.tensor",
			})
		}

		if rmk.IsTensorCoreFailure() {
			finding := "Tensor Core MMA instruction selection failed: fell back to FMA instructions"
			if rmk.Remark != "" {
				finding = rmk.Remark
			}
			if rmk.Reason != "" {
				finding += fmt.Sprintf(" (reason: %s)", rmk.Reason)
			}
			if rmk.Shape != "" {
				finding += fmt.Sprintf(" [shape: %s]", rmk.Shape)
			}

			report.Items = append(report.Items, Item{
				Severity:      SeverityCost,
				Category:      CategoryCompute,
				EvidenceLevel: EvidenceObserved,
				Location: Location{
					Kernel: rmk.Function,
					File:   rmk.File,
					Line:   rmk.Line,
				},
				Finding:                  finding,
				ActionableRecommendation: "Pad matrix dimensions to multiples of 8 or 16 and use supported precision types (e.g. FP16, BF16, TF32) to enable hardware Tensor Core acceleration.",
				NextMeasurement:          "inspect SASS for HMMA/IMMA instructions or measure wall-clock latency",
			})
		}
	}

	// 2. Register Pressure Curves (via cudaprof/pressure.go)
	launches := extractLaunches(cap)
	launchesByKernel := make(map[string]launchRecord, len(launches))
	for _, l := range launches {
		launchesByKernel[l.Kernel] = l
	}

	images, unread := kernelImages(cap)
	disasm, modules := adviseDisassembly(cap, images)
	resources := kernelResources(images)

	ptxData, ptxFile, _ := capturefiles.PTX(cap)
	var ptxMod *cudaptx.Module
	if len(ptxData) > 0 {
		if mod, err := cudaptx.ParseBytes(ptxData); err == nil {
			ptxMod = mod
		}
	}

	// Precise float math, charged only where it costs local memory.
	mathKernels, math := floatMathByKernel(disasm, ptxMod, ptxFile)
	frames := make(map[string]int)
	for _, k := range resources {
		frames[k.Name] = max(frames[k.Name], k.FrameSizeBytes)
	}
	auditPreciseFloatMath(cap, report, mathKernels, math, frames)

	if disasm != nil {
		arch := parseArchFromCapture(cap)
		if arch == "" {
			arch = "sm_121a"
		}
		for _, fn := range disasm.Functions {
			var peakGPR, peakUGPR int
			var peakGPRAddr, peakUGPRAddr uint64
			var peakGPRInst, peakUGPRInst cudasass.Instruction
			var spills []cudasass.Instruction
			var imadWides []cudasass.Instruction

			for _, inst := range fn.Instructions {
				if inst.LiveGPR > peakGPR {
					peakGPR = inst.LiveGPR
					peakGPRAddr = inst.Address
					peakGPRInst = inst
				}
				if inst.LiveUGPR > peakUGPR {
					peakUGPR = inst.LiveUGPR
					peakUGPRAddr = inst.Address
					peakUGPRInst = inst
				}

				base := cudasass.BaseOpcode(inst.Opcode)
				if base == "STL" || base == "LDL" || strings.HasPrefix(inst.Opcode, "STL") || strings.HasPrefix(inst.Opcode, "LDL") || strings.Contains(inst.Opcode, ".LOCAL") {
					spills = append(spills, inst)
				}

				// Check SASS for IMAD.WIDE scalar fallback
				if strings.HasPrefix(inst.Opcode, "IMAD.WIDE") || inst.Opcode == "IMAD.WIDE" {
					imadWides = append(imadWides, inst)
				}
			}

			// One finding per pattern per kernel: the pattern is one lead, and
			// a finding per instruction charged the kernel's time to each.
			if len(imadWides) > 0 {
				first := imadWides[0]
				severity := SeverityCost
				recommendation := "If address patterns are regular and aligned, use vectorized 128-bit memory addressing or hardware TMA descriptors on sm_90+ to eliminate scalar 64-bit integer address multiplications."
				if isStreamingKernelName(fn.Name) {
					severity = SeverityContext
					recommendation = "This kernel name suggests a streaming copy or elementwise workload. Scalar address arithmetic is a low-priority static-code observation here; check global-memory throughput and transaction efficiency before changing address calculations."
				}
				report.Items = append(report.Items, Item{
					rule:          "imad-wide",
					Severity:      severity,
					Category:      CategoryArchitecture,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: fn.Name,
						Module: modules[fn],
						PC:     first.Address,
						File:   first.SourceFile,
						Line:   first.SourceLine,
					},
					Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) in kernel %q: %d instructions, at %s", first.Opcode, fn.Name, len(imadWides), formatPCs(imadWides)),
					ActionableRecommendation: recommendation,
					NextMeasurement:          "A/B wall-clock timing or profiling with hardware instruction counters",
				})
			}

			if len(spills) > 0 {
				first := spills[0]
				rec := "Reduce register lifetimes or apply __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) to eliminate spills to slow local memory."
				if math[fn.Name].costs(frames[fn.Name]) {
					rec += fastMathLead
				}
				report.Items = append(report.Items, Item{
					rule:          "spill",
					Severity:      SeverityReadFirst,
					Category:      CategoryRegisterPressure,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
						Module: modules[fn],
						PC:     first.Address,
						File:   first.SourceFile,
						Line:   first.SourceLine,
					},
					Finding:                  fmt.Sprintf("Register spill point detected in kernel %q: %d local-memory instructions, first %s, at %s", fn.Name, len(spills), capturefiles.Instruction(first), formatPCs(spills)),
					ActionableRecommendation: rec,
					NextMeasurement:          "verify zero local memory traffic with l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
				})
			}

			// Occupancy cliffs / Peak GPR
			locGPR := Location{
				Kernel: fn.Name,
				Module: modules[fn],
				PC:     peakGPRAddr,
				File:   peakGPRInst.SourceFile,
				Line:   peakGPRInst.SourceLine,
			}
			threadsPerBlock := 128
			if l, ok := launchesByKernel[fn.Name]; ok && l.BlockThreads > 0 {
				threadsPerBlock = l.BlockThreads
			}
			occRes, occErr := cudaoccupancy.CalculateOccupancy(arch, peakGPR, 0, threadsPerBlock)

			if peakGPR > 255 {
				report.Items = append(report.Items, Item{
					rule:                     "gpr-limit",
					Severity:                 SeverityReadFirst,
					Category:                 CategoryRegisterPressure,
					EvidenceLevel:            EvidenceObserved,
					Location:                 locGPR,
					Finding:                  fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q exceeds hardware register limit (255 GPRs), forcing spilling to local memory", peakGPR, fn.Name),
					ActionableRecommendation: "Reduce register consumption below 255 using compiler flags (-maxrregcount=255 or __launch_bounds__) to prevent excessive local memory spills.",
					NextMeasurement:          "compile with -maxrregcount=255 and check for spills with cuobjdump -res-usage",
				})
			} else if peakGPR > 128 {
				finding := fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff (>128 GPRs, capping SM occupancy at <= 16 warps)", peakGPR, fn.Name)
				if occErr == nil {
					finding = fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff on %s (>128 GPRs, capping SM occupancy at %d/%d warps, %.1f%% across %d blocks/SM limited by %s)",
						peakGPR, fn.Name, occRes.Architecture, occRes.TheoreticalWarps, occRes.MaxWarps, occRes.Occupancy*100, occRes.ActiveBlocks, occRes.LimitingFactor)
				}
				report.Items = append(report.Items, Item{
					rule:                     "gpr-cliff",
					Severity:                 SeverityCost,
					Category:                 CategoryRegisterPressure,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 locGPR,
					Finding:                  finding,
					ActionableRecommendation: "Optimize register reuse, hoist expressions, or use __launch_bounds__ to keep register pressure at or below 128 GPRs for higher occupancy." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			} else if peakGPR > 64 {
				finding := fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff (>64 GPRs, capping SM occupancy at <= 32 warps)", peakGPR, fn.Name)
				if occErr == nil {
					finding = fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff on %s (>64 GPRs, capping SM occupancy at %d/%d warps, %.1f%% across %d blocks/SM limited by %s)",
						peakGPR, fn.Name, occRes.Architecture, occRes.TheoreticalWarps, occRes.MaxWarps, occRes.Occupancy*100, occRes.ActiveBlocks, occRes.LimitingFactor)
				}
				report.Items = append(report.Items, Item{
					rule:                     "gpr-cliff",
					Severity:                 SeverityCost,
					Category:                 CategoryRegisterPressure,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 locGPR,
					Finding:                  finding,
					ActionableRecommendation: "Evaluate whether limiting register usage to <= 64 GPRs with __launch_bounds__ raises warp occupancy." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			}

			// Peak UGPR
			if peakUGPR > 63 {
				report.Items = append(report.Items, Item{
					rule:          "ugpr",
					Severity:      SeverityCost,
					Category:      CategoryRegisterPressure,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
						Module: modules[fn],
						PC:     peakUGPRAddr,
						File:   peakUGPRInst.SourceFile,
						Line:   peakUGPRInst.SourceLine,
					},
					Finding:                  fmt.Sprintf("Peak uniform register pressure of %d UGPRs in kernel %q exceeds architecture limit of 63 UGPRs", peakUGPR, fn.Name),
					ActionableRecommendation: "Reduce warp-uniform temporary variables to prevent uniform register spills.",
					NextMeasurement:          "inspect SASS for UR spills or compile with fewer uniform variables",
				})
			}
		}
	}

	// 3. PTX Architecture Audits (via x/cudaptx)
	if mod := ptxMod; mod != nil {
		target := mod.Target
		for _, k := range mod.Kernels {
			// 1. Architecture compatibility check
			for _, inst := range k.Instructions {
				req := cudaptx.InstructionMinTarget(inst.Mnemonic)
				if req != "" && req != "all" && target != "" && !architectureCompatible(target, req) {
					loc := Location{
						Kernel: k.Name,
						File:   inst.Loc.File,
						Line:   inst.Loc.Line,
					}
					if loc.File == "" {
						loc.File = ptxFile
						loc.Line = inst.Line
					}
					report.Items = append(report.Items, Item{
						Severity:                 SeverityReadFirst,
						Category:                 CategoryArchitecture,
						EvidenceLevel:            EvidenceObserved,
						severityByKind:           true,
						Location:                 loc,
						Finding:                  fmt.Sprintf("Instruction %q (%s) requires architecture %s, but module targets %s", inst.Opcode, inst.Mnemonic, req, target),
						ActionableRecommendation: fmt.Sprintf("Compile with target architecture >= %s or replace with instructions compatible with %s.", req, target),
						NextMeasurement:          "recompile targeting supported compute capability",
					})
				}
			}

			// 2. Scalar fallbacks check on target architecture (sm_90, sm_100, sm_121a, or modern SM)
			// If SASS disassembly is available for this kernel, prefer SASS analysis
			// and skip redundant PTX-level scalar fallback warnings.
			if disasm != nil && disasm.Function(k.Name) != nil {
				continue
			}
			isModern := parseSM(target) >= 90
			if isModern {
				var scalarLoads []cudaptx.Instruction
				var madWides []cudaptx.Instruction
				for _, inst := range k.Instructions {
					// Scalar global loads: ld.global without .v2/.v4/.v8
					if inst.Mnemonic == "ld" && strings.Contains(inst.Opcode, ".global") {
						if !strings.Contains(inst.Opcode, ".v2") &&
							!strings.Contains(inst.Opcode, ".v4") &&
							!strings.Contains(inst.Opcode, ".v8") {
							scalarLoads = append(scalarLoads, inst)
						}
					}
					// Scalar address arithmetic fallback: mad.wide or mul.wide
					if (inst.Mnemonic == "mad" || inst.Mnemonic == "mul") && strings.Contains(inst.Opcode, ".wide") {
						madWides = append(madWides, inst)
					}
				}

				// Report scalar global loads (up to 3 distinct locations to avoid flooding)
				for i, sLoad := range scalarLoads {
					if i >= 3 {
						report.Items = append(report.Items, Item{
							rule:          "ptx-scalar-load",
							Severity:      SeverityCost,
							Category:      CategoryArchitecture,
							EvidenceLevel: EvidenceSuspected,
							Location: Location{
								Kernel: k.Name,
							},
							Finding:                  fmt.Sprintf("Kernel %q has %d scalar global load instructions on target %s", k.Name, len(scalarLoads), target),
							ActionableRecommendation: "If memory access is contiguous and 16-byte aligned, vectorize memory accesses using 128-bit vector loads (e.g. ld.global.v4) or, for multidimensional tensor tiles, Tensor Memory Accelerator (TMA) asynchronous copies (cp.async.bulk.tensor) on sm_90+ architectures.",
							NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or A/B wall-clock timing",
						})
						break
					}
					loc := Location{
						Kernel: k.Name,
						File:   sLoad.Loc.File,
						Line:   sLoad.Loc.Line,
					}
					if loc.File == "" {
						loc.File = ptxFile
						loc.Line = sLoad.Line
					}
					report.Items = append(report.Items, Item{
						rule:                     "ptx-scalar-load",
						Severity:                 SeverityCost,
						Category:                 CategoryArchitecture,
						EvidenceLevel:            EvidenceSuspected,
						Location:                 loc,
						Finding:                  fmt.Sprintf("Scalar global load (%s) detected on target %s without vectorized or TMA path", sLoad.Opcode, target),
						ActionableRecommendation: "If memory access is contiguous and 16-byte aligned, upgrade memory loads to 128-bit vector accesses (e.g. ld.global.v4) or, for multidimensional tensor tiles on sm_90+ architectures, use TMA asynchronous copies (cp.async.bulk.tensor) to maximize memory bandwidth utilization.",
						NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or A/B wall-clock timing",
					})
				}

				// Report scalar address arithmetic
				for i, mad := range madWides {
					if i >= 2 {
						break
					}
					loc := Location{
						Kernel: k.Name,
						File:   mad.Loc.File,
						Line:   mad.Loc.Line,
					}
					if loc.File == "" {
						loc.File = ptxFile
						loc.Line = mad.Line
					}
					report.Items = append(report.Items, Item{
						rule:                     "ptx-mad-wide",
						Severity:                 SeverityCost,
						Category:                 CategoryArchitecture,
						EvidenceLevel:            EvidenceSuspected,
						Location:                 loc,
						Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) detected in kernel %q on target %s", mad.Opcode, k.Name, target),
						ActionableRecommendation: "If indexing into contiguous arrays or structured tensors, use vectorized pointer indexing or TMA multidimensional tensor descriptors on sm_90+ to eliminate scalar 64-bit integer address multiplications.",
						NextMeasurement:          "A/B wall-clock timing",
					})
				}
			}
		}
	}

	// 0. Capture Status & Integrity Audit
	auditCaptureStatus(cap, report)

	// 3. CUBIN ELF Resource & Microarchitecture Occupancy Audit (via x/cudaoccupancy)
	auditCUBINResourceOccupancy(cap, report, resources, launchesByKernel, math)

	// 4. Launch Geometry Audits (Underpopulated Thread Block & SM Grid Starvation)
	auditLaunchGeometry(cap, report)

	// 5. Under-subscribed Launch Audit (Single-Threaded & Sub-Warp-Per-SM Grids)
	auditUnderSubscribedLaunches(cap, report)

	// 6. Uncoalesced Memory Access Audit (SASS & PC Sampling & Remarks)
	auditUncoalescedMemory(cap, report, disasm)

	// 7. Stream Kernel Sequences (CUDA Graph Candidates & Pointwise Fusion)
	auditStreamKernelSequences(cap, report)
	auditHostDispatchOverhead(cap, report)
	auditHostAPICost(cap, report)

	// 7a. Whole-capture duty cycle. This runs before the dominance audit so
	// that an idle-dominated capture says so above every kernel-time finding.
	auditDeviceDutyCycle(cap, report)

	// 8. Kernel Family Dominance Audit (Macro Architectural Bottlenecks)
	kernelTimes, totalTime := computeProfileTimes(cap)
	chargeModules(kernelTimes, images)
	hasTiming := hasDeviceTiming(cap)
	auditKernelFamilyDominance(cap, report, kernelTimes, totalTime, hasTiming)
	auditMissingModuleCode(cap, report, kernelTimes, totalTime, hasTiming)

	markUnreadSources(report, unread)

	// Finalize device times and rank findings by device time
	report.SampleOnly = !hasTiming
	finalizeAndRankFindingsWithTiming(report, kernelTimes, totalTime, hasTiming)
	classifyProfileShares(cap, report, source)
	for _, item := range report.Items {
		if item.EvidenceLevel == EvidenceSuspected || item.EvidenceLevel == EvidenceUnmeasured {
			report.EvidenceGuidance = "Suspected findings are heuristics. Capture with --pcsampling for per-PC evidence; bandwidth and occupancy claims require hardware counters through the CUPTI Profiling API with replay (for example, x/cuptirange)."
			break
		}
	}

	return report, nil
}

// auditMissingModuleCode says which module images the capture did not keep,
// and how much of the kernel time ran from them: code-based findings cannot
// see those kernels, and a short list of findings reads as a clean bill for
// code no finding inspected.
func auditMissingModuleCode(cap *cudacapture.Capture, report *Report, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	missing := capturefiles.MissingModuleCode(cap)
	if len(missing) == 0 {
		return
	}
	finding := "Code not captured for " + capturefiles.DescribeMissingCode(missing)
	// Most loaded modules never run, so the census alone overstates what
	// the findings below cannot see. Launches that name no module understate
	// it instead, and the share below is then only a floor.
	var unnamed bool
	if launches, err := capturefiles.CountModuleLaunches(cap); err == nil {
		if s := capturefiles.DescribeMissingLaunches(missing, launches); s != "" {
			finding += ". " + strings.ToUpper(s[:1]) + s[1:]
			unnamed = launches.Unnamed > 0
		}
	}
	unseen := make(map[uint32]bool)
	for _, m := range missing {
		if m.Module != 0 {
			unseen[m.Module] = true
		}
	}
	var kernels int
	var unseenTime int64
	for name, modules := range kernelModules(cap) {
		var share float64
		for id := range modules {
			if unseen[id] {
				share += moduleShare(modules, id)
			}
		}
		if share > 0 {
			kernels++
			unseenTime += int64(float64(kernelTimes[name]) * share)
		}
	}
	item := Item{
		Severity: SeverityReadFirst, severityByKind: true, Category: CategoryKernel, EvidenceLevel: EvidenceObserved,
		ActionableRecommendation: "Raise CUDAPROF_MODULE_FILE_BYTES, --capture-file-bytes or --capture-total-bytes on the capture that loads these modules; code-based findings are incomplete for them.",
		NextMeasurement:          "rerun cudaprof advise after capturing the omitted module images",
	}
	switch {
	case kernels > 0 && totalTime > 0:
		share := float64(unseenTime) / float64(totalTime) * 100
		unit := "of kernel time"
		if !hasTiming {
			unit = "of samples"
		}
		floor := ""
		if unnamed {
			floor = " at least"
		}
		verb := "kernels that ran from them hold"
		if kernels == 1 {
			verb = "kernel that ran from it holds"
		}
		finding += fmt.Sprintf(". %d %s%s %.1f%% %s", kernels, verb, floor, share, unit)
		item.ShareOfProfile = share
		if hasTiming {
			item.DeviceTimeNanos = unseenTime
		}
	case kernels > 0:
		finding += fmt.Sprintf(". %d kernels ran from them", kernels)
	}
	item.Finding = finding
	report.Items = append(report.Items, item)
}
