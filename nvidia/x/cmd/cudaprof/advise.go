package main

import (
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

const defaultAdviceLimit = 10

// AnalyzeCapture synthesizes diagnostics across compiler remarks, register pressure curves,
// and PTX architecture audits.
func AnalyzeCapture(cap *cudacapture.Capture) (*AdviceReport, error) {
	if cap == nil {
		return nil, errors.New("cudaprof: nil capture")
	}

	report := &AdviceReport{
		Items: make([]AdviceItem, 0),
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

			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityWarning,
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

			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityWarning,
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
	disasm := adviseDisassembly(cap, images)
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
			var fchkInsts []cudasass.Instruction

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

				// Check SASS for IEEE compliance check branches (missing --use_fast_math)
				if inst.Opcode == "FCHK" || strings.HasPrefix(inst.Opcode, "FCHK") {
					fchkInsts = append(fchkInsts, inst)
				}
			}

			// One finding per pattern per kernel: the pattern is one lead, and
			// a finding per instruction charged the kernel's time to each.
			if len(imadWides) > 0 {
				first := imadWides[0]
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryArchitecture,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: fn.Name,
						PC:     first.Address,
						File:   first.SourceFile,
						Line:   first.SourceLine,
					},
					Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) in kernel %q: %d instructions, at %s", first.Opcode, fn.Name, len(imadWides), formatPCs(imadWides)),
					ActionableRecommendation: "If address patterns are regular and aligned, use vectorized 128-bit memory addressing or hardware TMA descriptors on sm_90+ to eliminate scalar 64-bit integer address multiplications.",
					NextMeasurement:          "A/B wall-clock timing or profiling with hardware instruction counters",
				})
			}

			// Report FCHK arithmetic checks
			if len(fchkInsts) > 0 {
				first := fchkInsts[0]
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryArchitectureCompute,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
						PC:     first.Address,
						File:   first.SourceFile,
						Line:   first.SourceLine,
					},
					Finding:                  fmt.Sprintf("Kernel %q appears compiled without --use_fast_math: %d IEEE compliance check instructions (FCHK) detected in arithmetic paths", fn.Name, len(fchkInsts)),
					ActionableRecommendation: "If strict IEEE 754 bit-exact rounding is not required, compile with --use_fast_math to substitute hardware approximations (e.g. MUFU.RCP) and eliminate FCHK branches. NOTE: --use_fast_math changes division and transcendental rounding and flushes denormals to zero; if the workload enforces bit-exact logit invariants, keep IEEE math and prefer --extra-device-vectorization or explicit fast reciprocal intrinsics (__fdividef) over the global flag.",
					NextMeasurement:          "A/B wall-clock timing with --use_fast_math",
				})
			}

			if len(spills) > 0 {
				first := spills[0]
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityCritical,
					Category:      CategoryRegisterPressure,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
						PC:     first.Address,
						File:   first.SourceFile,
						Line:   first.SourceLine,
					},
					Finding:                  fmt.Sprintf("Register spill point detected in kernel %q: %d local-memory instructions, first %s, at %s", fn.Name, len(spills), capturefiles.Instruction(first), formatPCs(spills)),
					ActionableRecommendation: "Reduce register lifetimes or apply __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) to eliminate spills to slow local memory.",
					NextMeasurement:          "verify zero local memory traffic with l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
				})
			}

			// Occupancy cliffs / Peak GPR
			locGPR := Location{
				Kernel: fn.Name,
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
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityCritical,
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
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
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
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
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
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryRegisterPressure,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
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
	ptxData, ptxFile, _ := capturefiles.PTX(cap)
	if len(ptxData) > 0 {
		mod, err := cudaptx.ParseBytes(ptxData)
		if err == nil && mod != nil {
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
						report.Items = append(report.Items, AdviceItem{
							Severity:                 SeverityCritical,
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
							report.Items = append(report.Items, AdviceItem{
								Severity:      SeverityWarning,
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
						report.Items = append(report.Items, AdviceItem{
							Severity:                 SeverityWarning,
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
						report.Items = append(report.Items, AdviceItem{
							Severity:                 SeverityWarning,
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
	}

	// 0. Capture Status & Integrity Audit
	auditCaptureStatus(cap, report)

	// 3. CUBIN ELF Resource & Microarchitecture Occupancy Audit (via x/cudaoccupancy)
	auditCUBINResourceOccupancy(cap, report, images, launchesByKernel)

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
	hasTiming := hasDeviceTiming(cap)
	auditKernelFamilyDominance(cap, report, kernelTimes, totalTime, hasTiming)

	markUnreadSources(report, unread)

	// Finalize device times and rank findings by device time
	report.SampleOnly = !hasTiming
	finalizeAndRankFindingsWithTiming(report, kernelTimes, totalTime, hasTiming)

	return report, nil
}

func runAdviseCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	opts, err := parseAdviseArgs(args)
	if err != nil {
		return err
	}

	path, err := filepath.Abs(opts.capture)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		return fmt.Errorf("analyze capture: %w", err)
	}
	if opts.base != "" {
		if err := adviseAgainstBase(report, cap, path, opts, limits); err != nil {
			return fmt.Errorf("advise: %w", err)
		}
	}

	if opts.format == "json" {
		// -limit governs how much the markdown report prints, not how much
		// the tool knows. The summary counts every finding, so one format has
		// to be able to reach every finding; JSON is it. Truncating here left
		// the counted remainder unreachable by any flag combination.
		out, err := report.JSON()
		if err != nil {
			return err
		}
		fmt.Println(out)
		return nil
	}

	fmt.Print(report.MarkdownWithLimit(opts.limit))
	return nil
}

const adviseUsage = "usage: cudaprof advise [-e] [-limit N] [--format=json|markdown] [-base base.pprof [-ops ops.txt] [-phase i]] capture.pprof"

type adviseOptions struct {
	format  string
	limit   int
	capture string

	// base, ops and phase select a comparison against a base capture, as
	// compare's arguments of the same names do.
	base  string
	ops   string
	phase int
}

// parseAdviseArgs parses advise's flags, each spelled with one dash or two
// and taking its value as -flag=v or -flag v.
func parseAdviseArgs(args []string) (adviseOptions, error) {
	opts := adviseOptions{limit: defaultAdviceLimit, phase: -1}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") {
			if opts.capture != "" {
				return opts, errors.New("multiple capture files specified")
			}
			opts.capture = arg
			continue
		}
		name, val, hasVal := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-"), "=")
		value := func() (string, error) {
			if hasVal {
				return val, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("flag -%s requires an argument", name)
			}
			i++
			return args[i], nil
		}
		intValue := func() (int, error) {
			v, err := value()
			if err != nil {
				return 0, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return 0, fmt.Errorf("invalid -%s value: %s", name, v)
			}
			return n, nil
		}
		var err error
		switch {
		case !hasVal && (name == "json" || name == "markdown"):
			opts.format = name
		case hasVal && name == "format" && (val == "json" || val == "markdown"):
			opts.format = val
		case !hasVal && (name == "e" || name == "all"):
			opts.limit = 0
		case name == "limit" || name == "n":
			opts.limit, err = intValue()
		case name == "base":
			opts.base, err = value()
		case name == "ops":
			opts.ops, err = value()
		case name == "phase":
			opts.phase, err = intValue()
		default:
			return opts, fmt.Errorf("unknown flag: %s", arg)
		}
		if err != nil {
			return opts, err
		}
	}
	if opts.capture == "" {
		return opts, errors.New(adviseUsage)
	}
	if opts.base == "" && (opts.ops != "" || opts.phase >= 0) {
		return opts, errors.New("-ops and -phase select a comparison and need -base")
	}
	return opts, nil
}
