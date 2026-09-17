package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// occupancyCaveat qualifies every recommendation derived from an occupancy or
// register-residency figure. Occupancy states what a kernel CAN keep resident,
// never what it is waiting on. Raising it pays only when the kernel is
// latency-bound, and it can run backwards when the transformation that frees the
// resource costs something else: deleting the shared-memory staging in
// ds4_f16_pair_ordered_chunks raised occupancy from 10.4% to 100% and measured
// 2.84% slower end to end, because the staging was buying coalescing. Advice
// built on a capacity figure is a hypothesis to A/B, not an instruction.
//
// It replaces three ad-hoc hedges that said less in four different spellings.
// Sites where the figure is NOT capacity-read-as-throughput are deliberately
// left bare and are pinned bare by tests: a hardware legality maximum (block
// size > 1024), grid starvation (SMs are literally idle, which is an occupancy
// figure read AS occupancy), and register spills (real work, not residency).
const occupancyCaveat = " Occupancy bounds residency, not throughput: A/B this. It pays only if the kernel is latency-bound (a bandwidth-bound streaming kernel will see no speedup), and it can regress when freeing the resource costs coalescing."

// Severity denotes the impact of an advice finding.
type Severity string

const (
	SeverityCritical Severity = "Critical"
	SeverityWarning  Severity = "Warning"
	SeverityInfo     Severity = "Info"
)

// EvidenceLevel classifies the empirical basis of an advice finding.
type EvidenceLevel string

const (
	// EvidenceObserved indicates direct native hardware metrics, compiler diagnostics, or measured timings.
	EvidenceObserved EvidenceLevel = "observed"
	// EvidenceSuspected indicates static opcode heuristics, correlated PC samples without deciding transaction counters, or resource capacity bounds.
	EvidenceSuspected EvidenceLevel = "suspected"
	// EvidenceUnmeasured indicates missing required telemetry or skipped audits due to absent data.
	EvidenceUnmeasured EvidenceLevel = "unmeasured"
)

// Category classifies the optimization or architecture area of an advice finding.
type Category string

const (
	CategoryMemory               Category = "Memory"
	CategoryCompute              Category = "Compute"
	CategoryRegisterPressure     Category = "RegisterPressure"
	CategoryArchitecture         Category = "Architecture"
	CategoryArchitectureCompute  Category = "Architecture/Compute"
	CategoryArchitectureDispatch Category = "Architecture/Dispatch"
	CategoryMemoryCompute        Category = "Memory/Compute"
	CategoryDispatch             Category = "Dispatch"
)

// Location specifies the source or machine code position associated with an advice item.
type Location struct {
	Kernel string `json:"kernel,omitempty"`
	PC     uint64 `json:"pc,omitempty"`
	File   string `json:"file,omitempty"`
	Line   int    `json:"line,omitempty"`
}

func (l Location) String() string {
	var parts []string
	if l.Kernel != "" {
		parts = append(parts, fmt.Sprintf("kernel %s", l.Kernel))
	}
	if l.File != "" {
		if l.Line > 0 {
			parts = append(parts, fmt.Sprintf("%s:%d", l.File, l.Line))
		} else {
			parts = append(parts, l.File)
		}
	}
	if l.PC > 0 {
		parts = append(parts, fmt.Sprintf("PC 0x%04x", l.PC))
	}
	return strings.Join(parts, ", ")
}

// AdviceItem represents a single diagnosis or performance recommendation.
type AdviceItem struct {
	Severity                 Severity      `json:"severity"`
	Category                 Category      `json:"category"`
	EvidenceLevel            EvidenceLevel `json:"evidence_level,omitempty"`
	Location                 Location      `json:"location"`
	Finding                  string        `json:"finding"`
	ActionableRecommendation string        `json:"actionable_recommendation"`
	NextMeasurement          string        `json:"next_measurement,omitempty"`
	DeviceTimeNanos          int64         `json:"device_time_nanos,omitempty"`
	ShareOfProfile           float64       `json:"share_of_profile,omitempty"`

	// magnitude overrides the parenthetical the markdown header prints for
	// this item. The default reads "(N% of GPU time, M ms)", which is a claim
	// about device work: a finding whose numbers are idle time renders there
	// as though the idle were work a kernel did. An item whose numbers mean
	// something else says what they mean here.
	magnitude string

	// severityByKind exempts this item from the Critical share floor. A
	// finding that says the module may not run, or that the report's own
	// ordering is misleading, is Critical for what it is and not for how much
	// device time it accounts for. Everything else earns Critical by cost and
	// must clear the floor. Not serialized: it is how severity was decided,
	// which the severity field already reports.
	severityByKind bool
}

// AdviceReport contains all recommendations resulting from analyzing a capture.
type AdviceReport struct {
	Items []AdviceItem `json:"items"`

	// SampleOnly reports that the capture carries PC samples but no device
	// timing. A share of samples then says where execution was, never how
	// long anything took, and never whether the device was idle between
	// launches. That limit is a property of the instrument, not a defect,
	// but it has to be stated: a reader who does not know it reads
	// "17.66% of observed samples" as "17.66% of elapsed time".
	SampleOnly bool `json:"sample_only,omitempty"`

	// SourceFusion reports that the capture's PTX and SASS are not from one
	// compilation, so LoadCode refused the source binding and every finding
	// below carries an empty file and a zero line. Without this a reader cannot
	// tell a refused binding from a compiler that emitted no line information.
	SourceFusion string `json:"source_fusion_failure,omitempty"`
}

// sampleOnlyCaveat names what PC sampling cannot see. Activity tracing is the
// only tier that records launch counts and inter-launch gaps, so on a
// sample-only capture device idle is unmeasured rather than absent.
const sampleOnlyCaveat = "> **This capture is PC samples only.** A share of observed samples says where " +
	"execution was, and is *not a share of elapsed time*. PC sampling cannot observe " +
	"device idle between launches; capture with activity tracing to measure that.\n\n"

const defaultAdviceLimit = 10

var sassRegisterRE = regexp.MustCompile(`\bR[0-9]+\b`)

// Markdown formats the advice report as human-readable Markdown with the default limit of findings.
func (r *AdviceReport) Markdown() string {
	return r.MarkdownWithLimit(defaultAdviceLimit)
}

// MarkdownWithLimit formats up to limit findings as human-readable Markdown.
// If limit <= 0, all findings are printed without truncation.
func (r *AdviceReport) MarkdownWithLimit(limit int) string {
	if r == nil {
		return "# CUDA Kernel Advice Report\n\nNo optimization issues or architecture fallbacks identified.\n"
	}

	var sb strings.Builder
	sb.WriteString("# CUDA Kernel Advice Report\n\n")
	if r.SampleOnly {
		sb.WriteString(sampleOnlyCaveat)
	}
	if r.SourceFusion != "" {
		sb.WriteString(sourceFusionMarkdownCaveat(r.SourceFusion))
	}
	// The caveats above are written before this return, not after it: a report
	// with no findings on a capture that refused its source binding is not the
	// same as a clean report, and rendering them identically is how a refusal
	// reads as an all-clear.
	if len(r.Items) == 0 {
		sb.WriteString("No optimization issues or architecture fallbacks identified.\n")
		return sb.String()
	}

	var criticals, warnings, infos int
	for _, item := range r.Items {
		switch item.Severity {
		case SeverityCritical:
			criticals++
		case SeverityWarning:
			warnings++
		case SeverityInfo:
			infos++
		}
	}

	sb.WriteString("## Summary\n\n")
	sb.WriteString(fmt.Sprintf("- **Critical**: %d\n", criticals))
	sb.WriteString(fmt.Sprintf("- **Warning**: %d\n", warnings))
	sb.WriteString(fmt.Sprintf("- **Info**: %d\n\n", infos))

	sb.WriteString("## Findings\n\n")
	count := len(r.Items)
	if limit > 0 && limit < count {
		count = limit
		// The Summary above counts findings this report does not print. Say
		// so here, beside the count, rather than only in a footer the reader
		// reaches after forming an impression of how much there was.
		sb.WriteString(fmt.Sprintf("*Showing %d of %d findings (use --format=json for all, or -limit=0).*\n\n", count, len(r.Items)))
	}
	for i := 0; i < count; i++ {
		item := r.Items[i]
		header := fmt.Sprintf("### %d. [%s] %s: %s\n\n", i+1, item.Severity, item.Category, item.Finding)
		if item.magnitude != "" {
			header = fmt.Sprintf("### %d. [%s] (%s) %s: %s\n\n", i+1, item.Severity, item.magnitude, item.Category, item.Finding)
		} else if item.ShareOfProfile > 0 || item.DeviceTimeNanos > 0 {
			if item.DeviceTimeNanos > 0 {
				durMs := float64(item.DeviceTimeNanos) / 1e6
				header = fmt.Sprintf("### %d. [%s] (%.2f%% of GPU time, %.2f ms) %s: %s\n\n", i+1, item.Severity, item.ShareOfProfile, durMs, item.Category, item.Finding)
			} else {
				header = fmt.Sprintf("### %d. [%s] (%.2f%% of observed samples) %s: %s\n\n", i+1, item.Severity, item.ShareOfProfile, item.Category, item.Finding)
			}
		}
		sb.WriteString(header)
		if item.EvidenceLevel != "" {
			sb.WriteString(fmt.Sprintf("- **Evidence**: %s\n", item.EvidenceLevel))
		}
		locStr := item.Location.String()
		if locStr != "" {
			sb.WriteString(fmt.Sprintf("- **Location**: %s\n", locStr))
		}
		sb.WriteString(fmt.Sprintf("- **Recommendation**: %s\n", item.ActionableRecommendation))
		if item.NextMeasurement != "" {
			sb.WriteString(fmt.Sprintf("- **Next Measurement**: %s\n", item.NextMeasurement))
		}
		sb.WriteString("\n")
	}

	if limit > 0 && len(r.Items) > limit {
		omitted := len(r.Items) - limit
		sb.WriteString(fmt.Sprintf("*... %d additional findings omitted (use --format=json for all, -e, or -limit=N).*\n", omitted))
	}

	return sb.String()
}

// JSON formats the advice report as formatted JSON.
func (r *AdviceReport) JSON() (string, error) {
	if r == nil {
		return "{\n  \"items\": []\n}", nil
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal advice report: %w", err)
	}
	return string(data), nil
}

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
	remarks, _, _ := findRemarks(cap)
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

	disasm, _ := loadDisassemblyWithPressure(cap)
	if disasm == nil {
		if sassData, ok := findSASS(cap); ok && len(sassData) > 0 {
			if parsed, err := cudasass.Parse(bytes.NewReader(sassData)); err == nil {
				disasm = parsed
			}
		}
	}
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

			// Report SASS IMAD.WIDE scalar address arithmetic fallbacks (capped to avoid flooding)
			for i, inst := range imadWides {
				if i >= 3 {
					report.Items = append(report.Items, AdviceItem{
						Severity:      SeverityWarning,
						Category:      CategoryArchitecture,
						EvidenceLevel: EvidenceSuspected,
						Location: Location{
							Kernel: fn.Name,
						},
						Finding:                  fmt.Sprintf("Scalar address arithmetic fallback: kernel %q has %d instructions (%s) on target architecture", fn.Name, len(imadWides), inst.Opcode),
						ActionableRecommendation: "If address patterns are regular and aligned, use vectorized 128-bit memory addressing or hardware TMA descriptors on sm_90+ to eliminate scalar 64-bit integer address multiplications.",
						NextMeasurement:          "A/B wall-clock timing or profiling with hardware instruction counters",
					})
					break
				}
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryArchitecture,
					EvidenceLevel: EvidenceSuspected,
					Location: Location{
						Kernel: fn.Name,
						PC:     inst.Address,
						File:   inst.SourceFile,
						Line:   inst.SourceLine,
					},
					Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) detected at PC 0x%04x", inst.Opcode, inst.Address),
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

			// Report register spill points
			for i, spill := range spills {
				if i >= 5 {
					report.Items = append(report.Items, AdviceItem{
						Severity:      SeverityCritical,
						Category:      CategoryRegisterPressure,
						EvidenceLevel: EvidenceObserved,
						Location: Location{
							Kernel: fn.Name,
						},
						Finding:                  fmt.Sprintf("Kernel %q has %d total register spill instructions to local memory", fn.Name, len(spills)),
						ActionableRecommendation: "Reduce live variable ranges or specify __launch_bounds__ to eliminate register spills.",
						NextMeasurement:          "verify zero local memory traffic with l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
					})
					break
				}
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityCritical,
					Category:      CategoryRegisterPressure,
					EvidenceLevel: EvidenceObserved,
					Location: Location{
						Kernel: fn.Name,
						PC:     spill.Address,
						File:   spill.SourceFile,
						Line:   spill.SourceLine,
					},
					Finding:                  fmt.Sprintf("Register spill point detected at PC 0x%04x (%s): memory spilled to local stack", spill.Address, formatInstruction(spill)),
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
	ptxData, ptxFile, _ := findPTX(cap)
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
	auditCUBINResourceOccupancy(cap, report, launchesByKernel)

	// 4. Launch Geometry Audits (Underpopulated Thread Block & SM Grid Starvation)
	auditLaunchGeometry(cap, report)

	// 5. Under-subscribed Launch Audit (Single-Threaded & Sub-Warp-Per-SM Grids)
	auditUnderSubscribedLaunches(cap, report)

	// 6. Uncoalesced Memory Access Audit (SASS & PC Sampling & Remarks)
	auditUncoalescedMemory(cap, report)

	// 7. Stream Kernel Sequences (CUDA Graph Candidates & Pointwise Fusion)
	auditStreamKernelSequences(cap, report)
	auditHostDispatchOverhead(cap, report)

	// 7a. Whole-capture duty cycle. This runs before the dominance audit so
	// that an idle-dominated capture says so above every kernel-time finding.
	auditDeviceDutyCycle(cap, report)

	// 8. Kernel Family Dominance Audit (Macro Architectural Bottlenecks)
	kernelTimes, totalTime := computeProfileTimes(cap)
	hasTiming := hasDeviceTiming(cap)
	auditKernelFamilyDominance(cap, report, kernelTimes, totalTime, hasTiming)

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
	var format string
	var capturePath string
	limit := defaultAdviceLimit
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "-json" || arg == "--json" || arg == "--format=json" {
			format = "json"
		} else if arg == "-markdown" || arg == "--markdown" || arg == "--format=markdown" {
			format = "markdown"
		} else if arg == "-e" || arg == "--e" || arg == "-all" || arg == "--all" {
			limit = 0
		} else if strings.HasPrefix(arg, "-limit=") || strings.HasPrefix(arg, "--limit=") {
			val := strings.TrimPrefix(strings.TrimPrefix(arg, "--limit="), "-limit=")
			n, err := strconv.Atoi(val)
			if err != nil {
				return fmt.Errorf("invalid -limit value: %s", val)
			}
			limit = n
		} else if arg == "-limit" || arg == "--limit" {
			if i+1 >= len(args) {
				return errors.New("flag -limit requires an argument")
			}
			i++
			n, err := strconv.Atoi(args[i])
			if err != nil {
				return fmt.Errorf("invalid -limit value: %s", args[i])
			}
			limit = n
		} else if strings.HasPrefix(arg, "-n=") || strings.HasPrefix(arg, "--n=") {
			val := strings.TrimPrefix(strings.TrimPrefix(arg, "--n="), "-n=")
			n, err := strconv.Atoi(val)
			if err != nil {
				return fmt.Errorf("invalid -n value: %s", val)
			}
			limit = n
		} else if arg == "-n" {
			if i+1 >= len(args) {
				return errors.New("flag -n requires an argument")
			}
			i++
			n, err := strconv.Atoi(args[i])
			if err != nil {
				return fmt.Errorf("invalid -n value: %s", args[i])
			}
			limit = n
		} else if strings.HasPrefix(arg, "-") {
			return fmt.Errorf("unknown flag: %s", arg)
		} else if capturePath == "" {
			capturePath = arg
		} else {
			return errors.New("multiple capture files specified")
		}
	}
	if capturePath == "" {
		return errors.New("usage: cudaprof advise [-e] [-limit N] [--format=json|markdown] capture.pprof")
	}

	path, err := filepath.Abs(capturePath)
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

	if format == "json" {
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

	fmt.Print(report.MarkdownWithLimit(limit))
	return nil
}

func findPTX(cap *cudacapture.Capture) ([]byte, string, error) {
	if cap == nil || len(cap.Files) == 0 {
		return nil, "", nil
	}
	if cap.Code.PTX != "" {
		if data, ok := cap.Files[cap.Code.PTX]; ok && len(data) > 0 {
			return data, cap.Code.PTX, nil
		}
	}
	if data, ok := cap.Files["kernels.ptx"]; ok && len(data) > 0 {
		return data, "kernels.ptx", nil
	}
	for name, data := range cap.Files {
		if strings.HasSuffix(name, ".ptx") && len(data) > 0 {
			return data, name, nil
		}
	}
	return nil, "", nil
}

func architectureCompatible(target, req string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	req = strings.ToLower(strings.TrimSpace(req))
	if req == "" || req == "all" || target == req {
		return true
	}
	targetVal := parseSM(target)
	reqVal := parseSM(req)
	if targetVal > 0 && reqVal > 0 {
		return targetVal >= reqVal
	}
	return target >= req
}

func parseSM(s string) int {
	s = strings.TrimPrefix(s, "sm_")
	var num int
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
			num = num*10 + int(ch-'0')
		} else {
			break
		}
	}
	return num
}

// queryLiveDeviceSMCount attempts to query the physical SM count from a live CUDA device.
func queryLiveDeviceSMCount() int {
	defer func() {
		_ = recover()
	}()
	dev, err := cuda.NewDevice(0)
	if err != nil || dev == nil {
		return 0
	}
	defer dev.Close()
	count, err := dev.MultiprocessorCount()
	if err != nil || count <= 0 {
		return 0
	}
	return count
}

// queryLiveDeviceArch attempts to query the compute capability from a live CUDA device.
func queryLiveDeviceArch() string {
	defer func() {
		_ = recover()
	}()
	dev, err := cuda.NewDevice(0)
	if err != nil || dev == nil {
		return ""
	}
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil || major <= 0 {
		return ""
	}
	return fmt.Sprintf("sm_%d%d", major, minor)
}

type deviceSource int

const (
	sourceUnknown deviceSource = iota
	sourceCapture
	sourceLiveDevice
)

// determineArchWithSource determines the GPU architecture/compute capability by inspecting
// capture metadata/PTX/provenance, querying a live device if available, or returning empty string if unknown.
func determineArchWithSource(cap *cudacapture.Capture) (string, deviceSource) {
	if cap != nil {
		if arch := parseArchFromCapture(cap); arch != "" {
			return arch, sourceCapture
		}
	}
	if arch := queryLiveDeviceArch(); arch != "" {
		return arch, sourceLiveDevice
	}
	return "", sourceUnknown
}

func determineArch(cap *cudacapture.Capture) string {
	arch, _ := determineArchWithSource(cap)
	return arch
}

// determineDeviceLimits looks up the physical max warps per SM and max blocks per SM
// keyed by compute capability via x/cudaoccupancy. Returns ok=false if architecture is unknown.
func determineDeviceLimits(cap *cudacapture.Capture) (maxWarps int, maxBlocks int, src deviceSource, ok bool) {
	arch, src := determineArchWithSource(cap)
	if arch != "" {
		if dev, err := cudaoccupancy.LookupDevice(arch); err == nil && dev.MaxWarpsPerSM > 0 {
			return dev.MaxWarpsPerSM, dev.MaxBlocksPerSM, src, true
		}
	}
	return 0, 0, sourceUnknown, false
}

// determineSMCount determines the device SM count by inspecting capture metadata,
// looking up the capture's target architecture, or querying a live device as fallback.
// It returns ok=false when the hardware SM count cannot be determined.
func determineSMCount(cap *cudacapture.Capture) (int, deviceSource, bool) {
	// 1. Explicit SM count from capture metadata/provenance.
	if cap != nil {
		if count := parseSMCountFromCapture(cap); count > 0 {
			return count, sourceCapture, true
		}
	}

	// 2. SM count derived from the capture's own target architecture / PTX.
	if cap != nil {
		if arch := parseArchFromCapture(cap); arch != "" {
			if count := smCountForArch(arch); count > 0 {
				return count, sourceCapture, true
			}
		}
	}

	// 3. Fall back to live CUDA device hardware only when capture has no device identity.
	if count := queryLiveDeviceSMCount(); count > 0 {
		return count, sourceLiveDevice, true
	}

	return 0, sourceUnknown, false
}

func parseSMCountFromCapture(cap *cudacapture.Capture) int {
	if cap == nil {
		return 0
	}
	if len(cap.Metadata) > 0 {
		if count := parseSMCountFromJSON(cap.Metadata); count > 0 {
			return count
		}
	}
	for _, name := range []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			if count := parseSMCountFromJSON(data); count > 0 {
				return count
			}
		}
	}
	if cap.Profile != nil {
		for _, c := range cap.Profile.Comments {
			cLower := strings.ToLower(c)
			switch {
			case strings.Contains(cLower, "gb10"):
				return 48
			case strings.Contains(cLower, "gb200"):
				return 136
			case strings.Contains(cLower, "b200") || strings.Contains(cLower, "b100"):
				return 132
			case strings.Contains(cLower, "h100"):
				return 132
			case strings.Contains(cLower, "rtx 4090"):
				return 128
			case strings.Contains(cLower, "a100"):
				return 108
			}
		}
	}
	return 0
}

func parseSMCountFromJSON(data []byte) int {
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return 0
	}
	return findSMCountInMap(m)
}

func findSMCountInMap(m map[string]any) int {
	keys := []string{"sm_count", "smCount", "multiprocessor_count", "multiprocessorCount", "sms", "sm_num"}
	for _, k := range keys {
		if v, ok := m[k]; ok {
			if f, ok := v.(float64); ok && f > 0 {
				return int(f)
			}
			if s, ok := v.(string); ok {
				if n, err := strconv.Atoi(s); err == nil && n > 0 {
					return n
				}
			}
		}
	}
	if dev, ok := m["device"].(string); ok && dev != "" {
		devLower := strings.ToLower(dev)
		switch {
		case strings.Contains(devLower, "gb10"):
			return 48
		case strings.Contains(devLower, "gb200"):
			return 136
		case strings.Contains(devLower, "b200") || strings.Contains(devLower, "b100"):
			return 132
		case strings.Contains(devLower, "h100"):
			return 132
		case strings.Contains(devLower, "rtx 4090"):
			return 128
		case strings.Contains(devLower, "a100"):
			return 108
		}
	}
	for _, v := range m {
		if subMap, ok := v.(map[string]any); ok {
			if count := findSMCountInMap(subMap); count > 0 {
				return count
			}
		}
	}
	return 0
}

func parseArchFromCapture(cap *cudacapture.Capture) string {
	if cap == nil {
		return ""
	}
	if ptxData, _, _ := findPTX(cap); len(ptxData) > 0 {
		if mod, err := cudaptx.ParseBytes(ptxData); err == nil && mod != nil && mod.Target != "" {
			return mod.Target
		}
	}
	if len(cap.Metadata) > 0 {
		var m map[string]any
		if json.Unmarshal(cap.Metadata, &m) == nil {
			if arch := findArchInMap(m); arch != "" {
				return arch
			}
		}
	}
	for _, name := range []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			var m map[string]any
			if json.Unmarshal(data, &m) == nil {
				if arch := findArchInMap(m); arch != "" {
					return arch
				}
			}
		}
	}
	if sassData, ok := findSASS(cap); ok && len(sassData) > 0 {
		lines := strings.SplitN(string(sassData), "\n", 30)
		for _, line := range lines {
			lineLower := strings.ToLower(line)
			if idx := strings.Index(lineLower, "sm_"); idx != -1 {
				end := idx + 3
				for end < len(lineLower) && ((lineLower[end] >= '0' && lineLower[end] <= '9') || lineLower[end] == 'a') {
					end++
				}
				if end > idx+3 {
					return lineLower[idx:end]
				}
			}
		}
	}
	return ""
}

func findArchInMap(m map[string]any) string {
	for _, k := range []string{"arch", "architecture", "target", "compute_capability", "computeCapability", "chip", "chip_name", "sm"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	for _, v := range m {
		if subMap, ok := v.(map[string]any); ok {
			if arch := findArchInMap(subMap); arch != "" {
				return arch
			}
		}
	}
	return ""
}

func smCountForArch(arch string) int {
	arch = strings.ToLower(strings.TrimSpace(arch))
	arch = strings.TrimPrefix(arch, "compute_")
	arch = strings.TrimPrefix(arch, "sm_")
	switch {
	case strings.HasPrefix(arch, "121"):
		// GB10 (SM 12.1). Measured on hardware; see x/cudaoccupancy SM121.
		// GB200 is sm_100a, not sm_121a.
		return 48
	case strings.HasPrefix(arch, "120"):
		return 136 // Blackwell sm_120
	case strings.HasPrefix(arch, "100"):
		return 132 // Blackwell sm_100 (B100)
	case strings.HasPrefix(arch, "90"):
		return 132 // Hopper H100 SXM (132 SMs) / PCIe (114 SMs)
	case strings.HasPrefix(arch, "89"):
		return 128 // Ada Lovelace AD102 (RTX 4090: 128, L40S: 142)
	case strings.HasPrefix(arch, "86"):
		return 84 // Ampere GA102
	case strings.HasPrefix(arch, "80"):
		return 108 // Ampere A100 SXM (108 SMs)
	case strings.HasPrefix(arch, "75"):
		return 72 // Turing TU102
	case strings.HasPrefix(arch, "70"):
		return 80 // Volta GV100
	default:
		return 0
	}
}

type launchRecord struct {
	Kernel       string
	StreamID     uint32
	BlockThreads int
	GridBlocks   int
	Location     Location
}

func extractLaunches(cap *cudacapture.Capture) []launchRecord {
	if cap == nil {
		return nil
	}
	var launches []launchRecord

	// 1. From cap.Profile.Sample
	if cap.Profile != nil {
		for _, s := range cap.Profile.Sample {
			kName := extractKernelNameFromSample(s)
			if kName == "" {
				continue
			}
			var bThreads, gBlocks int
			if len(s.NumLabel["block_threads"]) > 0 {
				bThreads = int(s.NumLabel["block_threads"][0])
			} else if len(s.NumLabel["threads"]) > 0 {
				bThreads = int(s.NumLabel["threads"][0])
			} else if len(s.NumLabel["block_x"]) > 0 {
				bx := int(s.NumLabel["block_x"][0])
				by, bz := 1, 1
				if len(s.NumLabel["block_y"]) > 0 && s.NumLabel["block_y"][0] > 0 {
					by = int(s.NumLabel["block_y"][0])
				}
				if len(s.NumLabel["block_z"]) > 0 && s.NumLabel["block_z"][0] > 0 {
					bz = int(s.NumLabel["block_z"][0])
				}
				bThreads = bx * by * bz
			} else if len(s.Label["block"]) > 0 {
				bThreads = parseDimProduct(s.Label["block"][0])
			}

			if len(s.NumLabel["grid_blocks"]) > 0 {
				gBlocks = int(s.NumLabel["grid_blocks"][0])
			} else if len(s.NumLabel["blocks"]) > 0 {
				gBlocks = int(s.NumLabel["blocks"][0])
			} else if len(s.NumLabel["grid_x"]) > 0 {
				gx := int(s.NumLabel["grid_x"][0])
				gy, gz := 1, 1
				if len(s.NumLabel["grid_y"]) > 0 && s.NumLabel["grid_y"][0] > 0 {
					gy = int(s.NumLabel["grid_y"][0])
				}
				if len(s.NumLabel["grid_z"]) > 0 && s.NumLabel["grid_z"][0] > 0 {
					gz = int(s.NumLabel["grid_z"][0])
				}
				gBlocks = gx * gy * gz
			} else if len(s.Label["grid"]) > 0 {
				gBlocks = parseDimProduct(s.Label["grid"][0])
			}

			if bThreads > 0 || gBlocks > 0 {
				var loc Location
				loc.Kernel = kName
				if len(s.Location) > 0 && len(s.Location[0].Line) > 0 {
					line := s.Location[0].Line[0]
					if line.Function != nil && line.Function.Filename != "" {
						loc.File = line.Function.Filename
						loc.Line = int(line.Line)
					}
				}
				launches = append(launches, launchRecord{
					Kernel:       kName,
					BlockThreads: bThreads,
					GridBlocks:   gBlocks,
					Location:     loc,
				})
			}
		}
	}

	// 2. From cap.Files (activities.json, capture-status.json, etc.)
	for _, name := range []string{"activities.json", "capture-status.json", "status.json", "trace.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			fileLaunches := parseLaunchesFromJSON(data)
			launches = append(launches, fileLaunches...)
		}
	}

	// 3. From PTX
	if ptxData, ptxFile, _ := findPTX(cap); len(ptxData) > 0 {
		if mod, err := cudaptx.ParseBytes(ptxData); err == nil && mod != nil {
			for _, k := range mod.Kernels {
				threads := 0
				if k.Tuning.ReqNTID[0] > 0 {
					threads = k.Tuning.ReqNTID[0] * max(1, k.Tuning.ReqNTID[1]) * max(1, k.Tuning.ReqNTID[2])
				} else if k.Tuning.MaxNTID[0] > 0 {
					threads = k.Tuning.MaxNTID[0] * max(1, k.Tuning.MaxNTID[1]) * max(1, k.Tuning.MaxNTID[2])
				}
				if threads > 0 {
					launches = append(launches, launchRecord{
						Kernel:       k.Name,
						BlockThreads: threads,
						Location: Location{
							Kernel: k.Name,
							File:   ptxFile,
							Line:   k.StartLine,
						},
					})
				}
			}
		}
	}
	return launches
}

func auditLaunchGeometry(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	launches := extractLaunches(cap)

	if cap.Profile != nil && len(cap.Profile.Sample) > 0 && len(launches) == 0 {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityInfo,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceUnmeasured,
			Finding:                  "Launch geometry labels ('grid', 'block') are absent in this profile (requires profiling with updated cupti_agent); launch geometry and SM occupancy audits skipped.",
			ActionableRecommendation: "Re-capture the profile using an updated libcupti_agent that serializes kernel launch dimensions.",
			NextMeasurement:          "launch geometry attributes ('grid', 'block') in profile samples",
		})
		return
	}

	smCount, smSrc, hasSMCount := determineSMCount(cap)
	maxWarpsPerSM, maxBlocksPerSM, limitSrc, hasLimits := determineDeviceLimits(cap)

	if !hasLimits && !hasSMCount && cap.Profile != nil && len(launches) > 0 {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityInfo,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceUnmeasured,
			Finding:                  "Device architecture and SM count metadata are not recorded in this capture, and no local CUDA device is available; theoretical SM occupancy and SM grid starvation audits were skipped.",
			ActionableRecommendation: "Record device provenance in the capture (e.g. capture-provenance.json) or run advise on a host equipped with the target GPU architecture.",
			NextMeasurement:          "capture-provenance.json with device compute capability and SM count",
		})
	}

	// 1. Underpopulated Thread Block / Single-Warp Warning (<= 32 threads)
	if hasLimits {
		seenSingleWarp := make(map[string]bool)
		for _, launch := range launches {
			if launch.BlockThreads > 0 && launch.BlockThreads <= 32 {
				if seenSingleWarp[launch.Kernel] {
					continue
				}
				// Gating: suppress single-block launches (Grid(1)), which are whole-GPU
				// starvation issues rather than CTA-geometry issues.
				if launch.GridBlocks == 1 {
					continue
				}

				warpsPerBlock := (launch.BlockThreads + 31) / 32
				if warpsPerBlock <= 0 {
					warpsPerBlock = 1
				}
				blocksPerSM := maxBlocksPerSM
				if maxWarpsPerSM/warpsPerBlock < blocksPerSM {
					blocksPerSM = maxWarpsPerSM / warpsPerBlock
				}
				residentWarpsPerSM := blocksPerSM * warpsPerBlock
				occupancyPct := (residentWarpsPerSM * 100) / maxWarpsPerSM

				// Gating: fire only when the launch's geometry leaves the SM underfilled
				// (i.e. capped by maxBlocksPerSM below the hardware's warp limit).
				if residentWarpsPerSM >= maxWarpsPerSM {
					continue
				}
				seenSingleWarp[launch.Kernel] = true

				provSuffix := ""
				if limitSrc == sourceLiveDevice {
					provSuffix = " (hardware limits inferred from local host device, not recorded in capture)"
				}
				finding := fmt.Sprintf("Single-warp thread blocks limit theoretical SM occupancy: kernel %q launched with %d block threads (<= 32) reaches %d%% theoretical occupancy with %d/%d resident warps/SM across %d blocks/SM%s",
					launch.Kernel, launch.BlockThreads, occupancyPct, residentWarpsPerSM, maxWarpsPerSM, blocksPerSM, provSuffix)

				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
					Category:                 CategoryArchitectureCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 launch.Location,
					Finding:                  finding,
					ActionableRecommendation: "Restructure thread block geometry to >=128 threads (4 warps) to enable warp-level latency hiding." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			}
		}
	}

	// Multi-Warp Sequential Reduction Starvation (e.g. ds4_f16_pair_ordered_chunks)
	seenChunkReduction := make(map[string]bool)
	for _, launch := range launches {
		if launch.BlockThreads >= 128 {
			kLower := strings.ToLower(launch.Kernel)
			if strings.Contains(kLower, "ordered_chunks") || strings.Contains(kLower, "pair_ordered") || (strings.Contains(kLower, "chunk") && strings.Contains(kLower, "reduce")) {
				if seenChunkReduction[launch.Kernel] {
					continue
				}
				seenChunkReduction[launch.Kernel] = true
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
					Category:                 CategoryArchitectureCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 launch.Location,
					Finding:                  fmt.Sprintf("Multi-warp thread block underutilization in row reduction: kernel %q launched with %d threads/block (>= 4 warps) executes sequential chunk reductions with idle warps during accumulation", launch.Kernel, launch.BlockThreads),
					ActionableRecommendation: "Group multiple rows per thread block (e.g. 4 rows per 128-thread block so warp i computes row 4k+i) to maintain 100% warp utilization during chunk accumulation." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			}
		}
	}

	// 2. SM Grid Starvation (< SMCount)
	//
	// One finding for all of them, and it recommends nothing.
	//
	// This was one finding per kernel, each recommending 2D tiling or split-K.
	// On a ds4 capture that was 23 of 66 findings, all saying the same thing,
	// and the recommendation named a cause that did not exist: no grid in that
	// attention path consults any device property, so there was no tuning
	// decision to revisit. The grids were literals mirroring a .cu define, and
	// they were correct -- the work really is that shape.
	//
	// A capture cannot tell a grid that was tuned from one that was fixed: it
	// sees a grid and an SM count. Absence of a device-property query is not
	// evidence either, only absence of visibility. So the observation stays,
	// because it is true and sometimes the reason, and the prescription goes.
	if hasSMCount {
		type starved struct {
			kernel  string
			blocks  int
			idlePct int
		}
		var list []starved
		seenGridStarvation := make(map[string]bool)
		for _, launch := range launches {
			if launch.GridBlocks > 0 && launch.GridBlocks < smCount {
				if seenGridStarvation[launch.Kernel] {
					continue
				}
				seenGridStarvation[launch.Kernel] = true
				list = append(list, starved{launch.Kernel, launch.GridBlocks, (smCount - launch.GridBlocks) * 100 / smCount})
			}
		}
		if len(list) > 0 {
			sort.Slice(list, func(i, j int) bool {
				if list[i].idlePct != list[j].idlePct {
					return list[i].idlePct > list[j].idlePct
				}
				return list[i].kernel < list[j].kernel
			})
			// Named in the finding, not a table: the finding renders as a
			// markdown heading, so it has to stay one line. The cap keeps a
			// capture with hundreds of kernels from filling the report with
			// the same sentence it just collapsed.
			const named = 8
			var parts []string
			for i, st := range list {
				if i == named {
					parts = append(parts, fmt.Sprintf("and %d more", len(list)-named))
					break
				}
				parts = append(parts, fmt.Sprintf("%s (%d blocks, %d%% idle)", st.kernel, st.blocks, st.idlePct))
			}
			provSuffix := ""
			if smSrc == sourceLiveDevice {
				provSuffix = " (SM count inferred from local host device, not recorded in capture)"
			}
			loc := Location{}
			if len(list) == 1 {
				loc = Location{Kernel: list[0].kernel}
			}
			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityInfo,
				Category:      CategoryArchitecture,
				EvidenceLevel: EvidenceObserved,
				Location:      loc,
				Finding: fmt.Sprintf("%d kernel(s) launched with fewer grid blocks than the GPU has SMs (%d SMs)%s: %s. Whether that is a shortfall depends on how much work there is; this capture does not record how the grids were chosen.",
					len(list), smCount, provSuffix, strings.Join(parts, ", ")),
				ActionableRecommendation: "Compare achieved occupancy against these grids before changing any of them. A grid smaller than the SM count is a fact about the launch, not a defect: it is a problem only where there is more work available than the grid exposes.",
				NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active per kernel, against the grid sizes above",
			})
		}
	}
}

// dim3Threads returns the product of a launch dimension, or 0 when any
// component is unset, which means the capture did not record the geometry.
func dim3Threads(d [3]int) int {
	n := 1
	for _, v := range d {
		if v <= 0 {
			return 0
		}
		n *= v
	}
	return n
}

// underSubscribedStat accumulates the launches of one kernel that cannot
// occupy the device.
type underSubscribedStat struct {
	kernel  string
	threads int
	count   int
	totalNs uint64
}

// formatUnderSubscribed names the worst offenders by accumulated device time.
func formatUnderSubscribed(stats map[string]*underSubscribedStat) string {
	ordered := make([]*underSubscribedStat, 0, len(stats))
	for _, st := range stats {
		ordered = append(ordered, st)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].totalNs != ordered[j].totalNs {
			return ordered[i].totalNs > ordered[j].totalNs
		}
		return ordered[i].kernel < ordered[j].kernel
	})
	var parts []string
	for i := 0; i < len(ordered) && i < 3; i++ {
		st := ordered[i]
		meanUs := float64(st.totalNs) / float64(st.count) / 1000.0
		threadWord := "threads"
		if st.threads == 1 {
			threadWord = "thread"
		}
		parts = append(parts, fmt.Sprintf("%s (%d %s, %d launches, %.1f µs mean, %.2f ms total)", cleanKernelName(st.kernel), st.threads, threadWord, st.count, meanUs, float64(st.totalNs)/1e6))
	}
	return strings.Join(parts, ", ")
}

// worstKernel returns the kernel holding the most device time in stats.
func worstKernel(stats map[string]*underSubscribedStat) (string, uint64) {
	var name string
	var best uint64
	for _, st := range stats {
		if st.totalNs > best || (st.totalNs == best && (name == "" || st.kernel < name)) {
			name, best = st.kernel, st.totalNs
		}
	}
	return name, best
}

// auditUnderSubscribedLaunches reports launches whose total thread count cannot
// occupy the device: single-threaded launches, and launches that cannot supply
// one warp per SM. The per-SM rule needs the device SM count; when the capture
// does not record it and no local device answers, only the single-thread rule
// runs and the finding says so.
func auditUnderSubscribedLaunches(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	streamEvents := extractStreamEvents(cap)
	if len(streamEvents) == 0 {
		return
	}

	smCount, smSrc, hasSMCount := determineSMCount(cap)
	warpPerSMThreads := 0
	if hasSMCount {
		warpPerSMThreads = smCount * warpSize
	}

	single := make(map[string]*underSubscribedStat)
	subWarp := make(map[string]*underSubscribedStat)
	var singleCount, subWarpCount, geometryLaunches int
	var singleNs, subWarpNs uint64

	accumulate := func(stats map[string]*underSubscribedStat, ev streamEvent, threads int) {
		st := stats[ev.Name]
		if st == nil {
			st = &underSubscribedStat{kernel: ev.Name, threads: threads}
			stats[ev.Name] = st
		}
		if threads < st.threads {
			st.threads = threads
		}
		st.count++
		st.totalNs += ev.DurationNs
	}

	for _, events := range streamEvents {
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if sub != "kernel" && sub != "" {
				continue
			}
			threads := dim3Threads(ev.Grid) * dim3Threads(ev.Block)
			if threads <= 0 || ev.Name == "" {
				continue
			}
			geometryLaunches++
			switch {
			case threads == 1:
				singleCount++
				singleNs += ev.DurationNs
				accumulate(single, ev, threads)
			case warpPerSMThreads > 0 && threads < warpPerSMThreads:
				subWarpCount++
				subWarpNs += ev.DurationNs
				accumulate(subWarp, ev, threads)
			}
		}
	}
	if geometryLaunches == 0 {
		return
	}

	if singleCount > 0 {
		finding := fmt.Sprintf("Single-threaded kernel launches leave the device idle: %d of %d captured launches (%.1f%%) ran one thread (grid 1 x block 1), consuming %.2f ms of device time; worst offenders: %s",
			singleCount, geometryLaunches, float64(singleCount)*100.0/float64(geometryLaunches), float64(singleNs)/1e6, formatUnderSubscribed(single))
		if !hasSMCount {
			finding += " (device SM count is not recorded in this capture and no local CUDA device answered, so the one-warp-per-SM rule was skipped)"
		}
		kernel, _ := worstKernel(single)
		report.Items = append(report.Items, AdviceItem{
			Severity:        SeverityCritical,
			Category:        CategoryArchitectureCompute,
			EvidenceLevel:   EvidenceObserved,
			DeviceTimeNanos: int64(singleNs),
			Location: Location{
				Kernel: kernel,
			},
			Finding:                  finding,
			ActionableRecommendation: "Parallelize these kernels across threads and blocks, or fold their work into an adjacent kernel: a single-thread launch occupies one lane of one warp on one SM and leaves the rest of the device idle for its whole duration.",
			NextMeasurement:          "A/B wall-clock timing with parallelized kernel",
		})
	}

	if subWarpCount > 0 {
		provSuffix := ""
		if smSrc == sourceLiveDevice {
			provSuffix = " (SM count inferred from local host device, not recorded in capture)"
		}
		kernel, _ := worstKernel(subWarp)
		report.Items = append(report.Items, AdviceItem{
			Severity:        SeverityWarning,
			Category:        CategoryArchitectureCompute,
			EvidenceLevel:   EvidenceSuspected,
			DeviceTimeNanos: int64(subWarpNs),
			Location: Location{
				Kernel: kernel,
			},
			Finding: fmt.Sprintf("Launch geometry cannot fill one warp per SM: %d of %d captured launches (%.1f%%) ran fewer than %d total threads (%d SMs x %d threads per warp), consuming %.2f ms of device time; worst offenders: %s%s",
				subWarpCount, geometryLaunches, float64(subWarpCount)*100.0/float64(geometryLaunches), warpPerSMThreads, smCount, warpSize, float64(subWarpNs)/1e6, formatUnderSubscribed(subWarp), provSuffix),
			ActionableRecommendation: fmt.Sprintf("Grow the grid so at least one warp is resident per SM (>= %d total threads), or batch the independent launches of this kernel into a single dispatch.", warpPerSMThreads),
			NextMeasurement:          "A/B wall-clock timing with increased grid size",
		})
	}
}

// auditCUBINResourceOccupancy inspects CUBIN ELF .nv.info sections (EIATTR_REG_COUNT,
// EIATTR_SHARED_SIZE_SYSMEM, EIATTR_FRAME_SIZE, EIATTR_MAXREG_COUNT) using x/cudaoccupancy
// to calculate target-specific SM warp occupancy, occupancy cliffs, and hardware limiting factors.
func auditCUBINResourceOccupancy(cap *cudacapture.Capture, report *AdviceReport, launchesByKernel map[string]launchRecord) {
	if cap == nil {
		return
	}
	cubinData, ok := findCUBIN(cap)
	if !ok || len(cubinData) == 0 {
		return
	}

	infos, err := cudaoccupancy.ReadKernelInfoBytes(cubinData)
	if err != nil || len(infos) == 0 {
		return
	}

	arch := parseArchFromCapture(cap)
	dev, deviceErr := cudaoccupancy.LookupDevice(arch)

	for _, k := range infos {
		threadsPerBlock := 128
		if l, ok := launchesByKernel[k.Name]; ok && l.BlockThreads > 0 {
			threadsPerBlock = l.BlockThreads
		} else if k.ReqNTID[0] > 0 {
			threadsPerBlock = k.ReqNTID[0] * max(1, k.ReqNTID[1]) * max(1, k.ReqNTID[2])
		}

		// 1. Stack frame size (EIATTR_FRAME_SIZE)
		if k.FrameSizeBytes > 0 {
			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityCritical,
				Category:      CategoryRegisterPressure,
				EvidenceLevel: EvidenceObserved,
				Location:      Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q allocates %d bytes of local stack frame per thread in CUBIN ELF (EIATTR_FRAME_SIZE), causing spills to high-latency local memory",
					k.Name, k.FrameSizeBytes),
				ActionableRecommendation: "Reduce local variable lifetimes, eliminate large stack arrays, or apply __launch_bounds__ to eliminate local memory stack frame spills.",
				NextMeasurement:          "verify zero local memory traffic with l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
			})
		}

		// 2. Compiler register limit (EIATTR_MAXREG_COUNT)
		if k.MaxRegCount > 0 && k.Registers >= k.MaxRegCount {
			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityWarning,
				Category:      CategoryRegisterPressure,
				EvidenceLevel: EvidenceObserved,
				Location:      Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q reached the compiler register limit of %d GPRs (EIATTR_MAXREG_COUNT), which may force register spilling or sub-optimal instruction scheduling",
					k.Name, k.MaxRegCount),
				ActionableRecommendation: "Raise compiler register budget (-maxrregcount) or restructure compute loops to reduce register demand.",
				NextMeasurement:          "inspect SASS for spills or verify -maxrregcount with cuobjdump -res-usage",
			})
		}

		// 3. Analytical occupancy & hardware limiting factor
		// Unknown hardware cannot borrow another architecture's resource limits.
		if deviceErr != nil {
			continue
		}
		kRes := cudaoccupancy.KernelResource{
			Registers:       k.Registers,
			StaticSharedMem: k.SharedMemBytes,
		}
		rep := kRes.CalculateDevice(dev, threadsPerBlock, 0)
		if rep.Occupancy < 1.0 {
			warpsPerBlock := (threadsPerBlock + dev.WarpSize - 1) / dev.WarpSize
			if warpsPerBlock <= 0 {
				warpsPerBlock = 1
			}

			severity := SeverityWarning

			finding := fmt.Sprintf("Microarchitecture theoretical occupancy on %s for kernel %q is %.1f%% (%d/%d active warps, %d blocks/SM) constrained by %s (limits: %d blocks from registers, %d from smem, %d from block count, %d from block size)",
				dev.Architecture, k.Name, rep.Occupancy*100, rep.TheoreticalWarps, rep.MaxWarps, rep.ActiveBlocks, rep.LimitingFactor, rep.RegLimit, rep.SmemLimit, rep.BlockLimit, rep.WarpLimit)

			var rec string
			switch rep.LimitingFactor {
			case cudaoccupancy.LimitRegisters:
				targetBlocks := rep.ActiveBlocks + 1
				regsPerBlockTarget := dev.MaxRegistersPerSM / targetBlocks
				regsPerWarpTarget := regsPerBlockTarget / warpsPerBlock
				maxRegsTarget := regsPerWarpTarget / dev.WarpSize
				targetWarps := targetBlocks * warpsPerBlock
				targetOcc := (float64(targetWarps) / float64(dev.MaxWarpsPerSM)) * 100
				rec = fmt.Sprintf("Register file capacity (%d regs/SM) limits residency to %d blocks/SM. Reducing registers from %d to <=%d GPRs would allow %d blocks/SM (%d warps, %.1f%% occupancy). Test with __launch_bounds__ or -maxrregcount=%d.",
					dev.MaxRegistersPerSM, rep.ActiveBlocks, k.Registers, maxRegsTarget, targetBlocks, targetWarps, targetOcc, maxRegsTarget) + occupancyCaveat
			case cudaoccupancy.LimitSharedMemory:
				rec = fmt.Sprintf("Static shared memory (%d bytes/block) limits residency to %d blocks/SM on %s (%d bytes/SM). Reduce shared memory or dynamically allocate to allow higher CTA residency.",
					k.SharedMemBytes, rep.ActiveBlocks, dev.Architecture, dev.MaxSharedMemoryPerSM) + occupancyCaveat
			case cudaoccupancy.LimitBlocks:
				rec = fmt.Sprintf("Maximum thread blocks per SM (%d on %s) limits occupancy because block size (%d threads) is small. Restructure thread block geometry to >=128 or 256 threads to reach higher occupancy.",
					dev.MaxBlocksPerSM, dev.Architecture, threadsPerBlock) + occupancyCaveat
			case cudaoccupancy.LimitBlockSize:
				rec = "Restructure thread block geometry to fit within hardware block size limits (<= 1024 threads per block)."
			default:
				rec = "Tune thread block dimensions and resource consumption to maximize concurrent warp execution." + occupancyCaveat
			}

			report.Items = append(report.Items, AdviceItem{
				Severity:                 severity,
				Category:                 CategoryRegisterPressure,
				EvidenceLevel:            EvidenceSuspected,
				Location:                 Location{Kernel: k.Name},
				Finding:                  finding,
				ActionableRecommendation: rec,
				NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
			})
		}
	}
}

func parseDimProduct(s string) int {
	s = strings.Trim(s, "()[]{} ")
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == 'x' || r == 'X' || r == ' ' || r == '	'
	})
	prod := 1
	valid := false
	for _, p := range parts {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			prod *= n
			valid = true
		}
	}
	if !valid {
		return 0
	}
	return prod
}

func parseDim3(s string) [3]int {
	s = strings.Trim(s, "()[]{} ")
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == 'x' || r == 'X' || r == ' ' || r == '	'
	})
	var res [3]int
	res[0], res[1], res[2] = 1, 1, 1
	for i, p := range parts {
		if i >= 3 {
			break
		}
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			res[i] = n
		}
	}
	return res
}

func parseDim3FromSlice(items []any) [3]int {
	var res [3]int
	res[0], res[1], res[2] = 1, 1, 1
	for i, it := range items {
		if i >= 3 {
			break
		}
		if f, ok := it.(float64); ok && f > 0 {
			res[i] = int(f)
		}
	}
	return res
}

func parseLaunchesFromJSON(data []byte) []launchRecord {
	var launches []launchRecord
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}

	var items []any
	switch v := raw.(type) {
	case []any:
		items = v
	case map[string]any:
		if acts, ok := v["activities"].([]any); ok {
			items = acts
		} else if status, ok := v["status"].(map[string]any); ok {
			if acts, ok := status["activities"].([]any); ok {
				items = acts
			}
		}
	}

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		var kName string
		if s, ok := m["name"].(string); ok {
			kName = s
		} else if s, ok := m["kernel"].(string); ok {
			kName = s
		}
		if kName == "" {
			continue
		}

		var bThreads, gBlocks int
		if t, ok := m["threads"].(float64); ok && t > 0 {
			bThreads = int(t)
		}
		if t, ok := m["block_threads"].(float64); ok && t > 0 {
			bThreads = int(t)
		}
		if g, ok := m["grid_blocks"].(float64); ok && g > 0 {
			gBlocks = int(g)
		}
		if g, ok := m["blocks"].(float64); ok && g > 0 {
			gBlocks = int(g)
		}

		if blk, ok := m["block"].([]any); ok && len(blk) > 0 {
			dim := parseDim3FromSlice(blk)
			bThreads = dim[0] * max(1, dim[1]) * max(1, dim[2])
		}
		if grd, ok := m["grid"].([]any); ok && len(grd) > 0 {
			dim := parseDim3FromSlice(grd)
			gBlocks = dim[0] * max(1, dim[1]) * max(1, dim[2])
		}

		if bThreads > 0 || gBlocks > 0 {
			launches = append(launches, launchRecord{
				Kernel:       kName,
				BlockThreads: bThreads,
				GridBlocks:   gBlocks,
				Location:     Location{Kernel: kName},
			})
		}
	}
	return launches
}

func extractKernelNameFromSample(s *profile.Sample) string {
	if s == nil {
		return ""
	}
	if len(s.Label["kernel"]) > 0 && s.Label["kernel"][0] != "" {
		return s.Label["kernel"][0]
	}
	if len(s.Label["name"]) > 0 && s.Label["name"][0] != "" {
		return s.Label["name"][0]
	}
	if len(s.Location) > 0 {
		for _, loc := range s.Location {
			for _, line := range loc.Line {
				if line.Function != nil && line.Function.Name != "" {
					name := line.Function.Name
					name = strings.TrimSuffix(name, ".SASS")
					name = strings.TrimSuffix(name, ".PTX")
					name = strings.TrimSuffix(name, ".TILEIR")
					name = strings.TrimSuffix(name, ".inlined")
					return name
				}
			}
		}
	}
	return ""
}

func auditUncoalescedMemory(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}

	// A source-fusion failure zeroes every instruction's SourceFile and
	// SourceLine, which are exactly the fields the findings below cite. Left
	// undeclared, each finding reads as a location the analysis could not
	// determine rather than one the capture refused to bind.
	code, fusion, _ := loadCodeTiered(cap)
	if fusion != nil {
		report.SourceFusion = fusion.Error()
	}
	disasm, _ := loadDisassemblyWithPressure(cap)
	if disasm == nil {
		if sassData, ok := findSASS(cap); ok && len(sassData) > 0 {
			if parsed, err := cudasass.Parse(bytes.NewReader(sassData)); err == nil {
				disasm = parsed
			}
		}
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
				kName, pc = resolveLocationPC(s.Location[0], s, code, disasm)
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
							sev := SeverityWarning
							if ratio >= 0.50 {
								sev = SeverityCritical
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
						sev := SeverityWarning
						if sbRatio >= 0.50 {
							sev = SeverityCritical
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
						sev := SeverityWarning
						if brRatio >= 0.50 {
							sev = SeverityCritical
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
							ratio: wtRatio, stalls: stat.waitStalls, sev: SeverityWarning,
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
				report.Items = append(report.Items, AdviceItem{
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
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
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
				report.Items = append(report.Items, AdviceItem{
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
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
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
				brLimit = 3
			}
			for _, c := range brCands[:brLimit] {
				locationPC := c.pc
				attribution := fmt.Sprintf("sampled at %s", c.opcode)
				if c.branchFound {
					locationPC = c.branchPC
					attribution = fmt.Sprintf("created by preceding %s at PC 0x%04x; sampled at %s at PC 0x%04x, which is awaiting warp reconvergence", c.branchOpcode, c.branchPC, c.opcode, c.pc)
				}
				report.Items = append(report.Items, AdviceItem{
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

			// Report Barrier Synchronization Stalls (capped to 3)
			sort.Slice(wtCands, func(i, j int) bool { return wtCands[i].stalls > wtCands[j].stalls })
			wtLimit := len(wtCands)
			if wtLimit > 3 {
				wtLimit = 3
			}
			for _, c := range wtCands[:wtLimit] {
				report.Items = append(report.Items, AdviceItem{
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
	remarks, _, _ := findRemarks(cap)
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
				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
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
					report.Items = append(report.Items, AdviceItem{
						Severity:      SeverityWarning,
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

type streamEvent struct {
	Subtype    string
	Name       string
	Stream     uint32
	DurationNs uint64
	// Start and End are device timestamps. They are zero when the capture does
	// not carry per-activity timestamps.
	Start       uint64
	End         uint64
	Correlation uint32
	// GraphID names the CUDA graph this ran as a node of, or the graph a
	// replay envelope replayed. Zero means the record named none.
	GraphID uint32
	Shape   string
	Buffer  string
	Grid    [3]int
	Block   [3]int
}

const (
	// microKernelMedianNs is the median kernel duration below which a sequence
	// is short enough for per-launch host cost to be worth testing for.
	microKernelMedianNs = 10000
	// lowDutyCyclePct is the GPU duty cycle below which a sequence's device
	// span is mostly idle.
	lowDutyCyclePct = 75.0
	// hostBehindFraction is the share of a sequence's measured launches that
	// must start after the preceding kernel ended before the host counts as
	// persistently behind the device.
	hostBehindFraction = 0.75
	// warpSize is the hardware warp width, constant across every CUDA
	// architecture cudaprof supports.
	warpSize = 32
)

// apiCall is a host driver or runtime API call. Its timestamps describe the
// host call, not the GPU work it launched.
type apiCall struct {
	Start uint64
	End   uint64
}

// extractAPIRecords indexes host launch API records by correlation ID. Captures
// written through compactCaptureStatus omit the per-record array, so the result
// is empty unless the capture retains api_records.
func extractAPIRecords(cap *cudacapture.Capture) map[uint32]apiCall {
	res := make(map[uint32]apiCall)
	if cap == nil {
		return res
	}
	for _, name := range []string{"capture-status.json", "status.json", "activities.json", "trace.json"} {
		data, ok := cap.Files[name]
		if !ok || len(data) == 0 {
			continue
		}
		var raw map[string]any
		if json.Unmarshal(data, &raw) != nil {
			continue
		}
		items, _ := raw["api_records"].([]any)
		if items == nil {
			if nested, ok := raw["status"].(map[string]any); ok {
				items, _ = nested["api_records"].([]any)
			}
		}
		for _, item := range items {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			corr, ok := m["correlation"].(float64)
			if !ok || corr <= 0 {
				continue
			}
			start, _ := m["start"].(float64)
			if start <= 0 {
				continue
			}
			end, _ := m["end"].(float64)
			res[uint32(corr)] = apiCall{Start: uint64(start), End: uint64(end)}
		}
		if len(res) > 0 {
			return res
		}
	}
	// Same reason the timeline falls back to the journal: a status file that
	// omits the per-record array says nothing about the capture, and the
	// finding built on its silence claims the launches were never recorded.
	if fromJournal, err := journalAPIRecords(cap); err == nil && len(fromJournal) > 0 {
		return fromJournal
	}
	return res
}

// sequenceDutyCycle returns the percentage of the sequence's device span spent
// executing its kernels. ok is false when the capture does not carry usable
// timestamps for every kernel in the sequence.
func sequenceDutyCycle(seq []streamEvent) (pct float64, ok bool) {
	if len(seq) == 0 {
		return 0, false
	}
	var busy, first, last uint64
	for _, ev := range seq {
		if ev.Start == 0 || ev.End < ev.Start {
			return 0, false
		}
		busy += ev.DurationNs
		if first == 0 || ev.Start < first {
			first = ev.Start
		}
		if ev.End > last {
			last = ev.End
		}
	}
	span := last - first
	if span == 0 {
		return 0, false
	}
	return float64(busy) * 100.0 / float64(span), true
}

// hostBehindLaunches counts the sequence launches whose host API call started
// only after the preceding kernel had already finished on the device, i.e. the
// launches for which the device waited on the host. measured counts the
// launches for which both timestamps were available.
func hostBehindLaunches(seq []streamEvent, apis map[uint32]apiCall) (behind, measured int) {
	if len(apis) == 0 {
		return 0, 0
	}
	for i := 1; i < len(seq); i++ {
		prevEnd := seq[i-1].End
		call, ok := apis[seq[i].Correlation]
		if !ok || prevEnd == 0 || call.Start == 0 {
			continue
		}
		measured++
		if call.Start > prevEnd {
			behind++
		}
	}
	return behind, measured
}

// sampleCorrelation returns the CUPTI correlation ID recorded on a sample, or
// zero when the capture carries none.
func sampleCorrelation(s *profile.Sample) uint32 {
	if s == nil {
		return 0
	}
	if len(s.NumLabel["correlation"]) > 0 && s.NumLabel["correlation"][0] > 0 {
		return uint32(s.NumLabel["correlation"][0])
	}
	if len(s.Label["correlation"]) > 0 {
		if n, err := strconv.ParseUint(s.Label["correlation"][0], 10, 32); err == nil {
			return uint32(n)
		}
	}
	return 0
}

// sampleGraphID returns the CUDA graph ID recorded on a kernel sample, or zero
// when the sample carries none. CUPTI reports graphId zero for a kernel that is
// not a graph node, and cudacapture omits the label entirely in that case, so
// absence and zero mean the same thing and both answer "not a graph node".
func sampleGraphID(s *profile.Sample) uint32 {
	if s == nil {
		return 0
	}
	if len(s.NumLabel["graph_id"]) > 0 && s.NumLabel["graph_id"][0] > 0 {
		return uint32(s.NumLabel["graph_id"][0])
	}
	if len(s.Label["graph_id"]) > 0 {
		if n, err := strconv.ParseUint(s.Label["graph_id"][0], 10, 32); err == nil {
			return uint32(n)
		}
	}
	return 0
}

// capturedGraphReplays reports how many CUDA graph launches a capture holds.
// It answers zero for a capture with no graph work, and the callers use it only
// to suppress advice that recommends graph capture, never to raise advice.
//
// Two sources of evidence, in order of strength:
//
// Direct. CUPTI records graphId and graphNodeId on every kernel activity record
// that ran as a node of a graph. Each replay is one cudaGraphLaunch and so one
// correlation ID, so counting the distinct correlations among kernel samples
// that carry a graph ID counts replays.
//
// Indirect, for captures taken before the agent forwarded graphId. Under replay
// every node inherits the correlation ID of the single launch that issued it,
// so one correlation spread across several distinct kernel records is the
// device-side signature of a graph launch.
//
// The indirect form has a false negative the direct form does not: a graph of a
// SINGLE node gives each replay one kernel record with its own correlation,
// which is indistinguishable from an ordinary launch by multiplicity alone.
// Captures that carry graphId are not subject to this.
//
// Measured on GB10 (sm_121), one kernel, identical geometry, 20 executions per
// arm: launched 20 times directly the capture held 20 kernel records with 20
// distinct correlation IDs; a four-node graph replayed five times held 20
// kernel records with 5 correlation IDs, four records each, and each of those
// five resolved to a runtime cudaGraphLaunch API record.
func capturedGraphReplays(cap *cudacapture.Capture) int {
	if cap == nil || cap.Profile == nil {
		return 0
	}
	graphLaunches := make(map[uint32]struct{})
	counts := make(map[uint32]int)
	for _, s := range cap.Profile.Sample {
		if extractKernelNameFromSample(s) == "" {
			continue
		}
		corr := sampleCorrelation(s)
		if sampleGraphID(s) != 0 {
			// A graph node with no correlation still witnesses a replay; it
			// just cannot be told apart from another replay of the same graph.
			graphLaunches[corr] = struct{}{}
			continue
		}
		if corr != 0 {
			counts[corr]++
		}
	}
	if len(graphLaunches) > 0 {
		return len(graphLaunches)
	}
	shared := 0
	for _, n := range counts {
		if n > 1 {
			shared++
		}
	}
	return shared
}

func auditStreamKernelSequences(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}

	streamEvents := extractStreamEvents(cap)
	if len(streamEvents) == 0 {
		return
	}
	apis := extractAPIRecords(cap)
	graphReplays := capturedGraphReplays(cap)
	seenDutyUnmeasured := make(map[uint32]bool)

	streams := make([]uint32, 0, len(streamEvents))
	for s := range streamEvents {
		streams = append(streams, s)
	}
	sort.Slice(streams, func(i, j int) bool { return streams[i] < streams[j] })

	for _, streamID := range streams {
		events := streamEvents[streamID]
		if len(events) == 0 {
			continue
		}

		// Stream-wide micro-kernel telemetry (< 10 µs)
		var streamMicroCount int
		var streamMicroTotalNs uint64
		streamMicroKernelCounts := make(map[string]int)
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if (sub == "kernel" || sub == "") && ev.DurationNs < microKernelMedianNs && ev.DurationNs > 0 {
				streamMicroCount++
				streamMicroTotalNs += ev.DurationNs
				streamMicroKernelCounts[cleanKernelName(ev.Name)]++
			}
		}

		// 1. Heuristic 4: CUDA Graph Launch Sequence Candidates
		var kernelSeq []streamEvent
		checkGraphCandidate := func() {
			if len(kernelSeq) >= 5 {
				durations := make([]uint64, len(kernelSeq))
				for i, k := range kernelSeq {
					durations[i] = k.DurationNs
				}
				sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
				var median uint64
				n := len(durations)
				if n%2 == 1 {
					median = durations[n/2]
				} else {
					median = (durations[n/2-1] + durations[n/2]) / 2
				}

				if median < microKernelMedianNs {
					detail := fmt.Sprintf("sequence of %d consecutive kernels on stream %d has median duration %.2f µs (< %d µs)", len(kernelSeq), streamID, float64(median)/1000.0, microKernelMedianNs/1000)
					if streamMicroCount >= 5 {
						detail += fmt.Sprintf(" (stream totals %d micro-kernels accumulating %.2f ms)", streamMicroCount, float64(streamMicroTotalNs)/1e6)
					}
					if streamMicroCount >= 10 {
						type kCount struct {
							name  string
							count int
						}
						var sortedK []kCount
						for k, c := range streamMicroKernelCounts {
							sortedK = append(sortedK, kCount{name: k, count: c})
						}
						sort.Slice(sortedK, func(i, j int) bool {
							if sortedK[i].count != sortedK[j].count {
								return sortedK[i].count > sortedK[j].count
							}
							return sortedK[i].name < sortedK[j].name
						})
						var topList []string
						for i := 0; i < len(sortedK) && i < 3; i++ {
							topList = append(topList, fmt.Sprintf("%s (%dx)", sortedK[i].name, sortedK[i].count))
						}
						detail += fmt.Sprintf(" [top micro-kernels: %s]", strings.Join(topList, ", "))
					}

					// A short kernel does not imply that the host cannot keep the
					// device fed. Report submission overhead only where the device
					// is measurably idle and the host is measurably behind it.
					duty, dutyOK := sequenceDutyCycle(kernelSeq)
					switch {
					case !dutyOK:
						if !seenDutyUnmeasured[streamID] {
							seenDutyUnmeasured[streamID] = true
							report.Items = append(report.Items, AdviceItem{
								Severity:      SeverityInfo,
								Category:      CategoryArchitectureDispatch,
								EvidenceLevel: EvidenceUnmeasured,
								Location: Location{
									Kernel: kernelSeq[0].Name,
								},
								Finding:                  fmt.Sprintf("Short-kernel dispatch sequence with unmeasured GPU duty cycle: %s, but the timestamps this reader could reach carry no activity start and end, so neither device idle time nor host submission was measured%s", detail, journalRemedy(cap)),
								ActionableRecommendation: recaptureOrRead(cap),
								NextMeasurement:          "activity start/end timestamps in profile or capture-status.json",
							})
						}
					case duty >= lowDutyCyclePct:
						// The device is busy across the sequence span; per-launch
						// host cost is not costing device time here.
					default:
						behind, measured := hostBehindLaunches(kernelSeq, apis)
						sev := SeverityInfo
						why, rec := unmeasuredLaunchReason(len(apis))
						finding := fmt.Sprintf("Low GPU duty cycle across a short-kernel sequence: %s and its kernels occupy only %.1f%% of the sequence device span (< %.0f%%); %s, so the cause of the idle time was not measured", detail, duty, lowDutyCyclePct, why)
						evLevel := EvidenceSuspected
						if measured == 0 {
							evLevel = EvidenceUnmeasured
						} else if float64(behind) >= hostBehindFraction*float64(measured) {
							sev = SeverityWarning
							evLevel = EvidenceObserved
							finding = fmt.Sprintf("Driver submission overhead dominates short kernel execution: %s, its kernels occupy only %.1f%% of the sequence device span (< %.0f%%), and %d of %d measured launches (%.1f%%) began only after the preceding kernel had already finished", detail, duty, lowDutyCyclePct, behind, measured, float64(behind)*100.0/float64(measured))
							rec = "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency."
							if graphReplays > 0 {
								rec = "This sequence already runs as a CUDA graph replay, so submission latency is not removable by graph capture; look for host work between replays."
							}
						} else if measured > 0 {
							finding = fmt.Sprintf("Low GPU duty cycle across a short-kernel sequence: %s and its kernels occupy only %.1f%% of the sequence device span (< %.0f%%), but only %d of %d measured launches began after the preceding kernel had finished, so the host was not behind the device for most of the sequence", detail, duty, lowDutyCyclePct, behind, measured)
							rec = "Idle device time here is not explained by host launch submission; look for stream dependencies, synchronization, or concurrent work on other streams."
						}
						report.Items = append(report.Items, AdviceItem{
							Severity:      sev,
							Category:      CategoryArchitectureDispatch,
							EvidenceLevel: evLevel,
							Location: Location{
								Kernel: kernelSeq[0].Name,
							},
							Finding:                  finding,
							ActionableRecommendation: rec,
							NextMeasurement:          "host launch API records (cudaLaunchKernel) with correlation timestamps or A/B wall-clock timing",
						})
					}
				}
			}
			kernelSeq = kernelSeq[:0]
		}

		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			nameLower := strings.ToLower(ev.Name)
			if sub == "memcpy" || strings.Contains(nameLower, "memcpy") ||
				sub == "synchronization" || sub == "sync" || strings.Contains(nameLower, "synchroniz") {
				checkGraphCandidate()
				continue
			}
			if sub == "kernel" || sub == "" {
				kernelSeq = append(kernelSeq, ev)
			}
		}
		checkGraphCandidate()

		// 1b. Repeated fine-grained host loop dispatch detection
		consecutiveKernelCounts := make(map[string]int)
		var maxConsecutive int
		var maxKernel string
		var curRun int
		var curK string
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if sub == "kernel" || sub == "" {
				kClean := cleanKernelName(ev.Name)
				if kClean == curK {
					curRun++
				} else {
					curK = kClean
					curRun = 1
				}
				if curRun > consecutiveKernelCounts[kClean] {
					consecutiveKernelCounts[kClean] = curRun
				}
				if curRun > maxConsecutive {
					maxConsecutive = curRun
					maxKernel = curK
				}
			} else {
				curK = ""
				curRun = 0
			}
		}

		if maxConsecutive >= 50 && maxKernel != "" {
			report.Items = append(report.Items, AdviceItem{
				Severity:      SeverityWarning,
				Category:      CategoryArchitectureDispatch,
				EvidenceLevel: EvidenceObserved,
				Location: Location{
					Kernel: maxKernel,
				},
				Finding:                  fmt.Sprintf("Repeated fine-grained host loop dispatch: kernel %q was launched %d consecutive times on stream %d. This indicates an unbatched host-side token or row loop.", maxKernel, maxConsecutive, streamID),
				ActionableRecommendation: "Restructure caller to batch loop iterations into a single multi-row or multi-token cooperative grid launch, or capture the host loop into a CUDA Graph.",
				NextMeasurement:          "A/B wall-clock timing with batched dispatches or CUDA Graph",
			})
		}

		// 2. Heuristic 5: Pointwise Producer-Consumer Fusion & Sequential Misses
		type pairStat struct {
			count    int
			totalDur uint64
			ev1      streamEvent
			ev2      streamEvent
		}
		var pairKeys []string
		pairStats := make(map[string]*pairStat)
		for i := 0; i < len(events)-1; i++ {
			ev1 := events[i]
			ev2 := events[i+1]
			sub1 := strings.ToLower(ev1.Subtype)
			sub2 := strings.ToLower(ev2.Subtype)
			if (sub1 != "kernel" && sub1 != "") || (sub2 != "kernel" && sub2 != "") {
				continue
			}
			k1 := cleanKernelName(ev1.Name)
			k2 := cleanKernelName(ev2.Name)
			key := fmt.Sprintf("%s->%s:%d", k1, k2, streamID)
			ps := pairStats[key]
			if ps == nil {
				ps = &pairStat{ev1: ev1, ev2: ev2}
				pairStats[key] = ps
				pairKeys = append(pairKeys, key)
			}
			ps.count++
			ps.totalDur += ev1.DurationNs + ev2.DurationNs
		}

		seenFusion := make(map[string]bool)
		for _, key := range pairKeys {
			ps := pairStats[key]
			ev1 := ps.ev1
			ev2 := ps.ev2
			cleanName1 := cleanKernelName(ev1.Name)
			cleanName2 := cleanKernelName(ev2.Name)

			// Elementwise producer-consumer fusion
			if isElementwise(ev1) && isElementwise(ev2) && matchingShapesOrBuffers(ev1, ev2) {
				seenFusion[key] = true
				finding := fmt.Sprintf("Redundant global memory round-trips for intermediate tensors: sequential back-to-back elementwise kernels (%s followed by %s) on stream %d operate on matching tensors", ev1.Name, ev2.Name, streamID)
				if ps.count > 1 {
					finding += fmt.Sprintf(" (occurred %d times", ps.count)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
					}
					finding += ")"
				}
				rec := fmt.Sprintf("Fuse into a single kernel (e.g. fused %s + %s) to eliminate intermediate DRAM round-trips.", cleanName1, cleanName2)

				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryMemoryCompute,
					EvidenceLevel: EvidenceSuspected,
					// The pair is the unit of device time here, so the item carries the
					// sum the finding already names. Without it the item reaches a
					// consumer with neither device_time_nanos nor share_of_profile --
					// Location.Kernel is "A -> B", which is no kernel in the profile, so
					// the fill-in below looks it up and finds nothing. Pairs overlap, so
					// these shares do not sum with the per-kernel ones.
					DeviceTimeNanos: int64(ps.totalDur),
					Location: Location{
						Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
					},
					Finding:                  finding,
					ActionableRecommendation: rec,
					NextMeasurement:          "A/B wall-clock timing with fused kernel",
				})
			}

			// Sequential RMSNorm -> Block Quantization Fusion
			if isNormKernel(ev1.Name) && isQuantKernel(ev2.Name) {
				seenFusion[key] = true
				finding := fmt.Sprintf("Redundant global memory round-trips for normalized activations: %s writes normalized activations to DRAM and %s immediately reads them back for quantization on stream %d", ev1.Name, ev2.Name, streamID)
				if ps.count > 1 {
					finding += fmt.Sprintf(" (occurred %d times", ps.count)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
					}
					finding += ")"
				}
				rec := fmt.Sprintf("Fuse normalization and quantization into a single kernel (e.g. fused %s_%s) that normalizes in shared memory/registers and directly stores quantized blocks to eliminate %d DRAM round-trips.", cleanName1, cleanName2, ps.count)

				report.Items = append(report.Items, AdviceItem{
					Severity:      SeverityWarning,
					Category:      CategoryMemoryCompute,
					EvidenceLevel: EvidenceSuspected,
					// The pair is the unit of device time here, so the item carries the
					// sum the finding already names. Without it the item reaches a
					// consumer with neither device_time_nanos nor share_of_profile --
					// Location.Kernel is "A -> B", which is no kernel in the profile, so
					// the fill-in below looks it up and finds nothing. Pairs overlap, so
					// these shares do not sum with the per-kernel ones.
					DeviceTimeNanos: int64(ps.totalDur),
					Location: Location{
						Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
					},
					Finding:                  finding,
					ActionableRecommendation: rec,
					NextMeasurement:          "A/B wall-clock timing with fused kernel",
				})
			}

			// Dual-GEMV Paired Projections
			if isGEMVKernel(ev1.Name) && isGEMVKernel(ev2.Name) {
				n1 := strings.ToLower(ev1.Name)
				n2 := strings.ToLower(ev2.Name)
				isPaired := (ev1.Buffer != "" && ev1.Buffer == ev2.Buffer) ||
					(ev1.Shape != "" && ev1.Shape == ev2.Shape) ||
					(strings.Contains(n1, "qa") && strings.Contains(n2, "kv")) ||
					(strings.Contains(n1, "gate") && strings.Contains(n2, "up")) ||
					(strings.Contains(n1, "q_proj") && strings.Contains(n2, "k_proj"))
				if isPaired {
					seenFusion[key] = true
					finding := fmt.Sprintf("Redundant activation DRAM traffic: sequential GEMVs (%s and %s) on stream %d independently load the identical input activation vector", ev1.Name, ev2.Name, streamID)
					if ps.count > 1 {
						finding += fmt.Sprintf(" (occurred %d times", ps.count)
						if ps.totalDur > 0 {
							finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
						}
						finding += ")"
					}
					rec := fmt.Sprintf("Fuse into a paired dual-GEMV dispatch (e.g. fused %s + %s) that loads the activation vector into shared memory once and accumulates dual projection outputs.", cleanName1, cleanName2)

					report.Items = append(report.Items, AdviceItem{
						Severity:      SeverityInfo,
						Category:      CategoryMemoryCompute,
						EvidenceLevel: EvidenceSuspected,
						// The pair is the unit of device time here, so the item carries the
						// sum the finding already names. Without it the item reaches a
						// consumer with neither device_time_nanos nor share_of_profile --
						// Location.Kernel is "A -> B", which is no kernel in the profile, so
						// the fill-in below looks it up and finds nothing. Pairs overlap, so
						// these shares do not sum with the per-kernel ones.
						DeviceTimeNanos: int64(ps.totalDur),
						Location: Location{
							Kernel: fmt.Sprintf("%s + %s", ev1.Name, ev2.Name),
						},
						Finding:                  finding,
						ActionableRecommendation: rec,
						NextMeasurement:          "A/B wall-clock timing with paired GEMV dispatch",
					})
				}
			}

			// Pointwise Fusion Opportunity for generic short kernel pairs
			if !seenFusion[key] && ps.count >= 20 && cleanName1 != cleanName2 {
				avgDur := uint64(0)
				if ps.totalDur > 0 {
					avgDur = ps.totalDur / uint64(ps.count*2)
				}
				if avgDur == 0 || avgDur < 30000 {
					seenFusion[key] = true
					finding := fmt.Sprintf("Pointwise kernel fusion opportunity: sequential kernel pair (%s followed by %s) was launched %d times on stream %d", cleanName1, cleanName2, ps.count, streamID)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(" (accumulating %s)", time.Duration(ps.totalDur).String())
					}
					rec := fmt.Sprintf("Fuse %s and %s into a single kernel to keep intermediate activations in registers/shared memory, eliminating %d kernel launches and DRAM traffic.", cleanName1, cleanName2, ps.count)

					report.Items = append(report.Items, AdviceItem{
						Severity:      SeverityWarning,
						Category:      CategoryMemoryCompute,
						EvidenceLevel: EvidenceSuspected,
						// The pair is the unit of device time here, so the item carries the
						// sum the finding already names. Without it the item reaches a
						// consumer with neither device_time_nanos nor share_of_profile --
						// Location.Kernel is "A -> B", which is no kernel in the profile, so
						// the fill-in below looks it up and finds nothing. Pairs overlap, so
						// these shares do not sum with the per-kernel ones.
						DeviceTimeNanos: int64(ps.totalDur),
						Location: Location{
							Kernel: fmt.Sprintf("%s -> %s", cleanName1, cleanName2),
						},
						Finding:                  finding,
						ActionableRecommendation: rec,
						NextMeasurement:          "A/B wall-clock timing with fused kernel",
					})
				}
			}
		}
	}
}

// auditHostDispatchOverhead compares a complete host API dispatch span with
// the corresponding device kernel work. It requires both correlation-matched
// API timestamps and device activity timestamps; a launch count or a summed
// kernel duration alone cannot establish host overhead.
func auditHostDispatchOverhead(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	apis := extractAPIRecords(cap)
	if len(apis) == 0 {
		return
	}
	events := extractStreamEvents(cap)
	var hostStart, hostEnd, deviceNanos uint64
	seen := make(map[uint32]bool)
	measured := 0
	for _, stream := range events {
		for _, ev := range stream {
			if strings.ToLower(ev.Subtype) != "kernel" && ev.Subtype != "" {
				continue
			}
			if ev.Correlation == 0 || seen[ev.Correlation] || ev.Start == 0 || ev.End <= ev.Start {
				continue
			}
			api, ok := apis[ev.Correlation]
			if !ok || api.Start == 0 || api.End < api.Start {
				continue
			}
			seen[ev.Correlation] = true
			measured++
			deviceNanos += ev.End - ev.Start
			if hostStart == 0 || api.Start < hostStart {
				hostStart = api.Start
			}
			if api.End > hostEnd {
				hostEnd = api.End
			}
		}
	}
	if measured == 0 || hostEnd <= hostStart || hostEnd-hostStart <= deviceNanos {
		return
	}
	span := hostEnd - hostStart
	overhead := span - deviceNanos
	if float64(overhead)/float64(span) <= 0.15 {
		return
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:      SeverityCritical,
		Category:      CategoryArchitectureDispatch,
		EvidenceLevel: EvidenceObserved,
		Finding: fmt.Sprintf("Host dispatch overhead accounts for %.1f%% of the correlation-matched dispatch span (%d launches, %s host span, %s summed GPU execution)",
			100*float64(overhead)/float64(span), measured, time.Duration(span), time.Duration(deviceNanos)),
		ActionableRecommendation: "Batch fine-grained launches or capture the dispatch loop in a CUDA Graph and compare end-to-end wall-clock time; the measured gap is host/driver span, not device kernel time.",
		NextMeasurement:          "repeat with CUDA Graph capture or batched launches and retain correlation-matched host/API and device timestamps",
	})
}

func isElementwise(ev streamEvent) bool {
	name := strings.ToLower(ev.Name)
	if strings.Contains(name, "gemm") || strings.Contains(name, "matmul") ||
		strings.Contains(name, "matvec") || strings.Contains(name, "gemv") ||
		strings.Contains(name, "attention") || strings.Contains(name, "conv") ||
		strings.Contains(name, "allreduce") || strings.Contains(name, "reduce") ||
		strings.Contains(name, "reduction") || strings.Contains(name, "norm") ||
		strings.Contains(name, "softmax") || strings.Contains(name, "gather") ||
		strings.Contains(name, "scatter") || strings.Contains(name, "scan") {
		return false
	}
	keywords := []string{
		"swiglu", "silu", "gelu", "relu", "sigmoid", "tanh", "activation",
		"quant", "quantize", "dequant", "pointwise", "elementwise",
		"add", "mul", "scale", "bias", "dropout", "cast", "convert",
	}
	for _, kw := range keywords {
		if strings.Contains(name, kw) {
			return true
		}
	}
	return false
}

func isNormKernel(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gemm") || strings.Contains(n, "gemv") || strings.Contains(n, "matvec") || strings.Contains(n, "matmul") {
		return false
	}
	return strings.Contains(n, "rms") || strings.Contains(n, "layernorm") || strings.Contains(n, "norm")
}

func isQuantKernel(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gemm") || strings.Contains(n, "gemv") || strings.Contains(n, "matvec") || strings.Contains(n, "matmul") {
		return false
	}
	return strings.Contains(n, "quant") || strings.Contains(n, "quantize")
}

func isGEMVKernel(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "matvec") || strings.Contains(n, "gemv")
}

func matchingShapesOrBuffers(ev1, ev2 streamEvent) bool {
	if ev1.Shape != "" && ev2.Shape != "" {
		return ev1.Shape == ev2.Shape
	}
	if ev1.Buffer != "" && ev2.Buffer != "" {
		return ev1.Buffer == ev2.Buffer
	}
	if ev1.Grid != [3]int{} && ev1.Grid == ev2.Grid && ev1.Block != [3]int{} && ev1.Block == ev2.Block {
		return true
	}
	return false
}

func cleanKernelName(name string) string {
	name = strings.TrimSuffix(name, ".SASS")
	name = strings.TrimSuffix(name, ".PTX")
	name = strings.TrimSuffix(name, ".TILEIR")
	nameLower := strings.ToLower(name)
	switch {
	case strings.Contains(nameLower, "swiglu"):
		return "SwiGLU"
	case strings.Contains(nameLower, "quant"):
		return "quant"
	case strings.Contains(nameLower, "gelu"):
		return "GELU"
	case strings.Contains(nameLower, "silu"):
		return "SiLU"
	case strings.Contains(nameLower, "relu"):
		return "ReLU"
	case strings.Contains(nameLower, "rmsnorm") || strings.Contains(nameLower, "rms"):
		return "RMSNorm"
	case strings.Contains(nameLower, "layernorm"):
		return "LayerNorm"
	default:
		return name
	}
}

func extractStreamEvents(cap *cudacapture.Capture) map[uint32][]streamEvent {
	events, _ := extractStreamEventsDetail(cap)
	return events
}

// extractStreamEventsDetail also reports that the events came from a journal
// whose graph replays are opaque: the envelopes are on the timeline because no
// device record named a graph, so their interiors are counted as busy. A
// caller stating a duty cycle has to carry that caveat, and this is the only
// place that knows it -- re-deriving it would mean decoding the journal twice.
func extractStreamEventsDetail(cap *cudacapture.Capture) (map[uint32][]streamEvent, bool) {
	res := make(map[uint32][]streamEvent)
	if cap == nil {
		return res, false
	}

	if cap.Profile != nil && len(cap.Profile.Sample) > 0 {
		kernelTimeIdx := -1
		gpuTimeIdx := -1
		copyTimeIdx := -1
		for i, st := range cap.Profile.SampleType {
			switch st.Type {
			case "kernel_time":
				kernelTimeIdx = i
			case "gpu_time":
				gpuTimeIdx = i
			case "copy_time":
				copyTimeIdx = i
			}
		}

		for _, s := range cap.Profile.Sample {
			var streamID uint32
			if len(s.NumLabel["stream"]) > 0 {
				streamID = uint32(s.NumLabel["stream"][0])
			} else if len(s.NumLabel["stream_id"]) > 0 {
				streamID = uint32(s.NumLabel["stream_id"][0])
			} else if len(s.Label["stream"]) > 0 {
				if n, err := strconv.ParseUint(s.Label["stream"][0], 10, 32); err == nil {
					streamID = uint32(n)
				}
			}

			name := ""
			if len(s.Label["name"]) > 0 {
				name = s.Label["name"][0]
			} else if len(s.Label["kernel"]) > 0 {
				name = s.Label["kernel"][0]
			} else if len(s.Location) > 0 && len(s.Location[0].Line) > 0 && s.Location[0].Line[0].Function != nil {
				name = s.Location[0].Line[0].Function.Name
			}

			subtype := ""
			if len(s.Label["subtype"]) > 0 {
				subtype = s.Label["subtype"][0]
			} else {
				nLower := strings.ToLower(name)
				if copyTimeIdx >= 0 && s.Value[copyTimeIdx] > 0 {
					subtype = "memcpy"
				} else if strings.Contains(nLower, "memcpy") {
					subtype = "memcpy"
				} else if strings.Contains(nLower, "synchroniz") || strings.Contains(nLower, "sync") {
					subtype = "synchronization"
				} else if name != "" {
					subtype = "kernel"
				}
			}

			var durationNs uint64
			if kernelTimeIdx >= 0 && s.Value[kernelTimeIdx] > 0 {
				durationNs = uint64(s.Value[kernelTimeIdx])
			} else if gpuTimeIdx >= 0 && s.Value[gpuTimeIdx] > 0 {
				durationNs = uint64(s.Value[gpuTimeIdx])
			} else if copyTimeIdx >= 0 && s.Value[copyTimeIdx] > 0 {
				durationNs = uint64(s.Value[copyTimeIdx])
			} else if len(s.NumLabel["duration_ns"]) > 0 {
				durationNs = uint64(s.NumLabel["duration_ns"][0])
			} else if len(s.NumLabel["end"]) > 0 && len(s.NumLabel["start"]) > 0 && s.NumLabel["end"][0] >= s.NumLabel["start"][0] {
				durationNs = uint64(s.NumLabel["end"][0] - s.NumLabel["start"][0])
			} else if len(s.Label["duration_ns"]) > 0 {
				durationNs, _ = strconv.ParseUint(s.Label["duration_ns"][0], 10, 64)
			}

			var startNs, endNs uint64
			if len(s.NumLabel["start"]) > 0 && s.NumLabel["start"][0] > 0 {
				startNs = uint64(s.NumLabel["start"][0])
			} else if len(s.Label["start"]) > 0 {
				startNs, _ = strconv.ParseUint(s.Label["start"][0], 10, 64)
			}
			if len(s.NumLabel["end"]) > 0 && s.NumLabel["end"][0] > 0 {
				endNs = uint64(s.NumLabel["end"][0])
			} else if len(s.Label["end"]) > 0 {
				endNs, _ = strconv.ParseUint(s.Label["end"][0], 10, 64)
			}
			if endNs == 0 && startNs > 0 && durationNs > 0 {
				endNs = startNs + durationNs
			}

			hasStream := len(s.NumLabel["stream"]) > 0 || len(s.NumLabel["stream_id"]) > 0 || len(s.Label["stream"]) > 0
			hasTiming := durationNs > 0 || startNs > 0 || endNs > 0
			hasActivity := len(s.Label["subtype"]) > 0 || len(s.NumLabel["correlation"]) > 0 || len(s.Label["correlation_id"]) > 0 || len(s.Label["grid"]) > 0
			if !hasStream && !hasTiming && !hasActivity {
				continue
			}

			correlation := sampleCorrelation(s)
			var graphID uint32
			if len(s.NumLabel["graph_id"]) > 0 {
				graphID = uint32(s.NumLabel["graph_id"][0])
			}

			shape := ""
			if len(s.Label["shape"]) > 0 {
				shape = s.Label["shape"][0]
			} else if len(s.Label["tensor"]) > 0 {
				shape = s.Label["tensor"][0]
			}

			buffer := ""
			if len(s.Label["buffer"]) > 0 {
				buffer = s.Label["buffer"][0]
			}

			var grid, block [3]int
			if len(s.Label["grid"]) > 0 {
				grid = parseDim3(s.Label["grid"][0])
			}
			if len(s.Label["block"]) > 0 {
				block = parseDim3(s.Label["block"][0])
			}

			if name != "" || subtype != "" {
				res[streamID] = append(res[streamID], streamEvent{
					Subtype:     subtype,
					Name:        name,
					Stream:      streamID,
					DurationNs:  durationNs,
					Start:       startNs,
					End:         endNs,
					Correlation: correlation,
					GraphID:     graphID,
					Shape:       shape,
					Buffer:      buffer,
					Grid:        grid,
					Block:       block,
				})
			}
		}
	}

	// The journal comes first among the fallbacks because it is the only
	// source that carries per-activity timestamps. The pprof samples above
	// are an aggregate, and a reader that stops there reports the capture as
	// having no timestamps when the file it was handed has them.
	//
	// The test is whether a timestamp was actually produced, not whether the
	// projection produced anything. A projection yielding hundreds of samples
	// that all carry Start == End == 0 is the case this exists for: it fills
	// res, so a len(res) == 0 guard would never fire, and the reader would go
	// on to announce that the capture has no timestamps while holding them.
	if !anyStreamTimed(res) {
		if events, opaque, err := journalStreamEvents(cap); err == nil && anyEventTimed(events) {
			// Replaces what the projection produced rather than joining it.
			// The two sources describe the same device work at different
			// fidelities, and a union of them would double every activity the
			// projection also carries.
			timed := make(map[uint32][]streamEvent)
			for _, ev := range events {
				timed[ev.Stream] = append(timed[ev.Stream], ev)
			}
			return timed, opaque
		}
	}
	if len(res) == 0 && len(cap.Files) > 0 {
		for _, fname := range []string{"activities.json", "capture-status.json", "status.json", "trace.json"} {
			if data, ok := cap.Files[fname]; ok && len(data) > 0 {
				events := parseStreamEventsFromJSON(data)
				for _, ev := range events {
					res[ev.Stream] = append(res[ev.Stream], ev)
				}
				if len(res) > 0 {
					break
				}
			}
		}
	}

	return res, false
}

func parseStreamEventsFromJSON(data []byte) []streamEvent {
	var events []streamEvent
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}

	var items []any
	switch v := raw.(type) {
	case []any:
		items = v
	case map[string]any:
		if acts, ok := v["activities"].([]any); ok {
			items = acts
		} else if status, ok := v["status"].(map[string]any); ok {
			if acts, ok := status["activities"].([]any); ok {
				items = acts
			}
		}
	}

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		var ev streamEvent
		if s, ok := m["subtype"].(string); ok {
			ev.Subtype = s
		} else if s, ok := m["kind"].(string); ok {
			ev.Subtype = s
		}
		if s, ok := m["name"].(string); ok {
			ev.Name = s
		}
		if s, ok := m["stream"].(float64); ok {
			ev.Stream = uint32(s)
		} else if s, ok := m["stream_id"].(float64); ok {
			ev.Stream = uint32(s)
		}

		var start, end uint64
		if s, ok := m["start"].(float64); ok {
			start = uint64(s)
		}
		if e, ok := m["end"].(float64); ok {
			end = uint64(e)
		}
		if end >= start && start > 0 {
			ev.DurationNs = end - start
			ev.Start, ev.End = start, end
		}
		if c, ok := m["correlation"].(float64); ok && c > 0 {
			ev.Correlation = uint32(c)
		}
		if g, ok := m["graph_id"].(float64); ok && g > 0 {
			ev.GraphID = uint32(g)
		}
		if d, ok := m["duration"].(float64); ok && d > 0 {
			ev.DurationNs = uint64(d)
		}
		if d, ok := m["duration_ns"].(float64); ok && d > 0 {
			ev.DurationNs = uint64(d)
		}

		if s, ok := m["shape"].(string); ok {
			ev.Shape = s
		}
		if b, ok := m["buffer"].(string); ok {
			ev.Buffer = b
		}

		if blk, ok := m["block"].([]any); ok && len(blk) > 0 {
			ev.Block = parseDim3FromSlice(blk)
		}
		if grd, ok := m["grid"].([]any); ok && len(grd) > 0 {
			ev.Grid = parseDim3FromSlice(grd)
		}

		if ev.Name != "" || ev.Subtype != "" {
			events = append(events, ev)
		}
	}
	return events
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

// captureStatusFields returns the capture-status.json object carried by an
// artifact, unwrapping the optional "status" nesting.
func captureStatusFields(cap *cudacapture.Capture) map[string]any {
	if cap == nil {
		return nil
	}
	statusData, ok := cap.Files["capture-status.json"]
	if !ok {
		return nil
	}
	var raw map[string]any
	if err := json.Unmarshal(statusData, &raw); err != nil {
		return nil
	}
	if nested, ok := raw["status"].(map[string]any); ok {
		return nested
	}
	return raw
}

// captureFailure reports whether an artifact records a capture that did not
// succeed, and describes it in terms a reader can act on: the outcome names
// which transport gave up, and capture_error names the stage it gave up at.
// "transport-failed" alone is a token, not a diagnosis.
func captureFailure(cap *cudacapture.Capture) (string, bool) {
	status := captureStatusFields(cap)
	if status == nil {
		return "", false
	}
	success, ok := status["capture_success"].(bool)
	if !ok || success {
		return "", false
	}
	reason := "capture marked as failed"
	if failure, ok := status["failure"].(string); ok && failure != "" {
		reason = failure
	} else if captureErr, ok := status["capture_error"].(string); ok && captureErr != "" {
		reason = captureErr
	}
	if outcome, ok := status["outcome"].(string); ok && outcome != "" {
		reason = fmt.Sprintf("%s (outcome %s)", reason, outcome)
	}
	return reason, true
}

// preflightCapture refuses an artifact that records its own failure. A failed
// capture still writes a well-formed pprof -- a couple of kilobytes with zero
// activity records -- and pprof renders it as an empty but valid profile and
// exits 0. Nothing on that path says the numbers are missing rather than
// small, which is the mistake the artifact invites. Reading it fails closed;
// the status is still reachable through advise, which reports the failure as a
// finding rather than pretending to measure.
func preflightCapture(cap *cudacapture.Capture) error {
	reason, failed := captureFailure(cap)
	if !failed {
		return nil
	}
	return fmt.Errorf("this capture records its own failure: %s\n"+
		"\tIt carries no measurement, and an empty profile is not a fast one.\n"+
		"\tUse 'cudaprof advise' to see the failure as a finding, or re-run the capture.", reason)
}

func auditCaptureStatus(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	status := captureStatusFields(cap)
	if status == nil {
		return
	}
	if errMsg, failed := captureFailure(cap); failed {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityCritical,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Incomplete or failed profile capture: %s", errMsg),
			ActionableRecommendation: "Ensure target workload executes cleanly and does not exceed journal record capacity before analyzing performance.",
			NextMeasurement:          "Re-run capture with increased buffer size or filtered kernel scope to confirm clean execution",
		})
	}
	if dropped, ok := status["dropped_records"].(float64); ok && uint64(dropped) > 0 {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityWarning,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Lossy capture: %d records dropped due to buffer pressure", uint64(dropped)),
			ActionableRecommendation: "Whole-run totals and aggregate launch counts are incomplete (a lower bound). Increase buffer pool or reduce profiling overhead to capture a complete trace.",
			NextMeasurement:          "Increase trace buffer pool capacity and re-verify zero dropped records",
		})
	}
}

// hasDeviceTiming reports whether the capture carries measured device time.
// A wall-clock duration is not device time: every sample-only profile has one,
// and treating it as timing is what let a share of PC samples be rendered as a
// millisecond figure. Only a kernel_time or gpu_time sample type counts.
func hasDeviceTiming(cap *cudacapture.Capture) bool {
	if cap == nil {
		return false
	}
	if cap.Profile != nil {
		for i, vt := range cap.Profile.SampleType {
			if vt.Type == "kernel_time" || vt.Type == "gpu_time" {
				for _, s := range cap.Profile.Sample {
					if i < len(s.Value) && s.Value[i] > 0 {
						return true
					}
				}
			}
		}
	}
	return false
}

func computeProfileTimes(cap *cudacapture.Capture) (map[string]int64, int64) {
	kernelTimes := make(map[string]int64)
	var totalTime int64
	if cap == nil || cap.Profile == nil {
		return kernelTimes, totalTime
	}
	p := cap.Profile
	timeIdx := -1
	for i, vt := range p.SampleType {
		if vt.Type == "kernel_time" || vt.Type == "gpu_time" {
			timeIdx = i
			break
		}
	}
	if timeIdx >= 0 {
		for _, s := range p.Sample {
			val := int64(0)
			if timeIdx < len(s.Value) {
				val = s.Value[timeIdx]
			}
			totalTime += val
			kName := extractKernelNameFromSample(s)
			if kName != "" {
				kernelTimes[kName] += val
			}
		}
	}
	if totalTime == 0 {
		// Fall back to hardware PC samples or samples when timing is 0 or absent.
		sampleIdx := -1
		for i, vt := range p.SampleType {
			if vt.Type == "pc_samples" || vt.Type == "samples" {
				sampleIdx = i
				break
			}
		}
		if sampleIdx >= 0 {
			var totalSamples int64
			sampleCounts := make(map[string]int64)
			for _, s := range p.Sample {
				val := int64(0)
				if sampleIdx < len(s.Value) {
					val = s.Value[sampleIdx]
				}
				totalSamples += val
				kName := extractKernelNameFromSample(s)
				if kName != "" {
					sampleCounts[kName] += val
				}
			}
			if totalSamples > 0 {
				// Sample counts, not times. Scaling them by the capture's
				// wall-clock duration produces a per-kernel nanosecond figure
				// that was never measured; the caller distinguishes the two by
				// hasDeviceTiming and labels the share accordingly.
				totalTime = totalSamples
				for k, sc := range sampleCounts {
					kernelTimes[k] = sc
				}
			}
		}
	}
	return kernelTimes, totalTime
}

// dominanceRecommendation words the dominance advisory for what the capture
// shows. Recommending a CUDA Graph to a workload already running as one is
// advice its author cannot act on, so that clause is dropped whenever the
// capture carries graph replays.
func dominanceRecommendation(graphReplays int) string {
	if graphReplays > 0 {
		return "Restructure per-row dispatches into batched GEMM / GEMV operations. This capture already replays CUDA graphs, so graph capture is not available as a further remedy here."
	}
	return "Restructure per-row dispatches into batched GEMM / GEMV operations or capture the entire layer sequence into a CUDA Graph."
}

func auditKernelFamilyDominance(cap *cudacapture.Capture, report *AdviceReport, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	if totalTime <= 0 || len(kernelTimes) == 0 {
		return
	}
	graphReplays := capturedGraphReplays(cap)
	familyTimes := make(map[string]int64)
	familyCounts := make(map[string]int)

	kernelCounts := make(map[string]int)
	if cap.Profile != nil {
		for _, s := range cap.Profile.Sample {
			k := extractKernelNameFromSample(s)
			if k != "" {
				kernelCounts[k]++
			}
		}
	}

	for k, t := range kernelTimes {
		parts := strings.Split(k, "_")
		family := k
		if len(parts) >= 4 {
			family = strings.Join(parts[:len(parts)-1], "_")
		} else if len(parts) >= 2 {
			family = strings.Join(parts[:len(parts)-1], "_")
		}
		familyTimes[family] += t
		familyCounts[family] += kernelCounts[k]
	}

	for family, fTime := range familyTimes {
		pct := float64(fTime) / float64(totalTime) * 100.0
		cnt := familyCounts[family]
		if pct >= 50.0 && cnt >= 10 {
			var item AdviceItem
			if hasTiming {
				fTimeMs := float64(fTime) / 1e6
				item = AdviceItem{
					Severity:        SeverityCritical,
					Category:        CategoryArchitecture,
					EvidenceLevel:   EvidenceObserved,
					DeviceTimeNanos: fTime,
					ShareOfProfile:  pct,
					Location: Location{
						Kernel: family,
					},
					Finding: fmt.Sprintf("Kernel family %q dominates GPU execution, consuming %.1f%% of total device time (%.2f ms across %d dispatches). Fine-grained per-row dispatches incur massive submission and launch overhead.",
						family, pct, fTimeMs, cnt),
					ActionableRecommendation: dominanceRecommendation(graphReplays),
					NextMeasurement:          "Measure wall-clock latency improvement with CUDA graph or batched GEMM/GEMV",
				}
			} else {
				item = AdviceItem{
					Severity:        SeverityCritical,
					Category:        CategoryArchitecture,
					EvidenceLevel:   EvidenceObserved,
					DeviceTimeNanos: 0,
					ShareOfProfile:  pct,
					Location: Location{
						Kernel: family,
					},
					Finding: fmt.Sprintf("Kernel family %q dominates GPU execution, consuming %.1f%% of observed samples (%d samples across %d dispatches). Fine-grained per-row dispatches incur massive submission and launch overhead.",
						family, pct, fTime, cnt),
					ActionableRecommendation: dominanceRecommendation(graphReplays),
					NextMeasurement:          "Measure wall-clock latency improvement with CUDA graph or batched GEMM/GEMV",
				}
			}
			report.Items = append(report.Items, item)
		}
	}
}

func severityRank(s Severity) int {
	switch s {
	case SeverityCritical:
		return 0
	case SeverityWarning:
		return 1
	case SeverityInfo:
		return 2
	default:
		return 3
	}
}

func finalizeAndRankFindings(report *AdviceReport, kernelTimes map[string]int64, totalTime int64) {
	finalizeAndRankFindingsWithTiming(report, kernelTimes, totalTime, true)
}

func finalizeAndRankFindingsWithTiming(report *AdviceReport, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	if report == nil || len(report.Items) == 0 {
		return
	}
	for i := range report.Items {
		item := &report.Items[i]
		if hasTiming {
			if item.DeviceTimeNanos == 0 && item.Location.Kernel != "" {
				item.DeviceTimeNanos = kernelTimes[item.Location.Kernel]
			}
			if totalTime > 0 && item.DeviceTimeNanos > 0 && item.ShareOfProfile == 0 {
				item.ShareOfProfile = float64(item.DeviceTimeNanos) / float64(totalTime) * 100.0
			}
		} else {
			// Pure PC-sample or sample profile: do not assign sample counts to DeviceTimeNanos!
			// DeviceTimeNanos remains 0.
			if totalTime > 0 && item.ShareOfProfile == 0 && item.Location.Kernel != "" {
				kSamples := kernelTimes[item.Location.Kernel]
				if kSamples > 0 {
					item.ShareOfProfile = float64(kSamples) / float64(totalTime) * 100.0
				}
			}
		}
	}

	// Critical is a claim about what to work on first, so a finding that earns
	// it by cost has to account for some. Without this floor a 2-launch
	// single-thread kernel worth 0.006 ms was the only Critical in a 66-finding
	// report, above a kernel holding 536.58 ms as a Warning -- the reader's
	// first instruction pointed at 0.0002% of the profile.
	for i := range report.Items {
		item := &report.Items[i]
		if item.Severity == SeverityCritical && !item.severityByKind &&
			item.ShareOfProfile > 0 && item.ShareOfProfile < criticalShareFloorPct {
			item.Severity = SeverityWarning
		}
	}

	sort.SliceStable(report.Items, func(i, j int) bool {
		a, b := report.Items[i], report.Items[j]
		if a.Severity != b.Severity {
			return severityRank(a.Severity) < severityRank(b.Severity)
		}
		if a.DeviceTimeNanos != b.DeviceTimeNanos {
			return a.DeviceTimeNanos > b.DeviceTimeNanos
		}
		if a.ShareOfProfile != b.ShareOfProfile {
			return a.ShareOfProfile > b.ShareOfProfile
		}
		if a.Category != b.Category {
			return a.Category < b.Category
		}
		return a.Finding < b.Finding
	})
}

// criticalShareFloorPct is the share of the profile a cost-based finding must
// account for to stay Critical. A finding below it may still be true and worth
// fixing; it is not what to do first, and putting it first costs the reader
// the one position in the report they are certain to read.
const criticalShareFloorPct = 0.1

// deviceIdleCriticalPct is the whole-capture duty cycle below which the idle
// time outweighs everything a kernel-time ranking can see. It is deliberately
// far below lowDutyCyclePct, which judges one short-kernel sequence: this one
// decides whether the report's own ordering is misleading.
const deviceIdleCriticalPct = 50.0

// profileDutyCycle reports the fraction of the captured device span in which
// any device work was running.
//
// Busy time is the UNION of the activity intervals, not their sum. Kernels on
// different streams overlap, and summing durations across streams produces a
// busy figure larger than the span it is divided by -- a duty cycle above
// 100%, which is how this class of number announces that it was summed.
// deviceWorkSubtype reports whether an event occupied the device.
//
// The three it rejects are spans the capture carries for a different purpose:
// CUPTI's own cost, the window the host spent waiting, and the envelope of a
// graph replay. Counting a synchronization window as busy is the worst of the
// three, because it erases exactly the idle it explains -- the device was idle
// BECAUSE the host was in that call.
//
// graphTraceIsWork inverts the graph arm. The envelope is excluded because it
// spans the records of the nodes replayed inside it; when the capture holds no
// such record, it spans nothing, and excluding it deletes the whole replay
// from the timeline and reports its duration as idle.
func deviceWorkSubtype(subtype string, graphTraceIsWork bool) bool {
	switch strings.ToLower(subtype) {
	case "overhead", "sync", "synchronization":
		return false
	case "graph_trace", "graphtrace":
		return graphTraceIsWork
	}
	return true
}

func profileDutyCycle(streams map[uint32][]streamEvent) (pct float64, spanNs, busyNs uint64, ok bool) {
	// Decided from the capture, not assumed: a replay envelope is redundant
	// with its node records and is the only evidence of the replay without
	// them. See deviceWorkSubtype.
	graphTraceIsWork := true
	for _, evs := range streams {
		for _, ev := range evs {
			if ev.GraphID != 0 && !strings.EqualFold(ev.Subtype, "graph_trace") {
				graphTraceIsWork = false
			}
		}
	}
	type iv struct{ start, end uint64 }
	var ivs []iv
	var first, last uint64
	var haveFirst bool
	for _, evs := range streams {
		for _, ev := range evs {
			if !deviceWorkSubtype(ev.Subtype, graphTraceIsWork) {
				continue
			}
			// End is the sentinel for "this capture carries no activity
			// timestamps", not Start: a device timeline that begins at zero is
			// a legitimate one, and testing Start drops its first activity
			// silently -- which changes both the span and the busy time.
			if ev.End == 0 || ev.End <= ev.Start {
				continue
			}
			ivs = append(ivs, iv{ev.Start, ev.End})
			// Tracked with a flag, not a zero sentinel: a timeline whose
			// first activity starts at 0 would otherwise keep re-taking the
			// earliest start from every later activity, and the span would be
			// measured from the second one.
			if !haveFirst || ev.Start < first {
				first, haveFirst = ev.Start, true
			}
			if ev.End > last {
				last = ev.End
			}
		}
	}
	// One activity has no span worth calling a duty cycle: its span is its own
	// duration and the answer is always 100%.
	if len(ivs) < 2 || last <= first {
		return 0, 0, 0, false
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].start < ivs[j].start })
	cur := ivs[0]
	for _, x := range ivs[1:] {
		if x.start > cur.end {
			busyNs += cur.end - cur.start
			cur = x
			continue
		}
		if x.end > cur.end {
			cur.end = x.end
		}
	}
	busyNs += cur.end - cur.start
	spanNs = last - first
	return float64(busyNs) * 100.0 / float64(spanNs), spanNs, busyNs, true
}

// auditDeviceDutyCycle states, before any kernel finding, when most of the
// captured span was device idle.
//
// advise ranks by device time and launch count, so on an idle-dominated
// capture every finding it emits is ranking the minority of the problem while
// saying nothing about the majority. On one ds4 capture that ordering sent
// three independent readers -- this tool, its operator, and an external
// review -- to micro-kernel batching as the top parity fix, when the device
// was idle 76% of the span and the oracle they were losing to launched 48%
// MORE kernels per token. No count-based remedy could have touched it.
//
// This finding does not diagnose the idle time; it refuses to let the ranking
// below it be read as the whole account, and names the command that does
// attribute idle.
// dutyCycleFloor reports the busy share left when the graph replay envelopes
// are taken out of the busy union, measured against the same span so the two
// percentages are comparable.
//
// An opaque replay is one activity record covering the whole replay, so
// whatever the device did not do between its nodes is counted as work. The
// measured duty cycle is therefore a ceiling. This is the floor beneath it:
// the busy time the capture can actually show. The truth is somewhere in
// between and this capture cannot say where.
func dutyCycleFloor(streams map[uint32][]streamEvent, spanNs uint64) (float64, bool) {
	if spanNs == 0 {
		return 0, false
	}
	type iv struct{ start, end uint64 }
	var ivs []iv
	for _, evs := range streams {
		for _, ev := range evs {
			// The false is what makes this the floor: deviceWorkSubtype
			// reads it as "a graph replay envelope is not device work", so
			// the envelopes drop out and only records the capture can
			// resolve remain. Do not also skip graph_trace by name here --
			// two spellings of one rule mean neither can be tested.
			if !deviceWorkSubtype(ev.Subtype, false) || ev.End == 0 || ev.End <= ev.Start {
				continue
			}
			ivs = append(ivs, iv{ev.Start, ev.End})
		}
	}
	if len(ivs) == 0 {
		return 0, true
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].start < ivs[j].start })
	var busy uint64
	cur := ivs[0]
	for _, x := range ivs[1:] {
		if x.start > cur.end {
			busy += cur.end - cur.start
			cur = x
			continue
		}
		if x.end > cur.end {
			cur.end = x.end
		}
	}
	busy += cur.end - cur.start
	return float64(busy) * 100.0 / float64(spanNs), true
}

// auditOpaqueDutyCeiling reports a duty cycle that is a ceiling rather than a
// measurement.
//
// The idle finding does not fire above deviceIdleCriticalPct, and a reader who
// sees no idle finding concludes the device was busy. On a capture whose graph
// replays are opaque that inference is not available: each replay counts as
// busy from its envelope alone, so a run that spent most of its span inside
// replays reads as busy whatever it actually did. The absence of the finding
// then says nothing about the run -- the finding could not have fired.
//
// The gate is that removing the envelopes drops the busy share below the
// threshold. Only then could resolving the replays change the answer; a
// capture that is busy on its resolved records alone is busy, and saying
// anything here would be noise.
//
// That gate is also the whole test for opacity, which is why no separate
// opacity flag is consulted. A capture whose replays resolve into nodes has
// its envelopes excluded from the busy union already, so its floor and its
// measured duty cycle are the same number and this returns. Asking a second
// source -- journal record counts rather than the stream events these
// percentages come from -- would let the two disagree about one capture.
func auditOpaqueDutyCeiling(streams map[uint32][]streamEvent, pct float64, spanNs uint64, report *AdviceReport) {
	floor, ok := dutyCycleFloor(streams, spanNs)
	if !ok || floor >= deviceIdleCriticalPct {
		return
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:      SeverityWarning,
		Category:      CategoryArchitectureDispatch,
		EvidenceLevel: EvidenceUnmeasured,
		magnitude:     fmt.Sprintf("duty cycle %.2f%% measured, %.2f%% on resolved records alone", pct, floor),
		Finding: fmt.Sprintf("The device measured %.2f%% busy, which is why no idle finding appears in this report. That number is a ceiling, not a measurement: this capture counts each graph replay as busy from its envelope alone, so whatever the device did not do between a replay's nodes is counted as work. On the records this capture can resolve, the busy share is %.2f%%. The true duty cycle is somewhere between the two, and a run that is mostly replay reads as busy whichever it is.",
			pct, floor),
		ActionableRecommendation: "Do not read the absence of an idle finding here as evidence that the device was busy: on this capture that finding could not have fired regardless of the idle. Re-capture with graph node records resolved -- device records naming their graph ids -- before concluding anything about this run's duty cycle, and before comparing its duty cycle against a capture whose replays were resolved.",
		NextMeasurement:          "cudaprof gaps <capture> -- it reports the replay opacity directly and splits the idle it can see",
	})
}

func auditDeviceDutyCycle(cap *cudacapture.Capture, report *AdviceReport) {
	streams, graphsOpaque := extractStreamEventsDetail(cap)
	pct, spanNs, busyNs, ok := profileDutyCycle(streams)
	if !ok {
		return
	}
	if pct >= deviceIdleCriticalPct {
		// A high duty cycle is why no idle finding follows. On a capture
		// whose replays are opaque that reading is not available: see
		// auditOpaqueDutyCeiling.
		auditOpaqueDutyCeiling(streams, pct, spanNs, report)
		return
	}
	idleNs := spanNs - busyNs
	// Carried forward rather than left to the gaps reader: on such a capture
	// the busy figure includes whatever the device did NOT do between the
	// nodes of a replay, so the idle below is a floor.
	opaque := ""
	if graphsOpaque {
		opaque = " This capture's graph replays are opaque -- no device record names a graph, so each replay counts as busy from its envelope alone and the idle here is a lower bound."
	}
	// The attribution runs first so this finding can say whether the idle it
	// names was attributed, rather than sending the reader to a command that
	// has already run.
	rec := "Attribute the idle time before acting on any kernel finding below: run cudaprof gaps on this capture, which splits idle into host-behind, synchronization, and graph-replay time. A kernel-time ranking cannot distinguish a per-launch tax from a tail, and the two have opposite remedies."
	if auditIdleAttribution(cap, report) {
		rec = "Act on the idle attribution in this report -- the finding naming where this idle went -- before any kernel finding below it. A kernel-time ranking cannot distinguish a per-launch tax from a tail, and the two have opposite remedies."
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:        SeverityCritical,
		Category:        CategoryArchitectureDispatch,
		EvidenceLevel:   EvidenceObserved,
		severityByKind:  true,
		DeviceTimeNanos: int64(busyNs),
		ShareOfProfile:  pct,
		Finding: fmt.Sprintf("The device was idle for %.2f%% of the captured span: %.3f ms idle of a %.3f ms span, with %.3f ms of device work. Every finding below this one ranks that %.2f%% -- none of them can see the idle time, and on this capture the idle time is the larger quantity.%s",
			100.0-pct, float64(idleNs)/1e6, float64(spanNs)/1e6, float64(busyNs)/1e6, pct, opaque),
		ActionableRecommendation: rec,
		NextMeasurement:          "cudaprof gaps <capture> -- idle attribution and the gap distribution",
	})
}

// idleArm is one mechanism of the gaps idle attribution, paired with the
// remedy that mechanism admits.
//
// The remedies are mutually exclusive on purpose. Batching recovers driver
// submission cost and cannot touch hardware queue latency; submitting earlier
// recovers host-behind time and cannot touch either. Naming the wrong arm
// costs the reader the work of trying a remedy that could not have helped,
// and the arms are close enough in a duty-cycle number to be indistinguishable
// without this split.
type idleArm struct {
	name   string
	nanos  int64
	remedy string
}

// idleAttribution is the gaps attribution of one device's idle time, reduced
// to what a finding can state.
type idleAttribution struct {
	device uint32
	idle   int64
	duty   float64
	gaps   int
	median int64
	arms   []idleArm // largest first

	// opaqueGraphs reports that this capture held graph replay envelopes with
	// no per-node records. Those replays carry no launch correlation, so idle
	// beside them lands in the unattributed arm with a cause that is known
	// and nameable rather than unknown. Without saying so, the finding
	// reports that idle as a mechanism the capture could not identify, when
	// what actually happened is that the instrument could not see through the
	// replay.
	opaqueGraphs bool
}

func idleArms(a GapAttribution) []idleArm {
	arms := []idleArm{
		{
			name:   "host behind",
			nanos:  a.HostBehindNanos,
			remedy: "The device finished its work before the host called the next launch, so the host is the one behind. Submit further ahead: enqueue a run of work rather than one op at a time, move host-side work off the dispatch thread, or capture the sequence into a CUDA graph so a single call enqueues all of it.",
		},
		{
			name:   "driver submission",
			nanos:  a.DriverSubmitNanos,
			remedy: "The device went idle while the host was inside the launch call, so the cost is per call and paid in the driver. Make fewer, larger calls: fuse adjacent launches, or replay the sequence as a CUDA graph. Calling earlier cannot help -- the host had already called.",
		},
		{
			name:   "hardware queue",
			nanos:  a.HardwareNanos,
			remedy: "The launch had already returned before the previous op ended, so the work was queued and the device idled anyway. Neither batching nor earlier submission can recover this time: look at what keeps the queued work from starting -- same-stream ordering, an event wait, or a dependency on the op that just finished.",
		},
		{
			name:   "unattributed",
			nanos:  a.UnattributedNanos,
			remedy: "No host launch record correlated to the op that followed this idle, so this capture names no mechanism for it. Find why the correlation ids of these launches do not reach their activities before choosing a remedy: any remedy chosen here would be chosen from ignorance, not from the measurement.",
		},
	}
	sort.SliceStable(arms, func(i, j int) bool { return arms[i].nanos > arms[j].nanos })
	return arms
}

// captureIdleAttribution attributes the idle time of the device that spent the
// most of it, using the same analysis the gaps command reports.
//
// advise otherwise reads the pprof projection, which is an aggregate: it sums
// durations and drops the per-activity timestamps, so no attribution is
// recoverable from it. These numbers exist only in the capture journals. A
// capture without them, or whose journals are truncated, yields nothing here
// rather than an attribution computed from a surviving prefix.
func captureIdleAttribution(cap *cudacapture.Capture) (idleAttribution, bool) {
	gaps, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes})
	if err != nil {
		return idleAttribution{}, false
	}
	var best idleAttribution
	var found bool
	for _, dev := range gaps.Devices {
		s := dev.Overall
		if s == nil || s.IdleNanos <= 0 || (found && s.IdleNanos <= best.idle) {
			continue
		}
		best = idleAttribution{
			device:       dev.Device,
			idle:         s.IdleNanos,
			duty:         s.DutyCyclePct,
			gaps:         s.GapCount,
			median:       s.Distribution.Median,
			arms:         idleArms(s.Attribution),
			opaqueGraphs: gaps.OpaqueGraphReplays,
		}
		found = true
	}
	return best, found
}

// auditIdleAttribution names where an idle-dominated capture's idle time went,
// and reports whether it emitted a finding.
//
// auditDeviceDutyCycle states that the idle time is the larger quantity; it
// deliberately does not diagnose it. That left the report with no finding a
// reader could act on for the majority of the span, so the only actionable
// items remained the kernel-time ones -- the minority the duty cycle had just
// said not to act on first. This finding closes that: it ranks by the idle
// time it accounts for, so a device idling in the hardware queue outranks the
// kernel holding the largest share of a much smaller busy total.
func auditIdleAttribution(cap *cudacapture.Capture, report *AdviceReport) bool {
	attr, ok := captureIdleAttribution(cap)
	if !ok || attr.duty >= deviceIdleCriticalPct {
		return false
	}
	top := attr.arms[0]
	if top.nanos <= 0 {
		return false
	}
	share := float64(top.nanos) / float64(attr.idle) * 100.0
	var split strings.Builder
	for i, arm := range attr.arms {
		if i != 0 {
			split.WriteString(", ")
		}
		fmt.Fprintf(&split, "%s %.3f ms (%.1f%%)", arm.name, float64(arm.nanos)/1e6, float64(arm.nanos)/float64(attr.idle)*100.0)
	}
	finding := fmt.Sprintf("Device %d was idle %.3f ms across %d gaps (median %.3f us), and the largest share of that idle -- %.1f%% -- was %s. Idle attribution: %s.",
		attr.device, float64(attr.idle)/1e6, attr.gaps, float64(attr.median)/1e3, share, top.name, split.String())
	// A plurality is not a majority, and the four remedies are exclusive: a
	// reader who acts on the top arm alone recovers its share and no more.
	// Saying so here keeps the recommendation below from being read as an
	// account of all the idle time.
	if share < 50.0 {
		finding += " No single mechanism holds a majority of the idle time, so a remedy aimed at one arm can recover at most that arm's share."
	}
	// Replay opacity is a property of the instrument, not of the run. Saying
	// so here separates "this idle has no mechanism" from "this capture
	// cannot see the mechanism": the unattributed time is real either way,
	// and only the second is fixed by re-capturing rather than by changing
	// the program.
	if attr.opaqueGraphs {
		finding += " This capture holds graph replay envelopes with no per-node records, which carry no launch correlation, so idle beside a replay cannot be attributed and lands in the unattributed arm. That idle is real and unattributable here, not absent."
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:       SeverityCritical,
		Category:       CategoryArchitectureDispatch,
		EvidenceLevel:  EvidenceObserved,
		severityByKind: true,
		// The ranking key is time on the device timeline, and the time this
		// finding accounts for is idle time. Ranking it by the kernel time it
		// does not hold would sort the majority of the span below the
		// minority -- the ordering this finding exists to correct.
		DeviceTimeNanos:          top.nanos,
		ShareOfProfile:           100.0 - attr.duty,
		magnitude:                fmt.Sprintf("%s: %.2f ms, %.1f%% of idle", top.name, float64(top.nanos)/1e6, share),
		Finding:                  finding,
		ActionableRecommendation: top.remedy,
		NextMeasurement:          "cudaprof gaps <capture> -- the gap distribution and host concurrency classes behind this split",
	})
	return true
}

// journalRemedy and recaptureOrRead keep a diagnosis of missing timestamps
// from turning into a recommendation to produce a file that already exists.
//
// A re-capture costs a run under GPU lock. Telling someone to pay it for a
// capture whose journal is attached is the most expensive thing this tool can
// say, and it was saying it whenever the pprof projection -- which is an
// aggregate and carries no timestamps by construction -- was all that had been
// read.
func captureHasJournal(cap *cudacapture.Capture) bool {
	if cap == nil {
		return false
	}
	for name, data := range cap.Files {
		if cudacapture.IsJournalName(name) && len(data) > 0 {
			return true
		}
	}
	return false
}

func journalRemedy(cap *cudacapture.Capture) string {
	if !captureHasJournal(cap) {
		return ""
	}
	return ". This capture does carry a per-record journal, which holds them"
}

func recaptureOrRead(cap *cudacapture.Capture) string {
	if captureHasJournal(cap) {
		return "Read the journal already in this capture rather than re-running the workload: cudaprof gaps <capture> reports the duty cycle, the idle attribution and the gap distribution from it."
	}
	return "Re-capture with activity timestamps and host launch API records so duty cycle and submission latency can be measured before restructuring dispatch."
}

// anyEventTimed reports whether any event carries a usable device interval.
// End is the field tested, not Start: a timeline whose first activity begins
// at zero is a legitimate one.
func anyEventTimed(events []streamEvent) bool {
	for _, ev := range events {
		if ev.End != 0 && ev.End > ev.Start {
			return true
		}
	}
	return false
}

func anyStreamTimed(streams map[uint32][]streamEvent) bool {
	for _, evs := range streams {
		if anyEventTimed(evs) {
			return true
		}
	}
	return false
}

// unmeasuredLaunchReason states why host submission went unmeasured for a
// sequence, and what to do about it.
//
// Absent from the capture and absent for THESE kernels are different
// statements, and only the first is a reason to capture anything. A capture
// holding host launch records whose correlation ids do not reach a sequence is
// not missing data; something between the launch and the kernel lost the id,
// and that is where to look. Telling someone to capture what they already have
// costs a run under GPU lock and does not produce the missing link.
func unmeasuredLaunchReason(apiRecords int) (why, recommendation string) {
	if apiRecords > 0 {
		return fmt.Sprintf("this capture holds %d host launch record(s) but none correlates to these kernels", apiRecords),
			"Find why the correlation ids of these launches do not reach their kernels before capturing anything further: the host launch records are already here."
	}
	return "host launch timestamps are absent from this capture",
		"Capture host launch API records (correlation-matched) to test whether host submission is behind the device before restructuring dispatch."
}
