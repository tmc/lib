package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Severity denotes the impact of an advice finding.
type Severity string

const (
	SeverityCritical Severity = "Critical"
	SeverityWarning  Severity = "Warning"
	SeverityInfo     Severity = "Info"
)

// Category classifies the optimization or architecture area of an advice finding.
type Category string

const (
	CategoryMemory           Category = "Memory"
	CategoryCompute          Category = "Compute"
	CategoryRegisterPressure Category = "RegisterPressure"
	CategoryArchitecture     Category = "Architecture"
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
	Severity                 Severity `json:"severity"`
	Category                 Category `json:"category"`
	Location                 Location `json:"location"`
	Finding                  string   `json:"finding"`
	ActionableRecommendation string   `json:"actionable_recommendation"`
}

// AdviceReport contains all recommendations resulting from analyzing a capture.
type AdviceReport struct {
	Items []AdviceItem `json:"items"`
}

// Markdown formats the advice report as human-readable Markdown.
func (r *AdviceReport) Markdown() string {
	if r == nil || len(r.Items) == 0 {
		return "# CUDA Kernel Advice Report\n\nNo optimization issues or architecture fallbacks identified.\n"
	}

	var sb strings.Builder
	sb.WriteString("# CUDA Kernel Advice Report\n\n")

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
	for i, item := range r.Items {
		sb.WriteString(fmt.Sprintf("### %d. [%s] %s: %s\n\n", i+1, item.Severity, item.Category, item.Finding))
		locStr := item.Location.String()
		if locStr != "" {
			sb.WriteString(fmt.Sprintf("- **Location**: %s\n", locStr))
		}
		sb.WriteString(fmt.Sprintf("- **Recommendation**: %s\n\n", item.ActionableRecommendation))
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
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{
					Kernel: rmk.Function,
					File:   rmk.File,
					Line:   rmk.Line,
				},
				Finding:                  finding,
				ActionableRecommendation: rec,
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
				Severity: SeverityWarning,
				Category: CategoryCompute,
				Location: Location{
					Kernel: rmk.Function,
					File:   rmk.File,
					Line:   rmk.Line,
				},
				Finding:                  finding,
				ActionableRecommendation: "Pad matrix dimensions to multiples of 8 or 16 and use supported precision types (e.g. FP16, BF16, TF32) to enable hardware Tensor Core acceleration.",
			})
		}
	}

	// 2. Register Pressure Curves (via cudaprof/pressure.go)
	disasm, _ := loadDisassemblyWithPressure(cap)
	if disasm == nil {
		if sassData, ok := findSASS(cap); ok && len(sassData) > 0 {
			if parsed, err := cudasass.Parse(bytes.NewReader(sassData)); err == nil {
				disasm = parsed
			}
		}
	}
	if disasm != nil {
		for _, fn := range disasm.Functions {
			var peakGPR, peakUGPR int
			var peakGPRAddr, peakUGPRAddr uint64
			var peakGPRInst, peakUGPRInst cudasass.Instruction
			var spills []cudasass.Instruction

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
					report.Items = append(report.Items, AdviceItem{
						Severity: SeverityWarning,
						Category: CategoryArchitecture,
						Location: Location{
							Kernel: fn.Name,
							PC:     inst.Address,
							File:   inst.SourceFile,
							Line:   inst.SourceLine,
						},
						Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) detected at PC 0x%04x", inst.Opcode, inst.Address),
						ActionableRecommendation: "Use vectorized 128-bit memory addressing or hardware TMA descriptors to eliminate scalar 64-bit integer address multiplications.",
					})
				}
			}

			// Report register spill points
			for i, spill := range spills {
				if i >= 5 {
					report.Items = append(report.Items, AdviceItem{
						Severity: SeverityCritical,
						Category: CategoryRegisterPressure,
						Location: Location{
							Kernel: fn.Name,
						},
						Finding:                  fmt.Sprintf("Kernel %q has %d total register spill instructions to local memory", fn.Name, len(spills)),
						ActionableRecommendation: "Reduce live variable ranges or specify __launch_bounds__ to eliminate register spills.",
					})
					break
				}
				report.Items = append(report.Items, AdviceItem{
					Severity: SeverityCritical,
					Category: CategoryRegisterPressure,
					Location: Location{
						Kernel: fn.Name,
						PC:     spill.Address,
						File:   spill.SourceFile,
						Line:   spill.SourceLine,
					},
					Finding:                  fmt.Sprintf("Register spill point detected at PC 0x%04x (%s): memory spilled to local stack", spill.Address, formatInstruction(spill)),
					ActionableRecommendation: "Reduce register lifetimes or apply __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) to eliminate spills to slow local memory.",
				})
			}

			// Occupancy cliffs / Peak GPR
			locGPR := Location{
				Kernel: fn.Name,
				PC:     peakGPRAddr,
				File:   peakGPRInst.SourceFile,
				Line:   peakGPRInst.SourceLine,
			}
			if peakGPR > 255 {
				report.Items = append(report.Items, AdviceItem{
					Severity: SeverityCritical,
					Category: CategoryRegisterPressure,
					Location: locGPR,
					Finding:  fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q exceeds hardware register limit (255 GPRs), forcing spilling to local memory", peakGPR, fn.Name),
					ActionableRecommendation: "Reduce register consumption below 255 using compiler flags (-maxrregcount=255 or __launch_bounds__) to prevent excessive local memory spills.",
				})
			} else if peakGPR > 128 {
				report.Items = append(report.Items, AdviceItem{
					Severity: SeverityWarning,
					Category: CategoryRegisterPressure,
					Location: locGPR,
					Finding:  fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff (>128 GPRs, capping SM occupancy at <= 16 warps)", peakGPR, fn.Name),
					ActionableRecommendation: "Optimize register reuse, hoist expressions, or use __launch_bounds__ to keep register pressure at or below 128 GPRs for higher occupancy.",
				})
			} else if peakGPR > 64 {
				report.Items = append(report.Items, AdviceItem{
					Severity: SeverityWarning,
					Category: CategoryRegisterPressure,
					Location: locGPR,
					Finding:  fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q crosses occupancy cliff (>64 GPRs, capping SM occupancy at <= 32 warps)", peakGPR, fn.Name),
					ActionableRecommendation: "Evaluate whether limiting register usage to <= 64 GPRs with __launch_bounds__ improves throughput via higher warp occupancy.",
				})
			}

			// Peak UGPR
			if peakUGPR > 63 {
				report.Items = append(report.Items, AdviceItem{
					Severity: SeverityWarning,
					Category: CategoryRegisterPressure,
					Location: Location{
						Kernel: fn.Name,
						PC:     peakUGPRAddr,
						File:   peakUGPRInst.SourceFile,
						Line:   peakUGPRInst.SourceLine,
					},
					Finding:                  fmt.Sprintf("Peak uniform register pressure of %d UGPRs in kernel %q exceeds architecture limit of 63 UGPRs", peakUGPR, fn.Name),
					ActionableRecommendation: "Reduce warp-uniform temporary variables to prevent uniform register spills.",
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
							Severity: SeverityCritical,
							Category: CategoryArchitecture,
							Location: loc,
							Finding:  fmt.Sprintf("Instruction %q (%s) requires architecture %s, but module targets %s", inst.Opcode, inst.Mnemonic, req, target),
							ActionableRecommendation: fmt.Sprintf("Compile with target architecture >= %s or replace with instructions compatible with %s.", req, target),
						})
					}
				}

				// 2. Scalar fallbacks check on target architecture (sm_90, sm_100, sm_121a, or modern SM)
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
								Severity: SeverityWarning,
								Category: CategoryArchitecture,
								Location: Location{
									Kernel: k.Name,
								},
								Finding:                  fmt.Sprintf("Kernel %q has %d scalar global load instructions on target %s", k.Name, len(scalarLoads), target),
								ActionableRecommendation: "Vectorize memory accesses using 128-bit vector loads (e.g. ld.global.v4) or Tensor Memory Accelerator (TMA) asynchronous copies (cp.async.bulk.tensor) on sm_90+ architectures.",
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
							Severity: SeverityWarning,
							Category: CategoryArchitecture,
							Location: loc,
							Finding:  fmt.Sprintf("Scalar global load (%s) detected on target %s without vectorized or TMA path", sLoad.Opcode, target),
							ActionableRecommendation: "Upgrade memory loads to 128-bit vector accesses (e.g. ld.global.v4) or use TMA asynchronous copies (cp.async.bulk.tensor) on sm_90+ architectures to maximize memory bandwidth utilization.",
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
							Severity: SeverityWarning,
							Category: CategoryArchitecture,
							Location: loc,
							Finding:  fmt.Sprintf("Scalar address arithmetic fallback (%s) detected in kernel %q on target %s", mad.Opcode, k.Name, target),
							ActionableRecommendation: "Use vectorized pointer indexing or TMA multidimensional tensor descriptors to eliminate scalar 64-bit integer address multiplications.",
						})
					}
				}
			}
		}
	}

	return report, nil
}

func runAdviseCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	var format string
	var capturePath string
	for _, arg := range args {
		if arg == "-json" || arg == "--json" || arg == "--format=json" {
			format = "json"
		} else if arg == "-markdown" || arg == "--markdown" || arg == "--format=markdown" {
			format = "markdown"
		} else if strings.HasPrefix(arg, "-") {
			return fmt.Errorf("unknown flag: %s", arg)
		} else if capturePath == "" {
			capturePath = arg
		} else {
			return errors.New("multiple capture files specified")
		}
	}
	if capturePath == "" {
		return errors.New("usage: cudaprof advise [--format=json|markdown] capture.pprof")
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
		out, err := report.JSON()
		if err != nil {
			return err
		}
		fmt.Println(out)
		return nil
	}

	fmt.Print(report.Markdown())
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
