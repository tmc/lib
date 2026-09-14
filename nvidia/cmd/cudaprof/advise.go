package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
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
	Severity                 Severity `json:"severity"`
	Category                 Category `json:"category"`
	Location                 Location `json:"location"`
	Finding                  string   `json:"finding"`
	ActionableRecommendation string   `json:"actionable_recommendation"`
	DeviceTimeNanos          int64    `json:"device_time_nanos,omitempty"`
	ShareOfProfile           float64  `json:"share_of_profile,omitempty"`
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
		header := fmt.Sprintf("### %d. [%s] %s: %s\n\n", i+1, item.Severity, item.Category, item.Finding)
		if item.ShareOfProfile > 0 || item.DeviceTimeNanos > 0 {
			durMs := float64(item.DeviceTimeNanos) / 1e6
			header = fmt.Sprintf("### %d. [%s] (%.2f%% of GPU time, %.2f ms) %s: %s\n\n", i+1, item.Severity, item.ShareOfProfile, durMs, item.Category, item.Finding)
		}
		sb.WriteString(header)
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
			threadsPerBlock := 128
			if l, ok := launchesByKernel[fn.Name]; ok && l.BlockThreads > 0 {
				threadsPerBlock = l.BlockThreads
			}
			occRes, occErr := cudaoccupancy.CalculateOccupancy(arch, peakGPR, 0, threadsPerBlock)

			if peakGPR > 255 {
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityCritical,
					Category:                 CategoryRegisterPressure,
					Location:                 locGPR,
					Finding:                  fmt.Sprintf("Peak register pressure of %d GPRs in kernel %q exceeds hardware register limit (255 GPRs), forcing spilling to local memory", peakGPR, fn.Name),
					ActionableRecommendation: "Reduce register consumption below 255 using compiler flags (-maxrregcount=255 or __launch_bounds__) to prevent excessive local memory spills.",
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
					Location:                 locGPR,
					Finding:                  finding,
					ActionableRecommendation: "Optimize register reuse, hoist expressions, or use __launch_bounds__ to keep register pressure at or below 128 GPRs for higher occupancy.",
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
					Location:                 locGPR,
					Finding:                  finding,
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
							Severity:                 SeverityCritical,
							Category:                 CategoryArchitecture,
							Location:                 loc,
							Finding:                  fmt.Sprintf("Instruction %q (%s) requires architecture %s, but module targets %s", inst.Opcode, inst.Mnemonic, req, target),
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
							Severity:                 SeverityWarning,
							Category:                 CategoryArchitecture,
							Location:                 loc,
							Finding:                  fmt.Sprintf("Scalar global load (%s) detected on target %s without vectorized or TMA path", sLoad.Opcode, target),
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
							Severity:                 SeverityWarning,
							Category:                 CategoryArchitecture,
							Location:                 loc,
							Finding:                  fmt.Sprintf("Scalar address arithmetic fallback (%s) detected in kernel %q on target %s", mad.Opcode, k.Name, target),
							ActionableRecommendation: "Use vectorized pointer indexing or TMA multidimensional tensor descriptors to eliminate scalar 64-bit integer address multiplications.",
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

	// 5. Uncoalesced Memory Access Audit (SASS & PC Sampling & Remarks)
	auditUncoalescedMemory(cap, report)

	// 6. Stream Kernel Sequences (CUDA Graph Candidates & Pointwise Fusion)
	auditStreamKernelSequences(cap, report)

	// 7. Kernel Family Dominance Audit (Macro Architectural Bottlenecks)
	kernelTimes, totalTime := computeProfileTimes(cap)
	auditKernelFamilyDominance(cap, report, kernelTimes, totalTime)

	// Finalize device times and rank findings by device time
	finalizeAndRankFindings(report, kernelTimes, totalTime)

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
			Finding:                  "Launch geometry labels ('grid', 'block') are absent in this profile (requires profiling with updated cupti_agent); launch geometry and SM occupancy audits skipped.",
			ActionableRecommendation: "Re-capture the profile using an updated libcupti_agent that serializes kernel launch dimensions.",
		})
		return
	}

	smCount, smSrc, hasSMCount := determineSMCount(cap)
	maxWarpsPerSM, maxBlocksPerSM, limitSrc, hasLimits := determineDeviceLimits(cap)

	if !hasLimits && !hasSMCount && cap.Profile != nil && len(launches) > 0 {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityInfo,
			Category:                 CategoryArchitecture,
			Finding:                  "Device architecture and SM count metadata are not recorded in this capture, and no local CUDA device is available; theoretical SM occupancy and SM grid starvation audits were skipped.",
			ActionableRecommendation: "Record device provenance in the capture (e.g. capture-provenance.json) or run advise on a host equipped with the target GPU architecture.",
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
					Location:                 launch.Location,
					Finding:                  finding,
					ActionableRecommendation: "Restructure thread block geometry to >=128 threads (4 warps) to enable warp-level latency hiding (worth testing if kernel is latency-bound; note bandwidth-bound streaming kernels may see no speedup).",
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
					Severity: SeverityWarning,
					Category: CategoryArchitectureCompute,
					Location: launch.Location,
					Finding:  fmt.Sprintf("Multi-warp thread block underutilization in row reduction: kernel %q launched with %d threads/block (>= 4 warps) executes sequential chunk reductions with idle warps during accumulation", launch.Kernel, launch.BlockThreads),
					ActionableRecommendation: "Group multiple rows per thread block (e.g. 4 rows per 128-thread block so warp i computes row 4k+i) to maintain 100% warp utilization during chunk accumulation.",
				})
			}
		}
	}

	// 2. SM Grid Starvation (< SMCount)
	if hasSMCount {
		seenGridStarvation := make(map[string]bool)
		for _, launch := range launches {
			if launch.GridBlocks > 0 && launch.GridBlocks < smCount {
				if seenGridStarvation[launch.Kernel] {
					continue
				}
				seenGridStarvation[launch.Kernel] = true
				idlePct := (smCount - launch.GridBlocks) * 100 / smCount
				provSuffix := ""
				if smSrc == sourceLiveDevice {
					provSuffix = " (SM count inferred from local host device, not recorded in capture)"
				}
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
					Category:                 CategoryArchitecture,
					Location:                 launch.Location,
					Finding:                  fmt.Sprintf("Grid size leaves physical SMs completely idle: kernel %q launched with %d grid blocks on a %d-SM GPU leaves %d%% of SMs idle%s", launch.Kernel, launch.GridBlocks, smCount, idlePct, provSuffix),
					ActionableRecommendation: "Decompose grid using 2D tiling, split-K reduction, or sub-grid pipelining (griddepcontrol).",
				})
			}
		}
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
	if arch == "" {
		arch = "sm_121a"
	}
	dev, err := cudaoccupancy.LookupDevice(arch)
	if err != nil {
		dev = cudaoccupancy.SM121
	}

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
				Severity: SeverityCritical,
				Category: CategoryRegisterPressure,
				Location: Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q allocates %d bytes of local stack frame per thread in CUBIN ELF (EIATTR_FRAME_SIZE), causing spills to high-latency local memory",
					k.Name, k.FrameSizeBytes),
				ActionableRecommendation: "Reduce local variable lifetimes, eliminate large stack arrays, or apply __launch_bounds__ to eliminate local memory stack frame spills.",
			})
		}

		// 2. Compiler register limit (EIATTR_MAXREG_COUNT)
		if k.MaxRegCount > 0 && k.Registers >= k.MaxRegCount {
			report.Items = append(report.Items, AdviceItem{
				Severity: SeverityWarning,
				Category: CategoryRegisterPressure,
				Location: Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q reached the compiler register limit of %d GPRs (EIATTR_MAXREG_COUNT), which may force register spilling or sub-optimal instruction scheduling",
					k.Name, k.MaxRegCount),
				ActionableRecommendation: "Raise compiler register budget (-maxrregcount) or restructure compute loops to reduce register demand.",
			})
		}

		// 3. Analytical occupancy & hardware limiting factor
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
			if rep.Occupancy <= 0.25 {
				severity = SeverityCritical
			}

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
				rec = fmt.Sprintf("Register file capacity (%d regs/SM) limits residency to %d blocks/SM. Reducing registers from %d to <=%d GPRs would allow %d blocks/SM (%d warps, %.1f%% occupancy). Test with __launch_bounds__ or -maxrregcount=%d if latency-bound.",
					dev.MaxRegistersPerSM, rep.ActiveBlocks, k.Registers, maxRegsTarget, targetBlocks, targetWarps, targetOcc, maxRegsTarget)
			case cudaoccupancy.LimitSharedMemory:
				rec = fmt.Sprintf("Static shared memory (%d bytes/block) limits residency to %d blocks/SM on %s (%d bytes/SM). Reduce shared memory or dynamically allocate to allow higher CTA residency.",
					k.SharedMemBytes, rep.ActiveBlocks, dev.Architecture, dev.MaxSharedMemoryPerSM)
			case cudaoccupancy.LimitBlocks:
				rec = fmt.Sprintf("Maximum thread blocks per SM (%d on %s) limits occupancy because block size (%d threads) is small. Restructure thread block geometry to >=128 or 256 threads to reach higher occupancy (worth testing if latency-bound).",
					dev.MaxBlocksPerSM, dev.Architecture, threadsPerBlock)
			case cudaoccupancy.LimitBlockSize:
				rec = "Restructure thread block geometry to fit within hardware block size limits (<= 1024 threads per block)."
			default:
				rec = "Tune thread block dimensions and resource consumption to maximize concurrent warp execution."
			}

			report.Items = append(report.Items, AdviceItem{
				Severity:                 severity,
				Category:                 CategoryRegisterPressure,
				Location:                 Location{Kernel: k.Name},
				Finding:                  finding,
				ActionableRecommendation: rec,
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
	if len(s.Location) > 0 && len(s.Location[0].Line) > 0 {
		line := s.Location[0].Line[0]
		if line.Function != nil && line.Function.Name != "" {
			name := line.Function.Name
			name = strings.TrimSuffix(name, ".SASS")
			name = strings.TrimSuffix(name, ".PTX")
			name = strings.TrimSuffix(name, ".TILEIR")
			return name
		}
	}
	return ""
}

func auditUncoalescedMemory(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}

	code, _ := cudacapture.LoadCode(cap)
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
		for i, st := range cap.Profile.SampleType {
			if strings.HasPrefix(st.Type, "stall_") {
				stallIndices = append(stallIndices, i)
			}
			tLower := strings.ToLower(st.Type)
			if tLower == "stall_memory_throttle" || tLower == "stall_mem_throttle" || tLower == "stall_mem" {
				memThrottleIndices = append(memThrottleIndices, i)
			}
		}

		type pcStats struct {
			totalStalls int64
			memStalls   int64
			opcode      string
			file        string
			line        int
		}
		byKernelPC := make(map[string]map[uint64]*pcStats)

		for _, s := range cap.Profile.Sample {
			var totalStall, memStall int64
			for _, idx := range stallIndices {
				totalStall += s.Value[idx]
			}
			for _, idx := range memThrottleIndices {
				memStall += s.Value[idx]
			}

			if memStall == 0 && s.Label != nil {
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
						if totalStall == 0 {
							totalStall = s.Value[0]
						}
					}
				}
			}

			if totalStall == 0 && memStall == 0 {
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

			for pc, stat := range pcMap {
				if stat.totalStalls <= 0 {
					continue
				}
				ratio := float64(stat.memStalls) / float64(stat.totalStalls)
				if ratio <= 0.35 {
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

				opUpper := strings.ToUpper(opcode)
				if strings.HasPrefix(opUpper, "LDG.E") || strings.HasPrefix(opUpper, "LDG") || strings.Contains(opUpper, "LDG") {
					sev := SeverityWarning
					if ratio >= 0.50 {
						sev = SeverityCritical
					}
					locKey := fmt.Sprintf("%s:0x%x", kName, pc)
					reportedLocations[locKey] = true

					report.Items = append(report.Items, AdviceItem{
						Severity: sev,
						Category: CategoryMemory,
						Location: Location{
							Kernel: kName,
							PC:     pc,
							File:   file,
							Line:   line,
						},
						Finding:                  fmt.Sprintf("Threads within the warp generate strided or disparate 128-byte cache line requests, generating up to 32 cache transactions per instruction (stall_memory_throttle %.1f%% exceeds 35%% on %s at PC 0x%04x in kernel %q)", ratio*100, opcode, pc, kName),
						ActionableRecommendation: "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line.",
					})
				}
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
					Severity: SeverityWarning,
					Category: CategoryMemory,
					Location: Location{
						Kernel: rmk.Function,
						File:   rmk.File,
						Line:   rmk.Line,
					},
					Finding:                  finding,
					ActionableRecommendation: "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line.",
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
						Severity: SeverityWarning,
						Category: CategoryMemory,
						Location: Location{
							Kernel: kName,
						},
						Finding:                  finding,
						ActionableRecommendation: "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line.",
					})
				}
			}
		}
	}
}

type streamEvent struct {
	Subtype    string
	Name       string
	Stream     uint32
	DurationNs uint64
	Shape      string
	Buffer     string
	Grid       [3]int
	Block      [3]int
}

func auditStreamKernelSequences(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}

	streamEvents := extractStreamEvents(cap)
	if len(streamEvents) == 0 {
		return
	}

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
			if (sub == "kernel" || sub == "") && ev.DurationNs < 10000 && ev.DurationNs > 0 {
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

				if median < 10000 {
					sev := SeverityWarning
					if median >= 5000 {
						sev = SeverityInfo
					}
					finding := fmt.Sprintf("Driver submission overhead dominates short kernel execution: sequence of %d consecutive kernels on stream %d has median duration %.2f µs (< 10 µs)", len(kernelSeq), streamID, float64(median)/1000.0)
					if streamMicroCount >= 5 {
						finding += fmt.Sprintf(" (stream totals %d micro-kernels accumulating %.2f ms)", streamMicroCount, float64(streamMicroTotalNs)/1e6)
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
						finding += fmt.Sprintf(" [top micro-kernels: %s]", strings.Join(topList, ", "))
					}
					report.Items = append(report.Items, AdviceItem{
						Severity: sev,
						Category: CategoryArchitectureDispatch,
						Location: Location{
							Kernel: kernelSeq[0].Name,
						},
						Finding:                  finding,
						ActionableRecommendation: "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency.",
					})
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

		// 2. Heuristic 5: Pointwise Producer-Consumer Fusion & Sequential Misses
		seenFusion := make(map[string]bool)
		for i := 0; i < len(events)-1; i++ {
			ev1 := events[i]
			ev2 := events[i+1]
			sub1 := strings.ToLower(ev1.Subtype)
			sub2 := strings.ToLower(ev2.Subtype)
			if (sub1 != "kernel" && sub1 != "") || (sub2 != "kernel" && sub2 != "") {
				continue
			}

			// Elementwise producer-consumer fusion
			if isElementwise(ev1) && isElementwise(ev2) {
				if matchingShapesOrBuffers(ev1, ev2) {
					key := fmt.Sprintf("%s->%s:%d", ev1.Name, ev2.Name, streamID)
					if seenFusion[key] {
						continue
					}
					seenFusion[key] = true

					cleanName1 := cleanKernelName(ev1.Name)
					cleanName2 := cleanKernelName(ev2.Name)
					rec := fmt.Sprintf("Fuse into a single kernel (e.g. fused %s + %s).", cleanName1, cleanName2)

					report.Items = append(report.Items, AdviceItem{
						Severity: SeverityWarning,
						Category: CategoryMemoryCompute,
						Location: Location{
							Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
						},
						Finding:                  fmt.Sprintf("Redundant global memory round-trips for intermediate tensors: sequential back-to-back elementwise kernels (%s followed by %s) on stream %d operate on matching tensors", ev1.Name, ev2.Name, streamID),
						ActionableRecommendation: rec,
					})
				}
			}

			// Sequential RMSNorm -> Block Quantization Fusion
			if isNormKernel(ev1.Name) && isQuantKernel(ev2.Name) {
				key := fmt.Sprintf("norm_quant:%s->%s:%d", ev1.Name, ev2.Name, streamID)
				if !seenFusion[key] {
					seenFusion[key] = true
					cleanName1 := cleanKernelName(ev1.Name)
					cleanName2 := cleanKernelName(ev2.Name)
					report.Items = append(report.Items, AdviceItem{
						Severity: SeverityWarning,
						Category: CategoryMemoryCompute,
						Location: Location{
							Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
						},
						Finding:                  fmt.Sprintf("Redundant global memory round-trips for normalized activations: %s writes normalized activations to DRAM and %s immediately reads them back for quantization on stream %d", ev1.Name, ev2.Name, streamID),
						ActionableRecommendation: fmt.Sprintf("Fuse normalization and quantization into a single kernel (e.g. fused %s_%s) that normalizes in shared memory/registers and directly stores quantized blocks.", cleanName1, cleanName2),
					})
				}
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
					key := fmt.Sprintf("dual_gemv:%s+%s:%d", ev1.Name, ev2.Name, streamID)
					if !seenFusion[key] {
						seenFusion[key] = true
						cleanName1 := cleanKernelName(ev1.Name)
						cleanName2 := cleanKernelName(ev2.Name)
						report.Items = append(report.Items, AdviceItem{
							Severity: SeverityInfo,
							Category: CategoryMemoryCompute,
							Location: Location{
								Kernel: fmt.Sprintf("%s + %s", ev1.Name, ev2.Name),
							},
							Finding:                  fmt.Sprintf("Redundant activation DRAM traffic: sequential GEMVs (%s and %s) on stream %d independently load the identical input activation vector", ev1.Name, ev2.Name, streamID),
							ActionableRecommendation: fmt.Sprintf("Fuse into a paired dual-GEMV dispatch (e.g. fused %s + %s) that loads the activation vector into shared memory once and accumulates dual projection outputs.", cleanName1, cleanName2),
						})
					}
				}
			}
		}
	}
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
	res := make(map[uint32][]streamEvent)
	if cap == nil {
		return res
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
					Subtype:    subtype,
					Name:       name,
					Stream:     streamID,
					DurationNs: durationNs,
					Shape:      shape,
					Buffer:     buffer,
					Grid:       grid,
					Block:      block,
				})
			}
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

	return res
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

func auditCaptureStatus(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	if statusData, ok := cap.Files["capture-status.json"]; ok {
		var status map[string]any
		if err := json.Unmarshal(statusData, &status); err == nil {
			if success, ok := status["capture_success"].(bool); ok && !success {
				errMsg := "capture marked as failed"
				if failure, ok := status["failure"].(string); ok && failure != "" {
					errMsg = failure
				} else if captureErr, ok := status["capture_error"].(string); ok && captureErr != "" {
					errMsg = captureErr
				}
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityCritical,
					Category:                 CategoryArchitecture,
					Finding:                  fmt.Sprintf("Incomplete or failed profile capture: %s", errMsg),
					ActionableRecommendation: "Ensure target workload executes cleanly and does not exceed journal record capacity before analyzing performance.",
				})
			}
			if dropped, ok := status["dropped_records"].(float64); ok && uint64(dropped) > 0 {
				report.Items = append(report.Items, AdviceItem{
					Severity:                 SeverityWarning,
					Category:                 CategoryArchitecture,
					Finding:                  fmt.Sprintf("Lossy capture: %d records dropped due to CUPTI buffer pressure", uint64(dropped)),
					ActionableRecommendation: "Whole-run totals and aggregate launch counts are incomplete (a lower bound). Increase buffer pool or reduce profiling overhead to capture a complete trace.",
				})
			}
		}
	}
}

func computeProfileTimes(cap *cudacapture.Capture) (map[string]int64, int64) {
	kernelTimes := make(map[string]int64)
	var totalTime int64
	if cap == nil || cap.Profile == nil {
		return kernelTimes, totalTime
	}
	p := cap.Profile
	timeIdx := 0
	for i, vt := range p.SampleType {
		if vt.Type == "kernel_time" || vt.Type == "gpu_time" {
			timeIdx = i
			break
		}
	}
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
	return kernelTimes, totalTime
}

func auditKernelFamilyDominance(cap *cudacapture.Capture, report *AdviceReport, kernelTimes map[string]int64, totalTime int64) {
	if totalTime <= 0 || len(kernelTimes) == 0 {
		return
	}
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
			fTimeMs := float64(fTime) / 1e6
			report.Items = append(report.Items, AdviceItem{
				Severity:        SeverityCritical,
				Category:        CategoryArchitecture,
				DeviceTimeNanos: fTime,
				ShareOfProfile:  pct,
				Location: Location{
					Kernel: family,
				},
				Finding: fmt.Sprintf("Kernel family %q dominates GPU execution, consuming %.1f%% of total device time (%.2f ms across %d dispatches). Fine-grained per-row dispatches incur massive submission and launch overhead.",
					family, pct, fTimeMs, cnt),
				ActionableRecommendation: "Restructure per-row dispatches into batched GEMM / GEMV operations or capture the entire layer sequence into a CUDA Graph.",
			})
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
	if report == nil || len(report.Items) == 0 {
		return
	}
	for i := range report.Items {
		item := &report.Items[i]
		if item.DeviceTimeNanos == 0 && item.Location.Kernel != "" {
			item.DeviceTimeNanos = kernelTimes[item.Location.Kernel]
		}
		if totalTime > 0 && item.DeviceTimeNanos > 0 && item.ShareOfProfile == 0 {
			item.ShareOfProfile = float64(item.DeviceTimeNanos) / float64(totalTime) * 100.0
		}
	}

	sort.SliceStable(report.Items, func(i, j int) bool {
		a, b := report.Items[i], report.Items[j]
		if a.DeviceTimeNanos != b.DeviceTimeNanos {
			return a.DeviceTimeNanos > b.DeviceTimeNanos
		}
		if a.Severity != b.Severity {
			return severityRank(a.Severity) < severityRank(b.Severity)
		}
		if a.Category != b.Category {
			return a.Category < b.Category
		}
		return a.Finding < b.Finding
	})
}
