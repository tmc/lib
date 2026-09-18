package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/internal/cudaview"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func runViewCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New("usage: cudaprof view [-http=addr] [-html=file] [-no-browser] [kernel] capture.pprof")
	}

	cmd := args[0]
	if cmd == "view" || cmd == "web" || cmd == "-web" || cmd == "--web" {
		args = args[1:]
	}

	httpAddr := ":8088"
	htmlOut := ""
	noBrowser := false
	var diffPath string
	var kernel string
	var posArgs []string

	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "-http="):
			httpAddr = strings.TrimPrefix(a, "-http=")
		case strings.HasPrefix(a, "--http="):
			httpAddr = strings.TrimPrefix(a, "--http=")
		case strings.HasPrefix(a, "-html="):
			htmlOut = strings.TrimPrefix(a, "-html=")
		case strings.HasPrefix(a, "--html="):
			htmlOut = strings.TrimPrefix(a, "--html=")
		case strings.HasPrefix(a, "-output="):
			htmlOut = strings.TrimPrefix(a, "-output=")
		case strings.HasPrefix(a, "--output="):
			htmlOut = strings.TrimPrefix(a, "--output=")
		case a == "-no-browser" || a == "--no-browser" || a == "-no_browser":
			noBrowser = true
		case strings.HasPrefix(a, "-diff="):
			diffPath = strings.TrimPrefix(a, "-diff=")
		case strings.HasPrefix(a, "--diff="):
			diffPath = strings.TrimPrefix(a, "--diff=")
		default:
			posArgs = append(posArgs, a)
		}
	}

	var capturePath string
	switch len(posArgs) {
	case 0:
		// Default to capture.pprof in current directory if present
		if _, err := os.Stat("capture.pprof"); err == nil {
			capturePath = "capture.pprof"
		} else {
			return errors.New("usage: cudaprof view [-http=addr] [-html=file] [-diff=candidate.pprof] [kernel] capture.pprof")
		}
	case 1:
		capturePath = posArgs[0]
	case 2:
		kernel = posArgs[0]
		capturePath = posArgs[1]
	default:
		return errors.New("usage: cudaprof view [-http=addr] [-html=file] [-diff=candidate.pprof] [kernel] capture.pprof")
	}

	if diffPath != "" {
		return runDiffWeb(DiffWebOptions{
			BaseFile:     capturePath,
			TargetFile:   diffPath,
			KernelFilter: kernel,
			HTTPAddr:     httpAddr,
			HTMLOut:      htmlOut,
			NoBrowser:    noBrowser,
			Limits:       limits,
			Writer:       os.Stdout,
		})
	}

	absPath, err := filepath.Abs(capturePath)
	if err != nil {
		return fmt.Errorf("view: resolve capture path: %w", err)
	}

	cap, err := readCapturePathLimits(absPath, limits)
	if err != nil {
		return fmt.Errorf("view: read capture: %w", err)
	}

	data, err := buildPipelineViewData(cap, filepath.Base(absPath), kernel)
	if err != nil {
		return fmt.Errorf("view: build pipeline data: %w", err)
	}

	// Static HTML export mode. The NVRTC panel needs the live server to
	// compile, so the exported page does not offer it.
	if htmlOut != "" {
		data.Recompile = nil
		var buf bytes.Buffer
		if err := cudaview.Render(&buf, data); err != nil {
			return fmt.Errorf("view: render HTML: %w", err)
		}
		if err := os.WriteFile(htmlOut, buf.Bytes(), 0644); err != nil {
			return fmt.Errorf("view: write HTML output: %w", err)
		}
		fmt.Printf("Wrote 5-tier pipeline report to %s (%d bytes)\n", htmlOut, buf.Len())
		return nil
	}

	// Interactive HTTP Server mode
	handler := newPipelineServerHandler(data)
	listener, err := net.Listen("tcp", httpAddr)
	if err != nil {
		return fmt.Errorf("view: listen on %s: %w", httpAddr, err)
	}
	defer listener.Close()

	_, port, _ := net.SplitHostPort(listener.Addr().String())
	if port == "" {
		port = listener.Addr().String()
	}
	url := fmt.Sprintf("http://127.0.0.1:%s", port)
	fmt.Printf("Serving 5-tier GPU pipeline view at %s\n", url)
	fmt.Printf("Tiers: Go -> CUDA C++ -> Tile IR -> PTX -> SASS\n")
	fmt.Printf("Press Ctrl+C to stop.\n")

	if !noBrowser {
		openBrowser(url)
	}

	server := &http.Server{
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}
	return server.Serve(listener)
}

func buildSinglePipelineViewData(cap *cudacapture.Capture, capName, activeKernel string) (*cudaview.PipelineViewData, error) {
	data := &cudaview.PipelineViewData{
		Title:        "5-Tier GPU Compilation Pipeline: Go -> CUDA C++ -> Tile IR -> PTX -> SASS",
		CaptureFile:  capName,
		KernelTiers:  make(map[string][]cudaview.PipelineTier),
		ActiveKernel: activeKernel,
	}

	// 1. Extract metadata (device, arch, compile options)
	var nvrtcOptions map[string][]string
	if len(cap.Metadata) > 0 {
		var meta struct {
			Device string `json:"device"`
			Arch   string `json:"architecture"`
			// NVRTCOptions records the options each module was compiled
			// with, keyed by module ID or name. A program compiles its
			// modules separately and need not pass NVRTC the same options
			// for each, so there is no single list to record.
			NVRTCOptions map[string][]string `json:"nvrtc_options"`
		}
		if json.Unmarshal(cap.Metadata, &meta) == nil {
			data.Device = meta.Device
			data.Architecture = meta.Arch
			nvrtcOptions = meta.NVRTCOptions
		}
	}

	// 2. Discover full tier contents
	// Tier 0: Go Source (look for workload.go, main.go, or any .go file)
	goFileName := "workload.go"
	goContent := ""
	if b, ok := cap.Files[goFileName]; ok {
		goContent = string(b)
	} else {
		for fname, b := range cap.Files {
			if strings.HasSuffix(fname, ".go") && !strings.HasSuffix(fname, "_test.go") {
				goFileName = fname
				goContent = string(b)
				break
			}
		}
	}

	// Tier 1: CUDA C++ Source
	cuFileName := cap.Code.Source
	if cuFileName == "" {
		cuFileName = "kernels.cu"
	}
	cuContent := string(cap.Files[cuFileName])

	// Tier 2: Tile IR
	tileFileName := cap.Code.TileIR
	if tileFileName == "" {
		tileFileName = "kernels.tileir"
	}
	tileContent := ""
	if tileBytes, ok := cap.Files[tileFileName]; ok {
		if tileir.IsTileIRBytecode(tileBytes) {
			if bm, err := tileir.DecodeBytecodeBytes(tileBytes); err == nil && bm != nil {
				tileContent = bm.ToModule().String()
			} else {
				tileContent = string(tileBytes)
			}
		} else {
			tileContent = string(tileBytes)
		}
	}

	// Tier 3: PTX
	ptxFileName := cap.Code.PTX
	if ptxFileName == "" {
		ptxFileName = "kernels.ptx"
	}
	ptxContent := string(cap.Files[ptxFileName])

	// Tier 4: SASS
	sassFileName := cap.Code.SASS
	if sassFileName == "" {
		sassFileName = "kernels.sass"
	}
	sassContent := string(cap.Files[sassFileName])

	data.Recompile = recompileSources(cap, data.Architecture, nvrtcOptions)

	// Remarks & CFG if present
	var remarksList []tileiras.Remark
	if rmkBytes, ok := cap.Files["tileiras-remarks.yaml"]; ok {
		data.Remarks = string(rmkBytes)
		if rList, err := tileiras.ParseRemarksYAML(rmkBytes); err == nil {
			remarksList = rList
			data.RemarksCount = len(remarksList)
		}
	}
	if dotBytes, ok := cap.Files[cap.Code.CFG]; ok && len(dotBytes) > 0 {
		data.CFGDot = string(dotBytes)
	} else if dotBytes, ok := cap.Files["kernels.cfg.dot"]; ok {
		data.CFGDot = string(dotBytes)
	}

	// Analyze Tile IR module for verification & audit
	if tileContent != "" {
		if mod, err := tileir.ParseString(tileContent); err == nil && mod != nil {
			if len(remarksList) > 0 {
				mod.AttachRemarks(remarksList)
			}
			summary := mod.Summary()
			data.TileIROpcodes = summary.Counts
			hazards, shapeErrs := mod.Verify()
			status := "Verified"
			if len(hazards) > 0 || len(shapeErrs) > 0 {
				status = fmt.Sprintf("%d hazard(s)", len(hazards)+len(shapeErrs))
			}
			data.TileIRAudit = fmt.Sprintf("Tile IR: %s (%d ops • %d TMA, %d MMA)",
				status, summary.Total,
				summary.CategoryCount(tileir.CatTileMemory),
				summary.CategoryCount(tileir.CatComputeMMA),
			)
		}
	}

	// 3. Discover kernels across PTX, SASS, CUDA C++, and Tile IR
	kernelNames := discoverKernelNames(ptxContent, sassContent, cuContent, tileContent)
	data.Kernels = kernelNames

	// 4. Extract stall metrics from PC sampling data in profile (gpu.pprof)
	stallMap, fusion := extractSASSStallMap(cap, kernelNames)
	if fusion != nil {
		data.Diagnostics = append(data.Diagnostics, sourceFusionCaveat(fusion))
	}

	// 5. Extract live register pressure if available
	var pressureMap map[string]map[uint64]cudasass.RegisterPressure
	if disasmWithPressure, err := extractPreJoinedPressure(cap); err == nil && disasmWithPressure != nil {
		pressureMap = extractPressureMap(disasmWithPressure)
	}

	// Build full tiers
	data.Tiers = []cudaview.PipelineTier{
		buildTier("Go", goFileName, "go", goContent, nil, "", nil, kernelNames, nil),
		buildTier("CUDA C++", cuFileName, "cuda", cuContent, nil, "", nil, kernelNames, nil),
		buildTier("Tile IR", tileFileName, "mlir", tileContent, nil, "", remarksList, kernelNames, nil),
		buildTier("PTX", ptxFileName, "ptx", ptxContent, nil, "", nil, kernelNames, nil),
		buildTier("SASS", sassFileName, "sass", sassContent, stallMap, activeKernel, nil, kernelNames, pressureMap),
	}

	// Kernel listings select existing annotated lines; source and annotations
	// are built once per module.
	contents := []string{goContent, cuContent, tileContent, ptxContent, sassContent}
	languages := []string{"go", "cuda", "tileir", "ptx", "sass"}
	for _, kName := range kernelNames {
		tiers := make([]cudaview.PipelineTier, len(data.Tiers))
		for i, full := range data.Tiers {
			tiers[i] = full
			section := extractKernelSection(contents[i], kName, languages[i])
			if section == contents[i] {
				continue
			}
			if i == 0 {
				tiers[i].Lines = nil
				for _, line := range full.Lines {
					if strings.Contains(strings.ToLower(line.Text), strings.ToLower(kName)) {
						tiers[i].Lines = append(tiers[i].Lines, line)
					}
				}
				continue
			}
			if offset := strings.Index(contents[i], section); offset >= 0 {
				start := strings.Count(contents[i][:offset], "\n")
				end := start + strings.Count(section, "\n") + 1
				tiers[i].Lines = full.Lines[start:end]
			}
		}
		data.KernelTiers[kName] = tiers
	}

	return data, nil
}

type sassStallMap struct {
	hasStalls    bool
	byKernelPC   map[string]map[uint64]*pcStallInfo
	kernelTotals map[string]int64
	totalStalls  int64
}

type pcStallInfo struct {
	totalStalls int64
	breakdown   map[string]int64
}

// extractSASSStallMap returns the per-PC stall map and the source-fusion
// failure the capture's code was loaded through, if any. The failure travels
// with the map because the tiers this view renders beside it -- CUDA C++, Tile
// IR, PTX -- are the ones it invalidates, and an unannotated view of them is
// indistinguishable from a compilation that carried no line information.
func extractSASSStallMap(cap *cudacapture.Capture, knownKernels []string) (*sassStallMap, *cudacapture.SourceFusionError) {
	if cap == nil {
		return nil, nil
	}
	// Loaded before the early returns below, so a capture with no samples still
	// reports the failure: an empty stall map is not a reason to stop saying
	// that the source binding was refused.
	code, fusion, _ := loadCodeTiered(cap)
	prof := cap.Profile
	if prof == nil && cap.Files != nil && len(cap.Files["gpu.pprof"]) > 0 {
		if p, err := profile.ParseData(cap.Files["gpu.pprof"]); err == nil {
			prof = p
		}
	}
	if prof == nil || len(prof.Sample) == 0 {
		return nil, fusion
	}

	var stallIndices []int
	for i, st := range prof.SampleType {
		if strings.HasPrefix(st.Type, "stall_") {
			stallIndices = append(stallIndices, i)
		}
	}

	var disasm *cudasass.Disassembly
	if sassBytes, ok := cap.Files[cap.Code.SASS]; ok && len(sassBytes) > 0 {
		disasm, _ = cudasass.Parse(bytes.NewReader(sassBytes))
	} else if sassBytes, ok := cap.Files["kernels.sass"]; ok && len(sassBytes) > 0 {
		disasm, _ = cudasass.Parse(bytes.NewReader(sassBytes))
	}

	m := &sassStallMap{
		byKernelPC:   make(map[string]map[uint64]*pcStallInfo),
		kernelTotals: make(map[string]int64),
	}

	for _, s := range prof.Sample {
		if len(s.Location) == 0 {
			continue
		}
		loc := s.Location[0]
		kName, pc := resolveLocationPC(loc, s, code, disasm)
		if kName == "" {
			if disasm != nil && len(disasm.Functions) == 1 {
				kName = disasm.Functions[0].Name
			} else if len(knownKernels) == 1 {
				kName = knownKernels[0]
			}
		}

		sampleStalls := make(map[string]int64)
		var sampleTotal int64

		if len(stallIndices) > 0 {
			for _, idx := range stallIndices {
				val := s.Value[idx]
				if val > 0 {
					metricName := prof.SampleType[idx].Type
					sampleStalls[metricName] += val
					sampleTotal += val
				}
			}
		}

		if sampleTotal == 0 && s.Label != nil {
			var reason string
			if len(s.Label["sass.stall"]) > 0 {
				reason = s.Label["sass.stall"][0]
			} else if len(s.Label["stall"]) > 0 {
				reason = s.Label["stall"][0]
			}
			if reason != "" {
				if !strings.HasPrefix(reason, "stall_") {
					reason = "stall_" + reason
				}
				var val int64
				if len(s.Value) > 0 {
					val = s.Value[0]
				} else {
					val = 1
				}
				if val > 0 {
					sampleStalls[reason] += val
					sampleTotal += val
				}
			}
		}

		if sampleTotal == 0 {
			continue
		}

		m.hasStalls = true
		m.totalStalls += sampleTotal
		m.kernelTotals[kName] += sampleTotal

		if m.byKernelPC[kName] == nil {
			m.byKernelPC[kName] = make(map[uint64]*pcStallInfo)
		}
		info := m.byKernelPC[kName][pc]
		if info == nil {
			info = &pcStallInfo{breakdown: make(map[string]int64)}
			m.byKernelPC[kName][pc] = info
		}
		info.totalStalls += sampleTotal
		for r, v := range sampleStalls {
			info.breakdown[r] += v
		}
	}

	return m, fusion
}

func (m *sassStallMap) lookup(kernel string, pc uint64) (*pcStallInfo, int64) {
	if m == nil || !m.hasStalls || kernel == "" {
		return nil, 0
	}
	return m.byKernelPC[kernel][pc], m.kernelTotals[kernel]
}

func formatStallTooltip(pct float64, breakdown map[string]float64, totalCount int64) string {
	type kv struct {
		reason string
		pct    float64
	}
	var list []kv
	for k, v := range breakdown {
		list = append(list, kv{reason: k, pct: v})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].pct != list[j].pct {
			return list[i].pct > list[j].pct
		}
		return list[i].reason < list[j].reason
	})

	var parts []string
	for _, item := range list {
		parts = append(parts, fmt.Sprintf("%s: %.1f%%", item.reason, item.pct))
	}
	if len(parts) > 0 {
		return fmt.Sprintf("Stall: %.1f%% (%s)", pct, strings.Join(parts, ", "))
	}
	return fmt.Sprintf("Stall: %.1f%%", pct)
}

func extractTileIROpcode(line string) string {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") {
		return ""
	}
	if idx := strings.Index(trimmed, "cuda_tile."); idx >= 0 {
		rest := trimmed[idx+len("cuda_tile."):]
		end := len(rest)
		for i, r := range rest {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
				end = i
				break
			}
		}
		if end > 0 {
			return "cuda_tile." + rest[:end]
		}
	}
	return ""
}

func extractPressureMap(disasm *cudasass.Disassembly) map[string]map[uint64]cudasass.RegisterPressure {
	if disasm == nil {
		return nil
	}
	res := make(map[string]map[uint64]cudasass.RegisterPressure)
	for _, fn := range disasm.Functions {
		res[fn.Name] = make(map[uint64]cudasass.RegisterPressure)
		for _, inst := range fn.Instructions {
			if inst.LiveGPR > 0 || inst.LivePred > 0 || inst.LiveUGPR > 0 || inst.LiveUPred > 0 {
				res[fn.Name][inst.Address] = cudasass.RegisterPressure{
					GPR:   inst.LiveGPR,
					Pred:  inst.LivePred,
					UGPR:  inst.LiveUGPR,
					UPred: inst.LiveUPred,
				}
			}
		}
	}
	return res
}

func buildTier(name, fileName, lang, content string, stallMap *sassStallMap, activeKernel string, remarks []tileiras.Remark, knownKernels []string, pressureMap map[string]map[uint64]cudasass.RegisterPressure) cudaview.PipelineTier {
	rawLines := strings.Split(content, "\n")
	lines := make([]cudaview.PipelineLine, len(rawLines))

	currentKernel := activeKernel
	currentSourceLine := 0

	for i, text := range rawLines {
		line := cudaview.PipelineLine{
			Num:  i + 1,
			Text: text,
		}

		trimmed := strings.TrimSpace(text)

		switch lang {
		case "go":
			// Detect kernel launch calls in Go
			for _, k := range knownKernels {
				if k == "" {
					continue
				}
				if strings.Contains(trimmed, `"`+k+`"`) ||
					strings.Contains(trimmed, k+`(`) ||
					strings.Contains(strings.ToLower(trimmed), "launch"+strings.ToLower(k)) {
					line.LaunchKernel = k
					line.SemanticTag = "entry"
					break
				}
			}
			if strings.Contains(trimmed, "trace.WithRegion") || strings.Contains(trimmed, "trace.NewTask") || strings.Contains(trimmed, "cudaprof.Labels") {
				line.Category = "Trace"
			}

		case "cuda":
			line.SourceLine = line.Num
			if strings.Contains(trimmed, "__global__ void") {
				line.Category = "Global Kernel"
				line.SemanticTag = "entry"
				for _, k := range knownKernels {
					if strings.Contains(trimmed, k) {
						currentKernel = k
						break
					}
				}
			} else if strings.Contains(trimmed, "__device__") {
				line.Category = "Device Helper"
			} else if strings.Contains(trimmed, "extern __shared__") || strings.Contains(trimmed, "__shared__") {
				line.Category = "Shared Memory"
				line.SemanticTag = "memory"
			} else if strings.Contains(trimmed, "__launch_bounds__") {
				line.Category = "Launch Bounds"
			}

			if strings.Contains(trimmed, "__dp4a") {
				line.SemanticTag = "simd"
				line.HoverDoc = "SIMD: 4-way 8-bit dot product with 32-bit accumulator (__dp4a)"
			} else if strings.Contains(trimmed, "__shfl_down_sync") {
				line.SemanticTag = "barrier"
				line.HoverDoc = "Warp Shuffle: Downward intra-warp exchange (__shfl_down_sync)"
			} else if strings.Contains(trimmed, "__syncthreads") {
				line.SemanticTag = "barrier"
				line.HoverDoc = "Block Barrier: Synchronizes all threads in CTA (__syncthreads)"
			}

		case "mlir", "tileir":
			if strings.Contains(trimmed, "cuda_tile.entry @") || strings.Contains(trimmed, "cuda_tile.func @") {
				idx := strings.Index(trimmed, "@")
				if idx >= 0 {
					rest := trimmed[idx+1:]
					if end := strings.IndexAny(rest, "( \t{"); end >= 0 {
						currentKernel = rest[:end]
					} else {
						currentKernel = rest
					}
				}
				line.SemanticTag = "entry"
			}

			// Check for source loc references: e.g. loc("kernels.cu":45:10) or "kernels.cu":45
			if locIdx := strings.Index(trimmed, `loc("`); locIdx >= 0 {
				locStr := trimmed[locIdx+len(`loc("`):]
				if colon := strings.Index(locStr, `":`); colon >= 0 {
					numStr := locStr[colon+2:]
					if end := strings.IndexAny(numStr, ":)\""); end > 0 {
						numStr = numStr[:end]
					}
					if lNum, err := strconv.Atoi(numStr); err == nil {
						line.SourceLine = lNum
					}
				}
			} else if idx := strings.Index(trimmed, "kernels.cu"); idx >= 0 {
				sub := strings.TrimPrefix(trimmed[idx+len("kernels.cu"):], `"`)
				sub = strings.TrimPrefix(sub, ":")
				if end := strings.IndexAny(sub, ":)\" \t"); end > 0 {
					sub = sub[:end]
				}
				if lNum, err := strconv.Atoi(sub); err == nil {
					line.SourceLine = lNum
				}
			}

			if op := extractTileIROpcode(trimmed); op != "" {
				line.Opcode = op
				if doc := tileir.LookupDoc(op); doc != nil {
					line.Category = doc.Category.String()
					if doc.HardwareLowering != "" {
						line.HoverDoc = fmt.Sprintf("[%s] %s | Lowering: %s", doc.Category, doc.Summary, doc.HardwareLowering)
					} else {
						line.HoverDoc = fmt.Sprintf("[%s] %s", doc.Category, doc.Summary)
					}

					switch doc.Category {
					case tileir.CatTileMemory:
						line.SemanticTag = "tma"
					case tileir.CatComputeMMA:
						line.SemanticTag = "mma"
					case tileir.CatSyncToken:
						line.SemanticTag = "barrier"
					case tileir.CatComputeElement:
						line.SemanticTag = "simd"
					case tileir.CatControlFlow:
						if strings.Contains(op, "entry") {
							line.SemanticTag = "entry"
						} else if strings.Contains(op, "return") {
							line.SemanticTag = "return"
						}
					}
				}
			}

			// Correlate compiler remarks with Tile IR operations
			for _, r := range remarks {
				if r.Function != "" && currentKernel != "" {
					if !strings.EqualFold(r.Function, currentKernel) &&
						!strings.Contains(currentKernel, r.Function) &&
						!strings.Contains(r.Function, currentKernel) {
						continue
					}
				}

				match := false
				if r.Line > 0 && r.Line == line.Num {
					match = true
				} else if line.Opcode != "" {
					opLower := strings.ToLower(line.Opcode)
					if r.IsTMAFailure() || strings.Contains(r.Name, "TMA") {
						if strings.Contains(opLower, "load_view_tko") || strings.Contains(opLower, "store_view_tko") {
							match = true
						}
					} else if r.IsTensorCoreMMA() || strings.Contains(r.Name, "TensorCore") || strings.Contains(r.Name, "MMA") {
						if strings.Contains(opLower, "mma") {
							match = true
						}
					} else if strings.Contains(r.Name, "IDP4A") || strings.Contains(r.Name, "SIMD") {
						if strings.Contains(opLower, "mma") || strings.Contains(opLower, "mul") {
							match = true
						}
					} else if strings.Contains(r.Name, "SharedMemory") || strings.Contains(r.Name, "Staging") {
						if strings.Contains(opLower, "load_view_tko") {
							match = true
						}
					} else if strings.Contains(r.Name, "MoE") {
						if strings.Contains(opLower, "load_view_tko") || strings.Contains(opLower, "offset") {
							match = true
						}
					}
				}

				if match {
					line.RemarkBadge = r.Name
					desc := r.Remark
					if desc == "" && len(r.Args) > 0 {
						if s, ok := r.Args["String"]; ok {
							desc = s
						}
					}
					if desc != "" {
						line.RemarkTooltip = fmt.Sprintf("✓ %s: %s", r.Name, desc)
					} else {
						line.RemarkTooltip = fmt.Sprintf("✓ %s (Pass: %s)", r.Name, r.Pass)
					}
					break
				}
			}

		case "ptx":
			// Check for .loc directive: .loc <file_id> <line_num> <col_num>
			if strings.HasPrefix(trimmed, ".loc") {
				fields := strings.Fields(trimmed)
				if len(fields) >= 3 {
					if lNum, err := strconv.Atoi(fields[2]); err == nil {
						currentSourceLine = lNum
					}
				}
				line.Category = "Source Loc"
				line.SourceLine = currentSourceLine
			} else if currentSourceLine > 0 && !strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, ".") {
				line.SourceLine = currentSourceLine
			}

			fields := strings.Fields(trimmed)
			if len(fields) > 0 {
				rawOp := strings.TrimSuffix(fields[0], ";")
				line.Opcode = rawOp
				if doc := cudaptx.LookupDoc(rawOp); doc != nil {
					line.HoverDoc = doc.Summary
				}
				if strings.HasPrefix(rawOp, "cp.async.bulk") {
					line.SemanticTag = "tma"
				} else if strings.HasPrefix(rawOp, "mma.") || strings.HasPrefix(rawOp, "wgmma.") || strings.HasPrefix(rawOp, "tcgen05.") {
					line.SemanticTag = "mma"
				} else if strings.HasPrefix(rawOp, "mbarrier.") || strings.HasPrefix(rawOp, "bar.") {
					line.SemanticTag = "barrier"
				} else if strings.HasPrefix(rawOp, "vadd4") || strings.HasPrefix(rawOp, "dp4a") || strings.HasPrefix(rawOp, "hfma2") {
					line.SemanticTag = "simd"
				} else if strings.HasPrefix(rawOp, ".entry") {
					line.SemanticTag = "entry"
				} else if rawOp == "ret" {
					line.SemanticTag = "return"
				} else if strings.HasPrefix(rawOp, "ld.global") || strings.HasPrefix(rawOp, "st.global") || strings.HasPrefix(rawOp, "ld.shared") || strings.HasPrefix(rawOp, "st.shared") {
					line.SemanticTag = "memory"
				}
			}

		case "sass":
			if strings.HasPrefix(trimmed, "Function :") || strings.HasPrefix(trimmed, "Function ") {
				fields := strings.Fields(trimmed)
				if len(fields) >= 2 {
					if fields[1] == ":" && len(fields) >= 3 {
						currentKernel = fields[2]
					} else if fields[1] != ":" {
						currentKernel = fields[1]
					}
				}
				line.SemanticTag = "entry"
			}

			// Parse SASS DWARF line info: ## File "kernels.cu", line 47
			if strings.HasPrefix(trimmed, "## File") {
				if idx := strings.Index(trimmed, "line "); idx >= 0 {
					lStr := strings.TrimSpace(trimmed[idx+len("line "):])
					if end := strings.IndexAny(lStr, ", \t"); end >= 0 {
						lStr = lStr[:end]
					}
					if lNum, err := strconv.Atoi(lStr); err == nil {
						currentSourceLine = lNum
					}
				}
			}

			// Extract hex offset if present (/*0010*/ or /* 0010 */)
			if strings.HasPrefix(trimmed, "/*") {
				if currentSourceLine > 0 {
					line.SourceLine = currentSourceLine
				}
				if endIdx := strings.Index(trimmed, "*/"); endIdx != -1 {
					offsetStr := strings.TrimSpace(trimmed[2:endIdx])
					line.Offset = offsetStr

					if pc, err := strconv.ParseUint(strings.TrimPrefix(offsetStr, "0x"), 16, 64); err == nil {
						// Stall percentage
						if stallMap != nil && stallMap.hasStalls {
							if info, kernelTotal := stallMap.lookup(currentKernel, pc); info != nil && info.totalStalls > 0 && kernelTotal > 0 {
								pct := float64(info.totalStalls) / float64(kernelTotal) * 100.0
								line.StallPct = pct
								line.StallBreakdown = make(map[string]float64, len(info.breakdown))
								for reason, count := range info.breakdown {
									line.StallBreakdown[reason] = float64(count) / float64(kernelTotal) * 100.0
								}
								line.StallTooltip = formatStallTooltip(pct, line.StallBreakdown, info.totalStalls)
							}
						}

						// Register pressure
						if pressureMap != nil {
							var pres *cudasass.RegisterPressure
							if pKernel, ok := pressureMap[currentKernel]; ok {
								if p, ok := pKernel[pc]; ok {
									pres = &p
								}
							}
							if pres == nil && len(pressureMap) == 1 {
								for _, pKernel := range pressureMap {
									if p, ok := pKernel[pc]; ok {
										pres = &p
										break
									}
								}
							}
							if pres != nil {
								line.Pressure = fmt.Sprintf("R:%d", pres.GPR)
								if pres.UGPR > 0 {
									line.Pressure += fmt.Sprintf(" UR:%d", pres.UGPR)
								}
								line.PressureTooltip = fmt.Sprintf("Live Registers: %d GPR, %d PRED, %d UGPR, %d UPRED",
									pres.GPR, pres.Pred, pres.UGPR, pres.UPred)
							}
						}
					}
				}

				// Semantic tagging for SASS instructions
				fields := strings.Fields(trimmed)
				for _, f := range fields {
					if strings.HasPrefix(f, "/*") || strings.HasPrefix(f, "@") || strings.HasPrefix(f, "[") || strings.HasPrefix(f, "0x") {
						continue
					}
					op := strings.TrimSuffix(f, ";")
					opBase := strings.ToUpper(strings.Split(op, ".")[0])
					line.Opcode = op
					switch opBase {
					case "UTMA", "TMA":
						line.SemanticTag = "tma"
					case "HMMA", "HGMMA", "MMA":
						line.SemanticTag = "mma"
					case "BAR", "DEPBAR":
						line.SemanticTag = "barrier"
					case "IDP4A", "HFMA2":
						line.SemanticTag = "simd"
					case "EXIT", "RET":
						line.SemanticTag = "return"
					case "LDG", "STG", "LDS", "STS", "LDC":
						line.SemanticTag = "memory"
					}
					break
				}
			}
		}

		lines[i] = line
	}

	tier := cudaview.PipelineTier{
		Name:     name,
		FileName: fileName,
		Language: lang,
		Lines:    lines,
	}
	cudaview.Annotate(&tier)
	return tier
}

func discoverKernelNames(ptxText, sassText, cuText, tileText string) []string {
	seen := make(map[string]bool)
	var names []string

	add := func(name string) {
		name = strings.TrimSpace(name)
		if name != "" && !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}

	// SASS function headers: Function : <name>
	reSass := regexp.MustCompile(`(?:Function\s*:\s*|Function\s+)([a-zA-Z0-9_]+)`)
	for _, m := range reSass.FindAllStringSubmatch(sassText, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	// PTX entry: .visible .entry <name> or .entry <name>
	rePtx := regexp.MustCompile(`\.entry\s+([a-zA-Z0-9_]+)`)
	for _, m := range rePtx.FindAllStringSubmatch(ptxText, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	// CUDA C++: __global__ void <name>
	reCu := regexp.MustCompile(`__global__\s+void\s+([a-zA-Z0-9_]+)`)
	for _, m := range reCu.FindAllStringSubmatch(cuText, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	// Tile IR: cuda_tile.entry @<name> or cuda_tile.func @<name>
	reTile := regexp.MustCompile(`cuda_tile\.(?:entry|func)\s+@([a-zA-Z0-9_]+)`)
	for _, m := range reTile.FindAllStringSubmatch(tileText, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	return names
}

func extractKernelSection(content, kernelName, lang string) string {
	if content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	var result []string
	recording := false

	switch lang {
	case "sass":
		for _, line := range lines {
			if strings.Contains(line, "Function : "+kernelName) || strings.Contains(line, "Function "+kernelName) {
				recording = true
			} else if recording && strings.Contains(line, "Function :") {
				break
			}
			if recording {
				result = append(result, line)
			}
		}
	case "ptx":
		for _, line := range lines {
			if strings.Contains(line, ".entry "+kernelName) {
				recording = true
			} else if recording && strings.HasPrefix(strings.TrimSpace(line), ".entry ") {
				break
			}
			if recording {
				result = append(result, line)
			}
		}
	case "tileir":
		for _, line := range lines {
			if strings.Contains(line, "@"+kernelName) {
				recording = true
			} else if recording && strings.Contains(line, "cuda_tile.entry @") {
				break
			}
			if recording {
				result = append(result, line)
			}
		}
	case "cuda":
		braceCount := 0
		for _, line := range lines {
			if strings.Contains(line, "void "+kernelName) {
				recording = true
			}
			if recording {
				result = append(result, line)
				braceCount += strings.Count(line, "{") - strings.Count(line, "}")
				if braceCount == 0 && strings.Contains(line, "}") {
					break
				}
			}
		}
	case "go":
		for _, line := range lines {
			if strings.Contains(strings.ToLower(line), strings.ToLower(kernelName)) {
				result = append(result, line)
			}
		}
	}

	if len(result) == 0 {
		// If kernel-specific snippet wasn't cleanly isolated, return full text
		return content
	}
	return strings.Join(result, "\n")
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}
