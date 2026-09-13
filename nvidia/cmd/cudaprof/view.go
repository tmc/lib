package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
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

// PipelineTier represents one stage in the Go -> CUDA C++ -> Tile IR -> PTX -> SASS compilation hierarchy.
type PipelineTier struct {
	Name     string         `json:"name"`
	FileName string         `json:"file_name"`
	Language string         `json:"language"`
	Lines    []PipelineLine `json:"lines"`
}

// PipelineLine represents an annotated line in a pipeline tier.
type PipelineLine struct {
	Num             int                `json:"num"`
	Text            string             `json:"text"`
	Opcode          string             `json:"opcode,omitempty"`
	Category        string             `json:"category,omitempty"`
	Offset          string             `json:"offset,omitempty"`
	StallPct        float64            `json:"stall_pct,omitempty"`
	StallBreakdown  map[string]float64 `json:"stall_breakdown,omitempty"`
	StallTooltip    string             `json:"stall_tooltip,omitempty"`
	HoverDoc        string             `json:"hover_doc,omitempty"`
	RemarkBadge     string             `json:"remark_badge,omitempty"`
	RemarkTooltip   string             `json:"remark_tooltip,omitempty"`
	LaunchKernel    string             `json:"launch_kernel,omitempty"`
	SemanticTag     string             `json:"semantic_tag,omitempty"`
	SourceLine      int                `json:"source_line,omitempty"`
	Pressure        string             `json:"pressure,omitempty"`
	PressureTooltip string             `json:"pressure_tooltip,omitempty"`
}

// PipelineViewData is the data model passed to the web template and API endpoints.
type PipelineViewData struct {
	Title         string                    `json:"title"`
	Device        string                    `json:"device"`
	Architecture  string                    `json:"architecture"`
	CaptureFile   string                    `json:"capture_file"`
	Kernels       []string                  `json:"kernels"`
	ActiveKernel  string                    `json:"active_kernel"`
	Tiers         []PipelineTier            `json:"tiers"`
	KernelTiers   map[string][]PipelineTier `json:"kernel_tiers"`
	Remarks       string                    `json:"remarks,omitempty"`
	RemarksCount  int                       `json:"remarks_count,omitempty"`
	TileIRAudit   string                    `json:"tileir_audit,omitempty"`
	TileIROpcodes map[string]int            `json:"tileir_opcodes,omitempty"`
	CFGDot        string                    `json:"cfg_dot,omitempty"`
}

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
			return errors.New("usage: cudaprof view [-http=addr] [-html=file] [kernel] capture.pprof")
		}
	case 1:
		capturePath = posArgs[0]
	case 2:
		kernel = posArgs[0]
		capturePath = posArgs[1]
	default:
		return errors.New("usage: cudaprof view [-http=addr] [-html=file] [kernel] capture.pprof")
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

	// Static HTML export mode
	if htmlOut != "" {
		var buf bytes.Buffer
		if err := renderPipelineHTML(&buf, data); err != nil {
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

func buildPipelineViewData(cap *cudacapture.Capture, capName, activeKernel string) (*PipelineViewData, error) {
	data := &PipelineViewData{
		Title:        "5-Tier GPU Compilation Pipeline: Go -> CUDA C++ -> Tile IR -> PTX -> SASS",
		CaptureFile:  capName,
		KernelTiers:  make(map[string][]PipelineTier),
		ActiveKernel: activeKernel,
	}

	// 1. Extract metadata (device, arch)
	if len(cap.Metadata) > 0 {
		var meta struct {
			Device string `json:"device"`
			Arch   string `json:"architecture"`
		}
		if json.Unmarshal(cap.Metadata, &meta) == nil {
			data.Device = meta.Device
			data.Architecture = meta.Arch
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
	stallMap := extractSASSStallMap(cap, kernelNames)

	// 5. Extract live register pressure if available
	var pressureMap map[string]map[uint64]cudasass.RegisterPressure
	if disasmWithPressure, err := loadDisassemblyWithPressure(cap); err == nil && disasmWithPressure != nil {
		pressureMap = extractPressureMap(disasmWithPressure)
	}

	// Build full tiers
	data.Tiers = []PipelineTier{
		buildTier("Go", goFileName, "go", goContent, nil, "", nil, kernelNames, nil),
		buildTier("CUDA C++", cuFileName, "cuda", cuContent, nil, "", nil, kernelNames, nil),
		buildTier("Tile IR", tileFileName, "mlir", tileContent, nil, "", remarksList, kernelNames, nil),
		buildTier("PTX", ptxFileName, "ptx", ptxContent, nil, "", nil, kernelNames, nil),
		buildTier("SASS", sassFileName, "sass", sassContent, stallMap, activeKernel, nil, kernelNames, pressureMap),
	}

	// 6. Build per-kernel slice views
	for _, kName := range kernelNames {
		kGo := extractKernelSection(goContent, kName, "go")
		kCu := extractKernelSection(cuContent, kName, "cuda")
		kTile := extractKernelSection(tileContent, kName, "tileir")
		kPtx := extractKernelSection(ptxContent, kName, "ptx")
		kSass := extractKernelSection(sassContent, kName, "sass")

		data.KernelTiers[kName] = []PipelineTier{
			buildTier("Go", goFileName, "go", kGo, nil, kName, nil, kernelNames, nil),
			buildTier("CUDA C++", cuFileName, "cuda", kCu, nil, kName, nil, kernelNames, nil),
			buildTier("Tile IR", tileFileName, "mlir", kTile, nil, kName, remarksList, kernelNames, nil),
			buildTier("PTX", ptxFileName, "ptx", kPtx, nil, kName, nil, kernelNames, nil),
			buildTier("SASS", sassFileName, "sass", kSass, stallMap, kName, nil, kernelNames, pressureMap),
		}
	}

	if activeKernel != "" {
		if kTiers, ok := data.KernelTiers[activeKernel]; ok {
			data.Tiers = kTiers
		}
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

func extractSASSStallMap(cap *cudacapture.Capture, knownKernels []string) *sassStallMap {
	if cap == nil {
		return nil
	}
	prof := cap.Profile
	if prof == nil && cap.Files != nil && len(cap.Files["gpu.pprof"]) > 0 {
		if p, err := profile.ParseData(cap.Files["gpu.pprof"]); err == nil {
			prof = p
		}
	}
	if prof == nil || len(prof.Sample) == 0 {
		return nil
	}

	var stallIndices []int
	for i, st := range prof.SampleType {
		if strings.HasPrefix(st.Type, "stall_") {
			stallIndices = append(stallIndices, i)
		}
	}

	code, _ := cudacapture.LoadCode(cap)
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

	return m
}

func (m *sassStallMap) lookup(kernel string, pc uint64) (*pcStallInfo, int64) {
	if m == nil || !m.hasStalls {
		return nil, 0
	}
	if kernel != "" {
		if pcMap, ok := m.byKernelPC[kernel]; ok {
			return pcMap[pc], m.kernelTotals[kernel]
		}
		for k, pcMap := range m.byKernelPC {
			if strings.EqualFold(k, kernel) || strings.Contains(k, kernel) || strings.Contains(kernel, k) {
				return pcMap[pc], m.kernelTotals[k]
			}
		}
	}
	if len(m.byKernelPC) == 1 {
		for k, pcMap := range m.byKernelPC {
			return pcMap[pc], m.kernelTotals[k]
		}
	}
	if m.totalStalls > 0 {
		for _, pcMap := range m.byKernelPC {
			if info, ok := pcMap[pc]; ok {
				return info, m.totalStalls
			}
		}
	}
	return nil, 0
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

func buildTier(name, fileName, lang, content string, stallMap *sassStallMap, activeKernel string, remarks []tileiras.Remark, knownKernels []string, pressureMap map[string]map[uint64]cudasass.RegisterPressure) PipelineTier {
	rawLines := strings.Split(content, "\n")
	lines := make([]PipelineLine, len(rawLines))

	currentKernel := activeKernel
	currentSourceLine := 0

	for i, text := range rawLines {
		line := PipelineLine{
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

	return PipelineTier{
		Name:     name,
		FileName: fileName,
		Language: lang,
		Lines:    lines,
	}
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

func newPipelineServerHandler(data *PipelineViewData) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := renderPipelineHTML(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	mux.HandleFunc("/api/pipeline", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(data)
	})

	mux.HandleFunc("/api/kernel", func(w http.ResponseWriter, r *http.Request) {
		kName := r.URL.Query().Get("name")
		w.Header().Set("Content-Type", "application/json")
		if tiers, ok := data.KernelTiers[kName]; ok {
			json.NewEncoder(w).Encode(tiers)
		} else {
			json.NewEncoder(w).Encode(data.Tiers)
		}
	})

	return mux
}

func renderPipelineHTML(w io.Writer, data *PipelineViewData) error {
	tmpl, err := template.New("pipeline").Funcs(template.FuncMap{
		"stallWidth": func(pct float64) string {
			if pct <= 0 {
				return "0"
			}
			w := pct
			if w < 4 {
				w = 4
			}
			if w > 100 {
				w = 100
			}
			return fmt.Sprintf("%.1f", w)
		},
	}).Parse(pipelineHTMLTemplate)
	if err != nil {
		return fmt.Errorf("parse template: %w", err)
	}
	return tmpl.Execute(w, data)
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

const pipelineHTMLTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{.Title}}</title>
<style>
:root {
  --bg-primary: #0f141c;
  --bg-secondary: #18202c;
  --bg-tier: #131b26;
  --border-color: #2b384e;
  --text-primary: #e6edf3;
  --text-muted: #8b949e;
  --accent-go: #00add8;
  --accent-cuda: #76b900;
  --accent-tile: #f39c12;
  --accent-ptx: #9b59b6;
  --accent-sass: #e74c3c;
  --line-num-color: #48566a;
  --hover-bg: #1f2d3f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.4;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
header {
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.header-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-title h1 {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 12px;
  background-color: rgba(118, 185, 0, 0.15);
  color: #76b900;
  border: 1px solid rgba(118, 185, 0, 0.3);
  font-weight: 500;
}
.controls {
  display: flex;
  align-items: center;
  gap: 10px;
}
select, input[type="text"] {
  background-color: var(--bg-tier);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 5px 10px;
  border-radius: 6px;
  font-size: 12px;
  outline: none;
}
select:focus, input[type="text"]:focus {
  border-color: #76b900;
}
.btn {
  background-color: var(--bg-tier);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 5px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn:hover {
  background-color: var(--hover-bg);
}
.btn.active {
  background-color: #76b900;
  color: #000;
  font-weight: 600;
  border-color: #76b900;
}
.pipeline-container {
  display: flex;
  flex: 1;
  overflow: hidden;
  position: relative;
}
.tier-column {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 240px;
  border-right: 1px solid var(--border-color);
  background-color: var(--bg-tier);
  overflow: hidden;
}
.tier-column:last-child {
  border-right: none;
}
.tier-header {
  padding: 8px 12px;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
}
.tier-name {
  display: flex;
  align-items: center;
  gap: 6px;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.dot-go { background-color: var(--accent-go); }
.dot-cuda { background-color: var(--accent-cuda); }
.dot-tile { background-color: var(--accent-tile); }
.dot-ptx { background-color: var(--accent-ptx); }
.dot-sass { background-color: var(--accent-sass); }
.tier-filename {
  font-size: 10px;
  color: var(--text-muted);
  font-family: monospace;
}
.code-container {
  flex: 1;
  overflow-y: auto;
  overflow-x: auto;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 11px;
  line-height: 18px;
}
.code-table {
  width: 100%;
  border-collapse: collapse;
}
.code-row {
  display: flex;
}
.code-row:hover {
  background-color: var(--hover-bg);
}
.line-num {
  width: 44px;
  padding: 0 8px;
  text-align: right;
  color: var(--line-num-color);
  user-select: none;
  font-size: 10px;
  flex-shrink: 0;
}
.line-content {
  padding: 0 8px;
  white-space: pre;
  color: var(--text-primary);
  flex-grow: 1;
}
.opcode-hover {
  cursor: help;
  text-decoration: underline dotted var(--accent-ptx);
}
.opcode-hover-tile {
  cursor: help;
  text-decoration: underline dotted var(--accent-tile);
  color: #f6b93b;
  font-weight: 500;
}
.badge-category {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(155, 89, 182, 0.18);
  color: #c084fc;
  border: 1px solid rgba(155, 89, 182, 0.35);
  margin-right: 6px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  display: inline-block;
  vertical-align: middle;
}
.badge-remark {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(243, 156, 18, 0.15);
  color: #f39c12;
  border: 1px solid rgba(243, 156, 18, 0.35);
  margin-left: 8px;
  cursor: help;
  font-weight: 500;
  display: inline-block;
  vertical-align: middle;
}
.badge-audit {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 12px;
  background: rgba(243, 156, 18, 0.15);
  color: #f39c12;
  border: 1px solid rgba(243, 156, 18, 0.35);
  font-weight: 500;
}
.highlight-match {
  background-color: rgba(243, 156, 18, 0.22) !important;
  border-left: 2px solid var(--accent-tile);
}
.highlight-srcline {
  background-color: rgba(56, 189, 248, 0.18) !important;
  border-left: 2px solid #38bdf8 !important;
}
.highlight-semtag {
  background-color: rgba(245, 158, 11, 0.16) !important;
  box-shadow: inset 0 0 0 1px rgba(245, 158, 11, 0.4) !important;
}
.badge-reg {
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  color: #c7d2fe;
  background: #312e81;
  border: 1px solid rgba(165, 180, 252, 0.3);
  padding: 1px 4px;
  border-radius: 3px;
  margin-right: 5px;
  cursor: help;
  vertical-align: middle;
}
.badge-launch {
  display: inline-block;
  font-size: 10px;
  font-weight: 600;
  color: #6ee7b7;
  background: rgba(6, 78, 59, 0.6);
  border: 1px solid #059669;
  padding: 1px 6px;
  border-radius: 4px;
  cursor: pointer;
  margin-right: 6px;
  vertical-align: middle;
  transition: all 0.15s ease;
}
.badge-launch:hover {
  background: #047857;
  color: #ffffff;
  border-color: #10b981;
}
.badge-offset {
  color: var(--text-muted);
  font-size: 10px;
  margin-right: 4px;
}
.stall-heat {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-right: 6px;
  cursor: help;
  vertical-align: middle;
}
.stall-bar-bg {
  width: 44px;
  height: 7px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
  overflow: hidden;
  display: inline-block;
  vertical-align: middle;
}
.stall-bar {
  height: 100%;
  border-radius: 2px;
  display: block;
}
.stall-bar.stall-high {
  background: #ff4757;
}
.stall-bar.stall-med {
  background: #ffa502;
}
.stall-bar.stall-low {
  background: #2ed573;
}
.stall-text {
  font-size: 10px;
  font-family: monospace;
  color: var(--text-muted);
  min-width: 36px;
  text-align: right;
}
.stall-text.stall-high {
  color: #ff6b81;
  font-weight: 600;
}
.stall-text.stall-med {
  color: #ffa502;
}
.stall-text.stall-low {
  color: #2ed573;
}
.tab-bar {
  display: none;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: 0 8px;
}
@media (max-width: 1000px) {
  .pipeline-container { flex-direction: column; }
  .tab-bar { display: flex; gap: 4px; }
  .tier-column { display: none; }
  .tier-column.tab-active { display: flex; }
}
</style>
</head>
<body>

<header>
  <div class="header-title">
    <h1>{{.Title}}</h1>
    {{if .Device}}<span class="badge">{{.Device}} ({{.Architecture}})</span>{{end}}
    <span class="badge" style="border-color: #2b384e; color: #8b949e;">{{.CaptureFile}}</span>
    {{if .TileIRAudit}}<span class="badge-audit" title="Static Verification & Linearity Audit">{{.TileIRAudit}}</span>{{end}}
    {{if gt .RemarksCount 0}}<span class="badge" style="border-color: rgba(46, 213, 115, 0.4); color: #2ed573;">✓ {{.RemarksCount}} Remarks</span>{{end}}
  </div>

  <div class="controls">
    <label for="kernelSelect" style="font-size: 12px; color: var(--text-muted);">Kernel:</label>
    <select id="kernelSelect" onchange="onKernelChange(this.value)">
      <option value="">All Kernels (Full Bundle)</option>
      {{range .Kernels}}
      <option value="{{.}}" {{if eq . $.ActiveKernel}}selected{{end}}>{{.}}</option>
      {{end}}
    </select>

    <button class="btn" id="syncScrollBtn" onclick="toggleSyncScroll()">Sync Scroll: ON</button>
  </div>
</header>

<div class="tab-bar" id="tabBar">
  {{range $idx, $tier := .Tiers}}
  <button class="btn {{if eq $idx 0}}active{{end}}" onclick="showTab({{$idx}})">{{$tier.Name}}</button>
  {{end}}
</div>

<main class="pipeline-container" id="pipelineGrid">
  {{range $idx, $tier := .Tiers}}
  <div class="tier-column {{if eq $idx 0}}tab-active{{end}}" id="tier-col-{{$idx}}">
    <div class="tier-header">
      <div class="tier-name">
        <span class="dot {{if eq $tier.Language "go"}}dot-go{{else if eq $tier.Language "cuda"}}dot-cuda{{else if eq $tier.Language "mlir"}}dot-tile{{else if eq $tier.Language "ptx"}}dot-ptx{{else}}dot-sass{{end}}"></span>
        {{$tier.Name}}
      </div>
      <div class="tier-filename">{{$tier.FileName}}</div>
    </div>
    <div class="code-container" id="code-scroll-{{$idx}}">
      <div class="code-table">
        {{range $tier.Lines}}
        <div class="code-row" id="line-{{$idx}}-{{.Num}}" data-tier="{{$idx}}" data-opcode="{{.Opcode}}" data-sem-tag="{{.SemanticTag}}" data-source-line="{{.SourceLine}}">
          <div class="line-num">{{.Num}}</div>
          <div class="line-content">{{if .LaunchKernel}}<button class="badge-launch" onclick="onKernelChange('{{.LaunchKernel}}')" title="Switch view to {{.LaunchKernel}}">▶ Launch: {{.LaunchKernel}}</button>{{end}}{{if .Offset}}<span class="badge-offset">/*{{.Offset}}*/</span>{{end}}{{if .Pressure}}<span class="badge-reg" title="{{.PressureTooltip}}">{{.Pressure}}</span>{{end}}{{if gt .StallPct 0.0}}<span class="stall-heat" title="{{.StallTooltip}}"><span class="stall-bar-bg"><span class="stall-bar {{if ge .StallPct 30.0}}stall-high{{else if ge .StallPct 10.0}}stall-med{{else}}stall-low{{end}}" style="width: {{stallWidth .StallPct}}%;"></span></span><span class="stall-text {{if ge .StallPct 30.0}}stall-high{{else if ge .StallPct 10.0}}stall-med{{else}}stall-low{{end}}">{{printf "%.1f" .StallPct}}%</span></span>{{end}}{{if .Category}}<span class="badge-category" title="{{$tier.Name}} Category: {{.Category}}">{{.Category}}</span>{{end}}{{if .HoverDoc}}<span class="{{if eq $tier.Language "mlir"}}opcode-hover-tile{{else}}opcode-hover{{end}}" title="{{.HoverDoc}}">{{.Text}}</span>{{else}}{{.Text}}{{end}}{{if .RemarkBadge}}<span class="badge-remark" title="{{.RemarkTooltip}}">✓ {{.RemarkBadge}}</span>{{end}}</div>
        </div>
        {{end}}
      </div>
    </div>
  </div>
  {{end}}
</main>

<script>
let syncScroll = true;
const scrollContainers = [0, 1, 2, 3, 4].map(i => document.getElementById('code-scroll-' + i));

function toggleSyncScroll() {
  syncScroll = !syncScroll;
  const btn = document.getElementById('syncScrollBtn');
  btn.textContent = 'Sync Scroll: ' + (syncScroll ? 'ON' : 'OFF');
  btn.classList.toggle('active', syncScroll);
}

scrollContainers.forEach((c, idx) => {
  if (!c) return;
  c.addEventListener('scroll', () => {
    if (!syncScroll) return;
    const ratio = c.scrollTop / (c.scrollHeight - c.clientHeight || 1);
    scrollContainers.forEach((other, otherIdx) => {
      if (other && otherIdx !== idx) {
        other.scrollTop = ratio * (other.scrollHeight - other.clientHeight);
      }
    });
  });
});

document.addEventListener('mouseover', e => {
  const row = e.target.closest('.code-row');
  if (!row) return;
  const opcode = row.getAttribute('data-opcode');
  if (opcode && opcode !== '') {
    document.querySelectorAll('.code-row[data-opcode="' + opcode + '"]').forEach(r => r.classList.add('highlight-match'));
  }
  const semTag = row.getAttribute('data-sem-tag');
  if (semTag && semTag !== '') {
    document.querySelectorAll('.code-row[data-sem-tag="' + semTag + '"]').forEach(r => r.classList.add('highlight-semtag'));
  }
  const srcLine = row.getAttribute('data-source-line');
  if (srcLine && srcLine !== '' && srcLine !== '0') {
    document.querySelectorAll('.code-row[data-source-line="' + srcLine + '"]').forEach(r => r.classList.add('highlight-srcline'));
  }
});
document.addEventListener('mouseout', e => {
  const row = e.target.closest('.code-row');
  if (!row) return;
  document.querySelectorAll('.highlight-match').forEach(r => r.classList.remove('highlight-match'));
  document.querySelectorAll('.highlight-semtag').forEach(r => r.classList.remove('highlight-semtag'));
  document.querySelectorAll('.highlight-srcline').forEach(r => r.classList.remove('highlight-srcline'));
});

function showTab(tabIndex) {
  document.querySelectorAll('.tab-bar .btn').forEach((b, idx) => {
    b.classList.toggle('active', idx === tabIndex);
  });
  document.querySelectorAll('.tier-column').forEach((col, idx) => {
    col.classList.toggle('tab-active', idx === tabIndex);
  });
}

function onKernelChange(kernelName) {
  if (!kernelName) {
    window.location.href = window.location.pathname;
    return;
  }
  const sel = document.getElementById('kernelSelect');
  if (sel && sel.value !== kernelName) {
    sel.value = kernelName;
  }
  fetch('/api/kernel?name=' + encodeURIComponent(kernelName))
    .then(r => r.json())
    .then(tiers => {
      tiers.forEach((tier, idx) => {
        const container = document.getElementById('code-scroll-' + idx);
        if (!container) return;
        const table = container.querySelector('.code-table');
        if (!table) return;
        let html = '';
        tier.lines.forEach(l => {
          const launchBtn = l.launch_kernel ? '<button class="badge-launch" onclick="onKernelChange(\'' + escapeHtml(l.launch_kernel) + '\')">▶ Launch: ' + escapeHtml(l.launch_kernel) + '</button>' : '';
          const offsetSpan = l.offset ? '<span class="badge-offset">/*' + l.offset + '*/</span>' : '';
          const presSpan = l.pressure ? '<span class="badge-reg" title="' + escapeHtml(l.pressure_tooltip || '') + '">' + escapeHtml(l.pressure) + '</span>' : '';
          let stallSpan = '';
          if (l.stall_pct && l.stall_pct > 0) {
            const intensity = l.stall_pct >= 30 ? 'stall-high' : (l.stall_pct >= 10 ? 'stall-med' : 'stall-low');
            const tooltip = l.stall_tooltip ? escapeHtml(l.stall_tooltip) : (l.stall_pct.toFixed(1) + '% stall');
            const width = Math.min(100, Math.max(4, l.stall_pct));
            stallSpan = '<span class="stall-heat" title="' + tooltip + '">' +
              '<span class="stall-bar-bg"><span class="stall-bar ' + intensity + '" style="width: ' + width + '%;"></span></span>' +
              '<span class="stall-text ' + intensity + '">' + l.stall_pct.toFixed(1) + '%</span>' +
              '</span>';
          }
          const catSpan = l.category ? '<span class="badge-category" title="' + escapeHtml(tier.name) + ' Category: ' + escapeHtml(l.category) + '">' + escapeHtml(l.category) + '</span>' : '';
          const hoverClass = tier.language === 'mlir' ? 'opcode-hover-tile' : 'opcode-hover';
          const textSpan = l.hover_doc ? '<span class="' + hoverClass + '" title="' + escapeHtml(l.hover_doc) + '">' + escapeHtml(l.text) + '</span>' : escapeHtml(l.text);
          const remarkSpan = l.remark_badge ? '<span class="badge-remark" title="' + escapeHtml(l.remark_tooltip) + '">✓ ' + escapeHtml(l.remark_badge) + '</span>' : '';
          html += '<div class="code-row" data-tier="' + idx + '" data-opcode="' + (l.opcode || '') + '" data-sem-tag="' + (l.semantic_tag || '') + '" data-source-line="' + (l.source_line || '') + '"><div class="line-num">' + l.num + '</div><div class="line-content">' + launchBtn + offsetSpan + presSpan + stallSpan + catSpan + textSpan + remarkSpan + '</div></div>';
        });
        table.innerHTML = html;
      });
    })
    .catch(err => console.error('Failed to load kernel:', err));
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
</script>
</body>
</html>
`
