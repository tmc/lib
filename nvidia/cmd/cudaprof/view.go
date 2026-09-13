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
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/tileir"
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
	Num      int     `json:"num"`
	Text     string  `json:"text"`
	Opcode   string  `json:"opcode,omitempty"`
	Offset   string  `json:"offset,omitempty"`
	StallPct float64 `json:"stall_pct,omitempty"`
	HoverDoc string  `json:"hover_doc,omitempty"`
}

// PipelineViewData is the data model passed to the web template and API endpoints.
type PipelineViewData struct {
	Title        string                  `json:"title"`
	Device       string                  `json:"device"`
	Architecture string                  `json:"architecture"`
	CaptureFile  string                  `json:"capture_file"`
	Kernels      []string                `json:"kernels"`
	ActiveKernel string                  `json:"active_kernel"`
	Tiers        []PipelineTier          `json:"tiers"`
	KernelTiers  map[string][]PipelineTier `json:"kernel_tiers"`
	Remarks      string                  `json:"remarks,omitempty"`
	CFGDot       string                  `json:"cfg_dot,omitempty"`
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

	url := fmt.Sprintf("http://%s", listener.Addr().String())
	if strings.HasPrefix(httpAddr, ":") {
		url = fmt.Sprintf("http://127.0.0.1%s", httpAddr)
	}
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
	if rmkBytes, ok := cap.Files["tileiras-remarks.yaml"]; ok {
		data.Remarks = string(rmkBytes)
	}
	if dotBytes, ok := cap.Files[cap.Code.CFG]; ok && len(dotBytes) > 0 {
		data.CFGDot = string(dotBytes)
	} else if dotBytes, ok := cap.Files["kernels.cfg.dot"]; ok {
		data.CFGDot = string(dotBytes)
	}

	// Build full tiers
	data.Tiers = []PipelineTier{
		buildTier("Go", goFileName, "go", goContent),
		buildTier("CUDA C++", cuFileName, "cuda", cuContent),
		buildTier("Tile IR", tileFileName, "mlir", tileContent),
		buildTier("PTX", ptxFileName, "ptx", ptxContent),
		buildTier("SASS", sassFileName, "sass", sassContent),
	}

	// 3. Discover kernels across PTX and SASS
	kernelNames := discoverKernelNames(ptxContent, sassContent, cuContent)
	data.Kernels = kernelNames

	// 4. Build per-kernel slice views
	for _, kName := range kernelNames {
		kGo := extractKernelSection(goContent, kName, "go")
		kCu := extractKernelSection(cuContent, kName, "cuda")
		kTile := extractKernelSection(tileContent, kName, "tileir")
		kPtx := extractKernelSection(ptxContent, kName, "ptx")
		kSass := extractKernelSection(sassContent, kName, "sass")

		data.KernelTiers[kName] = []PipelineTier{
			buildTier("Go", goFileName, "go", kGo),
			buildTier("CUDA C++", cuFileName, "cuda", kCu),
			buildTier("Tile IR", tileFileName, "mlir", kTile),
			buildTier("PTX", ptxFileName, "ptx", kPtx),
			buildTier("SASS", sassFileName, "sass", kSass),
		}
	}

	return data, nil
}

func buildTier(name, fileName, lang, content string) PipelineTier {
	rawLines := strings.Split(content, "\n")
	lines := make([]PipelineLine, len(rawLines))

	for i, text := range rawLines {
		line := PipelineLine{
			Num:  i + 1,
			Text: text,
		}

		trimmed := strings.TrimSpace(text)
		if lang == "sass" {
			// Extract hex offset if present (/*0010*/)
			if strings.HasPrefix(trimmed, "/*") {
				if endIdx := strings.Index(trimmed, "*/"); endIdx != -1 {
					line.Offset = trimmed[2:endIdx]
				}
			}
		} else if lang == "ptx" {
			// Check for opcode documentation
			fields := strings.Fields(trimmed)
			if len(fields) > 0 {
				op := strings.TrimSuffix(fields[0], ";")
				if doc := cudaptx.LookupDoc(op); doc != nil {
					line.Opcode = op
					line.HoverDoc = doc.Summary
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

func discoverKernelNames(ptxText, sassText, cuText string) []string {
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
	tmpl, err := template.New("pipeline").Parse(pipelineHTMLTemplate)
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
.badge-offset {
  color: var(--text-muted);
  font-size: 10px;
  margin-right: 4px;
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
        <div class="code-row" id="line-{{$idx}}-{{.Num}}">
          <div class="line-num">{{.Num}}</div>
          <div class="line-content">{{if .Offset}}<span class="badge-offset">/*{{.Offset}}*/</span>{{end}}{{if .HoverDoc}}<span class="opcode-hover" title="{{.HoverDoc}}">{{.Text}}</span>{{else}}{{.Text}}{{end}}</div>
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
          const offsetSpan = l.offset ? '<span class="badge-offset">/*' + l.offset + '*/</span>' : '';
          const textSpan = l.hover_doc ? '<span class="opcode-hover" title="' + l.hover_doc + '">' + escapeHtml(l.text) + '</span>' : escapeHtml(l.text);
          html += '<div class="code-row"><div class="line-num">' + l.num + '</div><div class="line-content">' + offsetSpan + textSpan + '</div></div>';
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
