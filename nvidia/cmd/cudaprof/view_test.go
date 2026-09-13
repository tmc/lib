package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func createTest5TierCapture(t *testing.T) (*cudacapture.Capture, string) {
	t.Helper()

	metaJSON := `{"device":"NVIDIA GB10","architecture":"sm_121a","verified_results":100}`
	goSrc := `package main
func LaunchKernels() {
    // 1. RMSNorm
    // 2. SwiGLU GEMV
    // 3. Q8_0 Matvec
}`
	cuSrc := `extern "C" __global__ void ds4_q8_0_matvec(const void* W, const float* x, float* y) {
    int lane = threadIdx.x % 32;
    // SIMD reduction
}
extern "C" __global__ void ds4_swiglu(const float* x, float* out) {
    // SwiGLU activation
}`
	tileSrc := `// Tile IR module
cuda_tile.entry @ds4_q8_0_matvec(%arg0: !cuda_tile.tile<ptr<i8>>) {
    %tok = cuda_tile.make_token : !cuda_tile.token
    cuda_tile.return
}
cuda_tile.entry @ds4_swiglu(%arg0: !cuda_tile.tile<ptr<f32>>) {
    cuda_tile.return
}`
	ptxSrc := `.version 8.8
.target sm_121a
.visible .entry ds4_q8_0_matvec(
    .param .u64 W,
    .param .u64 x
) {
    ld.param.u64 %rd1, [W];
    ret;
}
.visible .entry ds4_swiglu(
    .param .u64 x
) {
    ret;
}`
	sassSrc := `Function : ds4_q8_0_matvec
/*0000*/ LDG.E.128 R0, [R2];
/*0010*/ RET;

Function : ds4_swiglu
/*0000*/ NOP;
/*0010*/ RET;
`

	files := map[string][]byte{
		"workload.go":     []byte(goSrc),
		"kernels.cu":      []byte(cuSrc),
		"kernels.cubin":   []byte("ELF_MOCK"),
		"kernels.tileir":  []byte(tileSrc),
		"kernels.ptx":     []byte(ptxSrc),
		"kernels.sass":    []byte(sassSrc),
		"kernels.cfg.dot": []byte("digraph G { n0 -> n1; }"),
	}

	fn := &profile.Function{ID: 1, Name: "ds4_q8_0_matvec", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile:  p,
		Metadata: []byte(metaJSON),
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			CUBIN:  "kernels.cubin",
			TileIR: "kernels.tileir",
			PTX:    "kernels.ptx",
			SASS:   "kernels.sass",
			CFG:    "kernels.cfg.dot",
		},
		Files: files,
	}

	tmpDir := t.TempDir()
	capPath := filepath.Join(tmpDir, "test-capture.pprof")

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, cap); err != nil {
		t.Fatalf("cudacapture.Write: %v", err)
	}
	if err := os.WriteFile(capPath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	return cap, capPath
}

func TestBuildPipelineViewData(t *testing.T) {
	cap, _ := createTest5TierCapture(t)
	data, err := buildPipelineViewData(cap, "test-capture.pprof", "")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	if data.Device != "NVIDIA GB10" {
		t.Errorf("Device = %q, want NVIDIA GB10", data.Device)
	}
	if data.Architecture != "sm_121a" {
		t.Errorf("Architecture = %q, want sm_121a", data.Architecture)
	}

	if len(data.Tiers) != 5 {
		t.Fatalf("len(Tiers) = %d, want 5 (Go, CUDA C++, Tile IR, PTX, SASS)", len(data.Tiers))
	}

	tierNames := []string{"Go", "CUDA C++", "Tile IR", "PTX", "SASS"}
	for i, name := range tierNames {
		if data.Tiers[i].Name != name {
			t.Errorf("Tiers[%d].Name = %q, want %q", i, data.Tiers[i].Name, name)
		}
		if len(data.Tiers[i].Lines) == 0 {
			t.Errorf("Tiers[%d] has 0 lines", i)
		}
	}

	// Verify kernel discovery
	if len(data.Kernels) < 2 {
		t.Errorf("discovered %d kernels, want at least 2", len(data.Kernels))
	}
	hasQ8 := false
	for _, k := range data.Kernels {
		if k == "ds4_q8_0_matvec" {
			hasQ8 = true
		}
	}
	if !hasQ8 {
		t.Errorf("Kernels does not contain ds4_q8_0_matvec: %v", data.Kernels)
	}
}

func TestRenderPipelineHTML(t *testing.T) {
	cap, _ := createTest5TierCapture(t)
	data, err := buildPipelineViewData(cap, "test-capture.pprof", "")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	var buf bytes.Buffer
	if err := renderPipelineHTML(&buf, data); err != nil {
		t.Fatalf("renderPipelineHTML: %v", err)
	}

	html := buf.String()
	if !strings.Contains(html, "<!DOCTYPE html>") {
		t.Error("missing DOCTYPE html")
	}
	for _, name := range []string{"Go", "CUDA C", "Tile IR", "PTX", "SASS"} {
		if !strings.Contains(html, name) {
			t.Errorf("rendered HTML missing tier %q", name)
		}
	}
	if !strings.Contains(html, "ds4_q8_0_matvec") {
		t.Error("rendered HTML missing kernel ds4_q8_0_matvec")
	}
}

func TestPipelineServerHandlers(t *testing.T) {
	cap, _ := createTest5TierCapture(t)
	data, err := buildPipelineViewData(cap, "test-capture.pprof", "")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	handler := newPipelineServerHandler(data)

	// Test GET /
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("GET / returned %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "5-Tier GPU Compilation Pipeline") {
		t.Error("GET / response missing title")
	}

	// Test GET /api/pipeline
	reqAPI := httptest.NewRequest("GET", "/api/pipeline", nil)
	recAPI := httptest.NewRecorder()
	handler.ServeHTTP(recAPI, reqAPI)
	if recAPI.Code != http.StatusOK {
		t.Errorf("GET /api/pipeline returned %d, want 200", recAPI.Code)
	}
	var respData PipelineViewData
	if err := json.Unmarshal(recAPI.Body.Bytes(), &respData); err != nil {
		t.Fatalf("unmarshal /api/pipeline: %v", err)
	}
	if len(respData.Tiers) != 5 {
		t.Errorf("respData len(Tiers) = %d, want 5", len(respData.Tiers))
	}

	// Test GET /api/kernel?name=ds4_q8_0_matvec
	reqK := httptest.NewRequest("GET", "/api/kernel?name=ds4_q8_0_matvec", nil)
	recK := httptest.NewRecorder()
	handler.ServeHTTP(recK, reqK)
	if recK.Code != http.StatusOK {
		t.Errorf("GET /api/kernel returned %d, want 200", recK.Code)
	}
	var tiers []PipelineTier
	if err := json.Unmarshal(recK.Body.Bytes(), &tiers); err != nil {
		t.Fatalf("unmarshal /api/kernel: %v", err)
	}
	if len(tiers) != 5 {
		t.Errorf("kernel tiers count = %d, want 5", len(tiers))
	}
}

func TestRunViewCommandHTMLExport(t *testing.T) {
	_, capPath := createTest5TierCapture(t)
	outHTML := filepath.Join(t.TempDir(), "report.html")

	err := runViewCommand([]string{"view", "-html=" + outHTML, capPath})
	if err != nil {
		t.Fatalf("runViewCommand: %v", err)
	}

	content, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", outHTML, err)
	}
	if len(content) == 0 {
		t.Fatal("exported HTML file is empty")
	}
	if !strings.Contains(string(content), "ds4_q8_0_matvec") {
		t.Error("exported HTML missing kernel ds4_q8_0_matvec")
	}
}

func TestFetcher_SecondaryProfileAndSourceResolution(t *testing.T) {
	cap, capPath := createTest5TierCapture(t)
	_, capPath2 := createTest5TierCapture(t)

	f := &fetcher{
		path:   capPath,
		dir:    t.TempDir(),
		cap:    cap,
		limits: cudacapture.Limits{FileBytes: 10 << 20, TotalBytes: 50 << 20},
	}

	// 1. Primary profile fetch
	p1, _, err := f.Fetch(capPath, 0, 0)
	if err != nil {
		t.Fatalf("Fetch primary profile: %v", err)
	}
	if len(p1.Function) == 0 {
		t.Fatal("primary profile has no functions")
	}
	// Verify source file resolution
	fn := p1.Function[0]
	if fn.Filename == "" {
		t.Errorf("function %q has empty Filename", fn.Name)
	}
	if fn.StartLine == 0 {
		t.Errorf("function %q has StartLine == 0", fn.Name)
	}
	if len(p1.Location) > 0 && len(p1.Location[0].Line) > 0 {
		line := p1.Location[0].Line[0].Line
		if line == 0 {
			t.Errorf("location line == 0")
		}
	}

	// 2. Secondary profile fetch (simulating -diff_base)
	p2, _, err := f.Fetch(capPath2, 0, 0)
	if err != nil {
		t.Fatalf("Fetch secondary profile (-diff_base): %v", err)
	}
	if len(p2.Function) == 0 {
		t.Fatal("secondary profile has no functions")
	}
}

func createTestStallCapture(t *testing.T) (*cudacapture.Capture, string) {
	t.Helper()

	metaJSON := `{"device":"NVIDIA H100","architecture":"sm_90a","verified_results":100}`
	goSrc := `package main
func Run() {}`
	cuSrc := `extern "C" __global__ void saxpy(float a, float *x, float *y) {}`
	ptxSrc := `.version 8.0
.target sm_90a
.visible .entry saxpy() { ret; }`
	sassSrc := `Function : saxpy
/*0000*/ LDG.E R0, [R2];
/*0010*/ FADD R0, R0, R1;
/*0020*/ BAR.SYNC 0x0;
/*0030*/ DEPBAR.LE SB0, 0x0;
/*0040*/ RET;
`

	files := map[string][]byte{
		"workload.go":   []byte(goSrc),
		"kernels.cu":    []byte(cuSrc),
		"kernels.cubin": []byte("CUBIN_DATA"),
		"kernels.ptx":   []byte(ptxSrc),
		"kernels.sass":  []byte(sassSrc),
	}

	fn := &profile.Function{ID: 1, Name: "saxpy", Filename: "kernels.cu"}
	loc0 := &profile.Location{ID: 1, Address: 0x0000, Line: []profile.Line{{Function: fn, Line: 1}}}
	loc1 := &profile.Location{ID: 2, Address: 0x0010, Line: []profile.Line{{Function: fn, Line: 2}}}
	loc2 := &profile.Location{ID: 3, Address: 0x0020, Line: []profile.Line{{Function: fn, Line: 3}}}
	loc3 := &profile.Location{ID: 4, Address: 0x0030, Line: []profile.Line{{Function: fn, Line: 4}}}

	// Sample with stall metrics breakdown:
	// PC 0x0000 (LDG): stall_mem=50
	// PC 0x0010 (FADD): stall_scoreboard=30
	// PC 0x0020 (BAR.SYNC): stall_barrier=15
	// PC 0x0030 (DEPBAR): stall_drain=5
	// Total stalls = 100
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "stall_mem", Unit: "count"},
			{Type: "stall_scoreboard", Unit: "count"},
			{Type: "stall_barrier", Unit: "count"},
			{Type: "stall_drain", Unit: "count"},
		},
		Sample: []*profile.Sample{
			{Location: []*profile.Location{loc0}, Value: []int64{50, 0, 0, 0}},
			{Location: []*profile.Location{loc1}, Value: []int64{0, 30, 0, 0}},
			{Location: []*profile.Location{loc2}, Value: []int64{0, 0, 15, 0}},
			{Location: []*profile.Location{loc3}, Value: []int64{0, 0, 0, 5}},
		},
		Location: []*profile.Location{loc0, loc1, loc2, loc3},
		Function: []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile:  p,
		Metadata: []byte(metaJSON),
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			CUBIN:  "kernels.cubin",
			PTX:    "kernels.ptx",
			SASS:   "kernels.sass",
		},
		Files: files,
	}

	tmpDir := t.TempDir()
	capPath := filepath.Join(tmpDir, "test-stall-capture.pprof")
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, cap); err != nil {
		t.Fatalf("cudacapture.Write: %v", err)
	}
	if err := os.WriteFile(capPath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	return cap, capPath
}

func TestSASSStallHeatmap(t *testing.T) {
	cap, capPath := createTestStallCapture(t)

	// 1. Verify buildPipelineViewData correlates stalls
	data, err := buildPipelineViewData(cap, filepath.Base(capPath), "saxpy")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	sassTier := data.Tiers[4]
	if sassTier.Name != "SASS" {
		t.Fatalf("tier 4 is %q, want SASS", sassTier.Name)
	}

	// Verify PC correlation on SASS lines
	linesByOffset := make(map[string]PipelineLine)
	for _, l := range sassTier.Lines {
		if l.Offset != "" {
			linesByOffset[l.Offset] = l
		}
	}

	// Check PC 0x0000
	l0, ok := linesByOffset["0000"]
	if !ok {
		t.Fatal("missing line with offset 0000")
	}
	if l0.StallPct != 50.0 {
		t.Errorf("line 0000 StallPct = %v, want 50.0", l0.StallPct)
	}
	if l0.StallBreakdown["stall_mem"] != 50.0 {
		t.Errorf("line 0000 breakdown stall_mem = %v, want 50.0", l0.StallBreakdown["stall_mem"])
	}
	if !strings.Contains(l0.StallTooltip, "stall_mem: 50.0%") {
		t.Errorf("line 0000 tooltip missing stall_mem: %s", l0.StallTooltip)
	}

	// Check PC 0x0010
	l1, ok := linesByOffset["0010"]
	if !ok {
		t.Fatal("missing line with offset 0010")
	}
	if l1.StallPct != 30.0 {
		t.Errorf("line 0010 StallPct = %v, want 30.0", l1.StallPct)
	}
	if l1.StallBreakdown["stall_scoreboard"] != 30.0 {
		t.Errorf("line 0010 breakdown stall_scoreboard = %v, want 30.0", l1.StallBreakdown["stall_scoreboard"])
	}

	// Check PC 0x0020
	l2, ok := linesByOffset["0020"]
	if !ok {
		t.Fatal("missing line with offset 0020")
	}
	if l2.StallPct != 15.0 {
		t.Errorf("line 0020 StallPct = %v, want 15.0", l2.StallPct)
	}

	// Check PC 0x0030
	l3, ok := linesByOffset["0030"]
	if !ok {
		t.Fatal("missing line with offset 0030")
	}
	if l3.StallPct != 5.0 {
		t.Errorf("line 0030 StallPct = %v, want 5.0", l3.StallPct)
	}

	// Check PC 0x0040 (no stalls)
	l4, ok := linesByOffset["0040"]
	if !ok {
		t.Fatal("missing line with offset 0040")
	}
	if l4.StallPct != 0 {
		t.Errorf("line 0040 StallPct = %v, want 0", l4.StallPct)
	}

	// 2. Verify HTML template rendering contains color-coded stall intensity bars and tooltips
	var buf bytes.Buffer
	if err := renderPipelineHTML(&buf, data); err != nil {
		t.Fatalf("renderPipelineHTML: %v", err)
	}
	html := buf.String()

	if !strings.Contains(html, "stall-heat") {
		t.Error("rendered HTML missing stall-heat class")
	}
	if !strings.Contains(html, "stall-high") {
		t.Error("rendered HTML missing stall-high class")
	}
	if !strings.Contains(html, "stall-med") {
		t.Error("rendered HTML missing stall-med class")
	}
	if !strings.Contains(html, "stall-low") {
		t.Error("rendered HTML missing stall-low class")
	}
	if !strings.Contains(html, "stall_mem: 50.0%") {
		t.Error("rendered HTML missing stall_mem tooltip content")
	}
	if !strings.Contains(html, "50.0%") {
		t.Error("rendered HTML missing 50.0% text")
	}

	// 3. Verify /api/kernel endpoint returns stall breakdown
	handler := newPipelineServerHandler(data)
	req := httptest.NewRequest("GET", "/api/kernel?name=saxpy", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/kernel returned %d", rec.Code)
	}
	var apiTiers []PipelineTier
	if err := json.Unmarshal(rec.Body.Bytes(), &apiTiers); err != nil {
		t.Fatalf("unmarshal api response: %v", err)
	}
	var apiL0 *PipelineLine
	for _, l := range apiTiers[4].Lines {
		if l.Offset == "0000" {
			apiL0 = &l
			break
		}
	}
	if apiL0 == nil {
		t.Fatal("API response missing SASS line 0000")
	}
	if apiL0.StallPct != 50.0 {
		t.Errorf("API line 0000 StallPct = %v, want 50.0", apiL0.StallPct)
	}
	if apiL0.StallBreakdown["stall_mem"] != 50.0 {
		t.Errorf("API line 0000 StallBreakdown[stall_mem] = %v, want 50.0", apiL0.StallBreakdown["stall_mem"])
	}
}

func TestTileIRRichSupportInView(t *testing.T) {
	tileSrc := `cuda_tile.module @gemm_mod {
  cuda_tile.entry @gemm(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}`
	remarksYAML := `--- !Passed
Pass:            tileiras-tma-lowering
Name:            TMALoadOptimized
Function:        gemm
Args:
  - String:          'Lowered load_view_tko on A to TMA 2D descriptor'
--- !Passed
Pass:            tileiras-mma-selection
Name:            TensorCoreMMASelected
Function:        gemm
Args:
  - String:          'Selected wgmma.mma_async for Tensor Core execution'
`

	files := map[string][]byte{
		"kernels.tileir":        []byte(tileSrc),
		"tileiras-remarks.yaml": []byte(remarksYAML),
		"kernels.ptx":           []byte(".version 8.8\n.target sm_121a\n.entry gemm() { ret; }"),
		"kernels.sass":          []byte("Function : gemm\n/*0000*/ RET;\n"),
	}

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			TileIR: "kernels.tileir",
			PTX:    "kernels.ptx",
			SASS:   "kernels.sass",
		},
		Files: files,
	}

	data, err := buildPipelineViewData(cap, "test-tileir.pprof", "")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	// 1. Verify Tile IR Audit & Verification
	if !strings.Contains(data.TileIRAudit, "Tile IR: Verified") {
		t.Errorf("TileIRAudit = %q, want Verified status", data.TileIRAudit)
	}
	if data.RemarksCount != 2 {
		t.Errorf("RemarksCount = %d, want 2", data.RemarksCount)
	}

	// 2. Verify Tile IR tier annotations
	if len(data.Tiers) < 3 {
		t.Fatalf("len(Tiers) = %d, want at least 3", len(data.Tiers))
	}
	tileTier := data.Tiers[2]
	if tileTier.Name != "Tile IR" {
		t.Fatalf("Tiers[2].Name = %q, want Tile IR", tileTier.Name)
	}

	var foundLoad, foundMMA bool
	for _, l := range tileTier.Lines {
		if strings.Contains(l.Text, "load_view_tko") {
			foundLoad = true
			if l.Category != "Tile Memory" {
				t.Errorf("load_view_tko Category = %q, want Tile Memory", l.Category)
			}
			if !strings.Contains(l.HoverDoc, "cp.async.bulk.tensor") {
				t.Errorf("load_view_tko HoverDoc missing TMA lowering: %q", l.HoverDoc)
			}
			if l.RemarkBadge != "TMALoadOptimized" {
				t.Errorf("load_view_tko RemarkBadge = %q, want TMALoadOptimized", l.RemarkBadge)
			}
		}
		if strings.Contains(l.Text, "mma_f") {
			foundMMA = true
			if l.Category != "Tensor Core MMA" {
				t.Errorf("mma_f Category = %q, want Tensor Core MMA", l.Category)
			}
			if !strings.Contains(l.HoverDoc, "wgmma") {
				t.Errorf("mma_f HoverDoc missing wgmma lowering: %q", l.HoverDoc)
			}
			if l.RemarkBadge != "TensorCoreMMASelected" {
				t.Errorf("mma_f RemarkBadge = %q, want TensorCoreMMASelected", l.RemarkBadge)
			}
		}
	}
	if !foundLoad {
		t.Error("missing load_view_tko line in Tile IR tier")
	}
	if !foundMMA {
		t.Error("missing mma_f line in Tile IR tier")
	}

	// 3. Verify HTML rendering has Tile IR styling and badges
	var buf bytes.Buffer
	if err := renderPipelineHTML(&buf, data); err != nil {
		t.Fatalf("renderPipelineHTML: %v", err)
	}
	html := buf.String()

	for _, expected := range []string{
		"badge-category",
		"badge-remark",
		"badge-audit",
		"opcode-hover-tile",
		"highlight-match",
		"TMALoadOptimized",
		"TensorCoreMMASelected",
		"Tile Memory",
		"Tensor Core MMA",
	} {
		if !strings.Contains(html, expected) {
			t.Errorf("rendered HTML missing expected token %q", expected)
		}
	}
}

func TestCrossTierLinkageAndOverlays(t *testing.T) {
	goSrc := `package main
func run() {
	// Launch kernel
	r.launch("gemm_kernel", grid, block)
}`
	cuSrc := `// CUDA source
__global__ void gemm_kernel(const half* A, const half* B, float* C) {
	__shared__ float smem[1024];
	int val = __dp4a(1, 2, 3);
}`
	tileSrc := `cuda_tile.module @gemm_mod {
  cuda_tile.entry @gemm_kernel(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16> loc("kernels.cu":3:10)
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vA, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}`
	ptxSrc := `.version 8.8
.target sm_121a
.visible .entry gemm_kernel() {
.loc 1 3 0
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%rd1], [%rd2, {%r1, %r2}];
.loc 1 4 0
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1, %f2}, {%r3, %r4}, {%r5}, {%f1, %f2};
ret;
}`
	sassSrc := `Function : gemm_kernel
## File "kernels.cu", line 3
/*0000*/ UTMA.LOAD.2D [R2], [R4];
## File "kernels.cu", line 4
/*0010*/ HMMA.16816.F32 R0, R2, R4, R0;
/*0020*/ RET;
`
	lrmSrc := `Function: gemm_kernel
/*0000*/ UTMA.LOAD.2D [R2], [R4] ; // | 4 | | 2 | |
/*0010*/ HMMA.16816.F32 R0, R2, R4, R0 ; // | 8 | | 2 | |
/*0020*/ RET ; // | 1 | | 0 | |
`

	files := map[string][]byte{
		"workload.go":    []byte(goSrc),
		"kernels.cu":     []byte(cuSrc),
		"kernels.tileir": []byte(tileSrc),
		"kernels.ptx":    []byte(ptxSrc),
		"kernels.sass":   []byte(sassSrc),
		"kernels.lrm":    []byte(lrmSrc),
	}

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			TileIR: "kernels.tileir",
			PTX:    "kernels.ptx",
			SASS:   "kernels.sass",
		},
		Files: files,
	}

	data, err := buildPipelineViewData(cap, "test-linkage.pprof", "")
	if err != nil {
		t.Fatalf("buildPipelineViewData: %v", err)
	}

	// 1. Verify Go tier launch detection
	goTier := data.Tiers[0]
	var foundGoLaunch bool
	for _, l := range goTier.Lines {
		if l.LaunchKernel == "gemm_kernel" {
			foundGoLaunch = true
			if l.SemanticTag != "entry" {
				t.Errorf("Go launch SemanticTag = %q, want entry", l.SemanticTag)
			}
		}
	}
	if !foundGoLaunch {
		t.Error("Go tier did not detect launch of gemm_kernel")
	}

	// 2. Verify CUDA C++ annotations and source lines
	cuTier := data.Tiers[1]
	if cuTier.Lines[1].SourceLine != 2 {
		t.Errorf("CUDA C++ line 2 SourceLine = %d, want 2", cuTier.Lines[1].SourceLine)
	}
	if cuTier.Lines[1].Category != "Global Kernel" || cuTier.Lines[1].SemanticTag != "entry" {
		t.Errorf("CUDA C++ kernel entry category/tag = %q/%q, want Global Kernel/entry", cuTier.Lines[1].Category, cuTier.Lines[1].SemanticTag)
	}

	// 3. Verify Tile IR semantic tags and source lines
	tileTier := data.Tiers[2]
	var foundTileTMA, foundTileMMA bool
	for _, l := range tileTier.Lines {
		if strings.Contains(l.Text, "load_view_tko") {
			foundTileTMA = true
			if l.SemanticTag != "tma" {
				t.Errorf("Tile load_view_tko SemanticTag = %q, want tma", l.SemanticTag)
			}
			if l.SourceLine != 3 {
				t.Errorf("Tile load_view_tko SourceLine = %d, want 3", l.SourceLine)
			}
		}
		if strings.Contains(l.Text, "mma_f") {
			foundTileMMA = true
			if l.SemanticTag != "mma" {
				t.Errorf("Tile mma_f SemanticTag = %q, want mma", l.SemanticTag)
			}
		}
	}
	if !foundTileTMA || !foundTileMMA {
		t.Error("Tile IR missing load_view_tko or mma_f")
	}

	// 4. Verify PTX source line mapping (.loc) and semantic tags
	ptxTier := data.Tiers[3]
	var foundPtxTMA, foundPtxMMA bool
	for _, l := range ptxTier.Lines {
		if strings.Contains(l.Text, "cp.async.bulk.tensor") {
			foundPtxTMA = true
			if l.SemanticTag != "tma" {
				t.Errorf("PTX cp.async.bulk SemanticTag = %q, want tma", l.SemanticTag)
			}
			if l.SourceLine != 3 {
				t.Errorf("PTX cp.async.bulk SourceLine = %d, want 3", l.SourceLine)
			}
		}
		if strings.Contains(l.Text, "mma.sync") {
			foundPtxMMA = true
			if l.SemanticTag != "mma" {
				t.Errorf("PTX mma.sync SemanticTag = %q, want mma", l.SemanticTag)
			}
			if l.SourceLine != 4 {
				t.Errorf("PTX mma.sync SourceLine = %d, want 4", l.SourceLine)
			}
		}
	}
	if !foundPtxTMA || !foundPtxMMA {
		t.Error("PTX missing cp.async.bulk or mma.sync")
	}

	// 5. Verify SASS DWARF line info, register pressure, and semantic tags
	sassTier := data.Tiers[4]
	var foundSassTMA, foundSassMMA bool
	for _, l := range sassTier.Lines {
		if strings.Contains(l.Text, "UTMA.LOAD.2D") {
			foundSassTMA = true
			if l.SemanticTag != "tma" {
				t.Errorf("SASS UTMA SemanticTag = %q, want tma", l.SemanticTag)
			}
			if l.SourceLine != 3 {
				t.Errorf("SASS UTMA SourceLine = %d, want 3", l.SourceLine)
			}
			if l.Pressure != "R:4 UR:2" {
				t.Errorf("SASS UTMA Pressure = %q, want R:4 UR:2", l.Pressure)
			}
		}
		if strings.Contains(l.Text, "HMMA.16816.F32") {
			foundSassMMA = true
			if l.SemanticTag != "mma" {
				t.Errorf("SASS HMMA SemanticTag = %q, want mma", l.SemanticTag)
			}
			if l.SourceLine != 4 {
				t.Errorf("SASS HMMA SourceLine = %d, want 4", l.SourceLine)
			}
			if l.Pressure != "R:8 UR:2" {
				t.Errorf("SASS HMMA Pressure = %q, want R:8 UR:2", l.Pressure)
			}
		}
	}
	if !foundSassTMA || !foundSassMMA {
		t.Error("SASS missing UTMA or HMMA")
	}

	// 6. Verify HTML output contains new interactive classes and attributes
	var buf bytes.Buffer
	if err := renderPipelineHTML(&buf, data); err != nil {
		t.Fatalf("renderPipelineHTML: %v", err)
	}
	html := buf.String()

	for _, expected := range []string{
		"highlight-srcline",
		"highlight-semtag",
		"badge-reg",
		"badge-launch",
		"data-sem-tag=\"tma\"",
		"data-sem-tag=\"mma\"",
		"data-source-line=\"3\"",
		"data-source-line=\"4\"",
		"▶ Launch: gemm_kernel",
		"R:4 UR:2",
		"R:8 UR:2",
	} {
		if !strings.Contains(html, expected) {
			t.Errorf("rendered HTML missing expected token %q", expected)
		}
	}
}




