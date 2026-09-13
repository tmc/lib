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
