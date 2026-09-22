package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// sampleTypeIsReal reports whether some producer this command reads can emit
// name. cmd/cudaprof reads two: x/cudaprof writes the counter and stall types,
// and x/cudacapture.BuildActivityProfile writes the activity ones. Asking only
// the first calls kernel_time invented when advise.go consumes it. Both halves
// are asked at runtime rather than transcribed, so neither list can drift from
// the producer that owns it.
func sampleTypeIsReal(name string) bool {
	if err := cudaprof.NewBuilder().SetSampleTypes(name); err == nil {
		return true
	}
	for _, st := range cudacapture.BuildActivityProfile("", 0, nil).SampleType {
		if st.Type == name {
			return true
		}
	}
	return false
}

// TestStallFixtureNamesAreReal keeps this file's fixtures inside the vocabulary
// the producers actually emit. A fixture is free to invent a sample type, and
// one that does makes the test exercise a shape no capture can have: that is
// how four stall narratives in diff.go stayed green while never firing on real
// data. TestFixtureSampleTypesAreReal sweeps every fixture in the package; this
// one carries the negative control that keeps the oracle honest.
func TestStallFixtureNamesAreReal(t *testing.T) {
	// kernel_time is here on purpose: it is real only through the activity
	// producer, so it fails if the oracle narrows back to one of the two.
	for _, name := range []string{"stall_mem", "stall_wait", "stall_scoreboard", "kernel_time"} {
		if !sampleTypeIsReal(name) {
			t.Errorf("sampleTypeIsReal(%q) = false, want a name some producer emits", name)
		}
	}
	// Negative control: an oracle that accepts everything would pass every
	// check above without checking anything. stall_memory_throttle is a
	// plausible-looking name neither producer writes.
	if sampleTypeIsReal("stall_memory_throttle") {
		t.Error("sampleTypeIsReal(\"stall_memory_throttle\") = true, so this test proves nothing")
	}
}

func createTestCapture(t *testing.T, kernelName, sassText string) string {
	t.Helper()
	fn := &profile.Function{ID: 1, Name: kernelName, Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void compute() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(sassText),
		},
	}

	dir := t.TempDir()
	capturePath := filepath.Join(dir, kernelName+".pprof")
	f, err := os.Create(capturePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return capturePath
}

func TestRunDiffSASSFiles(t *testing.T) {
	baseSASS := `
.section .text.gemv,"ax",@progbits
/*0100*/ LDG.E.F32 R4, [R2.64] ;
/*0110*/ SHR.B64 R5, R4, 0x8 ;
/*0120*/ IMAD.WIDE R0, R5, R6, R0 ;
/*0130*/ IMAD.WIDE R1, R7, R8, R1 ;
/*0140*/ EXIT ;
`
	targetSASS := `
.section .text.gemv,"ax",@progbits
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ LOP3.LUT R6, R4, R5, 0xca ;
/*0030*/ IDP4A.U8.U8 R0, R6, R7, R0 ;
/*0040*/ IDP4A.S8.S8 R1, R8, R9, R1 ;
/*0050*/ EXIT ;
`

	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.sass")
	targetPath := filepath.Join(dir, "target.sass")
	if err := os.WriteFile(basePath, []byte(baseSASS), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetPath, []byte(targetSASS), 0644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: gemv ===") {
		t.Errorf("output missing kernel header: %s", out)
	}
	if !strings.Contains(out, "+ /*0030*/ IDP4A.U8.U8 R0, R6, R7, R0") {
		t.Errorf("output missing + addition for IDP4A: %s", out)
	}
	if !strings.Contains(out, "- /*0120*/ IMAD.WIDE R0, R5, R6, R0") {
		t.Errorf("output missing - deletion for IMAD: %s", out)
	}
	if !strings.Contains(out, "IMAD -> IDP4A") {
		t.Errorf("output missing substitution summary: %s", out)
	}
	if !strings.Contains(out, "SIMD operations:   0 -> 2 (+2)") {
		t.Errorf("output missing SIMD summary: %s", out)
	}
}

func TestRunDiffPprofCaptures(t *testing.T) {
	baseSASS := `
.section .text.kernelA,"ax",@progbits
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`
	targetSASS := `
.section .text.kernelA,"ax",@progbits
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ STL.32 [R1+0x10], R4 ;
/*0030*/ LDL.32 R4, [R1+0x10] ;
/*0040*/ FFMA R0, R4, R5, R0 ;
/*0050*/ EXIT ;
`

	baseCapture := createTestCapture(t, "base", baseSASS)
	targetCapture := createTestCapture(t, "target", targetSASS)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", baseCapture, targetCapture})
	if err != nil {
		t.Fatalf("runDiff with pprof failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: kernelA ===") {
		t.Errorf("output missing kernel header: %s", out)
	}
	if !strings.Contains(out, "+ /*0020*/ STL.32 [R1+0x10], R4") {
		t.Errorf("output missing + STL addition: %s", out)
	}
	if !strings.Contains(out, "+ /*0030*/ LDL.32 R4, [R1+0x10]") {
		t.Errorf("output missing + LDL addition: %s", out)
	}
	if !strings.Contains(out, "Local-memory accesses: +2") {
		t.Errorf("output missing spill operation count: %s", out)
	}
}

func TestRunDiffKernelFilter(t *testing.T) {
	baseSASS := `
.section .text.k1,"ax",@progbits
/*0010*/ IMAD R0, R1, R2, R0 ;
/*0020*/ EXIT ;

.section .text.k2,"ax",@progbits
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`
	targetSASS := `
.section .text.k1,"ax",@progbits
/*0010*/ IDP4A.U8.U8 R0, R1, R2, R0 ;
/*0020*/ EXIT ;

.section .text.k2,"ax",@progbits
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`

	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.sass")
	targetPath := filepath.Join(dir, "target.sass")
	os.WriteFile(basePath, []byte(baseSASS), 0644)
	os.WriteFile(targetPath, []byte(targetSASS), 0644)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", "--kernel=k1", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: k1 ===") {
		t.Errorf("missing k1: %s", out)
	}
	if strings.Contains(out, "=== Kernel: k2 ===") {
		t.Errorf("k2 should have been filtered out: %s", out)
	}

	// Filter for non-existent kernel
	buf.Reset()
	err = runDiff(&buf, []string{"--no-color", "--kernel=k999", basePath, targetPath})
	if err == nil {
		t.Error("expected error for non-existent kernel filter")
	}
}

func TestRunDiffErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{
			name: "no arguments",
			args: []string{},
		},
		{
			name: "single argument",
			args: []string{"base.sass"},
		},
		{
			name: "missing file",
			args: []string{"nonexistent_base.sass", "nonexistent_target.sass"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := runDiff(&buf, tt.args); err == nil {
				t.Errorf("runDiff(%v) expected error, got nil", tt.args)
			}
		})
	}
}

func TestRunDiffExplain(t *testing.T) {
	createProfilingCapture := func(t *testing.T, kernelName string, durationNs int64, memStalls, barrierStalls int64) string {
		t.Helper()
		fn := &profile.Function{ID: 1, Name: kernelName, Filename: "kernels.cu"}
		loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
		p := &profile.Profile{
			SampleType: []*profile.ValueType{
				// These must be names x/cudaprof actually emits. The fixture
				// used to declare stall_memory_throttle and stall_barrier,
				// which it does not, so this test exercised a shape no capture
				// can have and asserted a narrative only that shape reached.
				{Type: "gpu_time", Unit: "nanoseconds"},
				{Type: "stall_mem", Unit: "samples"},
				{Type: "stall_wait", Unit: "samples"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{loc},
					Value:    []int64{durationNs, memStalls, barrierStalls},
				},
			},
			Location: []*profile.Location{loc},
			Function: []*profile.Function{fn},
		}

		cap := &cudacapture.Capture{
			Profile: p,
			Code: cudacapture.CodeFiles{
				Source: "kernels.cu",
				PTX:    "kernels.ptx",
				CUBIN:  "kernels.cubin",
				SASS:   "kernels.sass",
			},
			Files: map[string][]byte{
				"kernels.cu":    []byte("void compute() {}\n"),
				"kernels.ptx":   []byte("// PTX\n"),
				"kernels.cubin": []byte("CUBIN_MOCK"),
				"kernels.sass":  []byte("--- Kernel: " + kernelName + " ---\n/*0000*/ NOP ;\n/*0010*/ EXIT ;\n"),
			},
		}

		dir := t.TempDir()
		path := filepath.Join(dir, kernelName+".pprof")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := cudacapture.Write(f, cap); err != nil {
			t.Fatal(err)
		}
		return path
	}

	basePath := createProfilingCapture(t, "gemv_kernel", 100000, 5000, 1000)
	targetPath := createProfilingCapture(t, "gemv_kernel", 75000, 2000, 950)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--explain", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff(--explain) failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "CUDA Profile Diff & Optimization Explanation") {
		t.Errorf("output missing title: %s", out)
	}
	if !strings.Contains(out, "Execution Duration Delta") {
		t.Errorf("output missing duration delta section: %s", out)
	}
	if !strings.Contains(out, "stall_mem") {
		t.Errorf("output missing stall_mem: %s", out)
	}
	// The stall table is the part that works on real captures: 5000 -> 2000
	// samples is a 60% drop, and the row has to carry it.
	if !strings.Contains(out, "-60.0%") {
		t.Errorf("output missing the stall_mem shift: %s", out)
	}
	if !strings.Contains(out, "1.33×") && !strings.Contains(out, "1.33x") {
		t.Errorf("output missing expected speedup calculation: %s", out)
	}
}

func TestFindTimeIndex(t *testing.T) {
	makeVT := func(typ, unit string) *profile.ValueType {
		return &profile.ValueType{Type: typ, Unit: unit}
	}
	tests := []struct {
		name    string
		types   []*profile.ValueType
		wantIdx int
	}{
		{"gpu_time nanoseconds", []*profile.ValueType{makeVT("gpu_time", "nanoseconds")}, 0},
		{"kernel_time nanoseconds", []*profile.ValueType{makeVT("kernel_time", "nanoseconds")}, 0},
		{"duration_ns ns", []*profile.ValueType{makeVT("duration_ns", "ns")}, 0},
		{"duration_ns empty unit", []*profile.ValueType{makeVT("duration_ns", "")}, 0},
		{"duration nanoseconds", []*profile.ValueType{makeVT("duration", "nanoseconds")}, 0},
		{"time nanoseconds", []*profile.ValueType{makeVT("time", "nanoseconds")}, 0},
		{"cpu nanoseconds", []*profile.ValueType{makeVT("cpu", "nanoseconds")}, 0},
		{"pc_samples count (never duration)", []*profile.ValueType{makeVT("pc_samples", "count")}, -1},
		{"pc_samples samples (never duration)", []*profile.ValueType{makeVT("pc_samples", "samples")}, -1},
		{"samples count (never duration)", []*profile.ValueType{makeVT("samples", "count")}, -1},
		{"stall_mem samples (never duration)", []*profile.ValueType{makeVT("stall_mem", "samples")}, -1},
		{"inst_executed count (never duration)", []*profile.ValueType{makeVT("inst_executed", "count")}, -1},
		{"vram_alloc bytes", []*profile.ValueType{makeVT("vram_alloc", "bytes")}, -1},
		{"pc_samples at index 0 must not fall back", []*profile.ValueType{
			makeVT("pc_samples", "count"),
			makeVT("gpu_time", "nanoseconds"),
		}, 1},
		{"empty sample types", []*profile.ValueType{}, -1},
		{"nil profile", nil, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p *profile.Profile
			if tt.types != nil {
				p = &profile.Profile{SampleType: tt.types}
			}
			got := findTimeIndex(p)
			if got != tt.wantIdx {
				t.Errorf("findTimeIndex() = %d, want %d", got, tt.wantIdx)
			}
		})
	}
}

func TestFindSampleIndex(t *testing.T) {
	makeVT := func(typ, unit string) *profile.ValueType {
		return &profile.ValueType{Type: typ, Unit: unit}
	}
	tests := []struct {
		name    string
		types   []*profile.ValueType
		wantIdx int
	}{
		{"pc_samples count", []*profile.ValueType{makeVT("pc_samples", "count")}, 0},
		{"samples count", []*profile.ValueType{makeVT("samples", "count")}, 0},
		{"pc_samples preferred over samples", []*profile.ValueType{
			makeVT("samples", "count"),
			makeVT("pc_samples", "count"),
		}, 1},
		{"duration metric has no sample index", []*profile.ValueType{makeVT("gpu_time", "nanoseconds")}, -1},
		{"empty sample types", []*profile.ValueType{}, -1},
		{"nil profile", nil, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p *profile.Profile
			if tt.types != nil {
				p = &profile.Profile{SampleType: tt.types}
			}
			got := findSampleIndex(p)
			if got != tt.wantIdx {
				t.Errorf("findSampleIndex() = %d, want %d", got, tt.wantIdx)
			}
		})
	}
}

func createMultiKernelCapture(t *testing.T, sampleType, sampleUnit string, kernels map[string]int64) string {
	t.Helper()
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: sampleType, Unit: sampleUnit}},
	}
	var nextID uint64 = 1
	var sassBuilder strings.Builder
	for name, val := range kernels {
		fn := &profile.Function{ID: nextID, Name: name, Filename: "kernels.cu"}
		loc := &profile.Location{ID: nextID, Line: []profile.Line{{Function: fn, Line: 1}}}
		nextID++
		p.Function = append(p.Function, fn)
		p.Location = append(p.Location, loc)
		p.Sample = append(p.Sample, &profile.Sample{
			Location: []*profile.Location{loc},
			Value:    []int64{val},
		})
		sassBuilder.WriteString(fmt.Sprintf("--- Kernel: %s ---\n/*0000*/ NOP ;\n/*0010*/ EXIT ;\n", name))
	}

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void compute() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(sassBuilder.String()),
		},
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "capture.pprof")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunDiffExplainPCSamples(t *testing.T) {
	basePath := createMultiKernelCapture(t, "pc_samples", "count", map[string]int64{
		"q2_gather":  1338579,
		"iq2_gather": 444134,
	})
	targetPath := createMultiKernelCapture(t, "pc_samples", "count", map[string]int64{
		"q2_gather":  1083682,
		"iq2_gather": 528815,
	})

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--explain", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff(--explain) failed: %v", err)
	}

	out := buf.String()

	// Must contain PC sampling shift sections and statistics
	for _, want := range []string{
		"## Sample Count Shift",
		"- **Base Sample Count**:   1782713",
		"- **Target Sample Count**: 1612497",
		"- **Sample Count Shift**:  -170216 (170216 fewer PC samples",
		"q2_gather",
		"-254897 (254897 fewer PC samples)",
		"iq2_gather",
		"+84681 (84681 more PC samples)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing expected PC-sample string %q\nGot:\n%s", want, out)
		}
	}

	// Must NEVER claim runtime, µs, or speedup for PC sampling
	for _, forbidden := range []string{
		"Runtime",
		"µs",
		"speedup",
		"×",
	} {
		if strings.Contains(out, forbidden) {
			t.Errorf("output erroneously contains %q in PC-sample comparison\nGot:\n%s", forbidden, out)
		}
	}
}

func TestRunDiffExplainNativeDurationTable(t *testing.T) {
	tests := []struct {
		name       string
		sampleType string
		sampleUnit string
		baseNs     int64
		targetNs   int64
		wantSpeed  string
		wantSaved  string
	}{
		{"gpu_time", "gpu_time", "nanoseconds", 100000, 50000, "2.00×", "50.0%"},
		{"kernel_time", "kernel_time", "nanoseconds", 200000, 150000, "1.33×", "25.0%"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			basePath := createMultiKernelCapture(t, tt.sampleType, tt.sampleUnit, map[string]int64{"compute": tt.baseNs})
			targetPath := createMultiKernelCapture(t, tt.sampleType, tt.sampleUnit, map[string]int64{"compute": tt.targetNs})

			var buf bytes.Buffer
			if err := runDiff(&buf, []string{"--explain", basePath, targetPath}); err != nil {
				t.Fatalf("runDiff failed: %v", err)
			}
			out := buf.String()

			if !strings.Contains(out, "## Execution Duration Delta") {
				t.Errorf("missing Execution Duration Delta: %s", out)
			}
			if !strings.Contains(out, tt.wantSpeed) {
				t.Errorf("missing speedup %s: %s", tt.wantSpeed, out)
			}
			if !strings.Contains(out, tt.wantSaved) {
				t.Errorf("missing pct saved %s: %s", tt.wantSaved, out)
			}
			if !strings.Contains(out, "µs") {
				t.Errorf("missing µs unit in duration diff: %s", out)
			}
		})
	}
}

func TestRunDiffExplainMismatchedOrMissing(t *testing.T) {
	tests := []struct {
		name        string
		baseType    string
		baseUnit    string
		targetType  string
		targetUnit  string
		wantMessage string
	}{
		{
			name:        "duration vs pc_samples",
			baseType:    "gpu_time",
			baseUnit:    "nanoseconds",
			targetType:  "pc_samples",
			targetUnit:  "count",
			wantMessage: "Mismatched sample types",
		},
		{
			name:        "pc_samples vs duration",
			baseType:    "pc_samples",
			baseUnit:    "count",
			targetType:  "gpu_time",
			targetUnit:  "nanoseconds",
			wantMessage: "Mismatched sample types",
		},
		{
			name:        "both unrelated non-metrics",
			baseType:    "vram_alloc",
			baseUnit:    "bytes",
			targetType:  "vram_alloc",
			targetUnit:  "bytes",
			wantMessage: "No comparable execution duration or PC sampling metrics found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			basePath := createMultiKernelCapture(t, tt.baseType, tt.baseUnit, map[string]int64{"k": 100})
			targetPath := createMultiKernelCapture(t, tt.targetType, tt.targetUnit, map[string]int64{"k": 100})

			var buf bytes.Buffer
			if err := runDiff(&buf, []string{"--explain", basePath, targetPath}); err != nil {
				t.Fatalf("runDiff failed: %v", err)
			}
			out := buf.String()

			if !strings.Contains(out, tt.wantMessage) {
				t.Errorf("output missing expected message %q: %s", tt.wantMessage, out)
			}
			for _, forbidden := range []string{"Runtime:", "speedup", "×"} {
				if strings.Contains(out, forbidden) {
					t.Errorf("output erroneously contains %q on mismatched/missing metrics: %s", forbidden, out)
				}
			}
		})
	}
}

// TestRunDiffKernelFilterMangled exercises the filter against the symbol
// spelling real tools produce. nvdisasm names a kernel by its mangled symbol,
// so a filter written the way a user thinks of the kernel ("scale") has to
// reach _Z5scalePff. The rest of the corpus uses friendly names, where any
// matching rule works.
func TestRunDiffKernelFilterMangled(t *testing.T) {
	const baseSASS = `
.section .text._Z5scalePff,"ax",@progbits
/*0010*/ IMAD R0, R1, R2, R0 ;
/*0020*/ EXIT ;

.section .text._Z5gegluPKfPfi,"ax",@progbits
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`
	const targetSASS = `
.section .text._Z5scalePff,"ax",@progbits
/*0010*/ IDP4A.U8.U8 R0, R1, R2, R0 ;
/*0020*/ EXIT ;

.section .text._Z5gegluPKfPfi,"ax",@progbits
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`

	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.sass")
	targetPath := filepath.Join(dir, "target.sass")
	if err := os.WriteFile(basePath, []byte(baseSASS), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetPath, []byte(targetSASS), 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		filter  string
		want    string
		exclude string
		wantErr bool
	}{
		{name: "demangled", filter: "scale", want: "_Z5scalePff", exclude: "_Z5gegluPKfPfi"},
		{name: "mangled", filter: "_Z5scalePff", want: "_Z5scalePff", exclude: "_Z5gegluPKfPfi"},
		{name: "other kernel", filter: "geglu", want: "_Z5gegluPKfPfi", exclude: "_Z5scalePff"},
		{name: "absent", filter: "nosuchkernel", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := runDiff(&buf, []string{"--no-color", "--kernel=" + tt.filter, basePath, targetPath})
			if tt.wantErr {
				if err == nil {
					t.Fatalf("filter %q: got nil error, want a not-found error", tt.filter)
				}
				return
			}
			if err != nil {
				t.Fatalf("filter %q: %v", tt.filter, err)
			}
			out := buf.String()
			if !strings.Contains(out, "=== Kernel: "+tt.want+" ===") {
				t.Errorf("filter %q selected no kernel; want %s\n%s", tt.filter, tt.want, out)
			}
			if strings.Contains(out, "=== Kernel: "+tt.exclude+" ===") {
				t.Errorf("filter %q also selected %s\n%s", tt.filter, tt.exclude, out)
			}
		})
	}
}
