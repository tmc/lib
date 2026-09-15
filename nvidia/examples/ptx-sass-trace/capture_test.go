package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

//go:embed testdata/*
var testFS embed.FS

func TestCapture(t *testing.T) {
	path := os.Getenv("CUDA_PROFILE_CAPTURE")
	if path == "" {
		t.Skip("set CUDA_PROFILE_CAPTURE to a real capture (.pprof or dir)")
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat capture path: %v", err)
	}

	var cap *cudacapture.Capture
	if fi.IsDir() {
		cap, err = cudacapture.ReadDir(path)
	} else {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		cap, err = cudacapture.Read(f)
		if err != nil {
			cap, err = cudacapture.ReadDir(filepath.Dir(path))
		}
	}
	if err != nil {
		t.Fatalf("load capture: %v", err)
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("load code: %v", err)
	}
	if len(code.Kernels) == 0 {
		t.Fatal("expected kernels in code")
	}

	if cap.Profile == nil {
		t.Fatal("expected profile in capture")
	}
	t.Logf("profile has %d sample types, %d samples", len(cap.Profile.SampleType), len(cap.Profile.Sample))
	for i, st := range cap.Profile.SampleType {
		t.Logf("  sample_type[%d]: %s (%s)", i, st.Type, st.Unit)
	}
	t.Logf("verified capture %s: %d kernels, %d files", path, len(code.Kernels), len(cap.Files))

	// Verify SIMD vs scalar opcodes via cudasass.AssertOpcodes if SASS is present.
	if sassData, ok := cap.Files[cap.Code.SASS]; ok && len(sassData) > 0 {
		disasm, err := cudasass.ParseString(string(sassData))
		if err == nil {
			if disasm.Function("ds4_iq2_simd") != nil {
				if err := disasm.AssertOpcodes("ds4_iq2_simd", []string{"IDP4A"}, []string{"SHR"}); err != nil {
					t.Errorf("AssertOpcodes(ds4_iq2_simd): %v", err)
				}
			}
			if disasm.Function("ds4_iq2_scalar") != nil {
				if err := disasm.AssertOpcodes("ds4_iq2_scalar", []string{"IMAD"}, []string{"IDP4A"}); err != nil {
					t.Errorf("AssertOpcodes(ds4_iq2_scalar): %v", err)
				}
			}
		}
	}
}

func TestIsArithmeticOpcode(t *testing.T) {
	tests := []struct {
		opcode string
		want   bool
	}{
		{"fma.rn.f32", true},
		{"add.f32", true},
		{"sub.f32", true},
		{"mul.lo.s32", true},
		{"dp4a.s32.s32", true},
		{"shfl.sync.down.b32", true},
		{"ld.global.f32", true},
		{"st.global.f32", true},
		{"rsqrt.approx.f32", true},
		{"cvt.f32.f16", true},
		{"bar.sync", false},
		{"bra.uni", false},
		{"ret", false},
		{"nop", false},
	}

	for _, tt := range tests {
		got := isArithmeticOpcode(tt.opcode)
		if got != tt.want {
			t.Errorf("isArithmeticOpcode(%q) = %v, want %v", tt.opcode, got, tt.want)
		}
	}
}

func TestSanitizeCUDA12CUBIN(t *testing.T) {
	standard := make([]byte, 24)
	copy(standard, []byte{0x7f, 'E', 'L', 'F'})
	standard[7], standard[8], standard[20] = 0, 0, 1
	legacy := append([]byte(nil), standard...)
	legacy[7], legacy[8], legacy[20] = 0x33, 0x07, 0x81

	for _, tt := range []struct {
		name string
		in   []byte
		want []byte
	}{
		{name: "standard", in: standard, want: nil},
		{name: "legacy", in: legacy, want: standard},
		{name: "short", in: legacy[:23], want: nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeCUDA12CUBIN(tt.in)
			if !bytes.Equal(got, tt.want) {
				t.Fatalf("sanitizeCUDA12CUBIN() = %x, want %x", got, tt.want)
			}
			if tt.name == "legacy" && bytes.Equal(got, tt.in) {
				t.Fatal("sanitizer returned the input backing array")
			}
		})
	}
}

func TestKernelsNVRTCCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	src, err := sources.ReadFile("kernels.cu")
	if err != nil {
		t.Fatalf("read kernels.cu from embedded sources: %v", err)
	}
	if _, err := cudatest.CompileSource(string(src), major, minor); err != nil {
		t.Fatalf("kernels.cu does not compile with NVRTC: %v", err)
	}
}

func TestPTXInstructionAttribution(t *testing.T) {
	cubinHex, err := testFS.ReadFile("testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatalf("read kernels.cubin.hex: %v", err)
	}
	cubinBytes, err := hex.DecodeString(strings.TrimSpace(string(cubinHex)))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	cuBytes, err := testFS.ReadFile("testdata/kernels.cu")
	if err != nil {
		t.Fatalf("read kernels.cu: %v", err)
	}
	ptxBytes, err := testFS.ReadFile("testdata/kernels.ptx")
	if err != nil {
		t.Fatalf("read kernels.ptx: %v", err)
	}
	sassBytes, err := testFS.ReadFile("testdata/kernels.sass")
	if err != nil {
		t.Fatalf("read kernels.sass: %v", err)
	}

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    cuBytes,
			"kernels.ptx":   ptxBytes,
			"kernels.cubin": cubinBytes,
			"kernels.sass":  sassBytes,
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel("compute")
	if k == nil {
		t.Fatal("kernel compute not found")
	}
	if len(k.Instructions) != 2 {
		t.Fatalf("got %d instructions, want 2", len(k.Instructions))
	}
	if k.Instructions[0].PTXLine != 12 {
		t.Errorf("inst0.PTXLine = %d, want 12", k.Instructions[0].PTXLine)
	}
	if k.Instructions[1].PTXLine != 14 {
		t.Errorf("inst1.PTXLine = %d, want 14", k.Instructions[1].PTXLine)
	}

	res := cuptipcsampling.NewCaptureResolver(code, 0)
	loc0 := res.ResolvePC(cuptipcsampling.ModuleID{}, "compute", 0)
	if loc0.PTXLine != 12 || loc0.SASSLine != 3 || loc0.SourceLine != 2 {
		t.Errorf("loc0 = %+v, want PTX:12, SASS:3, Cu:2", loc0)
	}
	loc1 := res.ResolvePC(cuptipcsampling.ModuleID{}, "compute", 0x10)
	if loc1.PTXLine != 14 || loc1.SASSLine != 5 || loc1.SourceLine != 3 {
		t.Errorf("loc1 = %+v, want PTX:14, SASS:5, Cu:3", loc1)
	}

	var buf bytes.Buffer
	profSession := cudaprof.NewSession(&buf)
	if err := profSession.SetSampleTypes("samples"); err != nil {
		t.Fatalf("SetSampleTypes: %v", err)
	}
	bridge := &cuptipcsampling.ProfileBridge{
		Session:  profSession,
		Resolver: res,
		Inventory: cuptipcsampling.ReasonInventory{
			CUPTIVersion: 130001,
			Reasons: []cuptipcsampling.StallReason{
				{Index: 0, Name: "selected"},
			},
		},
	}
	batch := cuptipcsampling.Batch{
		Valid: true,
		PCs: []cuptipcsampling.PC{
			{Name: "compute", Offset: 0, Reasons: []cuptipcsampling.StallSamples{{Index: 0, Samples: 10}}},
			{Name: "compute", Offset: 0x10, Reasons: []cuptipcsampling.StallSamples{{Index: 0, Samples: 20}}},
		},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := profSession.Close(); err != nil {
		t.Fatalf("profSession.Close: %v", err)
	}

	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	if len(p.Sample) != 2 {
		t.Fatalf("got %d samples, want 2", len(p.Sample))
	}
	s0 := p.Sample[0]
	if len(s0.Location) != 3 {
		t.Fatalf("got %d locations in sample, want 3 (SASS, PTX, Cu)", len(s0.Location))
	}
	if s0.Location[0].Line[0].Line != 3 || s0.Location[1].Line[0].Line != 12 || s0.Location[2].Line[0].Line != 2 {
		t.Errorf("unexpected lines in s0 stack: SASS=%d, PTX=%d, Cu=%d",
			s0.Location[0].Line[0].Line, s0.Location[1].Line[0].Line, s0.Location[2].Line[0].Line)
	}
	s1 := p.Sample[1]
	if len(s1.Location) != 3 {
		t.Fatalf("got %d locations in sample 1, want 3", len(s1.Location))
	}
	if s1.Location[0].Line[0].Line != 5 || s1.Location[1].Line[0].Line != 14 || s1.Location[2].Line[0].Line != 3 {
		t.Errorf("unexpected lines in s1 stack: SASS=%d, PTX=%d, Cu=%d",
			s1.Location[0].Line[0].Line, s1.Location[1].Line[0].Line, s1.Location[2].Line[0].Line)
	}
}

func TestCFGAttachedInCapture(t *testing.T) {
	cfgBytes, err := testFS.ReadFile("testdata/kernels.cfg.dot")
	if err != nil {
		t.Fatalf("read kernels.cfg.dot: %v", err)
	}
	bbcfgBytes, err := testFS.ReadFile("testdata/kernels.bbcfg.dot")
	if err != nil {
		t.Fatalf("read kernels.bbcfg.dot: %v", err)
	}

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		},
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
			CFG:    "kernels.cfg.dot",
			BBCFG:  "kernels.bbcfg.dot",
		},
		Files: map[string][]byte{
			"kernels.cu":        []byte("void compute() {}\n"),
			"kernels.ptx":       []byte("// PTX\n"),
			"kernels.cubin":     []byte("CUBIN"),
			"kernels.sass":      []byte("// SASS\n"),
			"kernels.cfg.dot":   cfgBytes,
			"kernels.bbcfg.dot": bbcfgBytes,
		},
	}

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, cap); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.Code.CFG != "kernels.cfg.dot" || got.Code.BBCFG != "kernels.bbcfg.dot" {
		t.Fatalf("got Code = %+v", got.Code)
	}
	cfgData := string(got.Files["kernels.cfg.dot"])
	graphs := cudasass.ListDOTGraphs(cfgData)
	if len(graphs) != 1 || graphs[0] != "compute" {
		t.Fatalf("graphs = %v, want [compute]", graphs)
	}
	extracted, err := cudasass.ExtractDOTGraph(cfgData, "compute")
	if err != nil {
		t.Fatalf("ExtractDOTGraph: %v", err)
	}
	if !strings.Contains(extracted, "digraph compute") {
		t.Fatalf("extracted dot = %q, want digraph compute", extracted)
	}
}

func TestTileIRAttachedInCapture(t *testing.T) {
	tileIRBytes, err := sources.ReadFile("kernels.tileir")
	if err != nil {
		t.Fatalf("read embedded kernels.tileir: %v", err)
	}
	remarksBytes, err := sources.ReadFile("tileiras-remarks.yaml")
	if err != nil {
		t.Fatalf("read embedded tileiras-remarks.yaml: %v", err)
	}

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		},
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
			CFG:    "kernels.cfg.dot",
			BBCFG:  "kernels.bbcfg.dot",
			TileIR: "kernels.tileir",
		},
		Files: map[string][]byte{
			"kernels.cu":            []byte("void compute() {}\n"),
			"kernels.ptx":           []byte("// PTX\n"),
			"kernels.cubin":         []byte("CUBIN"),
			"kernels.sass":          []byte("// SASS\n"),
			"kernels.cfg.dot":       []byte("digraph G {}\n"),
			"kernels.bbcfg.dot":     []byte("digraph G {}\n"),
			"kernels.tileir":        tileIRBytes,
			"tileiras-remarks.yaml": remarksBytes,
		},
	}

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, cap); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.Code.TileIR != "kernels.tileir" {
		t.Fatalf("got Code.TileIR = %q, want kernels.tileir", got.Code.TileIR)
	}
	if len(got.Files["kernels.tileir"]) == 0 {
		t.Fatal("expected non-empty kernels.tileir in capture")
	}
	if len(got.Files["tileiras-remarks.yaml"]) == 0 {
		t.Fatal("expected non-empty tileiras-remarks.yaml in capture")
	}
}

// TestLoweringComparison runs the PTX-to-SASS comparison over the captured
// artifacts. The example ships both levels, so the check that they agree is
// available offline: it needs no device, only the two files.
func TestLoweringComparison(t *testing.T) {
	ptxBytes, err := testFS.ReadFile("testdata/kernels.ptx")
	if err != nil {
		t.Fatalf("read kernels.ptx: %v", err)
	}
	sassBytes, err := testFS.ReadFile("testdata/kernels.sass")
	if err != nil {
		t.Fatalf("read kernels.sass: %v", err)
	}
	mod, err := cudaptx.ParseString(string(ptxBytes))
	if err != nil {
		t.Fatalf("parse ptx: %v", err)
	}
	dis, err := cudasass.Parse(bytes.NewReader(sassBytes))
	if err != nil {
		t.Fatalf("parse sass: %v", err)
	}
	rep, err := cudalowering.Compare(mod, dis, nil)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if len(rep.Stats) == 0 {
		t.Fatal("no kernels compared; the comparison would report nothing for any input")
	}
	// Padding is the divergence this example has already been bitten by: a
	// throughput prediction was refuted by ptxas NOP padding that neither the
	// PTX nor a kernel-level timing shows.
	var nops, sass int
	for _, st := range rep.Stats {
		nops += st.NOPs
		sass += st.SASSInstructions
	}
	if sass == 0 {
		t.Fatal("compared 0 SASS instructions")
	}
	t.Logf("%d kernels, %d SASS instructions, %d NOPs, %d divergences", len(rep.Stats), sass, nops, len(rep.Divergences))
	for _, d := range rep.Divergences {
		t.Logf("divergence: %v", d)
	}
}

// TestLoweringDetectsPadding runs the comparison against a kernel that ptxas
// padded, so that a nonzero result is observed somewhere.
//
// TestLoweringComparison covers the wiring on this example's own kernels, and
// they report nothing: their NOPs are all trailing, so Padding is unreachable
// on that input and a Compare that had stopped working would produce the same
// empty report. This fixture is the case where the check has to speak.
func TestLoweringDetectsPadding(t *testing.T) {
	ptxSrc, err := os.ReadFile(filepath.Join("testdata", "padding.ptx"))
	if err != nil {
		t.Fatalf("read ptx: %v", err)
	}
	sassSrc, err := os.ReadFile(filepath.Join("testdata", "padding.sass"))
	if err != nil {
		t.Fatalf("read sass: %v", err)
	}
	mod, err := cudaptx.Parse(bytes.NewReader(ptxSrc))
	if err != nil {
		t.Fatalf("parse ptx: %v", err)
	}
	dis, err := cudasass.Parse(bytes.NewReader(sassSrc))
	if err != nil {
		t.Fatalf("parse sass: %v", err)
	}
	report, err := cudalowering.Compare(mod, dis, nil)
	if err != nil {
		t.Fatalf("compare: %v", err)
	}

	var padding int
	for _, d := range report.Divergences {
		if d.Kind == cudalowering.Padding {
			padding++
		}
	}
	if padding == 0 {
		t.Errorf("no Padding divergence on a kernel with interior NOP padding; divergences = %d", len(report.Divergences))
	}

	// The counts the Padding verdict rests on, so a change in how NOPs are
	// classified is visible here rather than only as a verdict that flips.
	if len(report.Stats) != 1 {
		t.Fatalf("stats for %d kernels, want 1", len(report.Stats))
	}
	st := report.Stats[0]
	if st.SASSInstructions != 1088 || st.NOPs != 187 || st.InteriorNOPs != 172 {
		t.Errorf("stats = %d SASS, %d NOPs, %d interior; want 1088, 187, 172", st.SASSInstructions, st.NOPs, st.InteriorNOPs)
	}
}

// TestProfileFrameFilesAreBundled checks that every file a profile frame names
// is one the capture ships.
//
// A pprof frame's file name is resolved against the bundle by every source
// surface -- list, weblist, the source view -- and a name that is not in it
// resolves to nothing, silently and for every reader forever. The failure is
// invisible in top and in the flame graph, which is why one survived here: an
// NCCL frame named "x/nccl/nccl.go", a file in another module, spelled as a
// path where every other frame is a base name.
//
// The sources FS is the embed list itself, so this test cannot drift from what
// is actually bundled.
//
// It sees string literals only, which is a real limit rather than a detail:
// goFrames builds its frames from trimPath(f.File), computed at run time, so
// most frames in any captured profile are invisible to it. Those resolve today
// only because every frame in the captured stacks happens to be in package
// main -- a property of the current call paths, not of the code. A callback
// routed through another package would produce a plausible base name that this
// test cannot see and no reader could resolve.
func TestProfileFrameFilesAreBundled(t *testing.T) {
	bundled := map[string]bool{}
	entries, err := sources.ReadDir(".")
	if err != nil {
		t.Fatalf("read embedded sources: %v", err)
	}
	for _, e := range entries {
		bundled[e.Name()] = true
	}
	// Written into the bundle at capture time rather than embedded, so they are
	// not in the FS but are present in the directory a reader opens.
	for _, generated := range []string{artifacts.SASS, artifacts.PTX, artifacts.Source, tileIRArtifact} {
		bundled[generated] = true
	}

	// The frames for the retained code are built by x/cudacapture from these
	// names, so they are no longer string literals this file's scan can see.
	// They are the names most likely to be wrong -- kernels.cu is written at
	// capture time rather than embedded -- so they are checked directly.
	for _, name := range []string{artifacts.Source, artifacts.PTX, artifacts.SASS, tileIRArtifact} {
		if name == "" {
			t.Error("a code artifact is unnamed; its frames would be omitted from every profile")
			continue
		}
		if !bundled[name] {
			t.Errorf("code artifact %q is not in the bundle; its frames resolve to nothing", name)
		}
	}

	frameFile := regexp.MustCompile(`FileName:\s*"([^"]*)"`)
	var checked int
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		src, err := sources.ReadFile(e.Name())
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		for _, m := range frameFile.FindAllStringSubmatch(string(src), -1) {
			name := m[1]
			checked++
			if strings.Contains(name, "/") {
				t.Errorf("%s: frame file %q is a path; frames use bundle base names (see trimPath)", e.Name(), name)
				continue
			}
			if !bundled[name] {
				t.Errorf("%s: frame file %q is not in the bundle; a source view resolves it to nothing", e.Name(), name)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no frame file names found; the pattern no longer matches and this test checks nothing")
	}
}
