package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
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
