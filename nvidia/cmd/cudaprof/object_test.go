package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/google/pprof/driver"
	"github.com/google/pprof/profile"
)

func TestObjectCoordinates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kernels.cubin")
	image := &objectImage{file: path, functions: []objectFunction{{name: "kernel.SASS", start: 0x100, end: 0x120, instructions: []driver.Inst{
		{Addr: 0x100, Text: "FADD R0, R1, R2;", Function: "kernel.SASS", File: "kernel.sass", Line: 20},
		{Addr: 0x110, Text: "EXIT;", Function: "kernel.SASS", File: "kernel.sass", Line: 21},
	}}}}
	tool := &objectTool{images: map[string]*objectImage{path: image}}
	f, err := tool.Open(path, 0x1000, 0x2000, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name        string
		input, want uint64
		wantErr     bool
	}{{"base", 0x1000, 0, false}, {"instruction", 0x1110, 0x110, false}, {"below", 0xfff, 0, true}, {"limit", 0x2000, 0, true}} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := f.ObjAddr(tt.input)
			if got != tt.want || (err != nil) != tt.wantErr {
				t.Fatalf("ObjAddr = %#x, %v", got, err)
			}
		})
	}
	inst, err := tool.Disasm(path, 0x110, 0x120, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(inst) != 1 || inst[0].Addr != 0x110 || inst[0].Text != "EXIT;" {
		t.Fatalf("disasm = %+v", inst)
	}
	frames, err := f.SourceLine(0x111f)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 1 || frames[0].Line != 21 || frames[0].Func != "kernel.SASS" {
		t.Fatalf("source = %+v", frames)
	}
	if _, err := f.SourceLine(0x1200); err == nil {
		t.Fatal("resolved an address outside the code")
	}
	syms, err := f.Symbols(regexp.MustCompile("kernel"), 0x11f)
	if err != nil {
		t.Fatal(err)
	}
	if len(syms) != 1 || syms[0].Start != 0x100 || syms[0].End != 0x11f {
		t.Fatalf("symbols = %+v", syms)
	}
	syms, err = f.Symbols(nil, 0x120)
	if err != nil || len(syms) != 0 {
		t.Fatalf("symbol past boundary = %+v, %v", syms, err)
	}
	if _, err := tool.Disasm(filepath.Join(dir, "other"), 0, 32, false); err == nil {
		t.Fatal("accepted another object")
	}
}

func TestPTXCoordinates(t *testing.T) {
	const text = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry kernel()\n{\n\tfma.rn.f32 %f0, %f1, %f2, %f3;\n\tret;\n}\n"
	path := filepath.Join(t.TempDir(), "kernels.ptx")
	image, err := ptxImage(path, "ptx-build", text)
	if err != nil {
		t.Fatal(err)
	}
	tool := &objectTool{images: map[string]*objectImage{path: image}, ptx: image}
	inst, err := tool.Disasm(path, 0, uint64(len(text)), false)
	if err != nil || len(inst) != 2 {
		t.Fatalf("Disasm = %+v, %v", inst, err)
	}
	if inst[0].Addr != uint64(strings.Index(text, "\tfma")) || inst[0].Line != 6 || inst[0].Text != "fma.rn.f32 %f0, %f1, %f2, %f3;" {
		t.Fatalf("PTX instruction = %+v", inst[0])
	}
	fn := &profile.Function{ID: 1, Name: "kernel.PTX", Filename: "kernels.ptx"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 6}}}
	p := &profile.Profile{Function: []*profile.Function{fn}, Location: []*profile.Location{loc}, Sample: []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{123}}}}
	tool.mapPTX(p)
	tool.mapPTX(p)
	if len(p.Mapping) != 1 || loc.Address != ptxBase+inst[0].Addr || p.Sample[0].Value[0] != 123 {
		t.Fatalf("PTX mapping changed samples or coordinates: %+v", p)
	}
	f, err := tool.Open(path, ptxBase, ptxBase+uint64(len(text)), 0, "")
	if err != nil {
		t.Fatal(err)
	}
	frames, err := f.SourceLine(loc.Address)
	if err != nil || len(frames) != 1 || frames[0].Func != "kernel.PTX" || frames[0].Line != 6 {
		t.Fatalf("PTX source = %+v, %v", frames, err)
	}
	if _, err := ptxImage(path, "", strings.TrimSuffix(text, "}\n")); err == nil {
		t.Fatal("accepted incomplete PTX function")
	}
}

func TestTileIRCoordinates(t *testing.T) {
	const text = `
cuda_tile.module @kernels {
  cuda_tile.entry @matmul(%arg0: !cuda_tile.tile<ptr<f32>>) {
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<16x16xf32>
    cuda_tile.return
  }
}
`
	path := filepath.Join(t.TempDir(), "kernels.tileir")
	image, err := tileIRImage(path, "tileir-build", []byte(text))
	if err != nil {
		t.Fatal(err)
	}
	tool := &objectTool{images: map[string]*objectImage{path: image}, tileir: image}
	inst, err := tool.Disasm(path, 0, 100, false)
	if err != nil || len(inst) != 2 {
		t.Fatalf("Disasm = %+v, %v", inst, err)
	}
	if !strings.Contains(inst[0].Text, "cuda_tile.constant") {
		t.Fatalf("unexpected instruction text: %s", inst[0].Text)
	}

	fn := &profile.Function{ID: 1, Name: "matmul.TILEIR", Filename: "kernels.tileir"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: int64(inst[0].Line)}}}
	p := &profile.Profile{Function: []*profile.Function{fn}, Location: []*profile.Location{loc}, Sample: []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{456}}}}
	tool.mapTileIR(p)
	tool.mapTileIR(p)
	if len(p.Mapping) != 1 || loc.Address != tileirBase+inst[0].Addr || p.Sample[0].Value[0] != 456 {
		t.Fatalf("TileIR mapping changed samples or coordinates: %+v", p)
	}

	f, err := tool.Open(path, tileirBase, tileirBase+uint64(len(text)), 0, "")
	if err != nil {
		t.Fatal(err)
	}
	frames, err := f.SourceLine(loc.Address)
	if err != nil || len(frames) != 1 || frames[0].Func != "matmul.TILEIR" {
		t.Fatalf("TileIR source = %+v, %v", frames, err)
	}
}

func TestTileIRBytecodeCoordinates(t *testing.T) {
	goldenPath := "/Users/tmc/go/src/github.com/NVIDIA/cuda-tile/test/Bytecode/encoding_stability/goldens/bf16_splat.golden.tileirbc"
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skip("golden file not found, skipping")
	}

	path := filepath.Join(t.TempDir(), "kernels.tilebc")
	image, err := tileIRImage(path, "tilebc-build", data)
	if err != nil {
		t.Fatal(err)
	}
	if len(image.functions) != 1 {
		t.Fatalf("got %d functions, want 1", len(image.functions))
	}
	if image.functions[0].name != "bf16_splat.TILEIR" {
		t.Errorf("function name = %s, want bf16_splat.TILEIR", image.functions[0].name)
	}
}
