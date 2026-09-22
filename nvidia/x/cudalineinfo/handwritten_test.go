package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// handWrittenSource is a hand-written CUDA file holding two kernels. Nothing
// was lowered into it, so it has no line map and its own file is the
// program: the lines a profile reports are lines of this text.
const handWrittenSource = `#include <cuda_fp16.h>

__global__ void geglu(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = x[i];
  y[i] = g * 0.5f * (1.f + tanhf(g));
}

__global__ void rmsnorm(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = v * rsqrtf(v * v + 1e-6f);
}`

// handWrittenSASS is the disassembly of handWrittenSource: one section per
// kernel, with the file name the compiler recorded for both.
const handWrittenSASS = `
//--------------------- .text.geglu          --------------------------
	.section	.text.geglu,"ax",@progbits
        .global         geglu
        .type           geglu,@function
geglu:
.text.geglu:
	//## File "forward.cu", line 4
        /*0000*/                   S2R R0, SR_TID.X ;
	//## File "forward.cu", line 6
        /*0010*/                   LDG.E R2, desc[UR4][R2.64] ;
	//## File "forward.cu", line 7
        /*0020*/                   FMUL R3, R2, 0.5 ;
        /*0030*/                   EXIT ;

//--------------------- .text.rmsnorm          --------------------------
	.section	.text.rmsnorm,"ax",@progbits
        .global         rmsnorm
        .type           rmsnorm,@function
rmsnorm:
.text.rmsnorm:
	//## File "forward.cu", line 13
        /*0000*/                   LDG.E R2, desc[UR4][R2.64] ;
	//## File "forward.cu", line 14
        /*0010*/                   MUFU.RSQ R3, R2 ;
        /*0020*/                   EXIT ;
`

func handWrittenDisassembly(t *testing.T) *cudasass.Disassembly {
	t.Helper()
	dis, err := cudasass.ParseString(handWrittenSASS)
	if err != nil {
		t.Fatal(err)
	}
	return dis
}

// TestAttributeHandWritten attributes a hand-written source with two
// kernels. A unit that carries source and no line map has no generated tier
// below it, so its own file is the program, and Kernels says which
// functions of the disassembly are its own.
func TestAttributeHandWritten(t *testing.T) {
	dis := handWrittenDisassembly(t)
	for _, tt := range []struct {
		name       string
		unit       Unit
		attributed int
		subs       map[string]int
	}{
		{
			name:       "both kernels",
			unit:       Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu", "rmsnorm"}},
			attributed: 7,
		},
		{
			name:       "one of the two",
			unit:       Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu"}},
			attributed: 4,
			subs:       map[string]int{"rmsnorm": 3},
		},
		{
			// Without Kernels the unit's name is its one kernel's name,
			// and no function of a multi-kernel module matches it.
			name:       "named but no kernels",
			unit:       Unit{Name: "forward", Source: handWrittenSource},
			attributed: 0,
			subs:       map[string]int{"geglu": 4, "rmsnorm": 3},
		},
		{
			// The unit knows nothing but a name, so it may not claim its
			// file is the program: this is what R4 alone would leave.
			name:       "kernels but no source",
			unit:       Unit{Name: "forward", Kernels: []string{"geglu", "rmsnorm"}},
			attributed: 0,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := Attribute(tt.unit, "", dis)
			if a.Instructions != 7 {
				t.Fatalf("instructions = %d, want 7", a.Instructions)
			}
			if a.Attributed != tt.attributed {
				t.Errorf("attributed %d of %d, want %d", a.Attributed, a.Instructions, tt.attributed)
			}
			if a.Remainder.Total() != a.Instructions-a.Attributed {
				t.Errorf("remainder totals %d, want %d", a.Remainder.Total(), a.Instructions-a.Attributed)
			}
			if len(a.Remainder.Subroutines) != len(tt.subs) {
				t.Errorf("subroutines = %v, want %v", a.Remainder.Subroutines, tt.subs)
			}
			for name, n := range tt.subs {
				if a.Remainder.Subroutines[name] != n {
					t.Errorf("subroutine %s = %d, want %d", name, a.Remainder.Subroutines[name], n)
				}
			}
		})
	}
}

// TestAttributeHandWrittenSites checks the sites, not only the counts: a
// hand-written unit reports lines of its own file.
func TestAttributeHandWrittenSites(t *testing.T) {
	a := Attribute(Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu", "rmsnorm"}}, "", handWrittenDisassembly(t))
	want := []Site{{"forward.cu", 4}, {"forward.cu", 6}, {"forward.cu", 7}, {"forward.cu", 13}, {"forward.cu", 14}}
	got := a.Sites()
	if len(got) != len(want) {
		t.Fatalf("sites = %v, want %v", got, want)
	}
	for i, s := range want {
		if got[i] != s {
			t.Errorf("site %d = %v, want %v", i, got[i], s)
		}
	}
	if ops := a.SASS[Site{"forward.cu", 7}]; len(ops) != 2 || ops[0].Opcode != "FMUL" {
		t.Errorf("forward.cu:7 = %v, want FMUL and EXIT", ops)
	}
}

// TestAttributeKernelNamedFile is the GeGLU case: a single-kernel source
// named after its kernel, reported through a unit that holds only that
// name. The unit's file and its kernel are then the same word, and the
// exclusion that keeps generated text out of program attribution swallows
// every instruction. Giving the unit its source is what says the file is
// the program.
func TestAttributeKernelNamedFile(t *testing.T) {
	const sass = `
	.section	.text.GeGLU,"ax",@progbits
        .global         GeGLU
        .type           GeGLU,@function
GeGLU:
.text.GeGLU:
	//## File "GeGLU.cu", line 3
        /*0000*/                   S2R R0, SR_TID.X ;
	//## File "GeGLU.cu", line 5
        /*0010*/                   FMUL R3, R2, 0.5 ;
        /*0020*/                   EXIT ;
`
	dis, err := cudasass.ParseString(sass)
	if err != nil {
		t.Fatal(err)
	}
	bare := Attribute(Unit{Name: "GeGLU"}, "", dis)
	if bare.Attributed != 0 || bare.Remainder.Generated != bare.Instructions {
		t.Errorf("bare unit attributed %d of %d, %d generated; want 0 attributed and all generated",
			bare.Attributed, bare.Instructions, bare.Remainder.Generated)
	}
	// The bare unit is right to refuse the file it cannot vouch for, so the
	// report must say the table below it is empty for want of a unit.
	if !bare.Unattributed() {
		t.Error("an attribution that reached no program line does not report it")
	}
	var buf bytes.Buffer
	if err := WriteTable(&buf, bare); err != nil {
		t.Fatal(err)
	}
	if want := `nothing attributed: none of the 3 instructions names a program line; the unit is bare, holding only the name "GeGLU"`; !strings.Contains(buf.String(), want) {
		t.Errorf("table lacks %q:\n%s", want, buf.String())
	}

	buf.Reset()
	if err := WriteProfile(&buf, bare, Profile{Measure: "samples", Sites: map[Site]int64{}, Other: 5}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "nothing attributed") {
		t.Errorf("profile does not say the join failed:\n%s", buf.String())
	}

	full := Attribute(Unit{Name: "GeGLU", Source: "__global__ void GeGLU() {\n}\n"}, "", dis)
	if full.Attributed != full.Instructions {
		t.Errorf("unit with source attributed %d of %d, want all", full.Attributed, full.Instructions)
	}
	if full.Unattributed() {
		t.Error("an attribution that reached every line reports nothing attributed")
	}
	buf.Reset()
	if err := WriteTable(&buf, full); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), "nothing attributed") {
		t.Errorf("a populated table says nothing was attributed:\n%s", buf.String())
	}
}

// TestProgramFileHandWritten is the discrimination R1 rests on. A unit
// holding source and no line map speaks for its own file; a bare unit knows
// only a name and does not.
func TestProgramFileHandWritten(t *testing.T) {
	for _, tt := range []struct {
		name string
		unit Unit
		want bool
	}{
		{"bare", Unit{Name: "k"}, false},
		{"hand-written", Unit{Name: "k", Source: "x"}, true},
		{"generated", Unit{Name: "k", Source: "x", LineMap: map[int]cudair.SourceLocation{1: {File: "m.go", Line: 3}}}, false},
	} {
		for _, file := range []string{"k.cu", "/abs/k.cu"} {
			if got := tt.unit.ProgramFile(file); got != tt.want {
				t.Errorf("%s: ProgramFile(%q) = %v, want %v", tt.name, file, got, tt.want)
			}
		}
		if !tt.unit.ProgramFile("kernels.go") {
			t.Errorf("%s: a file that is not the unit's own is not the program", tt.name)
		}
		if tt.unit.ProgramFile("cuda_fp16.hpp") {
			t.Errorf("%s: a header is the program", tt.name)
		}
	}
}

// TestUnitKernel checks which functions a unit claims.
func TestUnitKernel(t *testing.T) {
	for _, tt := range []struct {
		unit Unit
		yes  []string
		no   []string
	}{
		{Unit{Name: "k"}, []string{"k"}, []string{"other", ""}},
		{Unit{}, []string{"k", "other", ""}, nil},
		{Unit{Name: "forward", Kernels: []string{"geglu", "rmsnorm"}}, []string{"geglu", "rmsnorm"}, []string{"forward", "other"}},
	} {
		for _, name := range tt.yes {
			if !tt.unit.Kernel(name) {
				t.Errorf("%+v: Kernel(%q) = false", tt.unit, name)
			}
		}
		for _, name := range tt.no {
			if tt.unit.Kernel(name) {
				t.Errorf("%+v: Kernel(%q) = true", tt.unit, name)
			}
		}
	}
}

func TestHandWrittenSASSParses(t *testing.T) {
	dis := handWrittenDisassembly(t)
	var names []string
	for _, fn := range dis.Functions {
		names = append(names, fn.Name)
	}
	if strings.Join(names, " ") != "geglu rmsnorm" {
		t.Fatalf("functions = %v, want geglu rmsnorm", names)
	}
}

// TestReportSourceFromUnit reads the SOURCE column out of the unit. The
// fixture's forward.cu is not on disk anywhere, which is the ordinary case
// for a source compiled through NVRTC from an embedded string.
func TestReportSourceFromUnit(t *testing.T) {
	a := Attribute(Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu", "rmsnorm"}}, "", handWrittenDisassembly(t))
	var buf bytes.Buffer
	if err := WriteTable(&buf, a); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"float g = x[i];",
		"y[i] = v * rsqrtf(v * v + 1e-6f);",
	} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("table lacks %q:\n%s", want, buf.String())
		}
	}
	if readLines("forward.cu") != nil {
		t.Fatal("forward.cu is on disk, so the table may have read it there")
	}
}

// TestOptionsName checks the file name a unit is compiled under. A unit
// named for something other than its file needs the file name stated,
// since it reaches the line table and is what every tier then reports.
func TestOptionsName(t *testing.T) {
	u := Unit{Name: "forward"}
	if got := (Options{}).name(u); got != "forward.cu" {
		t.Errorf("default name = %q, want forward.cu", got)
	}
	if got := (Options{Name: "laya_forward.cu"}).name(u); got != "laya_forward.cu" {
		t.Errorf("name = %q, want laya_forward.cu", got)
	}
}
