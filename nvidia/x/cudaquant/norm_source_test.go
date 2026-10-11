package cudaquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type normFixture struct {
	name          string
	rows, cols    int
	epsilon       float32
	weighted      bool
	input, weight []float32
	want          []BlockQ8_1
}

// Expected blocks are analytic: constant rows normalize to 1 or 1/2. The
// weighted fixture has scale exactly 1 and includes positive/negative ties.
func normFixtures() []normFixture {
	var fixtures []normFixture
	for _, name := range []string{"unweighted", "epsilon", "weighted_ties", "zero"} {
		f := normFixture{name: name, rows: 2, cols: 64, input: make([]float32, 128), weight: make([]float32, 64), want: make([]BlockQ8_1, 4)}
		for i := range f.input {
			f.input[i] = 1
		}
		for i := range f.weight {
			f.weight[i] = []float32{127, 0.5, -0.5, 1.5, -1.5, 0, 0, 0}[i%8]
		}
		switch name {
		case "unweighted", "epsilon":
			d, s := uint16(0x2008), uint16(0x5000)
			if name == "epsilon" {
				f.epsilon = 3
				d = 0x1c08
				s = 0x4c00
			}
			for b := range f.want {
				f.want[b].D = d
				f.want[b].S = s
				for i := range f.want[b].Quants {
					f.want[b].Quants[i] = 127
				}
			}
		case "weighted_ties":
			f.weighted = true
			for b := range f.want {
				f.want[b].D = 0x3c00
				f.want[b].S = 0x5ff0
				for i := range f.want[b].Quants {
					f.want[b].Quants[i] = []int8{127, 1, -1, 2, -2, 0, 0, 0}[i%8]
				}
			}
		case "zero":
			f.epsilon = 1
			clear(f.input)
		}
		fixtures = append(fixtures, f)
	}
	return fixtures
}

func normBytes(blocks []BlockQ8_1) []byte {
	out := make([]byte, len(blocks)*36)
	for b, block := range blocks {
		binary.LittleEndian.PutUint16(out[b*36:], block.D)
		binary.LittleEndian.PutUint16(out[b*36+2:], block.S)
		for i, q := range block.Quants {
			out[b*36+4+i] = byte(q)
		}
	}
	return out
}

func TestNormAnalyticFixtures(t *testing.T) {
	for _, f := range normFixtures() {
		t.Run(f.name, func(t *testing.T) {
			got, err := RMSNormQ8_1(f.input, f.weight, f.rows, f.cols, f.epsilon, f.weighted)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(normBytes(got), normBytes(f.want)) {
				t.Fatalf("got=%+v want=%+v", got, f.want)
			}
		})
	}
}

// This executes the emitted scalar kernel on the host with IEEE _Float16
// conversion. It does not qualify NVRTC compilation or GPU arithmetic.
func TestNormEmittedNumerics(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++")
	}
	for _, tc := range []struct {
		name, old, replacement string
		fail                   bool
		unfused                bool
	}{
		{name: "source"},
		{name: "unfused", unfused: true},
		{name: "epsilon_removed", old: "(double)eps", replacement: "0.0", fail: true},
		{name: "rounding_changed", old: "roundf(values[i]/d)", replacement: "truncf(values[i]/d)", fail: true},
		{name: "weight_removed", old: "if (weighted)", replacement: "if (false)", fail: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			// The host shim supplies declarations NVRTC provides implicitly.
			header := `#include <stddef.h>
#include <math.h>
#define __global__
using __half = _Float16;
static_assert(sizeof(__half)==2,"half size");
inline __half __float2half_rn(float f){return (__half)f;}
struct dim {unsigned x;};
extern dim blockIdx,threadIdx;
`
			if err := os.WriteFile(filepath.Join(dir, "cuda_fp16.h"), []byte(header), 0600); err != nil {
				t.Fatal(err)
			}
			src := RMSNormQ8_1Source()
			if tc.unfused {
				src = RMSNormQ8_1UnfusedSource()
			}
			if tc.old != "" {
				if !strings.Contains(src, tc.old) {
					t.Fatal("missing mutation target")
				}
				src = strings.ReplaceAll(src, tc.old, tc.replacement)
			}
			var main strings.Builder
			main.WriteString("\n#include <cstdio>\n#include <cstring>\ndim blockIdx,threadIdx;\nint main(){\n")
			var want []byte
			for _, f := range normFixtures() {
				main.WriteString("{\nfloat input[]={")
				for _, v := range f.input {
					fmt.Fprintf(&main, "%g,", v)
				}
				main.WriteString("};\nfloat weight[]={")
				for _, v := range f.weight {
					fmt.Fprintf(&main, "%g,", v)
				}
				fmt.Fprintf(&main, "};\nQ8_1 out[%d]; memset(out,0xa5,sizeof(out));\n", len(f.want)+2)
				weighted := 0
				if f.weighted {
					weighted = 1
				}
				candidates, err := RMSNormQ8_1Candidates(f.rows, f.cols, f.weighted, tc.unfused)
				if err != nil {
					t.Fatal(err)
				}
				candidate := candidates[len(candidates)-1]
				if tc.unfused {
					fmt.Fprintf(&main, "float norm[%d]; for(float &v:norm)v=-999;\n", len(f.input)+2)
				}
				pointers := map[string]string{"input": "input", "norm": "norm+1", "output": "out+1"}
				for _, launch := range candidate.Launches {
					fmt.Fprintf(&main, "for(unsigned r=0;r<%d;r++)for(unsigned lane=0;lane<%d;lane++){blockIdx.x=r;threadIdx.x=lane;%s(%s,weight,%s,%d,%d,%g,%d);}\n", launch.Grid[0]+1, launch.Block[0], launch.Entry, pointers[launch.Input], pointers[launch.Output], f.rows, f.cols, f.epsilon, weighted)
				}
				if tc.unfused {
					fmt.Fprintf(&main, "if(norm[0]!=-999 || norm[%d]!=-999)return 3;\n", len(f.input)+1)
					for i := range f.input {
						want := float32(1)
						switch f.name {
						case "epsilon":
							want = 0.5
						case "zero":
							want = 0
						case "weighted_ties":
							want = f.weight[i%f.cols]
						}
						fmt.Fprintf(&main, "if(norm[%d]!=%g)return 4;\n", i+1, want)
					}
				}
				main.WriteString("for(unsigned j=0;j<sizeof(Q8_1);j++)if(((unsigned char*)out)[j]!=0xa5 || ((unsigned char*)&out[sizeof(out)/sizeof(Q8_1)-1])[j]!=0xa5)return 2;\n")
				fmt.Fprintf(&main, "fwrite(out+1,36,%d,stdout);\n}\n", len(f.want))
				want = append(want, normBytes(f.want)...)
			}
			main.WriteString("return 0;}\n")
			path := filepath.Join(dir, "norm.cc")
			if err := os.WriteFile(path, []byte(src+main.String()), 0600); err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(dir, "norm")
			if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-ffp-contract=off", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v\n%s", err, out)
			}
			got, err := exec.Command(binary).Output()
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			matches := bytes.Equal(got, want)
			if matches == tc.fail {
				t.Fatalf("output match=%v want failure=%v got=%x want=%x", matches, tc.fail, got, want)
			}
		})
	}
}
