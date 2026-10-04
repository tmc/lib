package cudalayer

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func backwardFixture(m *FusedMLP, n int) (x, w, dy []float32) {
	x = make([]float32, n*m.inputWidth)
	w = make([]float32, m.ParameterCount())
	dy = make([]float32, n*m.outputWidth)
	for i := range x {
		x[i] = float32((i*7)%19-9) / 8
	}
	for i := range w {
		w[i] = float32((i*11)%23-11) / 32
	}
	for i := range dy {
		dy[i] = float32((i*3)%17-8) / 16
	}
	return
}

// This reference is independent float64 evaluation and chain-rule arithmetic.
// It does not read the emitted source or use saved device context offsets.
func backwardReference(m *FusedMLP, x, w, dy []float64, n int) (y, dx, dw []float64) {
	y = make([]float64, n*m.outputWidth)
	dx = make([]float64, len(x))
	dw = make([]float64, len(w))
	for sample := 0; sample < n; sample++ {
		values := [][]float64{append([]float64(nil), x[sample*m.inputWidth:(sample+1)*m.inputWidth]...)}
		zs := make([][]float64, len(m.layers))
		offsets := make([]int, len(m.layers))
		offset := 0
		for k, l := range m.layers {
			offsets[k] = offset
			z := make([]float64, l.outputWidth)
			out := make([]float64, l.outputWidth)
			for j := range z {
				for i, v := range values[k] {
					z[j] += v * w[offset+i*l.outputWidth+j]
				}
				out[j] = z[j]
				switch l.activation {
				case "relu":
					out[j] = math.Max(0, z[j])
				case "silu":
					out[j] = z[j] / (1 + math.Exp(-z[j]))
				}
			}
			zs[k] = z
			values = append(values, out)
			offset += l.inputWidth * l.outputWidth
		}
		copy(y[sample*m.outputWidth:], values[len(m.layers)])
		grad := append([]float64(nil), dy[sample*m.outputWidth:(sample+1)*m.outputWidth]...)
		for i := range grad {
			grad[i] /= float64(n)
		}
		for k := len(m.layers) - 1; k >= 0; k-- {
			l := m.layers[k]
			for j := range grad {
				switch l.activation {
				case "relu":
					if zs[k][j] <= 0 {
						grad[j] = 0
					}
				case "silu":
					s := 1 / (1 + math.Exp(-zs[k][j]))
					grad[j] *= s + zs[k][j]*s*(1-s)
				}
			}
			inGrad := make([]float64, l.inputWidth)
			for i := range inGrad {
				for j, g := range grad {
					index := offsets[k] + i*l.outputWidth + j
					dw[index] += values[k][i] * g
					inGrad[i] += w[index] * g
				}
			}
			grad = inGrad
		}
		copy(dx[sample*m.inputWidth:], grad)
	}
	return
}

func float64s(v []float32) []float64 {
	r := make([]float64, len(v))
	for i := range v {
		r[i] = float64(v[i])
	}
	return r
}
func compareBackward(t *testing.T, name string, got []float32, want []float64) bool {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want%d", name, len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[i]) > 2e-5*(1+math.Abs(want[i])) {
			t.Logf("%s[%d]=%.9g want%.9g", name, i, v, want[i])
			return false
		}
	}
	return true
}

// Execute the actual generated CUDA bodies as host C++, replacing only the
// CUDA qualifiers and thread coordinates. This is host execution, not GPU proof.
func runBackwardHost(t *testing.T, m *FusedMLP, x, w, dy, previous []float32, o MLPBackwardOptions, mutate func(string) string, rounds int) MLPGradients {
	t.Helper()
	compiler := ""
	for _, c := range []string{os.Getenv("CXX"), "clang++", "c++", "g++"} {
		if c != "" {
			if p, e := exec.LookPath(c); e == nil {
				compiler = p
				break
			}
		}
	}
	if compiler == "" {
		t.Skip("generated backward host execution requires C++ compiler; native remains pending")
	}
	l, e := m.backwardLayout(x, w, dy, previous, o)
	if e != nil {
		t.Fatal(e)
	}
	source, _, e := m.backwardSource(o)
	if e != nil {
		t.Fatal(e)
	}
	if mutate != nil {
		changed := mutate(source)
		if changed == source {
			t.Fatal("mutant did not change emitted source")
		}
		source = changed
	}
	var main strings.Builder
	main.WriteString("\nint main(){\n")
	array := func(name string, v []float32) {
		fmt.Fprintf(&main, "float %s[]={", name)
		if len(v) == 0 {
			main.WriteString("0.0f")
		}
		for i, f := range v {
			if i > 0 {
				main.WriteString(",")
			}
			fmt.Fprintf(&main, "%.9ef", f)
		}
		main.WriteString("};\n")
	}
	array("x", x)
	array("w", w)
	array("dy", dy)
	init := make([]float32, l.parameters)
	copy(init, previous)
	array("dw", init)
	fmt.Fprintf(&main, "float y[%d]={}; float dx[%d]={}; float scratch[%d]={}; unsigned char context[%d]; memset(context,0xa5,sizeof(context));\n", l.output, l.input, max(1, l.scratch), l.context+64)
	fmt.Fprintf(&main, "for(int round=0;round<%d;++round){ for(int idx=0;idx<%d;++idx){blockIdx.x=idx/32;threadIdx.x=idx%%32;mlp_batch(%d,x,w,dy,y,dx,scratch,context+32);}\n", rounds, o.Samples, o.Samples)
	fmt.Fprintf(&main, "for(int idx=0;idx<%d;++idx){blockIdx.x=idx/32;threadIdx.x=idx%%32;mlp_backward_batch(%d,x,w,dy,y,dx,scratch,context+32);}\n", o.Samples, o.Samples)
	if l.scratch != 0 {
		fmt.Fprintf(&main, "for(int k=0;k<%d;++k){blockIdx.x=k/32;threadIdx.x=k%%32;mlp_reduce(%d,scratch,dw);}\n", l.parameters, o.Samples)
	}
	main.WriteString("}\nfor(int i=0;i<32;++i)if(context[i]!=0xa5||context[sizeof(context)-32+i]!=0xa5)return 3;\n")
	fmt.Fprintf(&main, "fwrite(y,4,%d,stdout);", l.output)
	if o.Input {
		fmt.Fprintf(&main, "fwrite(dx,4,%d,stdout);", l.input)
	}
	if l.scratch != 0 {
		fmt.Fprintf(&main, "fwrite(dw,4,%d,stdout);", l.parameters)
	}
	main.WriteString("return 0;}\n")
	preamble := `#include <cmath>
#include <cstdio>
#include <cstring>
#define __device__
#define __global__
#define __expf expf
struct Coordinate { int x; } blockIdx,threadIdx,blockDim{32};
`
	dir := t.TempDir()
	file := filepath.Join(dir, "backward.cc")
	binaryPath := filepath.Join(dir, "backward")
	if e := os.WriteFile(file, []byte(preamble+source+main.String()), 0600); e != nil {
		t.Fatal(e)
	}
	if out, e := exec.Command(compiler, "-std=c++17", "-O0", file, "-o", binaryPath).CombinedOutput(); e != nil {
		t.Fatalf("compile generated bodies: %v\n%s", e, out)
	}
	out, e := exec.Command(binaryPath).Output()
	if e != nil {
		t.Fatalf("run generated bodies: %v", e)
	}
	r := MLPGradients{Output: make([]float32, l.output)}
	if o.Input {
		r.Input = make([]float32, l.input)
	}
	if l.scratch != 0 {
		r.Parameters = make([]float32, l.parameters)
	}
	reader := bytes.NewReader(out)
	for _, v := range [][]float32{r.Output, r.Input, r.Parameters} {
		if e := binary.Read(reader, binary.LittleEndian, v); e != nil {
			t.Fatal(e)
		}
	}
	if reader.Len() != 0 {
		t.Fatal("unexpected emitted output bytes")
	}
	return r
}

func TestMLPBackward_ReluAndSiluDerivatives(t *testing.T) {
	for _, act := range []string{"none", "relu", "silu"} {
		t.Run(act, func(t *testing.T) {
			l, e := NewMLPLayer("activation", 1, 1, act)
			if e != nil {
				t.Fatal(e)
			}
			m := &FusedMLP{name: "activation", inputWidth: 1, outputWidth: 1, layers: []*MLPLayer{l}}
			x := []float32{-30, -2, -0.01, 0, 0.01, 2, 30}
			w := []float32{1}
			dy := []float32{1, 1, 1, 1, 1, 1, 1}
			o := MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 1 << 20}
			got := runBackwardHost(t, m, x, w, dy, nil, o, nil, 1)
			y, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
			if !compareBackward(t, "output", got.Output, y) || !compareBackward(t, "input", got.Input, dx) || !compareBackward(t, "parameters", got.Parameters, dw) {
				t.Fatal("activation derivative disagrees")
			}
			if act == "relu" && got.Input[3] != 0 {
				t.Fatal("ReLU derivative at zero must be zero")
			}
		})
	}
}

func TestMLPBackward_GeneratedGradientsAndFiniteDifferences(t *testing.T) {
	for _, dims := range [][4]int{{5, 3, 3, 1}, {5, 7, 3, 2}, {64, 64, 64, 3}} {
		t.Run(fmt.Sprint(dims), func(t *testing.T) {
			m, e := NewFusedMLP(dims[0], dims[1], dims[2], dims[3])
			if e != nil {
				t.Fatal(e)
			}
			x, w, dy := backwardFixture(m, 7)
			o := MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 64 << 20}
			got := runBackwardHost(t, m, x, w, dy, nil, o, nil, 1)
			y, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
			if !compareBackward(t, "output", got.Output, y) || !compareBackward(t, "input", got.Input, dx) || !compareBackward(t, "parameters", got.Parameters, dw) {
				t.Fatal("generated derivatives disagree with reference")
			}
			if dims[0] != 64 {
				checkFiniteDifference(t, m, x, w, dy, 7, dx, dw)
			}
		})
	}
}

func TestMLPBackward_SiluRectangularFiniteDifferences(t *testing.T) {
	m, e := NewFusedMLPWithActivation(5, 7, 3, 2, "silu")
	if e != nil {
		t.Fatal(e)
	}
	x, w, dy := backwardFixture(m, 7)
	o := MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 1 << 20}
	got := runBackwardHost(t, m, x, w, dy, nil, o, nil, 1)
	y, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
	if !compareBackward(t, "output", got.Output, y) || !compareBackward(t, "input", got.Input, dx) || !compareBackward(t, "parameters", got.Parameters, dw) {
		t.Fatal("SiLU chain disagreement")
	}
	checkFiniteDifference(t, m, x, w, dy, 7, dx, dw)
}

func checkFiniteDifference(t *testing.T, m *FusedMLP, x, w, dy []float32, n int, dx, dw []float64) {
	t.Helper()
	xx, ww, yy := float64s(x), float64s(w), float64s(dy)
	loss := func() float64 {
		y, _, _ := backwardReference(m, xx, ww, yy, n)
		sum := 0.0
		for i, v := range y {
			sum += v * yy[i] / float64(n)
		}
		return sum
	}
	const h = 1e-5
	for _, check := range []struct {
		name         string
		values, want []float64
	}{{"input", xx, dx}, {"parameters", ww, dw}} {
		for i := range check.values {
			old := check.values[i]
			check.values[i] = old + h
			high := loss()
			check.values[i] = old - h
			low := loss()
			check.values[i] = old
			finite := (high - low) / (2 * h)
			if math.Abs(finite-check.want[i]) > 1e-7 {
				t.Fatalf("finite difference %s[%d]=%.9g analytic%.9g", check.name, i, finite, check.want[i])
			}
		}
	}
}

func TestMLPBackward_AccumulationModes(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	x, w, dy := backwardFixture(m, 7)
	_, wantInput, wantParams := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
	for _, mode := range []cudair.GradientMode{cudair.GradientModeIgnore, cudair.GradientModeOverwrite, cudair.GradientModeAccumulate} {
		for _, input := range []bool{false, true} {
			if !input && mode == cudair.GradientModeIgnore {
				continue
			}
			t.Run(fmt.Sprintf("%s/input=%t", mode, input), func(t *testing.T) {
				o := MLPBackwardOptions{Samples: 7, Input: input, ParameterMode: mode, MaxBytes: 1 << 20}
				var previous []float32
				if mode == cudair.GradientModeAccumulate {
					previous = make([]float32, len(w))
					for i := range previous {
						previous[i] = float32(i%7-3) / 8
					}
				}
				got := runBackwardHost(t, m, x, w, dy, previous, o, nil, 2)
				if input && !compareBackward(t, "input", got.Input, wantInput) {
					t.Fatal("input mismatch")
				}
				if mode != cudair.GradientModeIgnore {
					want := append([]float64(nil), wantParams...)
					if mode == cudair.GradientModeAccumulate {
						for i := range want {
							want[i] = 2*want[i] + float64(previous[i])
						}
					}
					if !compareBackward(t, "parameters", got.Parameters, want) {
						t.Fatal("stale or doubled gradient")
					}
				}
			})
		}
	}
}

func TestMLPBackward_RequiredNegativeControls(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	x, w, dy := backwardFixture(m, 65)
	o := MLPBackwardOptions{Samples: 65, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 4 << 20}
	_, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 65)
	baseline := runBackwardHost(t, m, x, w, dy, nil, o, nil, 1)
	if !compareBackward(t, "baseline input", baseline.Input, dx) || !compareBackward(t, "baseline parameters", baseline.Parameters, dw) {
		t.Fatal("65-sample positive control failed")
	}
	controls := []struct {
		name    string
		replace func(string) string
	}{
		{"activation", func(s string) string { return strings.Replace(s, "(saved > 0.0f ? 1.0f : 0.0f)", "1.0f", 1) }},
		{"transpose dW", func(s string) string {
			return strings.Replace(s, "dL_dparams[in_idx * 7 + out_idx]", "dL_dparams[out_idx * 5 + in_idx]", 1)
		}},
		{"parameter offset", func(s string) string {
			i := strings.LastIndex(s, "params + 35,")
			if i < 0 {
				return s
			}
			return s[:i] + strings.Replace(s[i:], "params + 35,", "params + 0,", 1)
		}},
		{"context stride", func(s string) string { return strings.ReplaceAll(s, "fwd_ctx += lane_id() *", "fwd_ctx += 0 *") }},
		{"stale gradients", func(s string) string { return strings.Replace(s, "dw[k] = sum;", "dw[k] += sum;", 1) }},
	}
	for _, control := range controls {
		t.Run(control.name, func(t *testing.T) {
			rounds := 1
			if control.name == "stale gradients" {
				rounds = 2
			}
			got := runBackwardHost(t, m, x, w, dy, nil, o, control.replace, rounds)
			if compareBackward(t, "input", got.Input, dx) && compareBackward(t, "parameters", got.Parameters, dw) {
				t.Fatal("required behavior mutant survived")
			}
		})
	}
}

func TestMLPBackward_InvalidBuffersAndBudgets(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	x, w, dy := backwardFixture(m, 7)
	base := MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 1 << 20}
	for _, test := range []struct {
		name               string
		change             func(*MLPBackwardOptions)
		x, w, dy, previous []float32
	}{
		{"short input", nil, x[:len(x)-1], w, dy, nil}, {"short weights", nil, x, w[:len(w)-1], dy, nil}, {"short output gradient", nil, x, w, dy[:len(dy)-1], nil},
		{"mode", func(o *MLPBackwardOptions) { o.ParameterMode = 99 }, x, w, dy, nil}, {"samples", func(o *MLPBackwardOptions) { o.Samples = math.MaxInt32 }, x, w, dy, nil}, {"budget", func(o *MLPBackwardOptions) { o.MaxBytes = 1 }, x, w, dy, nil}, {"no outputs", func(o *MLPBackwardOptions) { o.Input = false; o.ParameterMode = cudair.GradientModeIgnore }, x, w, dy, nil}, {"missing accumulation", func(o *MLPBackwardOptions) { o.ParameterMode = cudair.GradientModeAccumulate }, x, w, dy, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			o := base
			if test.change != nil {
				test.change(&o)
			}
			if _, e := m.backwardLayout(test.x, test.w, test.dy, test.previous, o); e == nil {
				t.Fatal("invalid backward accepted")
			}
		})
	}
	if _, e := NewMLPLayer("overflow", math.MaxInt, 2, "relu"); e == nil {
		t.Fatal("overflow admitted")
	}
	valid, e := m.backwardLayout(x, w, dy, nil, base)
	if e != nil {
		t.Fatal(e)
	}
	exact := base
	exact.MaxBytes = uint64(valid.bytes)
	if _, e := m.backwardLayout(x, w, dy, nil, exact); e != nil {
		t.Fatalf("exact positive byte budget: %v", e)
	}
	exact.MaxBytes--
	if _, e := m.backwardLayout(x, w, dy, nil, exact); e == nil {
		t.Fatal("one-byte-short budget accepted")
	}
	bad := append([]float32(nil), x...)
	bad[0] = float32(math.NaN())
	if _, e := m.backwardLayout(bad, w, dy, nil, base); e == nil {
		t.Fatal("nonfinite input accepted")
	}
	if _, e := NewFusedMLPWithActivation(5, 7, 3, 2, "unknown"); e == nil {
		t.Fatal("unknown activation accepted")
	}
}
