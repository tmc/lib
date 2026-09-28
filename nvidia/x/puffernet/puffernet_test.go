package puffernet_test

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/puffernet"
)

// breakout is the policy ./puffer train builds for ocean/breakout with
// config/breakout.ini.
var breakout = puffernet.Config{InputSize: 118, HiddenSize: 32, NumLayers: 4, ActionSizes: []int{3}}

// goldenCases are the cases testdata/golden.c runs, in the same order.
var goldenCases = []struct {
	name  string
	cfg   puffernet.Config
	batch int
	steps int
}{
	{"breakout", breakout, 3, 6},
	{"continuous", puffernet.Config{InputSize: 5, HiddenSize: 12, NumLayers: 2, ActionSizes: []int{1, 1}}, 2, 6},
	{"multidiscrete", puffernet.Config{InputSize: 9, HiddenSize: 8, NumLayers: 3, ActionSizes: []int{2, 4}}, 2, 5},
}

// lcg is the generator in testdata/golden.c.
type lcg uint32

func (s *lcg) next() float32 {
	*s = *s*1664525 + 1013904223
	return float32(int32(*s>>9)%2001-1000) / 2000
}

// input is the weights and per-step observations and terminals of a case.
type input struct {
	w    *puffernet.Weights
	obs  [][]float32
	term [][]float32
}

func newInput(t *testing.T, cfg puffernet.Config, batch, steps int) input {
	t.Helper()
	s := lcg(1)
	flat := make([]float32, cfg.NumFloats())
	for i := range flat {
		flat[i] = s.next()
	}
	w, err := puffernet.NewWeights(cfg, flat)
	if err != nil {
		t.Fatal(err)
	}
	in := input{w: w}
	for step := range steps {
		obs := make([]float32, batch*cfg.InputSize)
		for i := range obs {
			obs[i] = 2 * s.next()
		}
		term := make([]float32, batch)
		for b := range term {
			if (step*batch+b)%5 == 4 {
				term[b] = 1
			}
		}
		in.obs = append(in.obs, obs)
		in.term = append(in.term, term)
	}
	return in
}

// readGolden returns the outputs of each step of each case in
// testdata/golden.txt.
func readGolden(t *testing.T) map[string][][]float32 {
	t.Helper()
	f, err := os.Open("testdata/golden.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	golden := make(map[string][][]float32)
	var name string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if n, ok := strings.CutPrefix(sc.Text(), "case "); ok {
			name = n
			continue
		}
		var row []float32
		for _, field := range strings.Fields(sc.Text()) {
			v, err := strconv.ParseFloat(field, 32)
			if err != nil {
				t.Fatal(err)
			}
			row = append(row, float32(v))
		}
		golden[name] = append(golden[name], row)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return golden
}

// closeTo reports whether got and want differ by at most tol relative to
// max(1, |want|).
func closeTo(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol*max(1, math.Abs(want))
}

func compare(t *testing.T, step int, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("step %d: got %d outputs, want %d", step, len(got), len(want))
	}
	for i := range got {
		if !closeTo(float64(got[i]), float64(want[i]), tol) {
			t.Fatalf("step %d: output %d = %v, want %v", step, i, got[i], want[i])
		}
	}
}

func TestNumFloats(t *testing.T) {
	// 32×118 + 4×32 + 4×(96×32), each already a multiple of 8. The
	// checkpoint ./puffer train wrote for breakout is 64768 bytes.
	if got, want := breakout.NumFloats(), 16192; got != want {
		t.Errorf("breakout.NumFloats() = %d, want %d", got, want)
	}
}

func TestWeightsLayout(t *testing.T) {
	cfg := goldenCases[1].cfg
	flat := make([]float32, cfg.NumFloats())
	for i := range flat {
		flat[i] = float32(i)
	}
	w, err := puffernet.NewWeights(cfg, flat)
	if err != nil {
		t.Fatal(err)
	}
	// encoder 60 → 64, decoder 36 → 104, logstd 2 → 112, layers 432 each.
	for _, tt := range []struct {
		name       string
		s          []float32
		start, len int
	}{
		{"Encoder", w.Encoder, 0, 60},
		{"Decoder", w.Decoder, 64, 36},
		{"LogStd", w.LogStd, 104, 2},
		{"Layers[0]", w.Layers[0], 112, 432},
		{"Layers[1]", w.Layers[1], 544, 432},
	} {
		if len(tt.s) != tt.len || tt.s[0] != float32(tt.start) {
			t.Errorf("%s: len %d starting at %v, want len %d starting at %d", tt.name, len(tt.s), tt.s[0], tt.len, tt.start)
		}
	}
}

func TestReadWeights(t *testing.T) {
	cfg := goldenCases[2].cfg
	need := cfg.NumFloats()
	for _, tt := range []struct {
		name string
		n    int
		ok   bool
	}{
		{"exact", need, true},
		{"unpadded", need - 7, true},
		{"short", need - 8, false},
		{"long", need + 1, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			for i := range tt.n {
				binary.Write(&buf, binary.LittleEndian, float32(i))
			}
			w, err := puffernet.ReadWeights(&buf, cfg)
			if (err == nil) != tt.ok {
				t.Fatalf("ReadWeights of %d floats: err = %v, want ok %v", tt.n, err, tt.ok)
			}
			if tt.ok && w.Layers[2][0] != float32(need-3*8*8) {
				t.Errorf("Layers[2][0] = %v, want %d", w.Layers[2][0], need-3*8*8)
			}
		})
	}
}

// TestHostGolden checks Host against the outputs of PufferLib's
// puffercpu.c.
func TestHostGolden(t *testing.T) {
	golden := readGolden(t)
	for _, c := range goldenCases {
		t.Run(c.name, func(t *testing.T) {
			in := newInput(t, c.cfg, c.batch, c.steps)
			want := golden[c.name]
			if len(want) != c.steps {
				t.Fatalf("golden has %d steps, want %d", len(want), c.steps)
			}
			h := puffernet.NewHost(in.w, c.batch)
			out := make([]float32, c.batch*c.cfg.OutputSize())
			for step := range c.steps {
				if err := h.Step(out, in.obs[step], in.term[step]); err != nil {
					t.Fatal(err)
				}
				compare(t, step, out, want[step], 1e-6)
			}
		})
	}
}

func TestKernelCompiles(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	if _, err := cudatest.CompileSource(puffernet.KernelSource, major, minor); err != nil {
		t.Fatal(err)
	}
}

// TestNetGolden checks Net against the outputs of PufferLib's puffercpu.c.
func TestNetGolden(t *testing.T) {
	dev := cudatest.Require(t)
	golden := readGolden(t)
	for _, c := range goldenCases {
		t.Run(c.name, func(t *testing.T) {
			in := newInput(t, c.cfg, c.batch, c.steps)
			got := runNet(t, dev, in, c.batch)
			for step, want := range golden[c.name] {
				compare(t, step, got[step], want, 1e-5)
			}
		})
	}
}

// TestNetBatch checks Net against Host for many agents, including one
// whose terminal flag is set at every step.
func TestNetBatch(t *testing.T) {
	dev := cudatest.Require(t)
	const batch, steps = 4096, 8
	in := newInput(t, breakout, batch, steps)
	for _, term := range in.term {
		term[7] = 1
	}
	got := runNet(t, dev, in, batch)
	h := puffernet.NewHost(in.w, batch)
	want := make([]float32, batch*breakout.OutputSize())
	for step := range steps {
		if err := h.Step(want, in.obs[step], in.term[step]); err != nil {
			t.Fatal(err)
		}
		compare(t, step, got[step], want, 1e-5)
	}
}

// runNet runs in on a Net and returns the output of each step.
func runNet(t *testing.T, dev *cuda.Device, in input, batch int) [][]float32 {
	t.Helper()
	cfg := in.w.Config
	net, err := puffernet.New(dev, in.w, batch)
	if err != nil {
		t.Fatal(err)
	}
	defer net.Close()
	a := cudamem.NewArena(dev)
	defer a.Free()
	obs := cudamem.AllocIn[float32](a, "obs", batch*cfg.InputSize)
	term := cudamem.AllocIn[float32](a, "terminals", batch)
	out := cudamem.AllocIn[float32](a, "out", batch*cfg.OutputSize())
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	stream := dev.DefaultStream()
	var outs [][]float32
	for step := range in.obs {
		if err := obs.CopyFromHost(in.obs[step]); err != nil {
			t.Fatal(err)
		}
		if err := term.CopyFromHost(in.term[step]); err != nil {
			t.Fatal(err)
		}
		if err := net.Step(stream, out, obs, term); err != nil {
			t.Fatal(err)
		}
		host := make([]float32, out.Len())
		if err := out.CopyToHost(host); err != nil {
			t.Fatal(err)
		}
		outs = append(outs, host)
	}
	return outs
}

// TestCheckpoint runs a checkpoint written by ./puffer train for breakout,
// named by $PUFFERNET_BREAKOUT_CHECKPOINT, on the device and the host.
func TestCheckpoint(t *testing.T) {
	path := os.Getenv("PUFFERNET_BREAKOUT_CHECKPOINT")
	if path == "" {
		t.Skip("PUFFERNET_BREAKOUT_CHECKPOINT not set")
	}
	w, err := puffernet.LoadWeights(path, breakout)
	if err != nil {
		t.Fatal(err)
	}
	dev := cudatest.Require(t)
	in := newInput(t, breakout, 64, 16)
	in.w = w
	got := runNet(t, dev, in, 64)
	h := puffernet.NewHost(w, 64)
	want := make([]float32, 64*breakout.OutputSize())
	for step := range in.obs {
		if err := h.Step(want, in.obs[step], in.term[step]); err != nil {
			t.Fatal(err)
		}
		// Trained weights reach magnitude 6, against 0.5 for the synthetic
		// ones, and float32 rounding grows with them: on the breakout
		// checkpoint both Host and Net are up to 2.5e-4 from a float64
		// forward pass, and neither is consistently closer.
		compare(t, step, got[step], want, 1e-3)
	}
}
