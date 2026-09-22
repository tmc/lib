package cudasearch

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprepack"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestPrefillFloat32Preparation(t *testing.T) {
	a, b := make([]float32, 256*16), make([]float32, 16*32)
	for i := range a {
		a[i] = math.Float32frombits(0x3f801000)
	} // halfway, rounds to 1.
	for row := 0; row < 16; row++ {
		for col := 0; col < 32; col++ {
			b[col*16+row] = float32(row*100 + col)
		}
	}
	c, err := NewPrefillCompilerFloat32(&cuda.Device{}, "sm_121a", 16, 32, a, b, true, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range c.a {
		if h != 0x3c00 {
			t.Fatalf("tie conversion=%04x", h)
		}
	}
	for row := 0; row < 16; row++ {
		for col := 0; col < 32; col++ {
			if got := cudaprepack.Float16ToFloat32(c.b[row*32+col]); got != float32(row*100+col) {
				t.Fatalf("packed [%d,%d]=%g", row, col, got)
			}
		}
	}
	p := c.Preparation()
	if len(p) != 4 {
		t.Fatalf("phases=%+v", p)
	}
	for i, name := range []string{"fp32_to_fp16", "b_row_major_pack", "bind_inputs"} {
		if p[i].Name != name || !p[i].Complete || p[i].WallNs < 0 || p[i].InputBytes <= 0 || p[i].OutputBytes <= 0 {
			t.Fatalf("phase=%+v", p[i])
		}
	}
	if p[0].InputBytes != 18432 || p[0].OutputBytes != 9216 || p[1].InputBytes != 1024 || p[1].OutputBytes != 1024 {
		t.Fatalf("byte counts=%+v", p)
	}
	if p[3].Complete || p[3].WallNs != cudatrial.Unmeasured {
		t.Fatal("invented grouping measurement")
	}
	p[0].Name = "changed"
	if c.Preparation()[0].Name != "fp32_to_fp16" {
		t.Fatal("preparation aliases compiler")
	}
	key := c.Bindings().Key()
	a[0] = math.Float32frombits(0x3f800800)
	other, err := NewPrefillCompilerFloat32(&cuda.Device{}, "sm_121a", 16, 32, a, b, true, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(c.a, other.a) || other.Bindings().Key() == key {
		t.Fatal("source identity lost when rounded inputs agree")
	}
	plain, err := NewPrefillCompiler(&cuda.Device{}, "sm_121a", 16, 32, c.a, c.b, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	for _, phase := range plain.Preparation() {
		if phase.Complete || phase.WallNs != cudatrial.Unmeasured {
			t.Fatal("invented upstream timing")
		}
	}
	for _, v := range []float32{65520, float32(math.NaN()), float32(math.Inf(1))} {
		a[0] = v
		if _, err := NewPrefillCompilerFloat32(&cuda.Device{}, "sm_121a", 16, 32, a, b, false, func() bool { return false }); err == nil {
			t.Fatalf("accepted %g", v)
		}
	}
}

type preparedTestExecutable struct {
	*testExecutable
	phases []PreparationPhase
}

func (x *preparedTestExecutable) Preparation() []PreparationPhase { return x.phases }

func TestPreparationDoesNotBecomeDeviceCost(t *testing.T) {
	e, base, compiled, _ := executableFixture(t)
	phases := []PreparationPhase{{Name: "conversion", WallNs: int64(time.Hour), Complete: true}, {Name: "external_prepack", WallNs: cudatrial.Unmeasured}}
	e.Baseline = &preparedTestExecutable{testExecutable: base, phases: phases}
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
		return &preparedTestExecutable{testExecutable: compiled, phases: phases}, nil
	}
	s := executableState()
	root, _ := s.CanonicalRoot("root")
	c, err := e.Evaluate(context.Background(), s, root)
	if err != nil {
		t.Fatal(err)
	}
	if c.Cost != float64(compiled.duration.Nanoseconds()) || len(c.Preparation) != 2 {
		t.Fatalf("candidate=%+v", c)
	}
	if c.Receipt.TimingKind != cudatrial.TimingSummedKernel {
		t.Fatal("changed device timing channel")
	}
	phases[0].Name = "changed"
	if c.Preparation[0].Name != "conversion" || e.Preparation["baseline"][0].Name != "conversion" || e.Preparation["candidate"][0].Name != "conversion" {
		t.Fatal("preparation receipt aliases source")
	}
	c.Preparation[0].Name = "candidate changed"
	if e.Preparation["candidate"][0].Name != "conversion" {
		t.Fatal("candidate preparation aliases attempt receipt")
	}
}

func TestEncodingPreparation(t *testing.T) {
	g, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", 2, 2, []float32{1, 2, 3, 4}, []int32{1, 0, 1}, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	n, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", normSearchWorkload(), func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		get   func() []PreparationPhase
		name  string
		bytes int64
	}{{g.Preparation, "input_encode", 28}, {n.Preparation, "input_copy_and_encode", 512}} {
		p := tc.get()
		if p[0].Name != tc.name || !p[0].Complete || p[0].WallNs < 0 || p[0].InputBytes != tc.bytes || p[0].OutputBytes != tc.bytes {
			t.Fatalf("phase=%+v", p)
		}
		for _, unknown := range p[1:] {
			if unknown.Complete || unknown.WallNs != cudatrial.Unmeasured {
				t.Fatalf("invented measurement: %+v", unknown)
			}
		}
		p[0].Name = "changed"
		if tc.get()[0].Name != tc.name {
			t.Fatal("compiler metadata aliases caller")
		}
	}
	if !slices.Equal(n.encodedInput, floatBytes(n.w.Input)) || len(n.encodedWeight) != 0 {
		t.Fatal("cached norm encoding differs from bound inputs")
	}
}

func TestFailedPreparationReceipt(t *testing.T) {
	var phases []PreparationPhase
	failure := errors.New("failed setup")
	if err := measurePreparation(&phases, "fixture", 4, 8, func() error { return failure }); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if len(phases) != 1 || phases[0].Complete || phases[0].WallNs < 0 || phases[0].InputBytes != 4 || phases[0].OutputBytes != 8 {
		t.Fatalf("failed phase=%+v", phases)
	}
	x := &sequenceExecutable{}
	if err := x.upload("missing", []byte{1, 2}); err == nil {
		t.Fatal("accepted missing upload buffer")
	}
	if err := x.load(nil); err == nil {
		t.Fatal("accepted missing artifacts")
	}
	p := x.Preparation()
	if len(p) != 2 || p[0].Name != "upload_missing" || p[1].Name != "load_modules_stream_events" {
		t.Fatalf("setup receipts=%+v", p)
	}
	for _, phase := range p {
		if phase.Complete || phase.WallNs < 0 {
			t.Fatalf("invalid failed receipt: %+v", phase)
		}
	}
	p[0].Name = "changed"
	if x.Preparation()[0].Name != "upload_missing" {
		t.Fatal("sequence metadata aliases caller")
	}
}
