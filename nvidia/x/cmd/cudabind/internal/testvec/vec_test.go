package testvec

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestKernels(t *testing.T) {
	dev := cudatest.Require(t)
	m, err := loadVecModule(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	stream := dev.DefaultStream()
	one := [3]uint32{1, 1, 1}
	block := [3]uint32{32, 1, 1}

	if err := m.setScale(3); err != nil {
		t.Fatal(err)
	}
	if s, err := m.scale(); err != nil || s != 3 {
		t.Fatalf("scale() = %v, %v, want 3", s, err)
	}
	x, err := cudamem.Alloc[float32](dev, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	if err := x.CopyFromHost([]float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := m.scaleAll(stream, one, block, x, 4); err != nil {
		t.Fatal(err)
	}

	if err := m.setOrigin(point{X: 1, Y: 0.5}); err != nil {
		t.Fatal(err)
	}
	p, err := cudamem.Alloc[point](dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Free()
	tags, err := cudamem.Alloc[pair](dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer tags.Free()
	if err := p.CopyFromHost([]point{{X: 1, Y: 1}, {X: 2, Y: 2}}); err != nil {
		t.Fatal(err)
	}
	if err := m.shift(stream, one, block, p, tags, 2, opMul); err != nil {
		t.Fatal(err)
	}

	filled, err := cudamem.Alloc[pair](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer filled.Free()
	fillWant := pair{A: 7, B: [4]uint32{1, 2, 3, 4}}
	if err := m.fill(stream, one, one, fillWant, filled); err != nil {
		t.Fatal(err)
	}

	in, err := cudamem.Alloc[int32](dev, 5)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Free()
	out, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := in.CopyFromHost([]int32{1, 2, 3, 4, 5}); err != nil {
		t.Fatal(err)
	}
	if err := m.sum(stream, one, block, 32*4, in, 5, out); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}

	gotX := make([]float32, 4)
	if err := x.CopyToHost(gotX); err != nil {
		t.Fatal(err)
	}
	if want := []float32{3, 6, 9, 12}; !equal(gotX, want) {
		t.Errorf("scaleAll: x = %v, want %v", gotX, want)
	}
	gotP := make([]point, 2)
	if err := p.CopyToHost(gotP); err != nil {
		t.Fatal(err)
	}
	if want := []point{{2, 1.5}, {3, 2.5}}; !equal(gotP, want) {
		t.Errorf("shift: p = %v, want %v", gotP, want)
	}
	gotTags := make([]pair, 2)
	if err := tags.CopyToHost(gotTags); err != nil {
		t.Fatal(err)
	}
	if want := []pair{{A: opMul}, {A: opMul, B: [4]uint32{1, 1, 1, 1}}}; !equal(gotTags, want) {
		t.Errorf("shift: tags = %v, want %v", gotTags, want)
	}
	gotFill := make([]pair, 1)
	if err := filled.CopyToHost(gotFill); err != nil {
		t.Fatal(err)
	}
	if gotFill[0] != fillWant {
		t.Errorf("fill: out = %v, want %v", gotFill[0], fillWant)
	}
	gotSum := make([]int32, 1)
	if err := out.CopyToHost(gotSum); err != nil {
		t.Fatal(err)
	}
	if gotSum[0] != 15 {
		t.Errorf("sum = %d, want 15", gotSum[0])
	}
}

func equal[T comparable](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
