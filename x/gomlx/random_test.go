package gomlx

import (
	"math"
	"os"
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativePhilox(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	builder := b.Builder("philox")
	f := builder.Main()
	state, err := f.Parameter("state", shapes.Make(dtypes.Uint64, 3), nil)
	if err != nil {
		t.Fatal(err)
	}
	next, bits, err := f.RNGBitGenerator(state, shapes.Make(dtypes.Uint32, 4))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{next, bits}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	run := func(state []uint64) ([]uint64, []uint32) {
		t.Helper()
		input, err := b.BufferFromFlatData(0, state, shapes.Make(dtypes.Uint64, 3))
		if err != nil {
			t.Fatal(err)
		}
		defer input.Finalize()
		out, err := e.Execute([]compute.Buffer{input}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range out {
			defer x.Finalize()
		}
		next, bits := make([]uint64, 3), make([]uint32, 4)
		if err := out[0].ToFlatData(next); err != nil {
			t.Fatal(err)
		}
		if err := out[1].ToFlatData(bits); err != nil {
			t.Fatal(err)
		}
		return next, bits
	}
	// Philox4x32-10 known answers from Random123 tests/kat_vectors:
	// https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
	for _, tc := range []struct {
		name  string
		state []uint64
		want  []uint32
	}{
		{"zero", []uint64{0, 0, 0}, []uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{"ones", []uint64{math.MaxUint64, math.MaxUint64, math.MaxUint64}, []uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{"pi", []uint64{0x299f31d0a4093822, 0x85a308d3243f6a88, 0x0370734413198a2e}, []uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			next, got := run(tc.state)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("bits %x, want %x", got, tc.want)
			}
			wantState := slices.Clone(tc.state)
			wantState[1]++
			if wantState[1] == 0 {
				wantState[2]++
			}
			if !slices.Equal(next, wantState) {
				t.Fatalf("state %x, want %x", next, wantState)
			}
			replay, replayed := run(tc.state)
			if !slices.Equal(replay, next) || !slices.Equal(replayed, got) {
				t.Fatal("same state did not replay")
			}
			_, advanced := run(next)
			if slices.Equal(advanced, got) {
				t.Fatal("advanced state repeated sequence")
			}
		})
	}
}

func TestNativeRandomDistributions(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	e, err := graph.NewExec(b, func(state *graph.Node) []*graph.Node {
		next, u := graph.RandomUniform(state, shapes.Make(dtypes.Float32, 4096))
		next, n := graph.RandomNormal(next, shapes.Make(dtypes.Float32, 4096))
		return []*graph.Node{next, u, n}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call([]uint64{42, 0, 0})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	for _, x := range out {
		defer x.FinalizeAll()
	}
	for index := 1; index <= 2; index++ {
		values := out[index].Value().([]float32)
		mean, square := 0.0, 0.0
		for _, v := range values {
			x := float64(v)
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatal("nonfinite sample")
			}
			if index == 1 && (x < 0 || x >= 1) {
				t.Fatalf("uniform sample %v outside [0,1)", x)
			}
			mean += x
			square += x * x
		}
		mean /= float64(len(values))
		variance := square/float64(len(values)) - mean*mean
		wantMean, wantVar, tolerance := 0.5, 1.0/12, 0.03
		if index == 2 {
			wantMean, wantVar, tolerance = 0, 1, 0.1
		}
		if math.Abs(mean-wantMean) > tolerance || math.Abs(variance-wantVar) > tolerance {
			t.Fatalf("distribution %d mean %v variance %v", index, mean, variance)
		}
		t.Logf("distribution %d: mean %.6f variance %.6f", index, mean, variance)
	}
}

func TestNativePhilox64(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	e, err := graph.NewExec(b, func(state *graph.Node) []*graph.Node {
		next, bits := graph.RNGStateSplit(state)
		return []*graph.Node{next, bits}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call([]uint64{0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		defer v.FinalizeAll()
	}
	next := out[0].Value().([]uint64)
	if !slices.Equal(next, []uint64{0, 2, 0}) {
		t.Fatalf("next state %x", next)
	}
	bits := out[1].Value().([]uint64)
	if !slices.Equal(bits[:2], []uint64{0xe169c58d6627e8d5, 0x9b00dbd8bc57ac4c}) {
		t.Fatalf("64-bit Philox prefix %x", bits)
	}
}
