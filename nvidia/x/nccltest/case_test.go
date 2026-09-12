package nccltest

import (
	"math"
	"slices"
	"testing"
	"time"
)

func TestLayout(t *testing.T) {
	for _, tc := range []struct {
		name Operation
		want Layout
	}{
		{AllReduce, Layout{65, 65, 65, 0, 0}},
		{AllGather, Layout{20, 60, 20, 20, 0}},
		{ReduceScatter, Layout{60, 20, 20, 0, 20}},
		{AllToAll, Layout{60, 60, 20, 0, 0}},
	} {
		t.Run(string(tc.name), func(t *testing.T) {
			got, err := (Case{Operation: tc.name, Bytes: 260}).Layout(3)
			if err != nil || got != tc.want {
				t.Fatalf("got %+v, %v; want %+v", got, err, tc.want)
			}
		})
	}
}
func TestOracleRankRouting(t *testing.T) {
	// These fixed expected values do not call the input or expected helpers.
	for _, tc := range []struct {
		op    Operation
		rank  int
		index int
		want  float32
	}{
		{AllGather, 2, 4, 12}, {AllToAll, 2, 4, 9}, {Scatter, 2, 0, 11},
		{Broadcast, 2, 1, 2}, {SendRecv, 0, 0, 13}, {AllReduce, 1, 0, 32},
		{ReduceScatter, 2, 0, 25},
	} {
		c := Case{Operation: tc.op, Bytes: 48}
		l, err := c.Layout(3)
		if err != nil {
			t.Fatal(err)
		}
		if got := expectedFloat[float32](c, l, 3, tc.rank, tc.index); got != tc.want {
			t.Errorf("%s rank %d index %d = %g, want %g", tc.op, tc.rank, tc.index, got, tc.want)
		}
	}
}
func TestCheckerRejectsCorruption(t *testing.T) {
	c := Case{Operation: AllGather, Bytes: 48}
	l, _ := c.Layout(3)
	got := make([]float32, l.RecvCount)
	for i := range got {
		got[i] = expectedFloat[float32](c, l, 3, 0, i)
	}
	n, bad, err := Check(c, 3, 0, got)
	if err != nil || n != 12 || bad != 0 {
		t.Fatalf("correct input: %d %d %v", n, bad, err)
	}
	got[5]++
	_, bad, err = Check(c, 3, 0, got)
	if err != nil || bad != 1 {
		t.Fatalf("corruption: %d %v", bad, err)
	}
	for i := range got {
		got[i] = float32(math.NaN())
	}
	_, bad, err = Check(c, 3, 0, got)
	if err != nil || bad != 12 {
		t.Fatalf("omitted operation: %d %v", bad, err)
	}
}
func TestBandwidth(t *testing.T) {
	for _, tc := range []struct {
		op       Operation
		ranks    int
		alg, bus float64
	}{
		{AllReduce, 4, 1, 1.5}, {AllGather, 4, 1, .75}, {Hypercube, 4, .75, .75}, {AllReduce, 1, 1, 0},
	} {
		c := Case{Operation: tc.op, Bytes: 1000000}
		a, b, err := c.Bandwidth(tc.ranks, time.Millisecond)
		if err != nil || a != tc.alg || b != tc.bus {
			t.Errorf("%s/%d: %g %g %v", tc.op, tc.ranks, a, b, err)
		}
	}
}
func TestZeroWorkRejected(t *testing.T) {
	if _, err := (Case{Operation: AllGather, Bytes: 4}).Layout(2); err == nil {
		t.Fatal("accepted zero-work case")
	}
	if _, err := (Case{Operation: Hypercube, Bytes: 64}).Layout(3); err == nil {
		t.Fatal("accepted non-power-of-two hypercube")
	}
}

func TestReductions(t *testing.T) {
	for _, tc := range []struct {
		op   Reduction
		want float32
	}{{PreMulSum, 44}, {Sum, 32}, {Product, 1092}, {Minimum, 7}, {Maximum, 13}, {Average, float32(32) / 3}} {
		c := Case{Operation: AllReduce, Bytes: 4, Reduction: tc.op}
		l, _ := c.Layout(3)
		if got := expectedFloat[float32](c, l, 3, 0, 0); got != tc.want {
			t.Errorf("%s = %g, want %g", tc.op, got, tc.want)
		}
	}
}

func TestCheckerRejectsRankPermutation(t *testing.T) {
	c := Case{Operation: AllGather, Bytes: 1024}
	l, _ := c.Layout(16)
	got := make([]float32, l.RecvCount)
	for i := range got {
		got[i] = expectedFloat[float32](c, l, 16, 0, i)
	}
	for i := 0; i < l.ParamCount; i++ {
		got[i], got[13*l.ParamCount+i] = got[13*l.ParamCount+i], got[i]
	}
	_, wrong, err := Check(c, 16, 0, got)
	if err != nil || wrong == 0 {
		t.Fatalf("rank permutation accepted: wrong=%d err=%v", wrong, err)
	}
}

func TestUnsupportedAliasing(t *testing.T) {
	for _, op := range []Operation{AllToAll, AllToAllV, SendRecv} {
		if _, err := (Case{Operation: op, Bytes: 64, InPlace: true}).Layout(1); err == nil {
			t.Fatalf("accepted unsupported %s aliasing", op)
		}
	}
}

func TestPreMulSumRejectsMissingScale(t *testing.T) {
	c := Case{Operation: AllReduce, Bytes: 4, Reduction: PreMulSum}
	// Rank inputs are 7, 12, 13. The middle rank's scalar is 2, so
	// ordinary sum (32) must fail while pre-multiplied sum (44) passes.
	for _, tc := range []struct {
		got   float32
		wrong int
	}{{32, 1}, {44, 0}} {
		count, wrong, err := Check(c, 3, 0, []float32{tc.got})
		if err != nil || count != 1 || wrong != tc.wrong {
			t.Fatalf("value=%g count=%d wrong=%d err=%v", tc.got, count, wrong, err)
		}
	}
}

func TestLinearSizes(t *testing.T) {
	max := int(^uint(0) >> 1)
	for _, tt := range []struct {
		name              string
		first, last, step int
		want              []int
	}{
		{"inclusive", 16, 64, 16, []int{16, 32, 48, 64}},
		{"unaligned end", 16, 63, 16, []int{16, 32, 48}},
		{"single", 16, 16, 1, []int{16}},
		{"large step", 1, 16, max, []int{1}},
		{"overflow boundary", max - 2, max, 2, []int{max - 2, max}},
		{"zero first", 0, 16, 1, nil},
		{"reverse", 16, 1, 1, nil},
		{"zero step", 1, 16, 0, nil},
		{"negative step", 1, 16, -1, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LinearSizes(tt.first, tt.last, tt.step)
			if (err != nil) != (tt.want == nil) || !slices.Equal(got, tt.want) {
				t.Fatalf("got %v, %v; want %v", got, err, tt.want)
			}
		})
	}
}
