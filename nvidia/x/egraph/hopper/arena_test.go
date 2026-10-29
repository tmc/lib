package hopper

import (
	"fmt"
	"math"
	"testing"
)

func TestPlanArena(t *testing.T) {
	tests := []struct {
		name   string
		args   [][]string
		output string
		reuse  bool
		want   uint64
		b      uint64
	}{
		{"same launch", [][]string{{"a", "b"}}, "", true, 512, 256},
		{"shared endpoint", [][]string{{"a"}, {"a", "b"}, {"b"}}, "", true, 512, 256},
		{"disjoint", [][]string{{"a"}, {"b"}}, "", true, 256, 0},
		{"output retained", [][]string{{"a"}, {"b"}}, "a", true, 512, 256},
		{"reuse disabled", [][]string{{"a"}, {"b"}}, "", false, 512, 256},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Plan{Temps: []Tensor{{"a", F32, []int{64}}, {"b", F32, []int{1}}}, Output: Tensor{Name: tt.output}}
			for _, args := range tt.args {
				p.Kernels = append(p.Kernels, PlanKernel{Args: args})
			}
			got, err := PlanArena([]*Plan{p}, 0, tt.reuse)
			if err != nil {
				t.Fatal(err)
			}
			if got.Bytes != tt.want || got.OriginalBytes != 260 || got.Offsets[0]["a"] != 0 || got.Offsets[0]["b"] != tt.b {
				t.Fatalf("arena=%+v", got)
			}
		})
	}
}

func TestPlanArenaSequential(t *testing.T) {
	p := &Plan{Temps: []Tensor{{"a", BF16, []int{129}}}, Kernels: []PlanKernel{{Args: []string{"a"}}}}
	a, err := PlanArena([]*Plan{p, p}, 512, true)
	if err != nil {
		t.Fatal(err)
	}
	if a.Bytes != 512 || a.OriginalBytes != 516 || a.Offsets[0]["a"] != 0 || a.Offsets[1]["a"] != 0 {
		t.Fatalf("arena=%+v", a)
	}
	if _, err := PlanArena([]*Plan{p}, 511, true); err == nil {
		t.Fatal("accepted insufficient budget")
	}
}

func TestPlanArenaInvalid(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		temps []Tensor
	}{
		{"dtype", []Tensor{{"x", 0, []int{1}}}},
		{"zero", []Tensor{{"x", F32, []int{0}}}},
		{"negative", []Tensor{{"x", F32, []int{-1}}}},
		{"shape overflow", []Tensor{{"x", U8, []int{maxInt, maxInt, 2}}}},
		{"storage overflow", []Tensor{{"x", I64, []int{maxInt}}}},
		{"duplicate", []Tensor{{"x", F32, []int{1}}, {"x", F32, []int{1}}}},
		{"empty name", []Tensor{{"", F32, []int{1}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := PlanArena([]*Plan{{Temps: tt.temps}}, 0, true); err == nil {
				t.Fatal("accepted invalid temporary")
			}
		})
	}
	if _, err := PlanArena([]*Plan{nil}, 0, true); err == nil {
		t.Fatal("accepted nil plan")
	}
	a, err := PlanArena(nil, 1, true)
	if err != nil || a.Bytes != 0 {
		t.Fatalf("empty arena=%v err=%v", a, err)
	}
}

func TestArenaTensorBytes(t *testing.T) {
	for _, tt := range []struct {
		dtype DType
		n     int
		want  uint64
	}{{E2M1, 3, 2}, {E2M1, 4, 2}, {F32, 3, 12}, {BF16, 3, 6}, {E4M3, 3, 3}, {U8, 3, 3}, {I32, 3, 12}, {I64, 3, 24}} {
		got, err := arenaTensorBytes(Tensor{"x", tt.dtype, []int{tt.n}})
		if err != nil || got != tt.want {
			t.Errorf("%v(%d)=%d,%v want%d", tt.dtype, tt.n, got, err, tt.want)
		}
	}
	if uint64(^uint(0)>>1) == math.MaxInt64 {
		p := &Plan{Temps: []Tensor{{"x", U8, []int{math.MaxInt64, 2}}}}
		if _, err := PlanArena([]*Plan{p}, 0, true); err == nil {
			t.Fatal("accepted aligned size overflow")
		}
	}
}

func ExamplePlanArena() {
	p := &Plan{Temps: []Tensor{{"a", F32, []int{64}}, {"b", F32, []int{64}}}, Kernels: []PlanKernel{{Args: []string{"a"}}, {Args: []string{"b"}}}}
	a, err := PlanArena([]*Plan{p}, 1024, true)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.Bytes, a.OriginalBytes, a.Offsets[0]["a"], a.Offsets[0]["b"])
	// Output: 256 512 0 0
}

func TestPlanArenaLifetimeOverlap(t *testing.T) {
	// All inclusive intervals over five launches exercise placement against an
	// independent pairwise overlap check, including holes and shared endpoints.
	p := &Plan{Kernels: make([]PlanKernel, 5)}
	type interval struct {
		name        string
		first, last int
		size        uint64
	}
	var intervals []interval
	for first := 0; first < 5; first++ {
		for last := first; last < 5; last++ {
			name := fmt.Sprintf("t%d_%d", first, last)
			n := 1 + (first+last)%3*64
			p.Temps = append(p.Temps, Tensor{name, F32, []int{n}})
			p.Kernels[first].Args = append(p.Kernels[first].Args, name)
			p.Kernels[last].Args = append(p.Kernels[last].Args, name)
			intervals = append(intervals, interval{name, first, last, uint64(n * 4)})
		}
	}
	a, err := PlanArena([]*Plan{p}, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range intervals {
		xo := a.Offsets[0][x.name]
		if xo%256 != 0 || xo+x.size > a.Bytes {
			t.Fatalf("invalid placement for %s", x.name)
		}
		for _, y := range intervals[:i] {
			if x.first > y.last || y.first > x.last {
				continue
			}
			yo := a.Offsets[0][y.name]
			if xo < yo+y.size && yo < xo+x.size {
				t.Errorf("live tensors %s and %s overlap", x.name, y.name)
			}
		}
	}
}
