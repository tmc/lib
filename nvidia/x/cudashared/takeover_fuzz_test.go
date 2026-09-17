package cudashared

import "testing"

func FuzzRequestArithmetic(f *testing.F) {
	for _, seed := range []struct{ size, align, first, last int }{
		{1, 1, 0, 0}, {17, 16, 1, 3}, {1 << 20, 256, 0, 7},
	} {
		f.Add(seed.size, seed.align, seed.first, seed.last)
	}
	f.Fuzz(func(t *testing.T, size, align, first, last int) {
		p, err := NewPlan(Limits{StaticMax: 1 << 20, DynamicMax: 1 << 20, TotalMax: 1 << 20, Source: "fuzz"})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = p.Add(Request{Name: "fuzz", Size: int64(size), Alignment: int64(align), FirstStage: first, LastStage: last, Dynamic: true})
	})
}
