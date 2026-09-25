package timeline

import "testing"

func TestUnion(t *testing.T) {
	tests := []struct {
		name string
		in   []Interval
		want uint64
	}{
		{name: "empty", in: nil, want: 0},
		{name: "single", in: []Interval{{10, 20}}, want: 10},
		{name: "disjoint", in: []Interval{{10, 20}, {30, 45}}, want: 25},
		{name: "unsorted disjoint", in: []Interval{{30, 45}, {10, 20}}, want: 25},
		{name: "touching", in: []Interval{{10, 20}, {20, 30}}, want: 20},
		// Concurrent streams: summing would report 20 and charge the excess to
		// the host as dispatch overhead on a device that was never idle.
		{name: "identical overlap", in: []Interval{{10, 20}, {10, 20}}, want: 10},
		{name: "partial overlap", in: []Interval{{10, 20}, {15, 30}}, want: 20},
		{name: "nested", in: []Interval{{10, 50}, {20, 30}}, want: 40},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Union(tt.in); got != tt.want {
				t.Errorf("Union(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}
