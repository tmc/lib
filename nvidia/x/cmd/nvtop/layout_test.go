package main

import (
	"reflect"
	"testing"
)

// The expected layouts are those of nvtop's compute_sizes_from_layout for
// the same arguments, with 3-row, 73-column device headers.
func TestComputeLayout(t *testing.T) {
	two := func(n int) []uint {
		info := make([]uint, n)
		for i := range info {
			info[i] = 2
		}
		return info
	}
	tests := []struct {
		name       string
		info       []uint
		rows, cols uint
		hideProcs  bool
		want       layout
	}{
		{
			name: "one device 120x40", info: two(1), rows: 39, cols: 120,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}},
				plots:       []rect{{0, 3, 119, 23}},
				plotDevices: [][]int{{0}},
				processes:   rect{0, 26, 120, 13},
			},
		},
		{
			name: "one device 80x24", info: two(1), rows: 23, cols: 80,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}},
				plots:       []rect{{0, 3, 79, 14}},
				plotDevices: [][]int{{0}},
				processes:   rect{0, 17, 80, 6},
			},
		},
		{
			name: "no plot", info: []uint{0}, rows: 39, cols: 120,
			want: layout{
				devices:   []rect{{1, 0, 73, 3}},
				processes: rect{0, 4, 120, 35},
			},
		},
		{
			name: "no processes", info: two(1), rows: 39, cols: 120, hideProcs: true,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}},
				plots:       []rect{{0, 3, 119, 36}},
				plotDevices: [][]int{{0}},
				processes:   rect{0, 39, 120, 0},
			},
		},
		{
			name: "two devices stacked", info: two(2), rows: 39, cols: 120,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}, {1, 4, 73, 3}},
				plots:       []rect{{0, 7, 119, 12}, {0, 19, 119, 12}},
				plotDevices: [][]int{{0}, {1}},
				processes:   rect{0, 31, 120, 8},
			},
		},
		{
			name: "two devices side by side", info: two(2), rows: 49, cols: 200,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}, {75, 0, 73, 3}},
				plots:       []rect{{0, 3, 199, 19}, {0, 22, 199, 19}},
				plotDevices: [][]int{{0}, {1}},
				processes:   rect{0, 41, 200, 8},
			},
		},
		{
			name: "four devices", info: two(4), rows: 59, cols: 160,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}, {75, 0, 73, 3}, {1, 4, 73, 3}, {75, 4, 73, 3}},
				plots:       []rect{{0, 7, 159, 13}, {0, 20, 159, 13}, {0, 33, 79, 13}, {80, 33, 79, 13}},
				plotDevices: [][]int{{0}, {1}, {2}, {3}},
				processes:   rect{0, 46, 160, 13},
			},
		},
		{
			name: "three devices wide", info: two(3), rows: 30, cols: 300,
			want: layout{
				devices:     []rect{{1, 0, 73, 3}, {75, 0, 73, 3}, {149, 0, 73, 3}},
				plots:       []rect{{0, 3, 99, 19}, {100, 3, 99, 19}, {200, 3, 99, 19}},
				plotDevices: [][]int{{0}, {1}, {2}},
				processes:   rect{0, 22, 300, 8},
			},
		},
		{
			name: "six devices", info: two(6), rows: 80, cols: 250,
			want: layout{
				devices: []rect{
					{1, 0, 73, 3}, {75, 0, 73, 3}, {149, 0, 73, 3},
					{1, 4, 73, 3}, {75, 4, 73, 3}, {149, 4, 73, 3},
				},
				plots: []rect{
					{0, 7, 249, 13}, {0, 20, 249, 13},
					{0, 33, 125, 13}, {125, 33, 125, 13},
					{0, 46, 125, 13}, {125, 46, 125, 13},
				},
				plotDevices: [][]int{{0}, {1}, {2}, {3}, {4}, {5}},
				processes:   rect{0, 59, 250, 21},
			},
		},
		{
			name: "too small for a plot", info: two(1), rows: 5, cols: 40,
			want: layout{
				devices:   []rect{{1, 0, 73, 3}},
				processes: rect{0, 3, 40, 2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeLayout(tt.info, 3, 73, tt.rows, tt.cols, tt.hideProcs)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("computeLayout:\ngot  %+v\nwant %+v", got, tt.want)
			}
		})
	}
}

func TestWhoToMerge(t *testing.T) {
	tests := []struct {
		per  []uint
		a, b uint
		ok   bool
	}{
		{[]uint{2, 2}, 0, 1, true},
		{[]uint{2, 2, 2}, 1, 2, true}, // ties prefer the end
		{[]uint{1, 2, 0, 1}, 1, 3, true},
		{[]uint{3, 2}, 0, 0, false},
		{[]uint{2}, 0, 0, false},
	}
	for _, tt := range tests {
		a, b, ok := whoToMerge(tt.per)
		if a != tt.a || b != tt.b || ok != tt.ok {
			t.Errorf("whoToMerge(%v) = %d, %d, %v; want %d, %d, %v", tt.per, a, b, ok, tt.a, tt.b, tt.ok)
		}
	}
}
