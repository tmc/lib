package cublasltx

import (
	"errors"
	"testing"
)

func TestPick(t *testing.T) {
	cands := []AlgoConfig{
		{ID: 1, Tile: 20},
		{ID: 21, Tile: 20, Stages: 15},
		{ID: 21, Tile: 20, Stages: 15, CustomOption: 1},
		{ID: 21, Tile: 20, Stages: 15},
	}
	config := func(i int) (AlgoConfig, error) { return cands[i], nil }
	id := func(id int32) func(AlgoConfig) bool {
		return func(c AlgoConfig) bool { return c.ID == id }
	}
	tests := []struct {
		name   string
		n      int
		config func(int) (AlgoConfig, error)
		match  func(AlgoConfig) bool
		want   int
		err    bool
	}{
		{name: "empty", n: 0, config: config, match: id(21), err: true},
		{name: "nil match", n: len(cands), config: config, want: 0},
		{name: "first match", n: len(cands), config: config, match: id(21), want: 1},
		{name: "exact", n: len(cands), config: config, match: func(c AlgoConfig) bool { return c == AlgoConfig{ID: 21, Tile: 20, Stages: 15, CustomOption: 1} }, want: 2},
		{name: "no match", n: len(cands), config: config, match: id(99), want: 0},
		{name: "bounded", n: 1, config: config, match: id(21), want: 0},
		{
			name: "unreadable skipped",
			n:    len(cands),
			config: func(i int) (AlgoConfig, error) {
				if i == 1 {
					return cands[i], errors.New("unreadable")
				}
				return cands[i], nil
			},
			match: id(21),
			want:  2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := pick(tt.n, tt.config, tt.match)
			if (err != nil) != tt.err {
				t.Fatalf("pick error = %v, want error %v", err, tt.err)
			}
			if got != tt.want {
				t.Errorf("pick = %d, want %d", got, tt.want)
			}
		})
	}
}
