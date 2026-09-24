package main

import (
	"maps"
	"slices"
	"strings"
	"testing"

	"golang.org/x/perf/benchfmt"
)

// results makes one Final result per configuration "k=v ...", by input.
func results(inputs map[string][]string) []*benchfmt.Result {
	var rs []*benchfmt.Result
	for _, file := range slices.Sorted(maps.Keys(inputs)) {
		for _, config := range inputs[file] {
			r := &benchfmt.Result{Name: benchfmt.Name("Final"), Iters: 1}
			for _, kv := range strings.Fields(config) {
				k, v, _ := strings.Cut(kv, "=")
				setFileConfig(r, k, v)
			}
			r.SetConfig(".file", file)
			rs = append(rs, r)
		}
	}
	return rs
}

func TestAutoColumns(t *testing.T) {
	tests := []struct {
		name   string
		inputs map[string][]string
		want   []string
	}{
		{
			"differ between files",
			map[string][]string{
				"a": {"lr=1 seed=1 mode=x", "lr=1 seed=2 mode=x"},
				"b": {"lr=2 seed=1 mode=x", "lr=2 seed=2 mode=x"},
			},
			[]string{"lr"},
		},
		{
			"differ within one file",
			map[string][]string{"a": {"lr=1 seed=1", "lr=2 seed=1"}},
			[]string{"lr"},
		},
		{
			"set by one input only",
			map[string][]string{
				"a": {"lr=1 goos=linux", "goos=linux"},
				"b": {"goos=linux"},
			},
			nil,
		},
		{
			"nothing differs",
			map[string][]string{"a": {"lr=1"}, "b": {"lr=1"}},
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := autoColumns(results(tt.inputs), []string{"seed"})
			if !slices.Equal(got, tt.want) {
				t.Errorf("autoColumns = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFillColumns(t *testing.T) {
	rs := results(map[string][]string{
		"a": {"mode=lora", "pkg=fit"},
		"b": {"mode=full", "mode=full2", "pkg=fit"},
	})
	fillColumns(rs, []string{"mode"})
	var got []string
	for _, r := range rs {
		got = append(got, r.GetConfig(".file")+":"+r.GetConfig("mode"))
	}
	want := []string{"a:lora", "a:lora", "b:full", "b:full2", "b:"}
	if !slices.Equal(got, want) {
		t.Errorf("after fillColumns: %q, want %q", got, want)
	}
}

func TestMerge(t *testing.T) {
	a := &analysis{d: newData()}
	tests := []struct {
		name     string
		unit     string
		xs       []float64
		reps     []string
		want     []float64
		wantReps []string
		bad      string
	}{
		{"one per replicate", "loss", []float64{3, 1, 2}, []string{"3", "1", "10"}, []float64{1, 3, 2}, []string{"1", "3", "10"}, ""},
		{"identical repeats", "loss", []float64{1, 1, 2}, []string{"1", "1", "2"}, []float64{1, 2}, []string{"1", "2"}, ""},
		{"differing repeats", "loss", []float64{1, 1.5}, []string{"1", "1"}, nil, nil, "not reproducible"},
		{"no replicate", "loss", []float64{1, 1}, []string{"", ""}, []float64{1}, []string{""}, ""},
		{"time keeps all", "sec/step", []float64{1, 1.5}, []string{"1", "1"}, []float64{1, 1.5}, nil, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := a.merge(tt.unit, tt.xs, tt.reps, make([]string, len(tt.xs)))
			if s.bad != tt.bad {
				t.Fatalf("bad = %q, want %q", s.bad, tt.bad)
			}
			if tt.bad != "" {
				return
			}
			if !slices.Equal(s.xs, tt.want) || !slices.Equal(s.reps, tt.wantReps) {
				t.Errorf("merge = %v %q, want %v %q", s.xs, s.reps, tt.want, tt.wantReps)
			}
		})
	}
}

func TestQuantile(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	for _, tt := range []struct{ p, want float64 }{{0, 1}, {0.5, 5}, {0.9, 9}, {0.99, 10}, {1, 10}} {
		if got := quantile(xs, tt.p); got != tt.want {
			t.Errorf("quantile(%v) = %v, want %v", tt.p, got, tt.want)
		}
	}
}
