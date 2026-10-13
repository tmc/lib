package main

import (
	"math"
	"reflect"
	"testing"
)

func TestFFNPlanRefusals(t *testing.T) {
	max := int(^uint(0) >> 1)
	for _, tt := range []struct {
		name   string
		h, i   int
		eps    float32
		budget int
	}{
		{"zero hidden", 0, 3, 1e-5, 100}, {"negative intermediate", 3, -1, 1e-5, 100},
		{"zero epsilon", 3, 5, 0, 100}, {"negative epsilon", 3, 5, -1, 100},
		{"nan epsilon", 3, 5, float32(math.NaN()), 100},
		{"infinite epsilon", 3, 5, float32(math.Inf(1)), 100},
		{"matrix bytes", max / 4, 2, 1e-5, max},
		{"scratch bytes", max / 4, 1, 1e-5, max},
		{"short budget", 3, 5, 1e-5, 83}, {"negative budget", 3, 5, 1e-5, -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := newFFNPlan(tt.h, tt.i, tt.eps, tt.budget); err == nil {
				t.Fatal("accepted invalid plan")
			}
		})
	}
	p, err := newFFNPlan(3, 5, 1e-5, 84)
	if err != nil {
		t.Fatal(err)
	}
	if p.scratchWords != 21 || p.weightWords != 15 {
		t.Fatalf("counts: %+v", p)
	}
}

func TestFFNPlanLengths(t *testing.T) {
	p, err := newFFNPlan(3, 5, 1e-5, 84)
	if err != nil {
		t.Fatal(err)
	}
	valid := func() ffnBuffers {
		return ffnBuffers{x: make([]float32, 3), norm: make([]float32, 3), gate: make([]float32, 15), up: make([]float32, 15), down: make([]float32, 15)}
	}
	for _, name := range []string{"input", "norm", "gate", "up", "down"} {
		for _, delta := range []int{-1, 1} {
			b := valid()
			var v *[]float32
			switch name {
			case "input":
				v = &b.x
			case "norm":
				v = &b.norm
			case "gate":
				v = &b.gate
			case "up":
				v = &b.up
			case "down":
				v = &b.down
			}
			*v = make([]float32, len(*v)+delta)
			out, err := p.reference(b)
			if err == nil || out != nil {
				t.Fatalf("%s delta %d accepted", name, delta)
			}
		}
	}
	if out, err := (ffnPlan{}).reference(valid()); err == nil || out != nil {
		t.Fatal("zero plan accepted")
	}
	p.scratchWords--
	if out, err := p.reference(valid()); err == nil || out != nil {
		t.Fatal("corrupt plan accepted")
	}
}

func TestFFNPlanDependencies(t *testing.T) {
	p, err := newFFNPlan(3, 5, 1e-5, 84)
	if err != nil {
		t.Fatal(err)
	}
	want := []ffnStage{{name: "normalize"}, {name: "gate", after: []string{"normalize"}}, {name: "up", after: []string{"normalize"}}, {name: "activate", after: []string{"gate", "up"}}, {name: "down", after: []string{"activate"}}, {name: "residual", after: []string{"down"}}}
	got := p.stages()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stages: %#v", got)
	}
	got[3].after[0] = "changed"
	if !reflect.DeepEqual(p.stages(), want) {
		t.Fatal("stage dependencies alias caller storage")
	}
}
