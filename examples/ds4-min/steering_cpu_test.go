package main

import (
	"math"
	"reflect"
	"testing"
)

func TestSteeringCPURows(t *testing.T) {
	v := &steeringVectors{values: []float32{1, 0, 2, 3}, layers: 2, width: 2}
	x := []float32{1, 2, 3, 4}
	if err := v.projectCPU(x, 1, 2, -0.5); err != nil {
		t.Fatal(err)
	}
	if want := []float32{9, 14, 21, 31}; !reflect.DeepEqual(x, want) {
		t.Fatalf("projection = %v, want %v", x, want)
	}
	if !reflect.DeepEqual(v.values, []float32{1, 0, 2, 3}) {
		t.Fatal("directions mutated")
	}
	if err := v.projectCPU(x, 1, 2, 0); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, []float32{9, 14, 21, 31}) {
		t.Fatal("zero scale changed rows")
	}
}

func TestSteeringCPUSerialRounding(t *testing.T) {
	v := &steeringVectors{values: []float32{1, 1, 1}, layers: 1, width: 3}
	x := []float32{1e8, 1, -1e8}
	// The pinned serial reduction loses the middle one. Reassociation would
	// produce dot=1 and change that element, so this is not a GPU oracle.
	if err := v.projectCPU(x, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, []float32{1e8, 1, -1e8}) {
		t.Fatalf("reassociated dot: %v", x)
	}
}

func TestSteeringCPUAdmission(t *testing.T) {
	v := &steeringVectors{values: []float32{1, 2}, layers: 1, width: 2}
	if err := v.projectCPU(nil, 0, 1, 1); err == nil {
		t.Fatal("nil nonzero rows accepted")
	}
	if err := (*steeringVectors)(nil).projectCPU([]float32{1, 2}, 0, 1, 1); err == nil {
		t.Fatal("nil directions accepted")
	}
	if err := v.projectCPU(nil, 0, 1, 0); err != nil {
		t.Fatal("zero scale did not skip projection")
	}
	for _, tt := range []struct {
		name        string
		vectors     *steeringVectors
		layer, rows int
		scale       float32
	}{
		{"layer", &steeringVectors{values: []float32{1, 2}, layers: 1, width: 2}, 1, 1, 1},
		{"rows", &steeringVectors{values: []float32{1, 2}, layers: 1, width: 2}, 0, -1, 1},
		{"extent", &steeringVectors{values: []float32{1}, layers: 1, width: 2}, 0, 1, 1},
		{"overflow", &steeringVectors{layers: math.MaxInt, width: 2}, 0, 1, 1},
		{"finite", &steeringVectors{values: []float32{1, 2}, layers: 1, width: 2}, 0, 1, float32(math.NaN())},
	} {
		t.Run(tt.name, func(t *testing.T) {
			x := []float32{3, 4}
			if err := tt.vectors.projectCPU(x, tt.layer, tt.rows, tt.scale); err == nil {
				t.Fatal("invalid projection accepted")
			}
			if !reflect.DeepEqual(x, []float32{3, 4}) {
				t.Fatal("rejected projection changed rows")
			}
		})
	}
}
