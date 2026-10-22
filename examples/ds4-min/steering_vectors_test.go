package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestSteeringVectorsAdmission(t *testing.T) {
	var raw bytes.Buffer
	for _, value := range []float32{1, 0, 0, 1} {
		if err := binary.Write(&raw, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	v, err := readSteeringVectors(bytes.NewReader(raw.Bytes()), 2, 2)
	if err != nil || !reflect.DeepEqual(v.values, []float32{1, 0, 0, 1}) {
		t.Fatalf("vectors %v error %v", v, err)
	}
	if v.digest != sha256.Sum256(raw.Bytes()) {
		t.Fatal("direction identity is not the exact file digest")
	}
	for _, data := range [][]byte{raw.Bytes()[:15], append(append([]byte(nil), raw.Bytes()...), 0)} {
		if _, err := readSteeringVectors(bytes.NewReader(data), 2, 2); err == nil {
			t.Fatal("accepted wrong extent")
		}
	}
	bad := append([]byte(nil), raw.Bytes()...)
	binary.LittleEndian.PutUint32(bad, math.Float32bits(float32(math.NaN())))
	if _, err := readSteeringVectors(bytes.NewReader(bad), 2, 2); err == nil {
		t.Fatal("accepted nonfinite direction")
	}
}

func TestSteeringProjectionValues(t *testing.T) {
	for _, tt := range []struct {
		scale float32
		want  []float32
	}{{0, []float32{2, -1}}, {0.5, []float32{1, -1}}, {-1, []float32{4, -1}}} {
		x := []float32{2, -1}
		if err := steeringProject(x, []float32{1, 0}, tt.scale); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(x, tt.want) {
			t.Fatalf("scale %v: %v want %v", tt.scale, x, tt.want)
		}
	}
	x := []float32{3, -2}
	if err := steeringProject(x, []float32{2, 1}, 0.5); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, []float32{-1, -4}) {
		t.Fatalf("raw nonunit directions normalized or projection wrong: %v", x)
	}
}
