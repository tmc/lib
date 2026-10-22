package dsv41

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

func TestEngramExecutionAndIsolation(t *testing.T) {
	spec := dsv41engram.LayoutSpec{TokenMap: []uint32{0, 1}, CompressedVocabSize: 2, Rows: [2]uint32{48, 48}}
	for i := range spec.Multipliers {
		for j := range spec.Multipliers[i] {
			spec.Multipliers[i][j] = 1
		}
		for j := range spec.Primes[i] {
			spec.Primes[i][j] = 2
		}
	}
	layout, err := dsv41engram.NewLayout(spec)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 48*dsv41engram.RowBytes)
	for row := 0; row < 48; row++ {
		for j := 0; j < 256; j++ {
			raw[row*264+j] = 0x38
		}
		for j := 256; j < 264; j++ {
			raw[row*264+j] = 127
		}
	}
	table, err := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 48)
	if err != nil {
		t.Fatal(err)
	}
	const features = 24 * 256
	weight := make([]byte, 5*features*2)
	for row := 0; row < 5; row++ {
		binary.LittleEndian.PutUint16(weight[row*features*2:], 0x3c00)
	}
	matrix, err := dsv4cpu.NewMatrix(bytes.NewReader(weight), 0, int64(len(weight)), 5, features, dsv4cpu.F16)
	if err != nil {
		t.Fatal(err)
	}
	w := EngramWeights{Table: table, Projection: matrix, Query: []float32{1, 1, 1, 1}, Key: []float32{1, 1, 1, 1}}
	a, err := NewEngramState(layout, [2]EngramWeights{w, w})
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewEngramState(layout, [2]EngramWeights{w, w})
	if err != nil {
		t.Fatal(err)
	}
	x, y := []float32{1, 1, 1, 1}, []float32{1, 1, 1, 1}
	if err := a.Apply(1, x); err == nil {
		t.Fatal("unprepared apply admitted")
	}
	if err := a.Prepare(0, true); err != nil {
		t.Fatal(err)
	}
	if err := a.Apply(1, x); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(x, y) {
		t.Fatal("projection/add did not execute")
	}
	if err := b.Prepare(1, false); err != nil {
		t.Fatal(err)
	}
	if err := b.Apply(14, y); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(y, []float32{1, 1, 1, 1}) {
		t.Fatal("image mask or state isolation failed")
	}
	if err := a.Prepare(9, true); err == nil {
		t.Fatal("invalid token admitted")
	}
	if err := a.Apply(1, x); err == nil {
		t.Fatal("failed preparation remained live")
	}
	a.Reset()
	if err := a.Apply(1, x); err == nil {
		t.Fatal("reset retained prepared token")
	}
}
