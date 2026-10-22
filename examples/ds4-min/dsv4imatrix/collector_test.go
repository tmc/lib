package dsv4imatrix

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"testing"
)

func TestSelectedValuesAndAdmission(t *testing.T) {
	c, err := New(1, 3, 2, 2, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Observe(0, []int{2, 0}, []float32{2, -3}, []float32{4, 5, 6, 7}); err != nil {
		t.Fatal(err)
	}
	if c.gate[4] != 4 || c.gate[5] != 9 || c.down[4] != 16 || c.down[5] != 25 || c.down[0] != 36 || c.down[1] != 49 {
		t.Fatalf("wrong slot/expert mapping: %v %v", c.gate, c.down)
	}
	var before bytes.Buffer
	if err := c.Write(&before, "fixture"); err != nil {
		t.Fatal(err)
	}
	for _, ids := range [][]int{{2, 2}, {3}, {-1}} {
		if err := c.Observe(0, ids, []float32{2, 3}, make([]float32, len(ids)*2)); err == nil {
			t.Fatal("invalid expert accepted")
		}
	}
	if err := c.Observe(0, []int{1}, []float32{float32(math.NaN()), 1}, []float32{1, 1}); err == nil {
		t.Fatal("NaN accepted")
	}
	var after bytes.Buffer
	_ = c.Write(&after, "fixture")
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatal("failed observation mutated output")
	}
	r := bytes.NewReader(before.Bytes())
	var entries int32
	_ = binary.Read(r, binary.LittleEndian, &entries)
	if entries != 3 {
		t.Fatal(entries)
	}
	for entry := 0; entry < 3; entry++ {
		var n, call, count int32
		_ = binary.Read(r, binary.LittleEndian, &n)
		name := make([]byte, n)
		_, _ = r.Read(name)
		_ = binary.Read(r, binary.LittleEndian, &call)
		_ = binary.Read(r, binary.LittleEndian, &count)
		if call != 1 || count != 6 {
			t.Fatal("bad entry header")
		}
		v := make([]float32, count)
		_ = binary.Read(r, binary.LittleEndian, &v)
		if v[2] != 0 || v[3] != 0 {
			t.Fatal("unobserved expert nonzero")
		}
	}
	mutant, _ := New(1, 3, 2, 2, 4096)
	_ = mutant.Observe(0, []int{0, 2}, []float32{2, -3}, []float32{4, 5, 6, 7})
	if mutant.down[0] == c.down[0] {
		t.Fatal("swapped-slot mutation escaped value oracle")
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) / 2, nil }

func TestLimitsAndShortWrite(t *testing.T) {
	if _, err := New(1, 3, 2, 2, 59); err == nil {
		t.Fatal("byte limit ignored")
	}
	c, _ := New(1, 3, 2, 2, 60)
	if err := c.Write(shortWriter{}, "x"); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write: %v", err)
	}
	if err := c.Observe(0, []int{0}, []float32{math.MaxFloat32, 1}, []float32{1, 1}); err == nil {
		t.Fatal("overflow accepted")
	}
	if c.counts[0] != 0 {
		t.Fatal("overflow mutated count")
	}
	// A float64 reference independently checks the squared-activation means.
	for _, x := range []float32{0.125, 0.25, 0.5} {
		if err := c.Observe(0, []int{0}, []float32{x, x}, []float32{x, x}); err != nil {
			t.Fatal(err)
		}
	}
	want := (float64(0.125)*0.125 + float64(0.25)*0.25 + float64(0.5)*0.5) / 3
	got := float64(c.gate[0] * (float32(1) / float32(c.counts[0])))
	if math.Abs(got-want) > 1e-7 {
		t.Fatalf("mean=%g want %g", got, want)
	}
}

func ExampleCollector() {
	c, _ := New(1, 2, 1, 1, 1024)
	_ = c.Observe(0, []int{1}, []float32{2}, []float32{3})
	var output bytes.Buffer
	_ = c.Write(&output, "sample")
	fmt.Println(output.Len() > 0)
	// Output: true
}

func TestBatchIsOneChunkAndAtomic(t *testing.T) {
	c, _ := New(1, 2, 1, 1, 1024)
	if err := c.ObserveBatch(0, 1, []int{0, 1}, []float32{2, 3}, []float32{4, 5}); err != nil {
		t.Fatal(err)
	}
	if c.Chunks() != 1 || c.MinSamples() != 1 || c.gate[0] != 4 || c.gate[1] != 9 || c.down[0] != 16 || c.down[1] != 25 {
		t.Fatalf("batch values/counts: %+v", c)
	}
	var before, after bytes.Buffer
	_ = c.Write(&before, "batch")
	if err := c.ObserveBatch(0, 1, []int{0, 2}, []float32{2, 3}, []float32{4, 5}); err == nil {
		t.Fatal("late invalid row accepted")
	}
	_ = c.Write(&after, "batch")
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatal("rejected batch changed serialized statistics")
	}
}
