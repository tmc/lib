package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

func ExampleMatrix() {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint32(b, math.Float32bits(2))
	binary.LittleEndian.PutUint32(b[4:], math.Float32bits(3))
	m, _ := NewMatrix(bytes.NewReader(b), 0, 8, 1, 2, F32)
	out := make([]float32, 1)
	_ = m.Mul(out, []float32{4, 5})
	fmt.Println(out)
	// Output: [23]
}

func ExampleRMSNorm() {
	x := []float32{3, 4}
	_ = RMSNorm(x, x, nil, 1e-6)
	fmt.Printf("%.3f %.3f\n", x[0], x[1])
	// Output: 0.849 1.131
}

func TestMatrixRealDimensions(t *testing.T) {
	for _, cols := range []int{512, 1024, 2048, 4096, 16384} {
		b := make([]byte, cols/32*34)
		want := make([]float32, cols)
		for i := range want {
			binary.LittleEndian.PutUint16(b[i/32*34:], 0x3800)
			q := int8(i%251 - 125)
			b[i/32*34+2+i%32] = byte(q)
			want[i] = .5 * float32(q)
		}
		m, err := NewMatrix(bytes.NewReader(b), 0, int64(len(b)), 1, cols, Q80)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, cols)
		if err := m.Row(got, 0); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("cols=%d index=%d got=%g want=%g", cols, i, got[i], want[i])
			}
		}
	}
}

func TestRMSRealDimensions(t *testing.T) {
	for _, n := range []int{128, 512, 1024, 4096, 16384} {
		x, w, got := make([]float32, n), make([]float32, n), make([]float32, n)
		var ss float64
		for i := range x {
			x[i] = float32(i%97-48) / 17
			w[i] = float32(i%13+1) / 9
			ss += float64(x[i]) * float64(x[i])
		}
		if err := RMSNorm(got, x, w, 1e-6); err != nil {
			t.Fatal(err)
		}
		scale := float32(1) / float32(math.Sqrt(float64(float32(ss/float64(n))+float32(1e-6))))
		for i := range got {
			if got[i] != x[i]*scale*w[i] {
				t.Fatalf("n=%d i=%d", n, i)
			}
		}
	}
}

func TestRowReadFailureDoesNotPublish(t *testing.T) {
	m, err := NewMatrix(bytes.NewReader([]byte{1}), 0, 4, 1, 1, F32)
	if err != nil {
		t.Fatal(err)
	}
	dst := []float32{99}
	if err := m.Row(dst, 0); err == nil || dst[0] != 99 {
		t.Fatalf("row: %v %v", dst, err)
	}
}

type shortReader struct{}

func (shortReader) ReadAt(b []byte, off int64) (int, error) { return 1, nil }

func TestInvalidMatrix(t *testing.T) {
	for _, m := range []*Matrix{nil, new(Matrix)} {
		if m.Row(nil, 0) == nil || m.Mul(nil, nil) == nil {
			t.Fatal("invalid matrix accepted")
		}
	}
	m, err := NewMatrix(shortReader{}, 0, 4, 1, 1, F32)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{99}
	if err := m.Row(x, 0); err == nil || x[0] != 99 {
		t.Fatalf("short read %v %v", x, err)
	}
}
