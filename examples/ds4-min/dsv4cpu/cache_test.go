package dsv4cpu

import (
	"fmt"
	"testing"
)

func ExampleCache() {
	c, _ := NewCache(2, 128)
	_ = c.Push([]float32{2, 4})
	out := make([]float32, 2)
	_ = c.Attend(out, []float32{0, 0}, []float32{0})
	fmt.Println(out)
	c.Reset()
	// Output: [1 2]
}

func ExampleNewCache() {
	c, err := NewCache(512, 128)
	fmt.Println(c != nil, err)
	// Output: true <nil>
}

func ExampleCache_Push() {
	c, _ := NewCache(2, 1)
	fmt.Println(c.Push([]float32{2, 4}))
	// Output: <nil>
}

func ExampleCache_Attend() {
	c, _ := NewCache(2, 1)
	_ = c.Push([]float32{2, 4})
	out := make([]float32, 2)
	_ = c.Attend(out, []float32{0, 0}, []float32{0})
	fmt.Println(out)
	// Output: [1 2]
}

func ExampleCache_Reset() {
	c, _ := NewCache(2, 1)
	_ = c.Push([]float32{2, 4})
	c.Reset()
	fmt.Println(c.count)
	// Output: 0
}

func TestCacheRequestIsolation(t *testing.T) {
	a, _ := NewCache(512, 128)
	b, _ := NewCache(512, 128)
	row := make([]float32, 512)
	for i := 0; i < 129; i++ {
		for j := range row {
			row[j] = float32(i)
		}
		if err := a.Push(row); err != nil {
			t.Fatal(err)
		}
	}
	if a.count != 128 || a.rows[0] != 1 || a.rows[127*512] != 128 || b.count != 0 {
		t.Fatal("request/window ownership")
	}
	a.Reset()
	if a.count != 0 || a.rows[127*512] != 0 {
		t.Fatal("reset")
	}
}

func TestMalformedShapes(t *testing.T) {
	if _, err := NewCache(0, 128); err == nil {
		t.Fatal("zero width")
	}
	if err := Attend(nil, nil, nil, nil, 512); err == nil {
		t.Fatal("empty attention")
	}
	if err := Split(nil, nil, nil, nil, 4, 20, 1e-6); err == nil {
		t.Fatal("empty split")
	}
	if err := Collapse(nil, nil, nil); err == nil {
		t.Fatal("empty collapse")
	}
	if err := Combine(nil, nil, nil, nil, nil); err == nil {
		t.Fatal("empty combine")
	}
}
