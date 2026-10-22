package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

func ExampleMatrix_MulQ8K_q4() {
	block := make([]byte, 144)
	binary.LittleEndian.PutUint16(block, 0x3c00)
	block[4], block[16] = 1, 1
	m, _ := NewMatrix(bytes.NewReader(block), 0, 144, 1, 256, Q4K)
	x := make([]Q8K, 1)
	x[0].Scale = 1
	x[0].Values[0] = 1
	out := make([]float32, 1)
	_ = m.MulQ8K(out, x)
	fmt.Println(out[0])
	// Output: 1
}

func TestQ4MatrixShapeAndExpertSlices(t *testing.T) {
	if _, err := NewMatrix(bytes.NewReader(nil), 0, 144, 1, 257, Q4K); err == nil {
		t.Fatal("partial Q4 block accepted")
	}
	data := make([]byte, 288)
	for expert := 0; expert < 2; expert++ {
		b := data[expert*144:]
		binary.LittleEndian.PutUint16(b, 0x3c00)
		b[4] = 1
		b[16] = byte(expert + 1)
	}
	c, err := NewCatalog(bytes.NewReader(data), int64(len(data)), []Tensor{{Name: "experts", Dimensions: []int{256, 1, 2}, Encoding: Q4K, Bytes: 288}})
	if err != nil {
		t.Fatal(err)
	}
	for expert := 0; expert < 2; expert++ {
		m, err := c.Matrix("experts", expert)
		if err != nil {
			t.Fatal(err)
		}
		row := make([]float32, 256)
		if err := m.Row(row, 0); err != nil {
			t.Fatal(err)
		}
		if row[0] != float32(expert+1) {
			t.Fatal("wrong expert slice", expert, row[0])
		}
	}
}
