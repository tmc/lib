package dsv41engram_test

import (
	"bytes"
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
)

func Example() {
	raw := make([]byte, dsv41engram.RowBytes)
	table, err := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	values := make([]float32, dsv41engram.Dim)
	err = table.ReadRows([]uint32{0}, values)
	fmt.Println(len(values), values[0], err)
	// Output: 256 0 <nil>
}

func exampleLayout() *dsv41engram.Layout {
	s := dsv41engram.LayoutSpec{TokenMap: []uint32{0, 1}, CompressedVocabSize: 2, PadID: 0}
	for l := range dsv41engram.Layers {
		s.Multipliers[l] = [4]uint64{1, 3, 5, 7}
		for c := range dsv41engram.Columns {
			s.Primes[l][c] = 2
			s.Rows[l] += 2
		}
	}
	l, _ := dsv41engram.NewLayout(s)
	return l
}

func ExampleNewLayout() {
	l := exampleLayout()
	var h dsv41engram.History
	rows := make([]uint32, dsv41engram.Layers*dsv41engram.Columns)
	err := l.Hash(&h, []int32{1}, nil, rows)
	fmt.Println(rows[0], rows[23], err)
	// Output: 1 47 <nil>
}

func ExampleLayout_Hash() {
	var history dsv41engram.History
	rows := make([]uint32, 2*dsv41engram.Layers*dsv41engram.Columns)
	err := exampleLayout().Hash(&history, []int32{1, 0}, []byte{1, 0}, rows)
	fmt.Println(rows[0], rows[48], err)
	// Output: 1 0 <nil>
}

func ExampleHistory_Reset() {
	var h dsv41engram.History
	h.Reset()
	rows := make([]uint32, dsv41engram.Layers*dsv41engram.Columns)
	fmt.Println(exampleLayout().Hash(&h, []int32{0}, nil, rows))
	// Output: <nil>
}

func ExampleNewTable() {
	raw := make([]byte, dsv41engram.RowBytes)
	_, err := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 1)
	fmt.Println(err)
	// Output: <nil>
}

func ExampleDecodeRow() {
	raw := make([]byte, dsv41engram.RowBytes)
	values := make([]float32, dsv41engram.Dim)
	fmt.Println(dsv41engram.DecodeRow(raw, values), values[0])
	// Output: <nil> 0
}

func ExampleTable_ReadEncodedRows() {
	raw := make([]byte, dsv41engram.RowBytes)
	table, _ := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 1)
	encoded := make([]byte, dsv41engram.RowBytes)
	fmt.Println(table.ReadEncodedRows([]uint32{0}, encoded), len(encoded))
	// Output: <nil> 264
}

func ExampleTable_ReadRows() {
	raw := make([]byte, dsv41engram.RowBytes)
	table, _ := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 1)
	values := make([]float32, dsv41engram.Dim)
	fmt.Println(table.ReadRows([]uint32{0}, values), values[0])
	// Output: <nil> 0
}

func ExampleTable_ReadBatch() {
	raw := make([]byte, dsv41engram.RowBytes)
	table, _ := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 1)
	values := make([]float32, dsv41engram.Columns*dsv41engram.Dim)
	fmt.Println(table.ReadBatch(make([]uint32, dsv41engram.Columns), 1, dsv41engram.Columns, values), len(values))
	// Output: <nil> 6144
}
