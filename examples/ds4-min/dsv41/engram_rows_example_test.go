package dsv41

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
)

func exampleEngramRowSources() (*dsv41engram.Layout, [2]*dsv41engram.Table) {
	spec := dsv41engram.LayoutSpec{TokenMap: []uint32{0, 1}, CompressedVocabSize: 2, Rows: [2]uint32{48, 48}}
	for i := range spec.Primes {
		spec.Multipliers[i] = [4]uint64{1, 3, 5, 7}
		for j := range spec.Primes[i] {
			spec.Primes[i][j] = 2
		}
	}
	layout, _ := dsv41engram.NewLayout(spec)
	raw := make([]byte, 48*dsv41engram.RowBytes)
	table, _ := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 48)
	return layout, [2]*dsv41engram.Table{table, table}
}

func ExampleNewEngramRows() {
	layout, tables := exampleEngramRowSources()
	s, err := NewEngramRows(layout, tables, 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer s.Close()
	fmt.Println(s.Prepare(1, true, 0))
	rows, text, err := s.Rows(1)
	fmt.Println(len(rows), text, err, s.Position())
	fmt.Println(s.Commit(), s.Position())
	// Output:
	// <nil>
	// 6144 true <nil> 0
	// <nil> 1
}

func ExampleModel_NewEngramRows() {
	var model *Model
	_, err := model.NewEngramRows(4)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleEngramRows_Prepare() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	defer s.Close()
	fmt.Println(s.Prepare(0, false, 0), s.Position())
	// Output: <nil> 0
}
func ExampleEngramRows_Rows() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	defer s.Close()
	_ = s.Prepare(0, false, 0)
	rows, text, err := s.Rows(14)
	fmt.Println(len(rows), text, err)
	// Output: 0 false <nil>
}
func ExampleEngramRows_Commit() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	defer s.Close()
	_ = s.Prepare(0, true, 0)
	fmt.Println(s.Commit(), s.Position())
	// Output: <nil> 1
}
func ExampleEngramRows_Abort() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	defer s.Close()
	_ = s.Prepare(0, true, 0)
	s.Abort()
	fmt.Println(s.Position(), s.Commit() != nil)
	// Output: 0 true
}
func ExampleEngramRows_Reset() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	defer s.Close()
	_ = s.Prepare(0, true, 0)
	_ = s.Commit()
	fmt.Println(s.Reset(), s.Position())
	// Output: <nil> 0
}
func ExampleEngramRows_Close() {
	layout, tables := exampleEngramRowSources()
	s, _ := NewEngramRows(layout, tables, 4)
	fmt.Println(s.Close(), s.Close())
	// Output: <nil> <nil>
}
