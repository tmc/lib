package dsparkstate_test

import (
	"bytes"
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/internal/dsparkstate"
)

func Example() {
	rows, values, err := dsparkstate.Shape(4, 1, 1, 4096)
	if err != nil {
		fmt.Println(err)
		return
	}
	j, err := dsparkstate.NewJournal(rows, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	v := &dsparkstate.Envelope{Platform: 3, Model: sha256.Sum256([]byte("model")), Context: 4, Block: 1, Targets: []int{1}, Journal: j, History: make([]float32, values)}
	var buf bytes.Buffer
	if err := dsparkstate.Write(&buf, v, 0, 0); err != nil {
		fmt.Println(err)
		return
	}
	decoded, err := dsparkstate.Read(buf.Bytes(), v, 0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(decoded.Platform, decoded.Journal.Frontier)
	// Output: 3 0
}
