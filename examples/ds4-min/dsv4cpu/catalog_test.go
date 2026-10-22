package dsv4cpu

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
)

func ExampleNewCatalog() {
	c, err := NewCatalog(bytes.NewReader(make([]byte, 8)), 8, []Tensor{{"norm", []int{2}, F32, 0, 8}})
	fmt.Println(c != nil, err)
	// Output: true <nil>
}

func ExampleCatalog_Matrix() {
	c, _ := NewCatalog(bytes.NewReader(make([]byte, 8)), 8, []Tensor{{"norm", []int{2}, F32, 0, 8}})
	m, err := c.Matrix("norm", 0)
	fmt.Println(m != nil, err)
	// Output: true <nil>
}

type noReadSource struct{}

func (noReadSource) ReadAt([]byte, int64) (int, error) { return 0, errors.New("unexpected model read") }

func TestFlashExpertCatalog(t *testing.T) {
	gateBytes := int64(4096/256*66) * 2048 * 256
	downBytes := int64(2048/256*84) * 4096 * 256
	dims := []int{4096, 2048, 256}
	c, err := NewCatalog(noReadSource{}, gateBytes+downBytes, []Tensor{{"blk.0.ffn_gate_exps.weight", dims, IQ2XXS, 0, gateBytes}, {"blk.0.ffn_down_exps.weight", []int{2048, 4096, 256}, Q2K, gateBytes, downBytes}})
	if err != nil {
		t.Fatal(err)
	}
	dims[0] = 1 // Caller mutation cannot alter the model view.
	m, err := c.Matrix("blk.0.ffn_gate_exps.weight", 255)
	if err != nil {
		t.Fatal(err)
	}
	if m.cols != 4096 || m.rows != 2048 || m.offset != gateBytes/256*255 {
		t.Fatal("expert layout")
	}
	if _, err := c.Matrix("blk.0.ffn_gate_exps.weight", 256); err == nil {
		t.Fatal("bad expert accepted")
	}
	if _, err := c.Matrix("missing", 0); err == nil {
		t.Fatal("missing tensor accepted")
	}
}
