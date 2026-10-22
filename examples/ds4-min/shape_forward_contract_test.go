package main

import (
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
)

func forwardTestBinding(t *testing.T, c dsv4shape.Config) *dsv4shape.Binding {
	t.Helper()
	specs, err := c.Weights()
	if err != nil {
		t.Fatal(err)
	}
	b := new(dsv4shape.Binding)
	for _, s := range specs {
		x := gguf.Tensor{TypeID: s.Types[0], Dimensions: s.Dimensions}
		n, err := x.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		x.Dimensions = s.Dimensions[:1]
		row, err := x.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		w := dsv4shape.WeightView{Name: s.Name, Type: s.Types[0], Bytes: n, RowBytes: row}
		if len(s.Dimensions) == 3 {
			w.ExpertBytes = n / s.Dimensions[2]
		}
		b.Weights = append(b.Weights, w)
	}
	return b
}

func TestShapeForwardPlan(t *testing.T) {
	for _, v := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, _ := dsv4shape.For(v)
		b := forwardTestBinding(t, c)
		p, err := planShapeForward(c, b, 32768)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.buffers) != 39 || len(p.norms) != 2*c.Layers {
			t.Fatalf("buffers=%d norms=%d", len(p.buffers), len(p.norms))
		}
		got := map[string]int{}
		var bytes uint64
		for _, x := range p.buffers {
			got[x.name] = x.elements
			bytes += uint64(x.elements) * 4
		}
		if bytes != p.bytes || got["cur"] != c.Embed*4 || got["q"] != c.Heads*512 || got["qr"] != c.QLoRA || got["attn_low"] != c.OutGroups*1024 || got["routed_mid_all"] != 6*c.ExpertFF || got["attn_score"] != 128+32768/4+2 {
			t.Fatalf("wrong scratch dimensions: %v", got)
		}
		b.Weights[1].Type = 1
		if _, err := planShapeForward(c, b, 32768); err == nil {
			t.Fatal("accepted half norm")
		}
		b.Weights[1].Type = 0
		b.Weights = b.Weights[1:]
		if _, err := planShapeForward(c, b, 32768); err == nil {
			t.Fatal("accepted missing norm")
		}
	}
	flash, _ := dsv4shape.For(dsv4shape.Flash)
	pro, _ := dsv4shape.For(dsv4shape.Pro)
	if _, err := planShapeForward(pro, forwardTestBinding(t, flash), 2048); err == nil {
		t.Fatal("admitted Flash binding as Pro")
	}
}
