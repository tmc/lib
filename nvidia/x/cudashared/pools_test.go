package cudashared

import (
	"math"
	"testing"
)

func TestIndependentPoolAccounting(t *testing.T) {
	p, _ := NewPlan(Limits{StaticMax: 64, DynamicMax: 64, TotalMax: 80, Source: "test"})
	if _, err := p.Add(Request{Name: "static", Size: 32, Alignment: 16}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Add(Request{Name: "dynamic", Size: 48, Alignment: 16, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	if p.StaticBytes() != 32 || p.DynamicBytes() != 48 || p.FootprintBytes() != 80 {
		t.Fatal("incorrect separate allocation totals")
	}
	if _, err := p.Add(Request{Name: "overflow", Size: 1, Alignment: 1, Dynamic: true}); err == nil {
		t.Fatal("accepted aggregate pool overflow")
	}
	if err := p.ValidateBase(math.MaxUint64 - 15); err == nil {
		t.Fatal("accepted wrapping region end")
	}
}
