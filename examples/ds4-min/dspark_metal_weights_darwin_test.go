//go:build darwin && arm64

package main

import (
	"strings"
	"testing"
)

func TestMetalDSparkWeightOwnership(t *testing.T) {
	w := &metalDSparkWeights{tensors: map[string]boundMetalTensor{"x": {dimensions: []uint64{32, 2}}}}
	if _, err := w.tensor("x"); err == nil {
		t.Fatal("unleased access admitted")
	}
	if err := w.acquire(); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err == nil {
		t.Fatal("live consumer release admitted")
	}
	v, err := w.tensor("x")
	if err != nil {
		t.Fatal(err)
	}
	v.dimensions[0] = 1
	if w.tensors["x"].dimensions[0] != 32 {
		t.Fatal("descriptor aliases caller")
	}
	if err := w.release(); err != nil {
		t.Fatal(err)
	}
	if err := w.release(); err == nil {
		t.Fatal("duplicate release admitted")
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.acquire(); err == nil {
		t.Fatal("closed weights admitted")
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMetalDSparkWeightPlanRejectsUnverified(t *testing.T) {
	if _, err := metalDSparkWeightPlan(nil); err == nil {
		t.Fatal("missing model admitted")
	}
	m, err := admitDSparkModel(supportFixture())
	if err != nil {
		t.Fatal(err)
	}
	m.sha256 = strings.Repeat("a", 64)
	if _, err := metalDSparkWeightPlan(m); err == nil {
		t.Fatal("missing model reader admitted")
	}
}
