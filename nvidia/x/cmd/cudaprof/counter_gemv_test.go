package main

import (
	"bytes"
	"github.com/tmc/lib/nvidia/x/cudaprof/counters"
	"reflect"
	"strings"
	"testing"
)

func TestCounterGEMVOptions(t *testing.T) {
	for _, args := range [][]string{{"-rows=0"}, {"-cols=8193"}, {"-max-passes=65"}, {"-implementation=unknown"}, {"-iterations=1"}} {
		if _, err := parseCounterGEMV("workload", args, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	if _, err := parseCounterGEMV("collect", nil, &bytes.Buffer{}); err == nil {
		t.Fatal("collect accepted missing selection")
	}
	o, err := parseCounterGEMV("collect", []string{"-selection=s.json", "-metrics=dram__bytes.sum", "-arm=target"}, &bytes.Buffer{})
	if err != nil || o.arm != "target" {
		t.Fatalf("options=%+v err=%v", o, err)
	}
}
func TestCounterGEMVInputs(t *testing.T) {
	for _, dims := range [][2]int{{1, 1}, {128, 256}, {3, 9}} {
		a, x, want := counterGEMVInputs(dims[0], dims[1])
		got := make([]float32, dims[0])
		for i := range got {
			for j, v := range x {
				got[i] += a[i*dims[1]+j] * v
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("CPU output=%v want=%v", got, want)
		}
		if counterGEMVFloatHash(got) != counterGEMVFloatHash(want) {
			t.Fatal("unstable completed output digest")
		}
	}
}
func TestCounterGEMVIdentity(t *testing.T) {
	o := counterGEMVOptions{rows: 129, cols: 256}
	id := counterGEMVIdentity(o, 3)
	if id.Operation != "controlled_gemv_129x256" || id.Token != 3 || id.Grid != [3]int{2, 1, 1} || id.Block != [3]int{128, 1, 1} {
		t.Fatalf("identity=%+v", id)
	}
	o.cols++
	if reflect.DeepEqual(id, counterGEMVIdentity(o, 3)) {
		t.Fatal("different dimensions share identity")
	}
}
func TestCounterGEMVUnlocked(t *testing.T) {
	t.Setenv("GPU_LOCK_HELD", "")
	err := runCounterGEMV([]string{"workload"}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "GPU_LOCK_HELD") {
		t.Fatalf("unlocked error=%v", err)
	}
}

func TestCounterGEMVHistoricalSourceRequired(t *testing.T) {
	source := testCounterShapeSource()
	selection, err := counters.Select(source.Shape, source.Shape.Groups[0].Identity)
	if err != nil {
		t.Fatal(err)
	}
	report := counterSelectionReport{Version: 1, Source: source, Selection: &selection}
	if err := revalidateCounterGEMVSelection(report); err == nil || !strings.Contains(err.Error(), "original source capture path") {
		t.Fatalf("source validation error=%v", err)
	}
}
