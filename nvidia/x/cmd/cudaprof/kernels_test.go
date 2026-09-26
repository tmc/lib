package main

import (
	"bytes"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

func TestKernelReport(t *testing.T) {
	cfg := &LaunchConfig{Grid: [3]uint32{4, 1, 1}, Block: [3]uint32{128, 1, 1}}
	recs := []rec{
		{subtype: "kernel", name: "LayerNorm", start: 1000, end: 1010, config: cfg},
		{subtype: "kernel", name: "LayerNorm", start: 1020, end: 1040, config: cfg},
		{subtype: "kernel", name: "LayerNorm", start: 1050, end: 1080, config: cfg},
		{subtype: "kernel", name: "LayerNormBackward", start: 1100, end: 1200},
		{subtype: "kernel", name: "gemm", start: 1200, end: 1300, config: cfg},
	}
	r, err := kernelReport(compareCapture(recs, phaseStatus(1000, 2000)), regexp.MustCompile("LayerNorm"), compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	var got []KernelStat
	for _, k := range r.Kernels {
		k.Configs = nil
		got = append(got, k)
	}
	want := []KernelStat{
		{Kernel: "LayerNormBackward", Launches: 1, SummedNanos: 100, ExclusiveNanos: 100, MeanNanos: 100, MedianNanos: 100, P90Nanos: 100, UnrecordedConfigs: 1},
		{Kernel: "LayerNorm", Launches: 3, SummedNanos: 60, ExclusiveNanos: 60, MeanNanos: 20, MedianNanos: 20, P90Nanos: 30},
	}
	if len(got) != len(want) {
		t.Fatalf("kernels = %+v, want %+v", got, want)
	}
	for i := range want {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Errorf("kernel %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	if n := len(r.Kernels[1].Configs); n != 1 || r.Kernels[1].Configs[0].Launches != 3 {
		t.Errorf("LayerNorm configs = %+v, want one config with 3 launches", r.Kernels[1].Configs)
	}
	var out bytes.Buffer
	if err := r.write(&out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"3 60 60 20 20 30 1: grid 4x1x1 block 128x1x1 smem 0 LayerNorm", "1 100 100 100 100 100 - LayerNormBackward"} {
		if !strings.Contains(squash(out.String()), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "gemm") {
		t.Errorf("output lists a kernel -grep excludes:\n%s", out.String())
	}
}
