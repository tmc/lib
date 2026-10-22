package main

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestShapeOutputWeightsSource(t *testing.T) {
	b, err := os.ReadFile("testdata/shape-output-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != "930bdc238f9a3ad86b44bae01ee026d9f89eb09e52e92d0aaceba556f6683e9e" {
		t.Fatal("output fixture pin mismatch")
	}
	want := strings.Replace(string(b), "__global__ static void output_hc_weights_kernel", "extern \"C\" __global__ void ds4_shape_output_hc_weights", 1)
	production, err := os.ReadFile("cuda_shape_output.cu")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(production), want) {
		t.Fatal("output kernel differs from pinned body")
	}
}

// This body contains no barriers or warp operations. Host C++ executes the
// actual scalar body, not an emulation of CUDA scheduling or native rounding.
func TestShapeOutputWeightsValues(t *testing.T) {
	path := "cuda_shape_output.cu"
	if p := os.Getenv("DS4_SHAPE_OUTPUT_TEST_SOURCE"); p != "" {
		path = p
	}
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	code := "#include <math.h>\n#include <stdio.h>\n#define __global__\nstruct {unsigned x;} blockIdx={0},blockDim={256},threadIdx={0};\n" + string(source)
	code += "\nint main(){float pre[4]={-4,-1,1,4},scale[1]={0.5f},base[4]={0.25f,-0.25f,0.5f,-0.5f},out[4]={};for(int i=0;i<4;i++){threadIdx.x=i;ds4_shape_output_hc_weights(out,pre,scale,base,4,1,1e-6f);}for(int i=0;i<4;i++)printf(\"%.9g \",out[i]);}\n"
	d := t.TempDir()
	input, output := filepath.Join(d, "output.cpp"), filepath.Join(d, "output")
	if err := os.WriteFile(input, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command(cc, input, "-o", output).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, b)
	}
	b, err := exec.Command(output).Output()
	if err != nil {
		t.Fatal(err)
	}
	fields := strings.Fields(string(b))
	if len(fields) != 4 {
		t.Fatalf("output value count=%d", len(fields))
	}
	pre, base := []float64{-4, -1, 1, 4}, []float64{0.25, -0.25, 0.5, -0.5}
	for i, field := range fields {
		got, err := strconv.ParseFloat(field, 64)
		if err != nil {
			t.Fatal(err)
		}
		want := 1/(1+math.Exp(-(pre[i]*0.5+base[i]))) + 1e-6
		if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 2e-7 {
			t.Fatalf("OUTPUT_WEIGHT_MISMATCH row=%d got=%g want=%g", i, got, want)
		}
	}
}
