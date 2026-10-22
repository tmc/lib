package main

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestShapeHCMixSource(t *testing.T) {
	b, err := os.ReadFile("testdata/shape-hc-mix-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "f09a0f73ad9b1139e0c27b3895043f557eff6f69ace5b2941053cdfd81ccf0da" {
		t.Fatal("HC mix pinned fixture changed")
	}
	want := string(b)
	for _, pair := range [][2]string{{"__global__ static void matmul_f16_ordered_chunks_kernel", "extern \"C\" __global__ void ds4_shape_hc_project"}, {"__global__ static void rms_norm_plain_kernel", "extern \"C\" __global__ void ds4_shape_hc_plain_rms"}, {"__global__ static void hc_split_sinkhorn_kernel", "extern \"C\" __global__ void ds4_shape_hc_split"}} {
		want = strings.ReplaceAll(want, pair[0], pair[1])
	}
	got, err := os.ReadFile("cuda_shape_hc_mix.cu")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(got), want) {
		t.Fatal("HC mix CUDA bodies differ from pinned source")
	}
}

func TestShapeHCSinkhornValues(t *testing.T) {
	p := "cuda_shape_hc_mix.cu"
	if v := os.Getenv("DS4_SHAPE_HC_MIX_TEST_SOURCE"); v != "" {
		p = v
	}
	src, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	a := strings.Index(body, "__device__ static void hc4_split_one")
	z := strings.Index(body, "extern \"C\" __global__ void ds4_shape_hc_split")
	if a < 0 || z <= a {
		t.Fatal("split source absent")
	}
	body = strings.Replace(body[a:z], "__device__", "", 1)
	cc, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	var code strings.Builder
	code.WriteString("#include <stdint.h>\n#include <math.h>\n#include <stdio.h>\n")
	code.WriteString(body)
	code.WriteString("int main(){float mix[24],base[24],scale[3]={0.3f,0.5f,0.7f},out[24];for(int i=0;i<24;i++){mix[i]=(float)(i-13)*0.125f;base[i]=(float)(i%5-2)*0.25f;}hc4_split_one(out,mix,scale,base,20,1e-6f);for(int i=0;i<24;i++)printf(\"%.9g \",out[i]);}\n")
	d := t.TempDir()
	input, out := filepath.Join(d, "split.cpp"), filepath.Join(d, "split")
	if err := os.WriteFile(input, []byte(code.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command(cc, input, "-o", out).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, b)
	}
	b, err := exec.Command(out).Output()
	if err != nil {
		t.Fatal(err)
	}
	fields := strings.Fields(string(b))
	if len(fields) != 24 {
		t.Fatal("missing split outputs")
	}
	var mix, base, want [24]float64
	scale := [3]float64{float64(float32(0.3)), 0.5, float64(float32(0.7))}
	epsilon := float64(float32(1e-6))
	for i := range mix {
		mix[i] = float64(i-13) * 0.125
		base[i] = float64(i%5-2) * 0.25
	}
	for i := 0; i < 4; i++ {
		want[i] = 1/(1+math.Exp(-(mix[i]*scale[0]+base[i]))) + epsilon
		want[4+i] = 2 / (1 + math.Exp(-(mix[4+i]*scale[1] + base[4+i])))
	}
	var comb [16]float64
	for r := 0; r < 4; r++ {
		max := math.Inf(-1)
		for c := 0; c < 4; c++ {
			v := mix[8+r*4+c]*scale[2] + base[8+r*4+c]
			comb[r*4+c] = v
			if v > max {
				max = v
			}
		}
		sum := 0.0
		for c := 0; c < 4; c++ {
			comb[r*4+c] = math.Exp(comb[r*4+c] - max)
			sum += comb[r*4+c]
		}
		for c := 0; c < 4; c++ {
			comb[r*4+c] = comb[r*4+c]/sum + epsilon
		}
	}
	for iteration := 0; iteration < 20; iteration++ {
		if iteration > 0 {
			for r := 0; r < 4; r++ {
				sum := epsilon
				for c := 0; c < 4; c++ {
					sum += comb[r*4+c]
				}
				for c := 0; c < 4; c++ {
					comb[r*4+c] /= sum
				}
			}
		}
		for c := 0; c < 4; c++ {
			sum := epsilon
			for r := 0; r < 4; r++ {
				sum += comb[r*4+c]
			}
			for r := 0; r < 4; r++ {
				comb[r*4+c] /= sum
			}
		}
	}
	copy(want[8:], comb[:])
	for i, f := range fields {
		got, err := strconv.ParseFloat(f, 64)
		if err != nil || math.IsNaN(got) || math.Abs(got-want[i]) > 2e-6 {
			t.Fatalf("HC_SPLIT_MISMATCH index=%d got=%s want=%g", i, f, want[i])
		}
	}
}
