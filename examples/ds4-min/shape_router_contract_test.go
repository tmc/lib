package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// This executes the production scalar CUDA body as ordinary C++. There are
// no barriers/warp intrinsics in that body; it is not a physical CUDA receipt.
func TestShapeRouterValues(t *testing.T) {
	source := "cuda_shape_router.cu"
	if p := os.Getenv("DS4_SHAPE_ROUTER_TEST_SOURCE"); p != "" {
		source = p
	}
	b, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{256, 384} {
		for _, mode := range []string{"topk", "hash", "nonfinite", "invalid-hash"} {
			t.Run(fmt.Sprintf("%d/%s", n, mode), func(t *testing.T) {
				scale := 1.5
				if n == 384 {
					scale = 2.5
				}
				code := "#include <math.h>\n#define __device__\n#define __global__\nstruct {unsigned x;} blockIdx={0},threadIdx={0};\n" + string(b)
				code += fmt.Sprintf("\n#include <stdio.h>\nint main(){float logits[%d],bias[%d],probs[%d]={},weights[6]={};int32_t ids[6]={},hash[6]={0,%d,13,31,127,128};uint32_t invalid=0;for(int i=0;i<%d;i++){logits[i]=(float)(i-128)*0.03125f;bias[i]=(float)(i%%5-2)*0.015625f;}", n, n, n, n-1, n)
				hashMode := 0
				if mode == "hash" || mode == "invalid-hash" {
					hashMode = 1
				}
				if mode == "nonfinite" {
					code += fmt.Sprintf("logits[%d]=NAN;", n-1)
				}
				if mode == "invalid-hash" {
					code += fmt.Sprintf("hash[1]=%d;", n)
				}
				code += fmt.Sprintf("ds4_shape_router_select(ids,weights,probs,bias,hash,logits,nullptr,0,1,1,1,%d,%d,%.1ff,&invalid);printf(\"%%u \",invalid);for(int i=0;i<6;i++)printf(\"%%d %%.9g \",ids[i],weights[i]);}\n", hashMode, n, scale)
				d := t.TempDir()
				input, out := filepath.Join(d, "router.cpp"), filepath.Join(d, "router")
				if err := os.WriteFile(input, []byte(code), 0600); err != nil {
					t.Fatal(err)
				}
				if b, err := exec.Command(cc, input, "-o", out).CombinedOutput(); err != nil {
					t.Fatalf("compile: %v %s", err, b)
				}
				output, err := exec.Command(out).Output()
				if err != nil {
					t.Fatal(err)
				}
				fields := strings.Fields(string(output))
				if len(fields) != 13 {
					t.Fatalf("router output count = %d", len(fields))
				}
				if mode == "nonfinite" || mode == "invalid-hash" {
					if fields[0] != "1" {
						t.Fatalf("ROUTER_FINITE_REFUSAL mode=%s invalid=%s", mode, fields[0])
					}
					return
				}
				if fields[0] != "0" {
					t.Fatal("valid router input rejected")
				}
				probs := make([]float64, n)
				ids := make([]int, n)
				for i := range probs {
					probs[i] = math.Sqrt(math.Log1p(math.Exp(float64(i-128) * 0.03125)))
					ids[i] = i
				}
				if hashMode != 0 {
					ids = []int{0, n - 1, 13, 31, 127, 128}
				} else {
					sort.SliceStable(ids, func(i, j int) bool {
						a, z := ids[i], ids[j]
						return probs[a]+float64(a%5-2)*0.015625 > probs[z]+float64(z%5-2)*0.015625
					})
					ids = ids[:6]
				}
				sum := 0.0
				for _, id := range ids {
					sum += probs[id]
				}
				for slot, id := range ids {
					gotID, err := strconv.Atoi(fields[1+slot*2])
					if err != nil || gotID != id {
						t.Fatalf("ROUTER_ID_MISMATCH slot=%d got=%s want=%d", slot, fields[1+slot*2], id)
					}
					got, err := strconv.ParseFloat(fields[2+slot*2], 64)
					want := probs[id] / math.Max(sum, 6.103515625e-5) * scale
					if err != nil || math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 2e-6 {
						t.Fatalf("ROUTER_WEIGHT_MISMATCH slot=%d got=%s want=%g", slot, fields[2+slot*2], want)
					}
				}
			})
		}
	}
}
