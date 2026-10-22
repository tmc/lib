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

// The production selector has no warp or barrier operations. Execute that
// body as C++ against an independently sorted host reference. This is not a
// CUDA execution receipt or a performance gate for the serial route.
func TestShapeIndexerSelectionValues(t *testing.T) {
	path := "cuda_shape_indexer.cu"
	if p := os.Getenv("DS4_SHAPE_INDEXER_TEST_SOURCE"); p != "" {
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
	for _, tc := range []struct {
		name      string
		rows, k   int
		nonfinite bool
	}{
		{"flash", 1100, 512, false}, {"pro", 1100, 1024, false},
		{"all", 512, 1024, false}, {"empty", 0, 1024, false},
		{"pro-nonfinite", 1100, 1024, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code := "#include <math.h>\n#undef INFINITY\nfloat __int_as_float(int x){union{int i;float f;}u={x};return u.f;}\n#define __global__\nstruct{unsigned x;}blockIdx={0},threadIdx={0};\n" + string(source)
			code += fmt.Sprintf("\n#include <stdio.h>\nint main(){float scores[%d],mask[%d];uint32_t invalid=99;for(int i=0;i<%d;i++){scores[i]=(float)((i*17)%%37-18)*0.125f;mask[i]=-9;}", max(1, tc.rows), max(1, tc.rows), tc.rows)
			if tc.nonfinite {
				code += fmt.Sprintf("scores[%d]=NAN;", tc.rows-1)
			}
			code += fmt.Sprintf("ds4_shape_indexer_select(scores,mask,&invalid,%d,%d);printf(\"%%u\",invalid);for(int i=0;i<%d;i++)printf(\" %%.0f\",mask[i]);}\n", tc.rows, tc.k, tc.rows)
			dir := t.TempDir()
			input, binary := filepath.Join(dir, "selector.cpp"), filepath.Join(dir, "selector")
			if err := os.WriteFile(input, []byte(code), 0600); err != nil {
				t.Fatal(err)
			}
			if log, err := exec.Command(cc, input, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v %s", err, log)
			}
			output, err := exec.Command(binary).Output()
			if err != nil {
				t.Fatal(err)
			}
			fields := strings.Fields(string(output))
			if len(fields) != tc.rows+1 {
				t.Fatalf("INDEXER_OUTPUT_COUNT got=%d want=%d", len(fields), tc.rows+1)
			}
			if tc.nonfinite {
				if fields[0] != "1" {
					t.Fatalf("INDEXER_FINITE_REFUSAL got=%s", fields[0])
				}
				return
			}
			if fields[0] != "0" {
				t.Fatalf("INDEXER_INVALID got=%s", fields[0])
			}
			ids := make([]int, tc.rows)
			for i := range ids {
				ids[i] = i
			}
			sort.SliceStable(ids, func(i, j int) bool { return float64((ids[i]*17)%37-18)*0.125 > float64((ids[j]*17)%37-18)*0.125 })
			want := make([]bool, tc.rows)
			for _, i := range ids[:min(tc.k, tc.rows)] {
				want[i] = true
			}
			for i := range want {
				got, err := strconv.ParseFloat(fields[i+1], 64)
				if err != nil || want[i] && got != 0 || !want[i] && !math.IsInf(got, -1) {
					t.Fatalf("INDEXER_MASK_MISMATCH row=%d got=%s want=%v", i, fields[i+1], want[i])
				}
			}
		})
	}
}
