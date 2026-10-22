package main

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// This executes the actual single-thread confidence kernel body as C++.
// It tests values and mutation sensitivity, not CUDA dispatch or arithmetic.
func TestDSparkConfidenceSourceValues(t *testing.T) {
	source, err := os.ReadFile("cuda_dspark_head.cu")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(source), "extern \"C\" __global__ void ds4_dspark_confidence_prefix(")
	if start < 0 {
		t.Fatal("confidence kernel missing")
	}
	kernel := string(source[start:])
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ unavailable: CPU kernel-body gate unmeasured")
	}
	for _, mode := range []string{"baseline", "extra row", "eager suffix", "global logit error"} {
		t.Run(mode, func(t *testing.T) {
			mutation := mode != "baseline"
			body := kernel
			switch mode {
			case "extra row":
				body = strings.Replace(body, "accepted = row;", "accepted = row + 1;", 1)
			case "eager suffix":
				body = strings.Replace(body, "{ accepted = row; break; }", "{ accepted = row; }", 1)
			case "global logit error":
				body = strings.Replace(body, "row_errors & (1u << row)", "row_errors", 1)
			}
			if mutation && body == kernel {
				t.Fatal("mutation did not apply")
			}
			dir := t.TempDir()
			path := filepath.Join(dir, "confidence.cc")
			program := `#include <cmath>
#include <cstdio>
#define __global__
using std::isfinite;
struct { unsigned int x; } threadIdx = {0};
static void atomicOr(unsigned int *p, unsigned int v) { *p |= v; }
` + body + `
int main() {
  float inputs[][4] = {{2, -2, 3, 4}, {2, -2, 3, 4}, {0, 0, 0, 0}, {2, NAN, 3, 4}, {2, -2, NAN, 4}, {2, -2, 3, 4}, {2, -2, 3, 4}, {2, -2, 3, 4}};
  float thresholds[] = {0.5f, 0, 0.5f, 0, 0.5f, 0.5f, 0.5f, 0.5f};
  unsigned int errors[] = {0, 0, 0, 0, 0, 1u << 2, 1u, 1u << 1};
  for (unsigned int i = 0; i != 8; ++i) {
    unsigned int length = 99, status = errors[i];
    ds4_dspark_confidence_prefix(inputs[i], &length, &status, 4, thresholds[i]);
    printf("%u %u", length, status);
    for (float v : inputs[i]) printf(" %.9g", v);
    puts("");
  }
}

`
			if err := os.WriteFile(path, []byte(program), 0600); err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(dir, "confidence")
			if out, err := exec.Command(compiler, "-std=c++17", path, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v\n%s", err, out)
			}
			out, err := exec.Command(binary).Output()
			if err != nil {
				t.Fatal(err)
			}
			inputs := [][4]float64{{2, -2, 3, 4}, {2, -2, 3, 4}, {0, 0, 0, 0}, {2, math.NaN(), 3, 4}, {2, -2, math.NaN(), 4}, {2, -2, 3, 4}, {2, -2, 3, 4}, {2, -2, 3, 4}}
			thresholds := []float64{0.5, 0, 0.5, 0, 0.5, 0.5, 0.5, 0.5}
			rowErrors := []uint32{0, 0, 0, 0, 0, 1 << 2, 1, 1 << 1}
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(lines) != len(inputs) {
				t.Fatalf("output rows: %q", out)
			}
			mismatches := 0
			for row, values := range inputs {
				fields := strings.Fields(lines[row])
				if len(fields) != 6 {
					t.Fatalf("output fields: %q", lines[row])
				}
				wantLength, wantStatus := 4, 0
				want := values
				for col, value := range values {
					if math.IsNaN(value) || math.IsInf(value, 0) {
						wantLength, wantStatus = 0, 2
						break
					}
					want[col] = 1 / (1 + math.Exp(-value))
					if thresholds[row] > 0 && want[col] < thresholds[row] {
						wantLength = col
						break
					}
					if rowErrors[row]&(1<<col) != 0 {
						wantLength, wantStatus = 0, 1
						break
					}
				}
				for col, value := range want {
					got, err := strconv.ParseFloat(fields[col+2], 64)
					if err != nil {
						t.Fatal(err)
					}
					if math.IsNaN(got) != math.IsNaN(value) || !math.IsNaN(value) && (math.IsInf(got, 0) || math.Abs(got-value) > 1e-7) {
						mismatches++
					}
				}
				if fields[0] != strconv.Itoa(wantLength) || fields[1] != strconv.Itoa(wantStatus) {
					mismatches++
				}
			}
			if mutation && mismatches == 0 || !mutation && mismatches != 0 {
				t.Fatalf("mutant=%t mismatches=%d", mutation, mismatches)
			}
		})
	}
}
