//go:build linux

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

	"github.com/tmc/lib/examples/ds4-min/dsv41"
)

func TestCUDAV41OpsValuesAndMutants(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in CUDA V4.1 operations gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	for _, variant := range []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"quant_scale", "float peak = 0;", "float peak = 64;"},
		{"rope_position", "float(start+row*stride)", "float(row*stride)"},
	} {
		t.Run(variant.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			source := cudaV41OpsSource
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutant site not unique")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			op, err := compileCUDAV41Ops(r, source)
			if op != nil {
				t.Cleanup(func() {
					if err := op.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			quantDiff, ropeDiff := 0, 0
			for mode := dsv41.BFloat16; mode <= dsv41.FP4E4M3; mode++ {
				input := make([]float32, 128)
				for i := range input {
					input[i] = float32(math.Sin(float64(i)*.71) * 3.2)
				}
				// Zero, BF16 ties, signed values and a scale-boundary peak.
				input[0] = 0
				input[1] = math.Float32frombits(0x3f808000)
				input[2] = math.Float32frombits(0x3f818000)
				input[32] = 4
				want := append([]float32(nil), input...)
				if err := dsv41.Quantize(want, 64, mode); err != nil {
					t.Fatal(err)
				}
				d := cudaSynthUpload(t, r, input)
				if err := op.Quantize(d, 64, 2, mode); err != nil {
					t.Fatal(err)
				}
				if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, len(input))
				if err := d.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						quantDiff++
					}
				}
			}
			for _, compressed := range []bool{false, true} {
				for _, inverse := range []bool{false, true} {
					for _, start := range []uint32{1, 64, 2048, 1048571} {
						const width, heads, rows = 128, 2, 3
						input := make([]float32, width*heads*rows)
						for i := range input {
							input[i] = float32(math.Sin(float64(i) * .19))
						}
						frequency := cudaV41Frequencies(compressed)
						want := append([]float32(nil), input...)
						for row := 0; row < rows; row++ {
							for head := 0; head < heads; head++ {
								for lane, f := range frequency {
									theta := float32(float32(start+uint32(row)*2) * f)
									c, s := float32(math.Cos(float64(theta))), float32(math.Sin(float64(theta)))
									if inverse {
										s = -s
									}
									i := (row*heads+head)*width + width - 64 + lane*2
									a, b := input[i], input[i+1]
									want[i] = dsv41.BF16(float32(a*c) - float32(b*s))
									want[i+1] = dsv41.BF16(float32(a*s) + float32(b*c))
								}
							}
						}
						d, frequencies := cudaSynthUpload(t, r, input), cudaSynthUpload(t, r, frequency)
						if err := op.Rope(d, frequencies, width, heads, rows, start, 2, inverse); err != nil {
							t.Fatal(err)
						}
						if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
							t.Fatal(err)
						}
						got := make([]float32, len(input))
						if err := d.CopyToHost(got); err != nil {
							t.Fatal(err)
						}
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								ropeDiff++
							}
						}
					}
				}
			}
			t.Logf("quantize differing=%d rotary differing=%d", quantDiff, ropeDiff)
			switch variant.name {
			case "baseline":
				if quantDiff != 0 || ropeDiff != 0 {
					t.Fatal("device values differ from scalar oracle")
				}
				input := make([]float32, 128)
				for i := range input {
					input[i] = float32(i) / 31
				}
				d := cudaSynthUpload(t, r, input)
				f := cudaSynthUpload(t, r, cudaV41Frequencies(false))
				if err := op.Rope(d, f, 128, 1, 1, 64, 0, false); err == nil {
					t.Fatal("zero stride accepted")
				}
				if err := op.Quantize(d, 31, 4, dsv41.FP8E8M0); err == nil {
					t.Fatal("invalid quantize shape accepted")
				}
				if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, len(input))
				if err := d.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(input[i]) {
						t.Fatalf("rejected operation changed operand %d", i)
					}
				}
			case "quant_scale":
				if quantDiff == 0 {
					t.Fatal("quantization mutant survived")
				}
				if ropeDiff != 0 {
					t.Fatal("quantization mutant altered adjacent rotary operation")
				}
			case "rope_position":
				if ropeDiff == 0 {
					t.Fatal("position mutant survived")
				}
				if quantDiff != 0 {
					t.Fatal("rotary mutant altered adjacent quantization operation")
				}
			}
		})
	}
}

// This compiles the literal pinned upstream constructor, not a Go facsimile.
// A mismatch is retained as an admission failure, not hidden by a tolerance.
func TestCUDAV41FrequenciesPinnedC(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 operations gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Fatal("DS41_ORACLE required")
	}
	source, err := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4_deepseek41_cuda.cuh").Output()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != "0dc287ea0e7db01eb1590ed73dfce0489595e6b6bc2fc3512d5a10e090f3041e" {
		t.Fatal("pinned CUDA source hash mismatch")
	}
	start := strings.Index(string(source), "    struct frequencies {")
	if start < 0 {
		t.Fatal("frequency constructor missing")
	}
	rest := string(source[start:])
	end := strings.Index(rest, "    static const frequencies table;")
	if end < 0 {
		t.Fatal("frequency constructor end missing")
	}
	constructor := rest[:end]
	compiler := os.Getenv("DS41_CXX")
	if compiler == "" {
		compiler = "c++"
	}
	run := func(body string) []uint32 {
		t.Helper()
		dir := t.TempDir()
		path := filepath.Join(dir, "frequency.cc")
		binary := filepath.Join(dir, "frequency")
		code := "#include <cmath>\n#include <cstdio>\n#include <cstdint>\n#include <cstring>\n" + body + "\nint main(){frequencies t; for(int k=0;k<2;k++)for(int i=0;i<32;i++){uint32_t u;memcpy(&u,&t.value[k][i],4);printf(\"%08x\\n\",u);}return 0;}\n"
		if err := os.WriteFile(path, []byte(code), 0600); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command(compiler, "-std=c++11", "-O2", "-ffp-contract=off", path, "-o", binary, "-lm").CombinedOutput(); err != nil {
			t.Fatalf("compile pinned frequency oracle: %v\n%s", err, out)
		}
		out, err := exec.Command(binary).Output()
		if err != nil {
			t.Fatal(err)
		}
		fields := strings.Fields(string(out))
		if len(fields) != 64 {
			t.Fatalf("oracle returned %d values", len(fields))
		}
		values := make([]uint32, 64)
		for i, s := range fields {
			u, err := strconv.ParseUint(s, 16, 32)
			if err != nil {
				t.Fatal(err)
			}
			values[i] = uint32(u)
		}
		return values
	}
	want := run(constructor)
	differences := 0
	for k := 0; k < 2; k++ {
		for i, f := range cudaV41Frequencies(k == 1) {
			if math.Float32bits(f) != want[k*32+i] {
				t.Errorf("frequency kind=%d lane=%d Go=%08x C=%08x", k, i, math.Float32bits(f), want[k*32+i])
				differences++
			}
		}
	}
	mutant := strings.Replace(constructor, "160000.0f : 10000.0f", "160000.0f : 20000.0f", 1)
	if mutant == constructor {
		t.Fatal("frequency mutation site missing")
	}
	changed := 0
	for i, u := range run(mutant) {
		if u != want[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("pinned frequency base mutant survived")
	}
	t.Logf("frequency comparisons=64 differing=%d mutant_changed=%d", differences, changed)
}
