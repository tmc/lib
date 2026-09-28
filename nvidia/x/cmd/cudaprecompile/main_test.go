package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
)

const scaleCU = `#include "scale.cuh"
extern "C" __global__ void scale(float* x, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) x[i] = twice(x[i]);
}
`

const scaleCUH = `__device__ inline float twice(float v) { return FACTOR * v; }
`

// newPackage writes a package inside the module, so that go vet can build
// the generated code against this checkout.
func newPackage(t *testing.T, files map[string]string) string {
	t.Helper()
	if err := os.MkdirAll("testdata", 0o777); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp("testdata", "pkg")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	for name, data := range files {
		name = filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(name), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(data), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func needCompiler(t *testing.T) {
	t.Helper()
	if _, _, err := nvrtc.Version(); err != nil {
		t.Skipf("nvrtc unavailable: %v", err)
	}
}

func goVet(t *testing.T, dir string, tags ...string) {
	t.Helper()
	args := []string{"vet"}
	if len(tags) > 0 {
		args = append(args, "-tags="+strings.Join(tags, ","))
	}
	cmd := exec.Command("go", append(args, "./"+filepath.ToSlash(dir))...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go vet: %v\n%s", err, out)
	}
}

func TestGenerate(t *testing.T) {
	needCompiler(t)
	tests := []struct {
		name  string
		files map[string]string
		args  []string
		want  []string // substrings of the generated file
	}{
		{
			name:  "single",
			files: map[string]string{"scale.cu": scaleCU, "scale.cuh": scaleCUH},
			args:  []string{"-arch=sm_90,sm_120", "-D", "FACTOR=2.0f", "scale.cu"},
			want:  []string{`Sources: []string{"scale.cu"}`, `Headers: []string{"scale.cuh"}`, `"-DFACTOR=2.0f"`, "//go:embed scale_precompiled.bin"},
		},
		{
			name: "multi",
			files: map[string]string{
				"main.cu":           "extern \"C\" __device__ float twice(float);\nextern \"C\" __global__ void scale(float* x) { x[threadIdx.x] = twice(x[threadIdx.x]); }\n",
				"kernels/twice.cu":  "#include <twice.cuh>\nextern \"C\" __device__ float twice(float v) { return helper(v); }\n",
				"include/twice.cuh": "__device__ inline float helper(float v) { return 2 * v; }\n",
			},
			args: []string{"-arch=sm_120", "-I", "include", "-o", "kernels.go", "main.cu", "kernels/twice.cu"},
			want: []string{`"main.cu", "kernels/twice.cu"`, `Include: []string{"include"}`, "//go:embed kernels.bin"},
		},
		{
			name:  "names",
			files: map[string]string{"tmpl.cu": "template <typename T> __global__ void fill(T* x, T v) { x[threadIdx.x] = v; }\n"},
			args:  []string{"-arch=sm_120", "-ptx=none", "-name", "fill<float>", "-name", "fill<int>", "tmpl.cu"},
			want:  []string{`"fill<float>": "_Z4fillIfEvPT_S0_"`, `"fill<int>":`},
		},
		{
			name:  "tag",
			files: map[string]string{"scale.cu": scaleCU, "scale.cuh": scaleCUH},
			args:  []string{"-arch=sm_120", "-DFACTOR=2.0f", "-tag=precompiled", "scale.cu"},
			want:  []string{"var precompiledFiles embed.FS\n\nvar precompiled = "},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := newPackage(t, tt.files)
			args := append([]string{"-pkg=p"}, tt.args...)
			if err := run(args, dir, io.Discard, io.Discard); err != nil {
				t.Fatal(err)
			}
			out := "kernels.go"
			if !strings.Contains(strings.Join(tt.args, " "), "-o") {
				out = strings.TrimSuffix(tt.args[len(tt.args)-1], ".cu") + "_precompiled.go"
			}
			src, err := os.ReadFile(filepath.Join(dir, out))
			if err != nil {
				t.Fatal(err)
			}
			for _, w := range tt.want {
				if !strings.Contains(string(src), w) {
					t.Errorf("generated file lacks %q:\n%s", w, src)
				}
			}
			goVet(t, dir)
			if strings.Contains(strings.Join(tt.args, " "), "-tag") {
				goVet(t, dir, "precompiled")
			}
			if err := run(append([]string{"-check"}, args...), dir, io.Discard, io.Discard); err != nil {
				t.Errorf("-check after generate: %v", err)
			}
		})
	}
}

func TestCheck(t *testing.T) {
	needCompiler(t)
	dir := newPackage(t, map[string]string{"scale.cu": scaleCU, "scale.cuh": scaleCUH})
	args := []string{"-pkg=p", "-arch=sm_120", "-ptx=none", "-DFACTOR=2.0f", "scale.cu"}
	if err := run(append([]string{"-check"}, args...), dir, io.Discard, io.Discard); err != errStale {
		t.Fatalf("-check before generate = %v, want errStale", err)
	}
	if err := run(args, dir, io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := run(append([]string{"-check"}, args...), dir, io.Discard, io.Discard); err != nil {
		t.Fatalf("-check = %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "scale.cuh"), []byte("__device__ inline float twice(float v) { return v + v; }\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := run(append([]string{"-check"}, args...), dir, io.Discard, io.Discard); err != errStale {
		t.Fatalf("-check after header edit = %v, want errStale", err)
	}
}

func TestErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"no sources", []string{"-pkg=p"}, "no sources"},
		{"no package", []string{"-pkg=", "a.cu"}, "no package name"},
		{"outside", []string{"-pkg=p", "../a.cu"}, "outside the package directory"},
		{"arch", []string{"-pkg=p", "-arch=sm_7", "a.cu"}, `unknown -arch "sm_7"`},
		{"ptx", []string{"-pkg=p", "-ptx=sm_80", "a.cu"}, `unknown -ptx "sm_80"`},
		{"var", []string{"-pkg=p", "-var=a-b", "a.cu"}, "invalid -var"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := run(tt.args, t.TempDir(), io.Discard, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("run(%q) = %v, want %q", tt.args, err, tt.want)
			}
		})
	}
}
