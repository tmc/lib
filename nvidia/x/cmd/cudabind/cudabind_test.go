package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/internal/libclang"
)

func TestCamel(t *testing.T) {
	tests := []struct {
		name, want string
	}{
		{"gpu_breakout_step_kernel", "GpuBreakoutStepKernel"},
		{"GpuBreakoutConfig", "GpuBreakoutConfig"},
		{"d_bcfg", "DBcfg"},
		{"BREAKOUT_STATE_GROUPS", "BreakoutStateGroups"},
		{"geom::Side::LEFT", "GeomSideLeft"},
		{"_x", "X"},
		{"N", "N"},
	}
	for _, tt := range tests {
		if got := camel(tt.name); got != tt.want {
			t.Errorf("camel(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

// TestGeneratedTypes checks the generated methods with the Go compiler. The
// test uses an overlay so it can check bad calls without changing the fixture.
func TestGeneratedTypes(t *testing.T) {
	probe := filepath.Join("internal", "testvec", "typed_compile_probe_test.go")
	probe, err := filepath.Abs(probe)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	overlay := filepath.Join(dir, "overlay.json")
	contents := filepath.Join(dir, "probe.go")
	mapFile := func(source string) error {
		if err := os.WriteFile(contents, []byte(source), 0o600); err != nil {
			return err
		}
		b, err := json.Marshal(struct {
			Replace map[string]string `json:"Replace"`
		}{map[string]string{probe: contents}})
		if err != nil {
			return err
		}
		return os.WriteFile(overlay, b, 0o600)
	}
	valid := `package testvec
import (
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)
func generatedTypesCompile(m *vecModule, s *cuda.Stream, x cudamem.Slice[float32]) error {
	return m.scaleAll(s, [3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, x, int32(4))
}
`
	badScalar := strings.Replace(valid, "int32(4)", "float32(4)", 1)
	badSlice := strings.Replace(valid, "Slice[float32]", "Slice[uint32]", 1)
	packageDir := filepath.Join("internal", "testvec")
	for _, tt := range []struct {
		name string
		src  string
		want bool
	}{
		{"valid", valid, true},
		{"wrong_scalar_type", badScalar, false},
		{"wrong_pointer_type", badSlice, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := mapFile(tt.src); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("go", "test", "-overlay", overlay, "-run", "^$", ".")
			cmd.Dir = packageDir
			out, err := cmd.CombinedOutput()
			if (err == nil) != tt.want {
				t.Fatalf("go test error = %v, want success %v\n%s", err, tt.want, out)
			}
		})
	}
}

func TestDocComment(t *testing.T) {
	tests := []struct {
		name, raw, want string
	}{
		{"step", "// step advances envs.\n// It is fast.", "step advances envs.\nIt is fast."},
		{"step", "/* step advances envs. */", "step advances envs."},
		{"step", "/**\n * step advances envs.\n */", "step advances envs."},
		{"step", "// ---- kernels ----", ""},
		{"step", "// stepper is unrelated.", ""},
		{"step", "", ""},
	}
	for _, tt := range tests {
		if got := docComment(tt.raw, tt.name); got != tt.want {
			t.Errorf("docComment(%q, %q) = %q, want %q", tt.raw, tt.name, got, tt.want)
		}
	}
}

func TestRelative(t *testing.T) {
	dir := t.TempDir()
	tests := []struct {
		path, want string
		ok         bool
	}{
		{filepath.Join(dir, "vec.cu"), "vec.cu", true},
		{filepath.Join(dir, "shim", "stdio.h"), "shim/stdio.h", true},
		{filepath.Join(dir, "..", "vec.cu"), "", false},
		{filepath.Join(dir, "..", filepath.Base(dir)+"x", "vec.cu"), "", false},
	}
	for _, tt := range tests {
		got, err := relative(dir, tt.path)
		if (err == nil) != tt.ok || got != tt.want {
			t.Errorf("relative(%q) = %q, %v, want %q, ok %v", tt.path, got, err, tt.want, tt.ok)
		}
	}
}

// TestGenerate checks that cudabind regenerates internal/testvec as it is
// checked in. Its own test runs the result on a device.
func TestGenerate(t *testing.T) {
	lib, err := libclang.Open(os.Getenv("CUDABIND_LIBCLANG"))
	if err != nil {
		t.Skip(err)
	}
	if _, err := os.Stat(filepath.Join(lib.ResourceDir(), "include", "__clang_cuda_runtime_wrapper.h")); err != nil {
		t.Skipf("%s has no CUDA headers; set $CUDABIND_LIBCLANG", lib.Path())
	}
	src, err := filepath.Abs("internal/testvec")
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "testvec")
	if err := os.Mkdir(dir, 0o777); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(src, "vec.cu"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "vec.cu"), b, 0o666); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	t.Setenv("GOPACKAGE", "testvec")
	cfg, err := newConfig([]string{"-type", "vecModule", "-shared", "sum", "-std", "c++17", "vec.cu"})
	if err != nil {
		t.Fatal(err)
	}
	if err := run(cfg); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"vec.cu.gen.go", "vec.cu.gen_test.go"} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s differs from internal/testvec/%s; run go generate there if the change is intended", name, name)
		}
	}
}
