package profilebundle

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManifestIntegrity(t *testing.T) {
	for _, tt := range []struct {
		name    string
		mutate  func(string) error
		wantErr bool
	}{
		{"unchanged", nil, false},
		{"assembly", func(dir string) error {
			return os.WriteFile(filepath.Join(dir, "kernels.sass"), []byte("changed"), 0644)
		}, true},
		{"missing source", func(dir string) error { return os.Remove(filepath.Join(dir, "kernels.cu")) }, true},
		{"version", func(dir string) error {
			return os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(`{"version":2}`), 0644)
		}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range Files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0644); err != nil {
					t.Fatal(err)
				}
			}
			if err := Write(dir, Manifest{Device: "test GPU"}); err != nil {
				t.Fatal(err)
			}
			if tt.mutate != nil {
				if err := tt.mutate(dir); err != nil {
					t.Fatal(err)
				}
			}
			_, err := Read(dir)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Read = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

func TestCapturedCode(t *testing.T) {
	dir := os.Getenv("CUDA_PROFILE_BUNDLE")
	if dir == "" {
		t.Skip("set CUDA_PROFILE_BUNDLE to a real capture")
	}
	m, err := Read(dir)
	if err != nil {
		t.Fatal(err)
	}
	code, err := LoadCode(dir)
	if err != nil {
		t.Fatal(err)
	}
	sass, err := os.ReadFile(filepath.Join(dir, "kernels.sass"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(sass), "\n")
	for name := range m.Intervals {
		k := code.Kernel(name)
		if k == nil {
			t.Fatalf("no code for measured kernel %s", name)
		}
		for _, in := range k.Instructions {
			if in.SASSLine < 1 || in.SASSLine > len(lines) {
				t.Fatalf("%s offset %#x has wrong source line %d", name, in.Offset, in.SASSLine)
			}
			line := lines[in.SASSLine-1]
			if !strings.Contains(line, fmt.Sprintf("/*%04x*/", in.Offset)) || !strings.Contains(line, strings.TrimSuffix(strings.Fields(in.Text)[0], ";")) {
				t.Fatalf("%s offset %#x mapped to %q", name, in.Offset, line)
			}
		}
	}
	cubin, err := os.ReadFile(filepath.Join(dir, "kernels.cubin"))
	if err != nil {
		t.Fatal(err)
	}
	ptx, err := os.ReadFile(filepath.Join(dir, "kernels.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile(filepath.Join(dir, "kernels.cu"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range code.Kernels {
		mutated := strings.Replace(string(source), "void "+k.Name+"(", "void absent(", 1)
		if _, err := ParseCode(cubin, string(ptx), string(sass), mutated); err == nil {
			t.Fatalf("accepted missing source for %s", k.Name)
		}
	}
}
