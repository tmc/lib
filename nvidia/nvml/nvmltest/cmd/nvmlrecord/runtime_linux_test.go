//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This is an ordinary mmap CPU fixture, not an NVML load or hardware proof.
func TestRuntimeActualSelfMappedCPU(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libnvidia-ml.so.fixture")
	if err := os.WriteFile(path, []byte(strings.Repeat("body", 1024)), 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	mapped, err := unix.Mmap(int(f.Fd()), 0, 4096, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	defer func() {
		if e := unix.Munmap(mapped); e != nil {
			t.Error(e)
		}
		if e := f.Close(); e != nil {
			t.Error(e)
		}
	}()
	t.Setenv("NVML_LIB_PATH", path)
	raw, err := captureRuntime()
	if err != nil {
		t.Fatal(err)
	}
	var proof runtimeEvidence
	if err = json.Unmarshal(raw, &proof); err != nil {
		t.Fatal(err)
	}
	p, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if proof.Version != 1 || proof.Process.PID != p.PID || proof.Process.Start != p.Start || proof.Process.Group != p.Group || len(proof.Process.ExecutableSHA256) != 64 || len(proof.Libraries) != 1 || proof.Libraries[0].Path != path || len(proof.Libraries[0].SHA256) != 64 {
		t.Fatalf("bad actual proof %+v", proof)
	}
	expected := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Repeat("body", 1024))))
	if proof.Libraries[0].SHA256 != expected {
		t.Fatal("actual descriptor digest differs", proof.Libraries[0].SHA256)
	}
	executable, err := os.ReadFile("/proc/self/exe")
	if err != nil {
		t.Fatal(err)
	}
	if proof.Process.ExecutableSHA256 != fmt.Sprintf("%x", sha256.Sum256(executable)) {
		t.Fatal("actual executable digest differs")
	}
	m := runtimeMapping{}
	for _, v := range proof.Mappings {
		if v.Path == path {
			m = v
		}
	}
	if m.Path == "" {
		t.Fatal("raw mapping missing")
	}
	for _, change := range []func(*runtimeMapping){func(m *runtimeMapping) { m.Inode++ }, func(m *runtimeMapping) { m.Major++ }, func(m *runtimeMapping) { m.Minor++ }} {
		wrong := m
		change(&wrong)
		if _, e := runtimeMappedBody(wrong); e == nil {
			t.Fatal("foreign descriptor accepted")
		}
	}
	t.Setenv("NVML_LIB_PATH", filepath.Join(dir, "other"))
	if err = os.WriteFile(filepath.Join(dir, "other"), []byte("other"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, e := captureRuntime(); e == nil {
		t.Fatal("unexpected actual mapping accepted")
	}
}
func TestRuntimeNoMappingAndConfiguration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ordinary")
	if err := os.WriteFile(path, []byte("ordinary"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"", "relative", filepath.Dir(path) + "/../" + filepath.Base(filepath.Dir(path)) + "/ordinary"} {
		t.Setenv("NVML_LIB_PATH", v)
		if e := runtimeProofSupported(); e == nil {
			t.Fatal("invalid config", v)
		}
	}
	t.Setenv("NVML_LIB_PATH", path)
	if _, e := captureRuntime(); e == nil {
		t.Fatal("prospective body mistaken for observed mapping")
	}
	alias := filepath.Join(filepath.Dir(path), "alias")
	if e := os.Symlink(path, alias); e != nil {
		t.Fatal(e)
	}
	t.Setenv("NVML_LIB_PATH", alias)
	if e := runtimeProofSupported(); e == nil {
		t.Fatal("symlink input accepted")
	}
}
func TestRuntimeParserBounds(t *testing.T) {
	valid := []byte("1000-2000 r--p 00000000 08:01 42 /ordinary/path with spaces\n")
	m, e := runtimeMaps(valid)
	if e != nil || len(m) != 1 || m[0].Path != "/ordinary/path with spaces" {
		t.Fatal(m, e)
	}
	for _, raw := range []string{"", "bad", "2000-1000 r--p 0 08:01 42 /a\n", "1000-2000 badp 0 08:01 42 /a\n", strings.Repeat("x", runtimeMapsLimit+1)} {
		if _, e := runtimeMaps([]byte(raw)); e == nil {
			t.Fatal("invalid maps accepted")
		}
	}
	path := filepath.Join(t.TempDir(), "large")
	if e := os.WriteFile(path, []byte("12345"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := runtimeRead(path, 4); e == nil {
		t.Fatal("read bound omitted")
	}
}
