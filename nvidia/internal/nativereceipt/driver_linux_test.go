//go:build linux

package nativereceipt

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These tests mmap ordinary synthetic files. They do not load CUDA libraries.
func driverReceipt(t *testing.T, nvrtc, builtins bool) (Receipt, Process, string, string) {
	t.Helper()
	mapFixture(t, "libcuda.so.driver-fixture")
	if nvrtc {
		mapFixture(t, "libnvrtc.so.driver-fixture")
	}
	if builtins {
		mapFixture(t, "libnvrtc-builtins.so.driver-fixture")
	}
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	source := fmt.Sprintf("%x", sha256.Sum256([]byte("prepared source")))
	image := fmt.Sprintf("%x", sha256.Sum256([]byte("prepared image")))
	if err := s.Capture("before-prepared-image", source, image); err != nil {
		t.Fatal(err)
	}
	if err := s.Capture("after-prepared-image", source, image); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "driver.json")
	if !nvrtc {
		if err := s.Write(path); err == nil {
			t.Fatal("strict writer accepted absent NVRTC")
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatal("strict refusal created a receipt")
		}
	}
	if err := s.WriteDriver(path); err != nil {
		t.Fatal(err)
	}
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(p.Executable)
	if err != nil {
		t.Fatal(err)
	}
	p.ExecutableSHA256 = fmt.Sprintf("%x", sha256.Sum256(body))
	for _, file := range r.Files {
		body, err := os.ReadFile(file.Path)
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(body)) != file.SHA256 {
			t.Fatal("retained descriptor hash differs from actual fixture body")
		}
	}
	if r.NVRTCObserved != nvrtc || r.BuiltinsObserved != builtins || !r.DriverObserved {
		t.Fatal("observed flags differ from actual synthetic mappings")
	}
	if err := r.CheckDriver(p, source, image); err != nil {
		t.Fatal(err)
	}
	if err := r.Check(p, source, image); (err == nil) != nvrtc {
		t.Fatalf("strict checker NVRTC=%v error=%v", nvrtc, err)
	}
	return r, p, source, image
}

func TestDriverReceiptObservedMatrix(t *testing.T) {
	for _, tc := range []struct {
		name            string
		nvrtc, builtins bool
	}{
		{"driver", false, false},
		{"driver-nvrtc", true, false},
		{"driver-builtins", false, true},
		{"driver-nvrtc-builtins", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) { driverReceipt(t, tc.nvrtc, tc.builtins) })
	}
}

func TestDriverReceiptRefusals(t *testing.T) {
	r, p, source, image := driverReceipt(t, false, false)
	cases := []struct {
		name   string
		change func(*Receipt, *Process, *string, *string)
	}{
		{"process", func(r *Receipt, p *Process, s, a *string) { p.PID++ }},
		{"parent", func(r *Receipt, p *Process, s, a *string) { p.Parent++ }},
		{"group", func(r *Receipt, p *Process, s, a *string) { p.Group++ }},
		{"start", func(r *Receipt, p *Process, s, a *string) { p.Start++ }},
		{"source", func(r *Receipt, p *Process, s, a *string) { *s = strings.Repeat("0", 64) }},
		{"image", func(r *Receipt, p *Process, s, a *string) { *a = strings.Repeat("0", 64) }},
		{"driver-claim", func(r *Receipt, p *Process, s, a *string) { r.DriverObserved = false }},
		{"nvrtc-claim", func(r *Receipt, p *Process, s, a *string) { r.NVRTCObserved = true }},
		{"builtin-claim", func(r *Receipt, p *Process, s, a *string) { r.BuiltinsObserved = true }},
		{"descriptor", func(r *Receipt, p *Process, s, a *string) { r.Files[0].Inode++ }},
		{"shadow", func(r *Receipt, p *Process, s, a *string) {
			f := r.Files[0]
			f.Inode++
			f.SHA256 = strings.Repeat("0", 64)
			r.Files = append([]File{f}, r.Files...)
		}},
		{"raw-map", func(r *Receipt, p *Process, s, a *string) { r.Stages[0].Mappings[0].Offset++ }},
		{"hash-absent", func(r *Receipt, p *Process, s, a *string) { r.Files[0].SHA256 = "" }},
		{"limitation-absent", func(r *Receipt, p *Process, s, a *string) { r.Unobserved = nil }},
		{"limitation-duplicate", func(r *Receipt, p *Process, s, a *string) { r.Unobserved[0] = r.Unobserved[1] }},
	}
	body, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var copy Receipt
			if err := json.Unmarshal(body, &copy); err != nil {
				t.Fatal(err)
			}
			pp, ss, aa := p, source, image
			tc.change(&copy, &pp, &ss, &aa)
			if err := copy.CheckDriver(pp, ss, aa); err == nil {
				t.Fatal("accepted changed driver proof")
			}
		})
	}
}

func TestDriverReceiptOptionalDescriptorRefused(t *testing.T) {
	r, p, source, image := driverReceipt(t, true, true)
	for _, name := range []string{"libnvrtc.so", "libnvrtc-builtins.so"} {
		t.Run(name, func(t *testing.T) {
			body, _ := json.Marshal(r)
			var copy Receipt
			if err := json.Unmarshal(body, &copy); err != nil {
				t.Fatal(err)
			}
			found := false
			for i := range copy.Files {
				if library(copy.Files[i].Path, name) {
					copy.Files[i].Inode++
					found = true
					break
				}
			}
			if !found {
				t.Fatal("optional descriptor not actually captured")
			}
			if err := copy.CheckDriver(p, source, image); err == nil {
				t.Fatal("accepted changed observed optional descriptor")
			}
		})
	}
}

func TestDriverReceiptMissingDriverRefused(t *testing.T) {
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Capture("no-driver", "", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.WriteDriver(filepath.Join(t.TempDir(), "proof.json")); err == nil {
		t.Fatal("driver writer accepted no actual driver mapping")
	}
}
