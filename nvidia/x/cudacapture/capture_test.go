package cudacapture_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func validFixture() *cudacapture.Capture {
	fn := &profile.Function{ID: 1, Name: "compute", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{500}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	return &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void compute() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_DATA"),
			"kernels.sass":  []byte("// SASS\n"),
			"extra.go":      []byte("package main\n"),
		},
		Metadata: json.RawMessage(`{"elements":1024,"device":"test GPU"}`),
	}
}

func TestRoundTripSingleFile(t *testing.T) {
	orig := validFixture()
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, orig); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if got.Code != orig.Code {
		t.Errorf("code files mismatch: got %+v, want %+v", got.Code, orig.Code)
	}
	if len(got.Files) != len(orig.Files) {
		t.Errorf("files count mismatch: got %d, want %d", len(got.Files), len(orig.Files))
	}
	for k, v := range orig.Files {
		if !bytes.Equal(got.Files[k], v) {
			t.Errorf("file %s content mismatch", k)
		}
	}
	if string(got.Metadata) != string(orig.Metadata) {
		t.Errorf("metadata mismatch: got %s, want %s", got.Metadata, orig.Metadata)
	}

	// Pprof parse/write round trip
	var pprofBuf bytes.Buffer
	if err := got.Profile.Write(&pprofBuf); err != nil {
		t.Fatalf("pprof Write failed: %v", err)
	}
	got2, err := cudacapture.Read(&pprofBuf)
	if err != nil {
		t.Fatalf("Read after pprof roundtrip failed: %v", err)
	}
	if len(got2.Files) != len(orig.Files) {
		t.Errorf("files count after pprof roundtrip: got %d, want %d", len(got2.Files), len(orig.Files))
	}
}

func TestRoundTripDirectory(t *testing.T) {
	dir := t.TempDir()
	orig := validFixture()

	if err := cudacapture.WriteDir(dir, orig); err != nil {
		t.Fatalf("WriteDir failed: %v", err)
	}

	got, err := cudacapture.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir failed: %v", err)
	}

	if got.Code != orig.Code {
		t.Errorf("code files mismatch: got %+v, want %+v", got.Code, orig.Code)
	}
	if len(got.Files) != len(orig.Files) {
		t.Errorf("files count mismatch: got %d, want %d", len(got.Files), len(orig.Files))
	}
	for k, v := range orig.Files {
		if !bytes.Equal(got.Files[k], v) {
			t.Errorf("file %s content mismatch", k)
		}
	}
}

func TestValidationErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		mutate  func(*cudacapture.Capture)
		wantErr string
	}{
		{
			name: "missing profile",
			mutate: func(c *cudacapture.Capture) {
				c.Profile = nil
			},
			wantErr: "missing profile",
		},
		{
			name: "empty files",
			mutate: func(c *cudacapture.Capture) {
				c.Files = nil
			},
			wantErr: "empty file set",
		},
		{
			name: "missing code role",
			mutate: func(c *cudacapture.Capture) {
				c.Code.SASS = ""
			},
			wantErr: "missing required code roles",
		},
		{
			name: "code file not in files map",
			mutate: func(c *cudacapture.Capture) {
				delete(c.Files, c.Code.SASS)
			},
			wantErr: "not found in files",
		},
		{
			name: "reserved name collision",
			mutate: func(c *cudacapture.Capture) {
				c.Files["manifest.json"] = []byte("boom")
			},
			wantErr: "reserved",
		},
		{
			name: "unsafe path with separator",
			mutate: func(c *cudacapture.Capture) {
				c.Files["sub/file.txt"] = []byte("boom")
			},
			wantErr: "not a flat relative name",
		},
		{
			name: "case collision",
			mutate: func(c *cudacapture.Capture) {
				c.Files["KERNELS.CU"] = []byte("boom")
			},
			wantErr: "case collision",
		},
		{
			name: "invalid metadata JSON",
			mutate: func(c *cudacapture.Capture) {
				c.Metadata = json.RawMessage(`{not json}`)
			},
			wantErr: "invalid metadata JSON",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := validFixture()
			tt.mutate(c)
			var buf bytes.Buffer
			err := cudacapture.Write(&buf, c)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Write error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestLegacyV1ReadDir(t *testing.T) {
	dir := t.TempDir()
	files := []string{"gpu.pprof", "go.trace", "main.go", "capture.go", "workload.go", "kernels.cu", "kernels.ptx", "kernels.cubin", "kernels.sass", "nvdisasm.txt", "README.md"}
	for _, f := range files {
		if f == "gpu.pprof" {
			fn := &profile.Function{ID: 1, Name: "k", Filename: "kernels.cu"}
			loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
			p := &profile.Profile{
				SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
				Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
				Location:   []*profile.Location{loc},
				Function:   []*profile.Function{fn},
			}
			raw, _ := os.Create(filepath.Join(dir, f))
			_ = p.Write(raw)
			_ = raw.Close()
		} else {
			_ = os.WriteFile(filepath.Join(dir, f), []byte("content of "+f), 0644)
		}
	}

	manifestJSON := `{
  "version": 1,
  "created": "2026-09-09T00:00:00Z",
  "device": "NVIDIA GB10",
  "architecture": "sm_121",
  "go_version": "go1.26.7",
  "nvrtc_version": "13.0",
  "elements": 257,
  "passes": 2,
  "fma_rounds": 8,
  "cuda_event_nanoseconds": {},
  "verified_results": 1028,
  "max_absolute_error": 0.0,
  "sha256": {}
}`
	_ = os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifestJSON), 0644)

	c, err := cudacapture.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir v1 failed: %v", err)
	}

	if c.Code.Source != "kernels.cu" || c.Code.PTX != "kernels.ptx" || c.Code.CUBIN != "kernels.cubin" || c.Code.SASS != "kernels.sass" {
		t.Errorf("v1 Code mapping mismatch: %+v", c.Code)
	}
	if len(c.Files) < 4 {
		t.Errorf("expected at least 4 files loaded, got %d", len(c.Files))
	}
}

func TestLegacyV1ReadEmbedded(t *testing.T) {
	fn := &profile.Function{ID: 1, Name: "k", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	payload := `{
  "manifest": {
    "version": 1,
    "created": "2026-09-09T00:00:00Z",
    "device": "NVIDIA GB10",
    "sha256": {
      "kernels.cu": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
    }
  },
  "files": {
    "kernels.cu": "Zm9v"
  }
}`
	p.Comments = append(p.Comments, "#cudaprof:capture:v1\n"+payload)

	var buf bytes.Buffer
	if err := p.Write(&buf); err != nil {
		t.Fatal(err)
	}

	c, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read v1 embedded failed: %v", err)
	}
	if string(c.Files["kernels.cu"]) != "foo" {
		t.Errorf("expected kernels.cu to be 'foo', got %q", string(c.Files["kernels.cu"]))
	}
}
