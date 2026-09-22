package cudacapture_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"reflect"
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

	if !reflect.DeepEqual(got.Code, orig.Code) {
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

func TestCFGAttachment(t *testing.T) {
	c := validFixture()
	c.Code.CFG = "kernels.cfg.dot"
	c.Code.BBCFG = "kernels.bbcfg.dot"
	c.Files["kernels.cfg.dot"] = []byte("digraph compute { bb0; }\n")
	c.Files["kernels.bbcfg.dot"] = []byte("digraph compute { bb0; bb1; bb0 -> bb1; }\n")

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatalf("Write with CFG failed: %v", err)
	}

	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read with CFG failed: %v", err)
	}
	if got.Code.CFG != "kernels.cfg.dot" || got.Code.BBCFG != "kernels.bbcfg.dot" {
		t.Errorf("Code CFG mismatch: %+v", got.Code)
	}
	if !bytes.Equal(got.Files["kernels.cfg.dot"], c.Files["kernels.cfg.dot"]) {
		t.Errorf("kernels.cfg.dot content mismatch")
	}
	if !bytes.Equal(got.Files["kernels.bbcfg.dot"], c.Files["kernels.bbcfg.dot"]) {
		t.Errorf("kernels.bbcfg.dot content mismatch")
	}

	dir := t.TempDir()
	if err := cudacapture.WriteDir(dir, c); err != nil {
		t.Fatalf("WriteDir with CFG failed: %v", err)
	}
	fromDir, err := cudacapture.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir with CFG failed: %v", err)
	}
	if fromDir.Code.CFG != "kernels.cfg.dot" || fromDir.Code.BBCFG != "kernels.bbcfg.dot" {
		t.Errorf("ReadDir Code CFG mismatch: %+v", fromDir.Code)
	}
	if !bytes.Equal(fromDir.Files["kernels.cfg.dot"], c.Files["kernels.cfg.dot"]) {
		t.Errorf("ReadDir kernels.cfg.dot content mismatch")
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

	if !reflect.DeepEqual(got.Code, orig.Code) {
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

func TestWriteDirDoesNotReplaceExistingOutput(t *testing.T) {
	dir := t.TempDir()
	marker := []byte("keep this file")
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), marker, 0600); err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.WriteDir(dir, validFixture()); err == nil || !strings.Contains(err.Error(), "output file already exists") {
		t.Fatalf("WriteDir error = %v, want collision", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, marker) {
		t.Fatal("existing manifest was replaced")
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
		{
			name: "file size limit",
			mutate: func(c *cudacapture.Capture) {
				c.Files["extra.go"] = bytes.Repeat([]byte{'x'}, cudacapture.MaxCaptureFileBytes+1)
			},
			wantErr: "exceeds size limit",
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

func TestVersion3OptionalArtifactsAndModules(t *testing.T) {
	c := validFixture()
	delete(c.Files, c.Code.Source)
	delete(c.Files, c.Code.PTX)
	c.Code.Source = ""
	c.Code.PTX = ""
	c.CodeUnavailable = map[string]string{
		"source": "third-party source was not present",
		"ptx":    "module was loaded from cubin only",
	}
	c.Modules = []cudacapture.Module{{
		ID: "module-7", Generation: 2, Name: "library-kernel",
		Code: cudacapture.CodeFiles{CUBIN: "kernels.cubin", SASS: "kernels.sass"},
	}}
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatal(err)
	}
	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Modules) != 1 || got.Modules[0].ID != "module-7" {
		t.Fatalf("modules = %+v", got.Modules)
	}
	if got.CodeUnavailable["source"] == "" || got.Code.Source != "" {
		t.Fatalf("unavailable code = %+v code=%+v", got.CodeUnavailable, got.Code)
	}
	if _, err := cudacapture.LoadCode(got); err == nil || !strings.Contains(err.Error(), "ptx unavailable") {
		t.Fatalf("LoadCode error = %v", err)
	}
	dir := t.TempDir()
	if err := cudacapture.WriteDir(dir, c); err != nil {
		t.Fatal(err)
	}
	fromDir, err := cudacapture.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(fromDir.Modules) != 1 || fromDir.CodeUnavailable["source"] == "" {
		t.Fatalf("directory extension = %+v", fromDir)
	}
}

func TestVersion3TypedObservationsAndHonestBanner(t *testing.T) {
	c := validFixture()
	delete(c.Files, c.Code.Source)
	delete(c.Files, c.Code.PTX)
	c.Code.Source, c.Code.PTX = "", ""
	c.CodeUnavailable = map[string]string{"source": "not retained", "ptx": "not retained"}
	c.Descriptors = []cudacapture.MetricDescriptor{{ID: "gpu.kernel_time", Namespace: "cupti", Name: "kernel time", Unit: "nanoseconds", NumericType: "uint64", Aggregation: "sum", Scope: "kernel", Collector: "native-cupti", Support: "cuda13"}}
	c.Observations = []cudacapture.Observation{{DescriptorID: "gpu.kernel_time", EntityID: "kernel:1", Start: 10, End: 20, Value: json.RawMessage(`10`), Availability: "collected"}}
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatal(err)
	}
	got, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Descriptors) != 1 || len(got.Observations) != 1 {
		t.Fatalf("typed data = %+v %+v", got.Descriptors, got.Observations)
	}
	joined := strings.Join(got.Profile.Comments, "\n")
	if !strings.Contains(joined, "#cudaprof:capture:v3\n") || strings.Contains(joined, "Source, PTX, SASS, CUBIN, and artifacts are embedded") || !strings.Contains(joined, "Embedded code artifacts: cubin, sass") || strings.Contains(joined, "No source, PTX, CUBIN, or SASS code artifacts were captured") {
		t.Fatalf("banner = %q", joined)
	}
}

func TestReadDirRejectsSymlinkedAttachment(t *testing.T) {
	c := validFixture()
	dir := t.TempDir()
	if err := cudacapture.WriteDir(dir, c); err != nil {
		t.Fatal(err)
	}
	attachment := filepath.Join(dir, "extra.go")
	if err := os.Remove(attachment); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/etc/hosts", attachment); err != nil {
		t.Fatal(err)
	}
	if _, err := cudacapture.ReadDir(dir); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("ReadDir error = %v, want symlink rejection", err)
	}
}

func TestReadDirRejectsSymlinkedControlFiles(t *testing.T) {
	for _, name := range []string{"manifest.json", "gpu.pprof"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if err := cudacapture.WriteDir(dir, validFixture()); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, name)
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(path+".target", path); err != nil {
				t.Fatal(err)
			}
			if _, err := cudacapture.ReadDir(dir); err == nil || !strings.Contains(err.Error(), "regular file") {
				t.Fatalf("ReadDir error = %v, want symlink rejection", err)
			}
		})
	}
}

func TestLoadCodeWithFixture(t *testing.T) {
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubinBytes, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    read("kernels.cu"),
			"kernels.ptx":   read("kernels.ptx"),
			"kernels.cubin": cubinBytes,
			"kernels.sass":  read("kernels.sass"),
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}

	// The fixture's two entry points differ in linkage, so the CUBIN spells
	// one of them Itanium-mangled and the other verbatim. Both must join, and
	// the mangled one must still resolve to its source definition, which the
	// source spells "scale".
	tests := []struct {
		name       string
		offset     uint64
		size       uint64
		sourceLine int
		ptxLine    int
		insts      int
	}{
		{name: "fill", offset: 3968, size: 384, sourceLine: 16, ptxLine: 49, insts: 24},
		{name: "_Z5scalePfPKff", offset: 4352, size: 384, sourceLine: 11, ptxLine: 15, insts: 24},
	}
	if len(code.Kernels) != len(tests) {
		t.Fatalf("got %d kernels, want %d", len(code.Kernels), len(tests))
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The kernels are ordered by CUBIN offset, which is ptxas's
			// layout and not the source order.
			k := code.Kernels[i]
			if k.Name != tt.name {
				t.Fatalf("Kernel.Name = %q, want %q", k.Name, tt.name)
			}
			if k.Offset != tt.offset {
				t.Errorf("Kernel.Offset = %d, want %d", k.Offset, tt.offset)
			}
			if k.Size != tt.size {
				t.Errorf("Kernel.Size = %d, want %d", k.Size, tt.size)
			}
			if k.SourceFile != "kernels.cu" {
				t.Errorf("Kernel.SourceFile = %q, want kernels.cu", k.SourceFile)
			}
			if k.SourceLine != tt.sourceLine {
				t.Errorf("Kernel.SourceLine = %d, want %d", k.SourceLine, tt.sourceLine)
			}
			if k.PTXLine != tt.ptxLine {
				t.Errorf("Kernel.PTXLine = %d, want %d", k.PTXLine, tt.ptxLine)
			}
			if len(k.Instructions) != tt.insts {
				t.Fatalf("got %d instructions, want %d", len(k.Instructions), tt.insts)
			}
			inst0 := k.Instruction(0)
			if inst0 == nil {
				t.Fatal("instruction at offset 0 not found")
			}
			// The entry instruction sets up the stack pointer and the
			// disassembler attributes it to the kernel's own declaration line.
			if inst0.SourceLine != tt.sourceLine {
				t.Errorf("inst0.SourceLine = %d, want %d", inst0.SourceLine, tt.sourceLine)
			}
		})
	}
}

func TestLoadCodeWithCaptureSmoke(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "captures", "v2-gb10-capture.pprof")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("skipping test without fixture: %v", err)
	}
	defer f.Close()

	cap, err := cudacapture.Read(f)
	if err != nil {
		t.Fatalf("Read capture: %v", err)
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}

	if len(code.Kernels) == 0 {
		t.Fatal("expected at least one kernel in capture")
	}
}
