package profilebundle

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
)

func packedFixture(t *testing.T) []byte {
	t.Helper()
	dir := t.TempDir()
	for _, name := range Files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0644); err != nil {
			t.Fatal(err)
		}
	}
	fn := &profile.Function{ID: 1, Name: "kernel", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{123}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}
	var raw bytes.Buffer
	if err := p.Write(&raw); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "gpu.pprof"), raw.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	if err := Write(dir, Manifest{Device: "test GPU"}); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := Pack(dir, &out); err != nil {
		t.Fatal(err)
	}
	// Subsequent reads must succeed without the original directory.
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

func TestSingleFile(t *testing.T) {
	data := packedFixture(t)
	// Exercise a normal pprof parse/write cycle, as used by pprof -proto.
	p, err := profile.ParseData(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.CheckValid(); err != nil {
		t.Fatal(err)
	}
	if p.SampleType[0].Type != "gpu_time" || len(p.Sample) != 1 || p.Sample[0].Value[0] != 123 {
		t.Fatalf("profile samples changed: %+v", p)
	}
	var roundtrip bytes.Buffer
	if err := p.Write(&roundtrip); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "renamed recording.pprof")
	if err := os.WriteFile(path, roundtrip.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	c, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for _, name := range Files {
		if name == "gpu.pprof" {
			continue
		}
		got, err := os.ReadFile(filepath.Join(c.Dir, name))
		if err != nil || string(got) != name {
			t.Fatalf("%s = %q, %v", name, got, err)
		}
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.Dir); !os.IsNotExist(err) {
		t.Fatalf("temporary files remain: %v", err)
	}
}

func TestEmbeddedCorruption(t *testing.T) {
	data := packedFixture(t)
	for _, name := range []string{"checksum", "missing", "path", "version", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			p, err := profile.ParseData(data)
			if err != nil {
				t.Fatal(err)
			}
			i := len(p.Comments) - 1
			var b embeddedCapture
			if err := json.Unmarshal([]byte(strings.TrimPrefix(p.Comments[i], capturePrefix)), &b); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "checksum":
				b.Files["kernels.sass"] = []byte("changed")
			case "missing":
				delete(b.Files, "kernels.cu")
			case "path":
				b.Files["../outside"] = b.Files["kernels.cu"]
				delete(b.Files, "kernels.cu")
			case "version":
				b.Manifest.Version++
			case "duplicate":
				p.Comments = append(p.Comments, p.Comments[i])
			}
			encoded, err := json.Marshal(b)
			if err != nil {
				t.Fatal(err)
			}
			p.Comments[i] = capturePrefix + string(encoded)
			if _, err := embedded(p); err == nil {
				t.Fatal("accepted corrupted capture")
			}
		})
	}
}
