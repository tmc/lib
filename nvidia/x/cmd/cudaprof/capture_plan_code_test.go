package main

import (
	"bytes"
	"compress/gzip"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func planCodeFixture(t *testing.T) (*cudacapture.Capture, cudaprof.CapturePublication) {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		b, e := os.ReadFile(filepath.Join("..", "..", "cudacapture", "testdata", name))
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	raw, e := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if e != nil {
		t.Fatal(e)
	}
	schema, e := cudakernel.NewSchema()
	if e != nil {
		t.Fatal(e)
	}
	a, e := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "fill", schema, "sm_90")
	if e != nil {
		t.Fatal(e)
	}
	w := cudacache.WorkloadIdentity{ProgramID: "retained fill fixture", ShapeDims: []int{64}, NumericalMode: "fp32"}
	key := cudacache.CompilerCacheKey(a.Digest, cudacache.CompilerIdentity{TargetArch: a.Target, OutputKind: nvrtcx.OutputCUBIN, Flags: []string{"fixture producer receipt"}})
	id, e := cudaprof.BindEvidence(a, w, key)
	if e != nil {
		t.Fatal(e)
	}
	code := cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"}
	c := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}}, Code: code, Modules: []cudacapture.Module{{ID: "module", Generation: 1, Code: code}}, Files: map[string][]byte{"kernels.cu": read("kernels.cu"), "kernels.ptx": read("kernels.ptx"), "kernels.cubin": raw, "kernels.sass": read("kernels.sass")}}
	return c, cudaprof.CapturePublication{Version: 1, Artifact: a, Workload: w, CompilerKey: key, Result: cudaprof.CaptureProviderResult{Provider: cudaprof.CaptureRange, Identity: id}}
}
func planCodePacket(t *testing.T, c *cudacapture.Capture) []byte {
	t.Helper()
	var b bytes.Buffer
	if e := cudacapture.WriteLimits(&b, c, cudacapture.Limits{}); e != nil {
		t.Fatal(e)
	}
	return b.Bytes()
}
func planCodeWrite(t *testing.T, b []byte) string {
	t.Helper()
	d := t.TempDir()
	if e := os.WriteFile(filepath.Join(d, "capture-provider-range.pprof"), b, 0600); e != nil {
		t.Fatal(e)
	}
	return d
}
func TestCapturePlanCodeRoundTrip(t *testing.T) {
	c, p := planCodeFixture(t)
	raw := planCodePacket(t, c)
	got, e := admitCapturePlanCode(planCodeWrite(t, raw), 1<<20, p)
	if e != nil {
		t.Fatal(e)
	}
	if !bytes.Equal(got, raw) {
		t.Fatal("original packet changed")
	}
	parsed, e := cudacapture.ReadLimits(bytes.NewReader(got), cudacapture.Limits{})
	if e != nil {
		t.Fatal(e)
	}
	if _, e = cudacapture.VerifyModule(parsed, "module", p.Artifact.Digest, []string{"fill"}); e != nil {
		t.Fatal(e)
	}
}
func TestCapturePlanCodeRefusals(t *testing.T) {
	for _, tc := range []struct {
		name    string
		change  func(*cudacapture.Capture, *cudaprof.CapturePublication)
		missing bool
	}{
		{"cubin role", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { c.Modules[0].Code.CUBIN = "" }, true},
		{"ptx role", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { c.Modules[0].Code.PTX = "" }, true},
		{"source role", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { c.Modules[0].Code.Source = "" }, true},
		{"sass role", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { c.Modules[0].Code.SASS = "" }, true},
		{"stale image", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) {
			c.Files["kernels.cubin"][len(c.Files["kernels.cubin"])-1] ^= 1
		}, false},
		{"duplicate generation", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) {
			m := c.Modules[0]
			m.ID = "second"
			m.Generation = 2
			c.Modules = append(c.Modules, m)
		}, false},
		{"corrupt artifact", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { p.Artifact.Raw[0] ^= 1 }, false},
		{"missing entry", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { p.Artifact.EntryPoint = "" }, false},
		{"identity entry", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { p.Result.Identity.Entry = "other" }, false},
		{"exact entry", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) {
			p.Artifact.EntryPoint = "scale"
			p.Result.Identity.Entry = "scale"
		}, false},
		{"native target", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) {
			p.Artifact.Target = "sm_80"
			p.Result.Identity.Target = "sm_80"
		}, false},
		{"feature suffix", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) {
			p.Artifact.Target = "sm_90a"
			p.Result.Identity.Target = "sm_90a"
		}, true},
		{"virtual ptx", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { p.Artifact.Kind = cudakernel.KindPTX }, true},
		{"missing artifact", func(c *cudacapture.Capture, p *cudaprof.CapturePublication) { p.Artifact = nil }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, p := planCodeFixture(t)
			tc.change(c, &p)
			raw := planCodePacket(t, c)
			got, e := admitCapturePlanCode(planCodeWrite(t, raw), 1<<20, p)
			if e == nil {
				t.Fatal("admitted invalid association")
			}
			if tc.missing && !errors.Is(e, os.ErrNotExist) {
				t.Fatalf("want unavailable: %v", e)
			}
			if !bytes.Equal(got, raw) {
				t.Fatal("diagnostic packet lost")
			}
		})
	}
}
func TestCapturePlanCodeInputBounds(t *testing.T) {
	c, p := planCodeFixture(t)
	raw := planCodePacket(t, c)
	t.Run("missing", func(t *testing.T) {
		_, e := admitCapturePlanCode(t.TempDir(), 1<<20, p)
		if !errors.Is(e, os.ErrNotExist) {
			t.Fatal(e)
		}
	})
	t.Run("symlink", func(t *testing.T) {
		d := t.TempDir()
		src := filepath.Join(d, "original")
		if e := os.WriteFile(src, raw, 0600); e != nil {
			t.Fatal(e)
		}
		if e := os.Symlink(src, filepath.Join(d, "capture-provider-range.pprof")); e != nil {
			t.Fatal(e)
		}
		if _, e := admitCapturePlanCode(d, 1<<20, p); e == nil {
			t.Fatal("accepted symlink")
		}
	})
	t.Run("oversize", func(t *testing.T) {
		if _, e := admitCapturePlanCode(planCodeWrite(t, raw), len(raw)-1, p); e == nil {
			t.Fatal("accepted oversized file")
		}
	})
	t.Run("truncated", func(t *testing.T) {
		b := raw[:len(raw)/2]
		got, e := admitCapturePlanCode(planCodeWrite(t, b), 1<<20, p)
		if e == nil || !bytes.Equal(got, b) {
			t.Fatalf("raw retained=%v err=%v", bytes.Equal(got, b), e)
		}
	})
	t.Run("gzip expansion", func(t *testing.T) {
		var b bytes.Buffer
		z := gzip.NewWriter(&b)
		if _, e := z.Write(bytes.Repeat([]byte{0}, 8192)); e != nil {
			t.Fatal(e)
		}
		if e := z.Close(); e != nil {
			t.Fatal(e)
		}
		got, e := admitCapturePlanCode(planCodeWrite(t, b.Bytes()), 1024, p)
		if e == nil || !strings.Contains(e.Error(), "decompressed byte budget") || !bytes.Equal(got, b.Bytes()) {
			t.Fatalf("raw retained=%v err=%v", bytes.Equal(got, b.Bytes()), e)
		}
	})
}
