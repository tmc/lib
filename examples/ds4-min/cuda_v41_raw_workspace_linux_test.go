//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var v41RawWorkspaceOutput = flag.String("v41-raw-output", "", "new absolute raw-attention vector directory")

type v41RawWorkspaceCapture struct {
	root  string
	files []struct {
		Path   string
		Bytes  int
		SHA256 string
	}
}

func newV41RawWorkspaceCapture(t *testing.T) *v41RawWorkspaceCapture {
	t.Helper()
	p := *v41RawWorkspaceOutput
	if !filepath.IsAbs(p) || filepath.Clean(p) != p {
		t.Fatal("-v41-raw-output must be a clean absolute fresh directory")
	}
	if err := os.Mkdir(p, 0700); err != nil {
		t.Fatal(err)
	}
	return &v41RawWorkspaceCapture{root: p}
}

func (c *v41RawWorkspaceCapture) save(t *testing.T, name string, v []float32) {
	t.Helper()
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	c.saveBytes(t, name, b)
}

func (c *v41RawWorkspaceCapture) saveBytes(t *testing.T, name string, b []byte) {
	t.Helper()
	if err := c.write(t.Name(), name, b); err != nil {
		t.Fatal(err)
	}
}

func (c *v41RawWorkspaceCapture) write(group, name string, b []byte) error {
	p := filepath.Join(group, name+".bin")
	full := filepath.Join(c.root, p)
	if err := os.MkdirAll(filepath.Dir(full), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(full, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	n, we := f.Write(b)
	ce := f.Close()
	if we != nil || ce != nil || n != len(b) {
		return fmt.Errorf("vector write %d/%d: %w", n, len(b), errors.Join(we, ce))
	}
	got, err := os.ReadFile(full)
	if err != nil || !bytes.Equal(got, b) {
		return fmt.Errorf("vector readback differs: %w", err)
	}
	c.files = append(c.files, struct {
		Path   string
		Bytes  int
		SHA256 string
	}{filepath.ToSlash(p), len(b), fmt.Sprintf("%x", sha256.Sum256(b))})
	j, err := json.MarshalIndent(c.files, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(c.root, "manifest.json"), j, 0600); err != nil {
		return err
	}
	return nil
}

func (c *v41RawWorkspaceCapture) compiled(group string, r *cudaFoundation, p *nvrtcx.CompiledModule) error {
	if len(p.CUBIN()) == 0 {
		return errors.New("executed NVRTC CUBIN missing")
	}
	flags, err := nvrtcx.CompileFlags(nvrtcx.WithName("cuda_v41_raw_attention.cu"), nvrtcx.WithDeviceArch(r.device), nvrtcx.WithNoCache())
	if err != nil {
		return err
	}
	meta, err := json.MarshalIndent(map[string]any{"flags": flags, "implicit_toolkit_include": nvrtcx.ToolkitIncludeDir(), "dependencies": p.Dependencies(), "note": "flags exclude automatic source-triggered include insertion; retain toolkit closure"}, "", "  ")
	if err != nil {
		return err
	}
	for _, v := range []struct {
		name string
		b    []byte
	}{{"executed-cubin", p.CUBIN()}, {"executed-ptx", []byte(p.PTX())}, {"nvrtc-log", []byte(p.Log())}, {"compile-meta", meta}, {"compile-prefix", []byte(cudaV41CompilePrefix)}, {"stdint-header", []byte(cudaV41DeviceStdint)}} {
		if err := c.write(group, v.name, v.b); err != nil {
			return err
		}
	}
	return nil
}

func captureV41RawReplay(t *testing.T, c *v41RawWorkspaceCapture, resources *v41RawCleanup, s *cudaV41RawAttention, out, sinks cudamem.Slice[float32], prefix string, position uint32) {
	t.Helper()
	// The primary Append result and scratch are already retained. This replay
	// reads that exact post-RoPE query/ring and executes its actual loaded image.
	s.pending = []cudamem.Slice[float32]{out, sinks}
	err := s.r.device.WithContext(func() error {
		return cudalaunch.Kernel(s.attend).Grid(2, 1, 1).Block(32, 1, 1).On(s.r.stream()).Launch(out, s.q, s.ring, sinks, s.scores, position)
	})
	drain := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) })
	if drain == nil {
		s.pending = nil
	}
	if err != nil || drain != nil {
		if drain != nil {
			resources.err = drain
		}
		t.Fatal("pre-BF16 replay", errors.Join(err, drain))
	}
	for _, v := range []struct {
		name string
		dev  cudamem.Slice[float32]
	}{{"executed-heads", out}, {"executed-q", s.q}, {"executed-kv", s.kv}} {
		x := make([]float32, v.dev.Len())
		if err := v.dev.CopyToHost(x); err != nil {
			t.Fatal(err)
		}
		c.save(t, prefix+v.name, x)
	}
}

func testV41RawScoreABI(t *testing.T, s *cudaV41RawAttention) {
	t.Helper()
	err := s.r.device.WithContext(func() error {
		for i, p := range [][2]uint{{0, 8}, {8, 8}, {16, 8}, {24, 8}, {32, 8}, {40, 4}} {
			o, n, e := s.attend.ParamInfo(uint(i))
			if e != nil {
				return e
			}
			if o != p[0] || n != p[1] {
				return fmt.Errorf("attend parameter %d offset/size %d/%d", i, o, n)
			}
		}
		_, _, err := s.attend.ParamInfo(6)
		if !errors.Is(err, driver.ErrInvalidValue) {
			return fmt.Errorf("attend parameter count exhaustion: %w", err)
		}
		return nil
	})
	if err != nil {
		t.Fatal("actual score workspace ABI", err)
	}
}

func poisonV41RawScores(t *testing.T, s *cudaV41RawAttention, repeat int) {
	t.Helper()
	v := make([]float32, s.scoresStorage.Len())
	for i := range v {
		bits := uint32(0x7fc12345)
		if repeat != 0 {
			bits = 0xc2f68000
		}
		if i < 32 || i >= 32+64*128 {
			bits = 0x4e13579b + uint32(i)
		}
		v[i] = math.Float32frombits(bits)
	}
	if err := s.scoresStorage.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
}

func checkV41RawScores(t *testing.T, c *v41RawWorkspaceCapture, s *cudaV41RawAttention, prefix string, q, ring, sinks []float32, pos int, variant string) {
	t.Helper()
	got := make([]float32, s.scoresStorage.Len())
	if err := s.scoresStorage.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	c.save(t, prefix+"workspace-guarded", got)
	n := min(pos+1, 128)
	if variant == "window" {
		n = min(n, 127)
	}
	first := pos + 1 - n
	want := make([]float32, 64*128)
	for h := 0; h < 64; h++ {
		peak := sinks[h]
		for row := 0; row < n; row++ {
			dot := float32(0)
			for d := 0; d < 512; d++ {
				dot = float32(dot + float32(q[h*512+d]*ring[((first+row)%128)*512+d]))
			}
			v := float32(dot * float32(1/float32(math.Sqrt(512))))
			want[h*128+row] = v
			peak = max(peak, v)
		}
		for row := 0; row < n; row++ {
			want[h*128+row] = float32(math.Exp(float64(float32(want[h*128+row] - peak))))
		}
	}
	c.save(t, prefix+"workspace-reference", want)
	for i, v := range got {
		if i < 32 || i >= 32+64*128 {
			if math.Float32bits(v) != 0x4e13579b+uint32(i) {
				t.Fatal("score workspace canary changed")
			}
		}
	}
	changed := 0
	for i, w := range want {
		v := got[32+i]
		if i%128 >= n && math.Float32bits(v) != 0 {
			t.Fatal("score workspace not fully written")
		}
		if math.Float32bits(v) != math.Float32bits(w) {
			changed++
		}
	}
	if variant == "scratch" {
		if changed == 0 {
			t.Fatal("score write mutant survived")
		}
	} else if changed != 0 {
		t.Fatalf("score workspace arithmetic differs: %d", changed)
	}
}

// Source mutation of the workspace admission guard must be rejected before a
// launch, without touching validity/frontier. Device ranges stay real.
func testV41RawWorkspaceAdmission(t *testing.T, s *cudaV41RawAttention, q, kv, sinks, out cudamem.Slice[float32]) {
	t.Helper()
	old := s.scores
	s.scores = s.ring.Slice(0, 64*128)
	err := s.Append(out, q, kv, sinks, s.position)
	s.scores = old
	if err == nil || !s.valid || s.position != 0 {
		t.Fatal("foreign score workspace admitted")
	}
	// The owned storage itself must be covered by the normal overlap gate.
	oldStorage := s.scoresStorage
	s.scoresStorage = q.Slice(0, 64*128+64)
	s.scores = s.scoresStorage.Slice(32, 32+64*128)
	err = s.Append(out, q, kv, sinks, s.position)
	s.scoresStorage = oldStorage
	s.scores = old
	if err == nil || !s.valid || s.position != 0 {
		t.Fatal("aliased score workspace admitted")
	}
}
