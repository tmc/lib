//go:build linux && native

package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type sidecarImageFakeKernel struct {
	function   driver.Function
	closeError error
}

func (k *sidecarImageFakeKernel) Raw() driver.Function                     { return k.function }
func (k *sidecarImageFakeKernel) Launch(cudalaunch.Config, ...any) error   { return nil }
func (k *sidecarImageFakeKernel) Wait(context.Context, *cuda.Stream) error { return nil }
func (k *sidecarImageFakeKernel) Close() error                             { return k.closeError }

type sidecarImageFakeLoader struct {
	kernel    *sidecarImageFakeKernel
	loadError error
	loadCalls int
}

func (l *sidecarImageFakeLoader) Compile(_ context.Context, s cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return &cudahotswap.Compilation{Entry: s.Entry, Key: s.Code}, nil
}
func (l *sidecarImageFakeLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	l.loadCalls++
	return l.kernel, l.loadError
}
func sidecarImageControl(t *testing.T) (*sidecarImageLoader, *sidecarImageFakeLoader, *cudahotswap.Compilation, cudahotswap.State) {
	t.Helper()
	base := &sidecarImageFakeLoader{kernel: &sidecarImageFakeKernel{function: 23}}
	l := newSidecarImageLoader(base, "literal-source", []byte("explicit CPU fixture metadata"))
	l.image = func(*cudahotswap.Compilation) ([]byte, error) { return []byte{127, 69, 76, 70, 17}, nil }
	c, err := l.Compile(context.Background(), cudahotswap.Source{Code: "literal-source", Entry: "chain"})
	if err != nil {
		t.Fatal(err)
	}
	m := l.compiled[c]
	state := cudahotswap.State{Name: "kernel", Kind: "kernel", Version: 1, Generation: 1, SourceDigest: m.Digest, CompilerKey: m.Key}
	return l, base, c, state
}
func TestSidecarImageLifetimeCPU(t *testing.T) {
	l, base, c, state := sidecarImageControl(t)
	k, err := l.Load(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	record, body, err := l.snapshot(state, 23)
	if err != nil || record.Bytes != 5 || record.Entry != "chain" || record.Function != 23 {
		t.Fatalf("record=%+v body=%v err=%v", record, body, err)
	}
	if record.ReferenceSourceSHA256 != state.SourceDigest || record.ReferenceSidecarSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte("explicit CPU fixture metadata"))) {
		t.Fatal("reference identity differs")
	}
	body[0] = 0
	_, again, _ := l.snapshot(state, 23)
	if again[0] != 127 {
		t.Fatal("image aliases snapshot")
	}
	for _, bad := range []cudahotswap.State{{}, {Name: "kernel", Kind: "kernel", Version: 1, Generation: 1, SourceDigest: "wrong", CompilerKey: state.CompilerKey}, {Name: "kernel", Kind: "kernel", Version: 1, Generation: 1, SourceDigest: state.SourceDigest, CompilerKey: "wrong"}} {
		if _, _, err := l.snapshot(bad, 23); err == nil {
			t.Fatal("forged state accepted")
		}
	}
	if _, _, err := l.snapshot(state, 24); err == nil {
		t.Fatal("wrong function accepted")
	}
	base.kernel.closeError = errors.New("owned close failure")
	if k.Close() == nil {
		t.Fatal("close failure lost")
	}
	if _, _, err := l.snapshot(state, 23); err != nil {
		t.Fatal("failed close retired owner")
	}
	base.kernel.closeError = nil
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.snapshot(state, 23); err == nil {
		t.Fatal("closed function accepted")
	}
	c2, err := l.Compile(context.Background(), cudahotswap.Source{Code: "new-source", Entry: "chain"})
	if err != nil {
		t.Fatal(err)
	}
	m := l.compiled[c2]
	state.SourceDigest = m.Digest
	state.CompilerKey = m.Key
	state.Version = 2
	state.Generation = 2
	if _, err := l.Load(context.Background(), c2); err != nil {
		t.Fatal("retired raw handle could not be reused", err)
	}
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.snapshot(state, 23); err != nil {
		t.Fatal(err)
	}
}
func TestSidecarImageRefusalsCPU(t *testing.T) {
	t.Run("active reuse", func(t *testing.T) {
		l, _, c, _ := sidecarImageControl(t)
		if _, e := l.Load(context.Background(), c); e != nil {
			t.Fatal(e)
		}
		c2, _ := l.Compile(context.Background(), cudahotswap.Source{Code: "second", Entry: "chain"})
		if k, e := l.Load(context.Background(), c2); e == nil || k == nil {
			t.Fatal("must refuse and retain returned owner")
		}
	})
	t.Run("partial load", func(t *testing.T) {
		l, b, c, s := sidecarImageControl(t)
		b.loadError = errors.New("partial load")
		if k, e := l.Load(context.Background(), c); e == nil || k == nil {
			t.Fatal("partial owner lost")
		}
		if _, _, e := l.snapshot(s, 23); e == nil {
			t.Fatal("partial load recorded")
		}
	})
	for _, n := range []int{0, sidecarImageBytes + 1} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			l, b, c, _ := sidecarImageControl(t)
			l.image = func(*cudahotswap.Compilation) ([]byte, error) { return make([]byte, n), nil }
			if _, e := l.Load(context.Background(), c); e == nil || b.loadCalls != 0 {
				t.Fatal("image bound failed before load")
			}
		})
	}

	t.Run("entry drift", func(t *testing.T) {
		l, b, c, _ := sidecarImageControl(t)
		c.Entry = "changed"
		if _, e := l.Load(context.Background(), c); e == nil || b.loadCalls != 0 {
			t.Fatal("entry drift accepted")
		}
	})
	t.Run("image error", func(t *testing.T) {
		l, b, c, _ := sidecarImageControl(t)
		l.image = func(*cudahotswap.Compilation) ([]byte, error) { return nil, errors.New("image refused") }
		if _, e := l.Load(context.Background(), c); e == nil || b.loadCalls != 0 {
			t.Fatal("image refusal ignored")
		}
	})
	t.Run("missing function", func(t *testing.T) {
		l, b, c, s := sidecarImageControl(t)
		b.kernel.function = 0
		if k, e := l.Load(context.Background(), c); e == nil || k == nil {
			t.Fatal("must refuse and retain owner")
		}
		if _, _, e := l.snapshot(s, 0); e == nil {
			t.Fatal("zero function accepted")
		}
	})
	t.Run("metadata drift", func(t *testing.T) {
		l, b, c, _ := sidecarImageControl(t)
		c.Key = "forged"
		if _, e := l.Load(context.Background(), c); e == nil || b.loadCalls != 0 {
			t.Fatal("metadata drift accepted")
		}
	})
	t.Run("unassociated", func(t *testing.T) {
		l, b, _, _ := sidecarImageControl(t)
		if _, e := l.Load(context.Background(), &cudahotswap.Compilation{}); e == nil || b.loadCalls != 0 {
			t.Fatal("unassociated accepted")
		}
	})
	t.Run("capacity", func(t *testing.T) {
		l, _, _, _ := sidecarImageControl(t)
		for i := 1; i < sidecarImageLimit; i++ {
			if _, e := l.Compile(context.Background(), cudahotswap.Source{Code: "same", Entry: "chain"}); e != nil {
				t.Fatal(e)
			}
		}
		if _, e := l.Compile(context.Background(), cudahotswap.Source{Code: "same", Entry: "chain"}); e == nil {
			t.Fatal("capacity exceeded")
		}
	})
}

func FuzzSidecarImageSnapshotCPU(f *testing.F) {
	f.Add("literal-source", "literal-source", uint64(1), uint64(1), uint64(23))
	f.Add("wrong", "literal-source", uint64(0), uint64(2), uint64(24))
	f.Fuzz(func(t *testing.T, digest, key string, version, generation, function uint64) {
		l, _, c, state := sidecarImageControl(t)
		if _, err := l.Load(context.Background(), c); err != nil {
			t.Fatal(err)
		}
		goodDigest := state.SourceDigest
		if digest == "literal-source" {
			digest = goodDigest
		}
		state.SourceDigest = digest
		state.CompilerKey = key
		state.Version = version
		state.Generation = generation
		_, _, err := l.snapshot(state, driver.Function(function))
		want := digest == goodDigest && key == "literal-source" && version != 0 && generation != 0 && function == 23
		if (err == nil) != want {
			t.Fatalf("accepted=%v want=%v", err == nil, want)
		}
	})
}

func TestSidecarImageEarlyRefusalCleanupCPU(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{{"nil-context", nil}, {"missing-inputs", context.Background()}, {"missing-sidecar", context.Background()}} {
		t.Run(tc.name, func(t *testing.T) {
			wrapped := false
			c := config{}
			if tc.name == "missing-sidecar" {
				c.path = t.TempDir() + "/missing.cu"
				c.sidecar = t.TempDir() + "/missing.json"
				c.inputs = t.TempDir() + "/missing.inputs"
			}
			service, close, err := newSidecarSessionLoader(tc.ctx, c, nil, nil, func(l cudahotswap.Loader) cudahotswap.Loader { wrapped = true; return l })
			if err == nil || service != nil || close != nil || wrapped {
				t.Fatalf("early refusal reached loader or created owner: %v/%v/%v/%v", service, close != nil, err, wrapped)
			}
			cleanup := sidecarImageCleanup(close, nil)
			if cleanup != nil {
				if err := cleanup(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	// An empty recorder alone still has a safe cleanup without a session owner.
	images := newSidecarImageLoader(nil, "source", []byte("metadata"))
	cleanup := sidecarImageCleanup(nil, images)
	if cleanup == nil {
		t.Fatal("recorder cleanup absent")
	}
	if err := cleanup(); err != nil {
		t.Fatal(err)
	}
	calls := 0
	want := errors.New("retained session close error")
	cleanup = sidecarImageCleanup(func() error { calls++; return want }, nil)
	if err := cleanup(); !errors.Is(err, want) || calls != 1 {
		t.Fatalf("cleanup error/calls=%v/%d", err, calls)
	}
}
