//go:build linux

package main

import (
	"errors"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

type foundationCloseFixture struct {
	r             *cudaFoundation
	fail, restore string
	cause         error
	calls         map[string]int
	last          string
	destroyed     bool
}

func newFoundationCloseFixture(t *testing.T) *foundationCloseFixture {
	t.Helper()
	f := &foundationCloseFixture{cause: errors.New("release fault"), calls: make(map[string]int)}
	file, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	direct, err := os.Open("/dev/null")
	if err != nil {
		file.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { file.Close(); direct.Close() })
	f.r = &cudaFoundation{device: new(cuda.Device), workStream: new(cuda.Stream), file: file, direct: direct,
		quantQ80Mod: driver.Module(1), matrixMod: driver.Module(2), model: []byte{1}, mapped: cudaModelRange{host: 42, view: []byte{2}}}
	f.r.closeOps = &cudaFoundationCloseOperations{
		run: func(_ *cudaFoundation, fn func() error) error {
			if err := fn(); err != nil {
				return err
			}
			if f.restore == f.last {
				return f.cause
			}
			return nil
		},
		synchronize: func(*cuda.Stream) error {
			if f.destroyed {
				return errors.New("synchronize destroyed stream")
			}
			return f.call("sync")
		},
		synchronizeContext: func() error { return f.call("context") },
		unload: func(m driver.Module) error {
			if m == 1 {
				return f.call("quant")
			}
			return f.call("matrix")
		},
		freeResident: func(*cudaFoundation) error { return f.call("resident") },
		unregister:   func(uintptr) error { return f.call("unregister") },
		unmap: func(b []byte) error {
			if b[0] == 2 {
				return f.call("mapped")
			}
			return f.call("model")
		},
		destroy: func(*cuda.Stream) error {
			if f.destroyed {
				return nil
			}
			if err := f.call("stream"); err != nil {
				return err
			}
			f.destroyed = true
			if f.restore == "stream" {
				return f.cause
			}
			return nil
		},
		closeDevice: func(*cuda.Device) error { return f.call("device") },
		closeFile: func(p *os.File) error {
			name := "file"
			if p == direct {
				name = "direct"
			}
			err := p.Close()
			if fault := f.call(name); fault != nil {
				return errors.Join(err, fault)
			}
			return err
		},
	}
	return f
}

func (f *foundationCloseFixture) call(name string) error {
	f.calls[name]++
	f.last = name
	if f.fail == name {
		return f.cause
	}
	return nil
}

func TestCUDAFoundationCloseRetry(t *testing.T) {
	order := []string{"sync", "context", "quant", "matrix", "resident", "unregister", "mapped", "stream", "device", "model", "file", "direct"}
	for i, stage := range order {
		t.Run(stage, func(t *testing.T) {
			f := newFoundationCloseFixture(t)
			f.fail = stage
			if err := f.r.Close(); !errors.Is(err, f.cause) {
				t.Fatalf("close=%v", err)
			}
			if !f.r.closing {
				t.Fatal("close did not enter teardown")
			}
			for _, after := range order[i+1:] {
				if f.calls[after] != 0 {
					t.Fatalf("released %s after failed %s", after, stage)
				}
			}
			if i < 8 && f.r.device == nil {
				t.Fatal("device released before its dependents")
			}
			f.fail = ""
			if err := f.r.Close(); err != nil {
				t.Fatal(err)
			}
			if err := f.r.Close(); err != nil {
				t.Fatal(err)
			}
			for _, name := range order {
				want := 1
				if name == stage && name != "file" && name != "direct" {
					want = 2
				}
				if stage == "context" && name == "sync" {
					want = 2
				}
				if f.calls[name] != want {
					t.Fatalf("%s calls=%d, want %d", name, f.calls[name], want)
				}
			}
			if f.r.device != nil || f.r.workStream != nil || f.r.model != nil || f.r.file != nil || f.r.direct != nil || f.r.mapped.view != nil {
				t.Fatal("successful cleanup retained resources")
			}
		})
	}
}

func TestCUDAFoundationCloseRestoreFailure(t *testing.T) {
	for _, stage := range []string{"quant", "matrix", "unregister", "stream"} {
		t.Run(stage, func(t *testing.T) {
			f := newFoundationCloseFixture(t)
			f.restore = stage
			if err := f.r.Close(); !errors.Is(err, f.cause) {
				t.Fatalf("close=%v", err)
			}
			if f.r.device == nil {
				t.Fatal("lost device before restoration error was reported")
			}
			f.restore = ""
			if err := f.r.Close(); err != nil {
				t.Fatal(err)
			}
			if f.calls[stage] != 1 || f.calls["sync"] != 1 {
				t.Fatalf("raw %s repeated: calls=%v", stage, f.calls)
			}
		})
	}
}

func TestCUDAFoundationCloseWithoutStream(t *testing.T) {
	f := newFoundationCloseFixture(t)
	f.r.workStream = nil
	if err := f.r.Close(); err != nil {
		t.Fatal(err)
	}
	if f.calls["context"] != 1 || f.calls["sync"] != 0 || f.calls["stream"] != 0 {
		t.Fatalf("calls=%v", f.calls)
	}
}
