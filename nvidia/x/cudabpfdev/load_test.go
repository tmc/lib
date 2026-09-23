package cudabpfdev

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

type fakeInjectionLoad struct {
	t               *testing.T
	fail            string
	calls           []string
	images          [][]byte
	live            map[driver.Module]bool
	next            driver.Module
	base, candidate compiledResources
	badGlobal       bool
	after           func(string)
}

func (f *fakeInjectionLoad) call(s string) error {
	f.calls = append(f.calls, s)
	if f.after != nil {
		f.after(s)
	}
	if s == f.fail {
		return errRun
	}
	return nil
}

func (f *fakeInjectionLoad) ops() injectionLoadOps {
	f.live = map[driver.Module]bool{}
	return injectionLoadOps{
		run: func(fn func() error) error {
			if err := f.call("enter"); err != nil {
				return err
			}
			return errors.Join(fn(), f.call("restore"))
		},
		load: func(image []byte, _ cudakernel.JITOptions) (driver.Module, error) {
			f.next++
			f.images = append(f.images, bytes.Clone(image))
			if err := f.call(fmt.Sprintf("load%d", f.next)); err != nil {
				return 0, err
			}
			f.live[f.next] = true
			return f.next, nil
		},
		function: func(m driver.Module, name string) (driver.Function, error) {
			if name != "affine" || !f.live[m] {
				f.t.Fatal("wrong function lookup")
			}
			return driver.Function(m), f.call(fmt.Sprintf("function%d", m))
		},
		resources: func(fn driver.Function) (compiledResources, error) {
			r := f.base
			if fn == 2 {
				r = f.candidate
			}
			return r, f.call(fmt.Sprintf("resources%d", fn))
		},
		compile: func(a *cudakernel.Artifact) (*cudakernel.Artifact, *nvjitlinkx.Result, error) {
			if err := f.call("compile"); err != nil {
				return nil, &nvjitlinkx.Result{ErrorLog: "fake linker"}, err
			}
			// Synthetic native boundary: no actual CUBIN or native acceptance.
			return cloneArtifact(a), &nvjitlinkx.Result{}, nil
		},
		global: func(m driver.Module, name string) (driver.DevicePtr, uint, error) {
			if m != 2 || !f.live[m] {
				f.t.Fatal("global lookup on original")
			}
			if err := f.call(name); err != nil {
				return 0, 0, err
			}
			if f.badGlobal {
				return 1, 3, nil
			}
			if name == "__cudabpfdev_map" {
				return 4096, 16, nil
			}
			return 8192, 8, nil
		},
		unload: func(m driver.Module) error {
			if !f.live[m] {
				f.t.Fatal("double unload")
			}
			if err := f.call(fmt.Sprintf("unload%d", m)); err != nil {
				return err
			}
			delete(f.live, m)
			return nil
		},
	}
}

func loadedFixture(t *testing.T) (*Injection, *fakeInjectionLoad) {
	t.Helper()
	s, policy, p, b := injectionFixture(t, injectionPTX)
	i, err := PrepareInjection(s, p, b, policy)
	if err != nil {
		t.Fatal(err)
	}
	if a, reason := i.Candidate(); a == nil {
		t.Fatal(reason)
	}
	return i, &fakeInjectionLoad{t: t, base: compiledResources{8, 0, 0, 1024}, candidate: compiledResources{32, 0, 0, 1024}}
}

func TestInjectionLoadSelection(t *testing.T) {
	i, f := loadedFixture(t)
	l, err := i.load(f.ops())
	if err != nil {
		t.Fatal(err)
	}
	selection, err := l.Selection()
	if err != nil || !selection.Candidate || selection.Reason != "" || l.function != 2 {
		t.Fatalf("selection=%+v fn=%v err=%v", selection, l.function, err)
	}
	if selection.Kernel.Digest != i.candidate.launch.Artifact.Digest {
		t.Fatal("wrong selected image")
	}
	if !bytes.Equal(f.images[0], append(i.original.launch.Artifact.Bytes(), 0)) {
		t.Fatal("original image modified")
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if len(f.live) != 0 {
		t.Fatal("leaked modules")
	}
	n := len(f.calls)
	if err := l.Close(); err != nil || len(f.calls) != n {
		t.Fatal("close is not idempotent")
	}
	got, err := l.Selection()
	if err != nil || got != selection {
		t.Fatal("lost selection receipt")
	}
}

func TestInjectionFailureLoadsOriginalArtifact(t *testing.T) {
	for _, stage := range []string{"resources1", "compile", "load2", "function2", "resources2", "__cudabpfdev_map", "__cudabpfdev_misses"} {
		t.Run(stage, func(t *testing.T) {
			i, f := loadedFixture(t)
			f.fail = stage
			l, err := i.load(f.ops())
			if err != nil {
				t.Fatal(err)
			}
			sel, err := l.Selection()
			if err != nil || sel.Candidate || sel.Reason == "" || l.function != 1 || sel.Kernel.Digest != i.original.launch.Artifact.Digest {
				t.Fatalf("fallback=%+v fn=%v err=%v", sel, l.function, err)
			}
			if !bytes.Equal(f.images[0], append(i.original.launch.Artifact.Bytes(), 0)) {
				t.Fatal("fallback bytes changed")
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			if len(f.live) != 0 {
				t.Fatal("fallback leaked modules")
			}
		})
	}
}

func TestInjectionLoadResourceFallback(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Injection, *fakeInjectionLoad)
	}{
		{"register total", func(_ *Injection, f *fakeInjectionLoad) { f.candidate.registers = 129 }},
		{"register delta", func(i *Injection, f *fakeInjectionLoad) {
			f.candidate.registers = f.base.registers + int32(i.addedRegisters) + 1
		}},
		{"shared", func(_ *Injection, f *fakeInjectionLoad) { f.candidate.sharedBytes = 2048 }},
		{"local", func(_ *Injection, f *fakeInjectionLoad) { f.candidate.localBytes = 4 }},
		{"block", func(i *Injection, _ *fakeInjectionLoad) {
			i.original.launch.Geometry.Block = [3]uint32{1024, 1024, 1024}
		}},
		{"dynamic shared", func(i *Injection, _ *fakeInjectionLoad) { i.original.launch.Geometry.SharedBytes = 2048 }},
		{"invalid baseline", func(_ *Injection, f *fakeInjectionLoad) { f.base.registers = -1 }},
		{"global layout", func(_ *Injection, f *fakeInjectionLoad) { f.badGlobal = true }},
		{"map bound", func(i *Injection, _ *fakeInjectionLoad) { i.slots = 1<<20 + 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			i, f := loadedFixture(t)
			tc.change(i, f)
			l, err := i.load(f.ops())
			if err != nil {
				t.Fatal(err)
			}
			sel, err := l.Selection()
			if err != nil || sel.Candidate || sel.Reason == "" || l.function != 1 {
				t.Fatalf("selection=%+v error=%v", sel, err)
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestInjectionLoadErrorsRetainCleanup(t *testing.T) {
	for _, stage := range []string{"enter", "load1", "function1", "restore"} {
		t.Run(stage, func(t *testing.T) {
			i, f := loadedFixture(t)
			f.fail = stage
			l, err := i.load(f.ops())
			if !errors.Is(err, errRun) {
				t.Fatalf("error=%v", err)
			}
			if _, err := l.Selection(); err == nil {
				t.Fatal("failed load has usable selection")
			}
			f.fail = ""
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			if len(f.live) != 0 {
				t.Fatal("error lost module owner")
			}
		})
	}
	for _, stage := range []string{"unload2", "unload1", "restore"} {
		t.Run("cleanup/"+stage, func(t *testing.T) {
			i, f := loadedFixture(t)
			l, err := i.load(f.ops())
			if err != nil {
				t.Fatal(err)
			}
			f.fail = stage
			if err := l.Close(); !errors.Is(err, errRun) {
				t.Fatalf("error=%v", err)
			}
			f.fail = ""
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			if len(f.live) != 0 {
				t.Fatal("cleanup leaked modules")
			}
		})
	}
}

func TestInjectionDisabledDoesNotCompile(t *testing.T) {
	i, f := loadedFixture(t)
	i.candidate = nil
	i.reason = "injection disabled"
	l, err := i.load(f.ops())
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(f.calls, "compile") || len(f.images) != 1 {
		t.Fatal("disabled path compiled candidate")
	}
	sel, err := l.Selection()
	if err != nil || sel.Candidate || sel.Reason != i.reason {
		t.Fatalf("selection=%+v error=%v", sel, err)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestInjectionCompilerUsesPTX(t *testing.T) {
	i, _ := loadedFixture(t)
	a := i.candidate.launch.Artifact
	result := &nvjitlinkx.Result{ErrorLog: "synthetic failure"}
	artifact, got, err := compileInjection(a, func(in []nvjitlinkx.Input, opts ...string) (*nvjitlinkx.Result, error) {
		if len(in) != 1 || in[0].Kind != nvjitlink.InputPtx || !bytes.Equal(in[0].Data, a.Raw) || !reflect.DeepEqual(opts, []string{"-arch=" + a.Target}) {
			t.Fatalf("bad link input: %+v %v", in, opts)
		}
		return result, errRun
	})
	if artifact != nil || got != result || !errors.Is(err, errRun) {
		t.Fatal("lost failed linker diagnostics or owner")
	}
}

func TestInjectionGlobalsMustBeDisjoint(t *testing.T) {
	for _, start := range []driver.DevicePtr{4096, 4104, ^driver.DevicePtr(0) - 7} {
		i, f := loadedFixture(t)
		ops := f.ops()
		global := ops.global
		ops.global = func(m driver.Module, name string) (driver.DevicePtr, uint, error) {
			if name == "__cudabpfdev_misses" {
				return start, 8, nil
			}
			return global(m, name)
		}
		l, err := i.load(ops)
		if err != nil {
			t.Fatal(err)
		}
		sel, err := l.Selection()
		if err != nil || sel.Candidate || sel.Reason == "" {
			t.Fatalf("overlap/wrap selected: %+v %v", sel, err)
		}
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
