package cudabpfdev

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func sourceFixture(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "u64", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 9.0\n.target sm_121a\n.address_size 64\n.visible .entry kernel(.param .u64 out) { ret; }\n"), "kernel", schema, "sm_121a",
		cudakernel.WithCompilerOptions([]string{"-lineinfo"}),
		cudakernel.WithDependencies([]cudakernel.Dependency{{Path: "kernel.ptx", Digest: "source", Tracked: true}}),
		cudakernel.WithJIT(cudakernel.JITOptions{MaxRegisters: 32}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestArtifactSourceOwnership(t *testing.T) {
	a := sourceFixture(t)
	want := cloneArtifact(a)
	g := Geometry{Grid: [3]uint32{2, 1, 1}, Block: [3]uint32{32, 1, 1}, SharedBytes: 64}
	s, err := NewArtifactSource(a, g)
	if err != nil {
		t.Fatal(err)
	}
	mutate := func(a *cudakernel.Artifact) {
		a.Raw[0] = '!'
		a.Schema.Params[0].Name = "changed"
		a.CompilerOpts[0] = "changed"
		a.Dependencies[0].Path = "changed"
		a.EntryPoint = "changed"
	}
	mutate(a)
	l, err := s.DescribeLaunch()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(l.Artifact, want) || l.Geometry != g {
		t.Fatalf("source changed: %+v", l)
	}
	mutate(l.Artifact)
	l, err = s.DescribeLaunch()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(l.Artifact, want) {
		t.Fatal("description aliases source")
	}
	_, err = s.load(func(f func() error) error { return f() }, func(image []byte, jit cudakernel.JITOptions) (driver.Module, error) {
		if !bytes.Equal(image, append(want.Bytes(), 0)) || jit != want.JITOptions {
			t.Fatal("original image or options changed")
		}
		image[0] = '!'
		return 7, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	l, _ = s.DescribeLaunch()
	if !reflect.DeepEqual(l.Artifact, want) {
		t.Fatal("loader aliases source")
	}
}

func TestArtifactSourceRejects(t *testing.T) {
	g := Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}}
	for _, tc := range []struct {
		name   string
		change func(*cudakernel.Artifact, *Geometry)
	}{
		{"digest", func(a *cudakernel.Artifact, _ *Geometry) { a.Raw[0]++ }},
		{"kind", func(a *cudakernel.Artifact, _ *Geometry) { a.Kind = cudakernel.KindLTOIR }},
		{"entry", func(a *cudakernel.Artifact, _ *Geometry) { a.EntryPoint = "kernel\x00other" }},
		{"grid", func(_ *cudakernel.Artifact, g *Geometry) { g.Grid[1] = 0 }},
		{"block", func(_ *cudakernel.Artifact, g *Geometry) { g.Block[2] = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, geom := sourceFixture(t), g
			tc.change(a, &geom)
			if _, err := NewArtifactSource(a, geom); err == nil {
				t.Fatal("accepted invalid source")
			}
		})
	}
	a := sourceFixture(t)
	a, err := cudakernel.NewArtifact(a.Kind, append([]byte{'x', 0}, a.Raw...), a.EntryPoint, a.Schema, a.Target)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewArtifactSource(a, g); err == nil {
		t.Fatal("accepted truncated PTX")
	}
	if _, err := NewArtifactSource(nil, g); err == nil {
		t.Fatal("accepted nil artifact")
	}
	for _, s := range []*ArtifactSource{nil, {}} {
		if _, err := s.DescribeLaunch(); err == nil {
			t.Fatal("accepted empty source")
		}
		if _, err := s.Load(nil); err == nil {
			t.Fatal("accepted nil device")
		}
	}
}

func TestArtifactSourceLoadErrors(t *testing.T) {
	s, err := NewArtifactSource(sourceFixture(t), Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	failure := errors.New("native failure")
	for _, tc := range []struct {
		name                 string
		enter, load, restore bool
		want                 driver.Module
	}{
		{"enter", true, false, false, 0},
		{"load", false, true, false, 0},
		{"restore", false, false, true, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			mod, err := s.load(func(f func() error) error {
				if tc.enter {
					return failure
				}
				err := f()
				if tc.restore {
					return errors.Join(err, failure)
				}
				return err
			}, func([]byte, cudakernel.JITOptions) (driver.Module, error) {
				called = true
				if tc.load {
					return 0, failure
				}
				return 7, nil
			})
			if !errors.Is(err, failure) || mod != tc.want || called == tc.enter {
				t.Fatalf("module=%v called=%v error=%v", mod, called, err)
			}
		})
	}
}
