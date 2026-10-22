package cudakernel

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

func immutableSpec() ArtifactSpec {
	return ArtifactSpec{
		Kind: KindPTX, Raw: []byte("fixture ptx"), EntryPoint: "first", Target: "sm_80",
		Schema:     Schema{Params: []Param{{Name: "scale", Type: "f32", ByteWidth: 4, Alignment: 4}}, TotalParamBytes: 8},
		Resources:  ResourceRequirements{StaticSharedMemory: 4, DynamicSharedMemory: 8, AllocatedRegisters: 12, PeakLiveGPR: 7, RegisterQuantum: 256, ExecutionGroup: "cta"},
		JITOptions: JITOptions{MaxRegisters: 12, ThreadsPerBlock: 64, OptLevelSet: true},
		Toolchain:  "fixture compiler", CompilerOpts: []string{"--std=c++17"},
		Dependencies: []Dependency{{Path: "input.h", Digest: "header digest", Tracked: true, ErrorMsg: ""}},
	}
}

func assertImmutableArtifact(t *testing.T, a Artifact) {
	t.Helper()
	if got, want := a.Spec(), immutableSpec(); !reflect.DeepEqual(got, want) {
		t.Fatalf("artifact spec = %+v, want %+v", got, want)
	}
	if a.Digest() != fmt.Sprintf("%x", sha256.Sum256([]byte("fixture ptx"))) {
		t.Fatal("raw-byte digest changed")
	}
	packed, err := a.Pack(float32(1))
	if err != nil || !bytes.Equal(packed, []byte{0, 0, 128, 63, 0, 0, 0, 0}) {
		t.Fatalf("actual immutable Pack = %x, %v", packed, err)
	}
	image, err := a.prepareLoad()
	if err != nil || string(image.raw) != "fixture ptx\x00" || image.kind != KindPTX || image.target != "sm_80" {
		t.Fatalf("actual load preparation = %+v, %v", image, err)
	}
	wantValues := []uintptr{12, 64, 0}
	if !reflect.DeepEqual(image.values, wantValues) {
		t.Fatalf("load options = %v, want %v", image.values, wantValues)
	}
}

func mutateSpec(s *ArtifactSpec) {
	s.Raw[0] = '!'
	s.Schema.Params[0].Name = "changed"
	s.Schema.Params[0].Type = "u32"
	s.CompilerOpts[0] = "changed option"
	s.Dependencies[0].Tracked = false
	s.Dependencies[0].Digest = "changed digest"
	s.EntryPoint, s.Target, s.Toolchain = "second", "sm_90", "other compiler"
	s.Resources.AllocatedRegisters++
	s.JITOptions.OptLevel = 2
}

func TestArtifactConstructorOwnsInputs(t *testing.T) {
	s := immutableSpec()
	a, err := NewArtifact(s.Kind, s.Raw, s.EntryPoint, s.Schema, s.Target,
		WithResources(s.Resources), WithJIT(s.JITOptions), WithToolchain(s.Toolchain),
		WithCompilerOptions(s.CompilerOpts), WithDependencies(s.Dependencies))
	if err != nil {
		t.Fatal(err)
	}
	mutateSpec(&s)
	assertImmutableArtifact(t, *a)
}

func TestArtifactOptionCannotRetainPublishedData(t *testing.T) {
	var retained *ArtifactSpec
	option := func(s *ArtifactSpec) {
		*s = immutableSpec()
		retained = s
	}
	a, err := NewArtifact(KindPTX, []byte("initial"), "initial", Schema{}, "sm_80", option)
	if err != nil {
		t.Fatal(err)
	}
	mutateSpec(retained)
	assertImmutableArtifact(t, *a)
}

func TestArtifactAccessorsReturnDetachedViews(t *testing.T) {
	a, err := NewArtifactFromSpec(immutableSpec())
	if err != nil {
		t.Fatal(err)
	}
	s := a.Spec()
	mutateSpec(&s)
	a.Bytes()[0] = '!'
	a.Schema().Params[0].Type = "u32"
	a.CompilerOptions()[0] = "changed"
	a.Dependencies()[0].Tracked = false
	image, err := a.prepareLoad()
	if err != nil {
		t.Fatal(err)
	}
	image.raw[0], image.values[0] = '!', 99
	assertImmutableArtifact(t, *a)
}

func TestArtifactDecodeIsTransactional(t *testing.T) {
	a, err := NewArtifactFromSpec(immutableSpec())
	if err != nil {
		t.Fatal(err)
	}
	captured := *a
	bad := artifactWire{ArtifactSpec: immutableSpec(), Digest: "wrong digest"}
	data, err := json.Marshal(bad)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, a); err == nil {
		t.Fatal("bad digest admitted")
	}
	if *a != captured {
		t.Fatal("failed decode replaced handle")
	}
	assertImmutableArtifact(t, *a)
	other := immutableSpec()
	other.EntryPoint = "second"
	b, err := NewArtifactFromSpec(other)
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, a); err != nil {
		t.Fatal(err)
	}
	if a.EntryPoint() != "second" || captured.EntryPoint() != "first" {
		t.Fatal("decode changed captured handle")
	}
	assertImmutableArtifact(t, captured)
}

func TestArtifactJSONPreservesEmptySchema(t *testing.T) {
	s := immutableSpec()
	s.Schema = Schema{Params: []Param{}}
	a, err := NewArtifactFromSpec(s)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var b Artifact
	if err := json.Unmarshal(data, &b); err != nil {
		t.Fatal(err)
	}
	if b.Schema().Params == nil || !reflect.DeepEqual(a.Spec(), b.Spec()) {
		t.Fatal("empty schema wire meaning changed")
	}
}

func TestArtifactValueJSON(t *testing.T) {
	a, err := NewArtifactFromSpec(immutableSpec())
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"pointer", a}, {"value", *a}, {"slice", []Artifact{*a}}, {"map", map[string]Artifact{"a": *a}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatal(err)
			}
			switch tc.name {
			case "slice":
				var parts []json.RawMessage
				if err := json.Unmarshal(data, &parts); err != nil {
					t.Fatal(err)
				}
				data = parts[0]
			case "map":
				var parts map[string]json.RawMessage
				if err := json.Unmarshal(data, &parts); err != nil {
					t.Fatal(err)
				}
				data = parts["a"]
			}
			var got Artifact
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			assertImmutableArtifact(t, got)
		})
	}
}

func TestArtifactDetachedMutationConcurrentWithUse(t *testing.T) {
	s := immutableSpec()
	a, err := NewArtifactFromSpec(s)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 100 {
			mutateSpec(&s)
		}
	}()
	for range 100 {
		assertImmutableArtifact(t, *a)
	}
	wg.Wait()
}

func TestArtifactPackInvalidHandles(t *testing.T) {
	for _, a := range []*Artifact{nil, new(Artifact)} {
		if _, err := a.Pack(); err == nil {
			t.Fatal("invalid handle packed arguments")
		}
	}
}
