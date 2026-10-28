package cudakernel_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func parseFixture(t *testing.T) (*cudakernel.Artifact, string) {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "scale", Type: "f32", ByteWidth: 4, Alignment: 4})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifactFromSpec(cudakernel.ArtifactSpec{
		Kind: cudakernel.KindPTX, Raw: []byte("fixture ptx"), EntryPoint: "first", Target: "sm_80", Schema: schema,
		CompilerOpts: []string{"--std=c++17"},
		Dependencies: []cudakernel.Dependency{{Path: "input.h", Digest: "header digest", Tracked: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	return a, string(data)
}

func TestParseArtifact(t *testing.T) {
	a, data := parseFixture(t)
	got, err := cudakernel.ParseArtifact([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if got.Digest() != a.Digest() || got.EntryPoint() != a.EntryPoint() || got.Target() != a.Target() || !bytes.Equal(got.Bytes(), a.Bytes()) {
		t.Fatalf("ParseArtifact = %+v, want %+v", got.Spec(), a.Spec())
	}
	if *got == *a {
		t.Fatal("parsed artifact shares the encoder's identity")
	}
	again, err := json.Marshal(got)
	if err != nil || string(again) != data {
		t.Fatalf("re-encoding = %s, %v; want %s", again, err, data)
	}
}

func TestParseArtifactRejects(t *testing.T) {
	a, data := parseFixture(t)
	edit := func(old, new string) string {
		t.Helper()
		s := strings.Replace(data, old, new, 1)
		if s == data {
			t.Fatalf("fixture lacks %s", old)
		}
		return s
	}
	for _, tc := range []struct {
		name string
		data string
	}{
		{"empty", ""},
		{"null", "null"},
		{"truncated", data[:len(data)/2]},
		{"unknown field", edit(`"kind":`, `"extra":1,"kind":`)},
		{"nested unknown field", edit(`"params":[{`, `"params":[{"extra":1,`)},
		{"trailing object", data + ` {}`},
		{"trailing artifact", data + data},
		{"trailing garbage", data + ` x`},
		{"raw under old digest", edit(`"raw":"`, `"raw":"AAAA`)},
		{"wrong digest", edit(`"digest":"`+a.Digest(), `"digest":"`+strings.Repeat("0", 64))},
		{"missing digest", edit(`,"digest":"`+a.Digest()+`"`, ``)},
		{"empty digest", edit(`"digest":"`+a.Digest(), `"digest":"`)},
		{"empty entry", edit(`"entry_point":"first"`, `"entry_point":""`)},
		{"invalid target", edit(`"target":"sm_80"`, `"target":"bogus"`)},
		{"deep nesting", edit(`"compiler_opts":[`, `"compiler_opts":[`+strings.Repeat("[", 100)+strings.Repeat("]", 100)+`,`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cudakernel.ParseArtifact([]byte(tc.data))
			if err == nil || got != nil {
				t.Fatalf("ParseArtifact = %v, %v; want error", got, err)
			}
		})
	}
}

// TestParseArtifactDecodeContract pins the encoding/json behavior ParseArtifact
// keeps: a repeated key takes its last value and keys match case-insensitively.
// Consumers that must reject these check keys before parsing.
func TestParseArtifactDecodeContract(t *testing.T) {
	_, data := parseFixture(t)
	for _, tc := range []struct {
		name, old, new string
	}{
		{"duplicate key", `"target":"sm_80"`, `"target":"sm_70","target":"sm_80"`},
		{"case alias", `"target":`, `"Target":`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := strings.Replace(data, tc.old, tc.new, 1)
			got, err := cudakernel.ParseArtifact([]byte(s))
			if err != nil || got.Target() != "sm_80" {
				t.Fatalf("ParseArtifact = %v, %v; want target sm_80", got, err)
			}
		})
	}
}

// TestParseArtifactStricterThanUnmarshal checks that UnmarshalJSON keeps
// its permissive unknown-field contract while sharing digest validation.
func TestParseArtifactStricterThanUnmarshal(t *testing.T) {
	a, data := parseFixture(t)
	extra := strings.Replace(data, `"kind":`, `"extra":1,"kind":`, 1)
	var u cudakernel.Artifact
	if err := json.Unmarshal([]byte(extra), &u); err != nil || u.Digest() != a.Digest() {
		t.Fatalf("UnmarshalJSON with unknown field = %v", err)
	}
	if _, err := cudakernel.ParseArtifact([]byte(extra)); err == nil {
		t.Fatal("ParseArtifact accepted unknown field")
	}
	corrupt := strings.Replace(data, `"raw":"`, `"raw":"AAAA`, 1)
	before := u
	if err := json.Unmarshal([]byte(corrupt), &u); err == nil || u != before {
		t.Fatalf("UnmarshalJSON of corrupt raw = %v; receiver changed %t", err, u != before)
	}
}

func TestParseArtifactDetachesInput(t *testing.T) {
	a, data := parseFixture(t)
	b := []byte(data)
	got, err := cudakernel.ParseArtifact(b)
	if err != nil {
		t.Fatal(err)
	}
	for i := range b {
		b[i] = 'x'
	}
	if err := got.Validate(); err != nil || !bytes.Equal(got.Bytes(), a.Bytes()) {
		t.Fatalf("input reuse changed artifact: %v", err)
	}
}

func TestArtifactSize(t *testing.T) {
	a, _ := parseFixture(t)
	var zero cudakernel.Artifact
	var nilArtifact *cudakernel.Artifact
	for _, tc := range []struct {
		name string
		a    *cudakernel.Artifact
		want int
	}{
		{"nil", nilArtifact, 0},
		{"zero", &zero, 0},
		{"artifact", a, len("fixture ptx")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.a.Size(); got != tc.want || got != len(tc.a.Bytes()) {
				t.Fatalf("Size = %d, want %d and len(Bytes) %d", got, tc.want, len(tc.a.Bytes()))
			}
		})
	}
	b := a.Bytes()
	b = append(b, "more"...)
	b[0] = 'X'
	if a.Size() != len("fixture ptx") || a.Bytes()[0] != 'f' {
		t.Fatal("Bytes result aliases the artifact")
	}
}

func BenchmarkArtifactSize(b *testing.B) {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, make([]byte, 64<<20), "k", cudakernel.Schema{}, "sm_80")
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if a.Size() != 64<<20 {
			b.Fatal("wrong size")
		}
	}
}

// TestArtifactMarshalFieldOrder compares MarshalJSON with the encoding of
// the struct Artifact was before it was sealed, and checks that ParseArtifact
// reads those bytes back unchanged. Hashes of encoded artifacts, such as
// history observation keys, depend on these exact bytes.
func TestArtifactMarshalFieldOrder(t *testing.T) {
	type legacyArtifact struct {
		Kind         cudakernel.Kind                 `json:"kind"`
		Digest       string                          `json:"digest"`
		Raw          []byte                          `json:"raw"`
		EntryPoint   string                          `json:"entry_point"`
		Schema       cudakernel.Schema               `json:"schema"`
		Target       string                          `json:"target"`
		Resources    cudakernel.ResourceRequirements `json:"resources"`
		JITOptions   cudakernel.JITOptions           `json:"jit_options,omitempty"`
		Toolchain    string                          `json:"toolchain,omitempty"`
		CompilerOpts []string                        `json:"compiler_opts,omitempty"`
		Dependencies []cudakernel.Dependency         `json:"dependencies,omitempty"`
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "x", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4},
	)
	if err != nil {
		t.Fatal(err)
	}
	base := func() cudakernel.ArtifactSpec {
		return cudakernel.ArtifactSpec{Kind: cudakernel.KindPTX, Raw: []byte("variant ptx"), EntryPoint: "k", Target: "sm_90"}
	}
	tests := []struct {
		name string
		edit func(*cudakernel.ArtifactSpec)
	}{
		{"minimal", func(*cudakernel.ArtifactSpec) {}},
		{"ltoir", func(s *cudakernel.ArtifactSpec) { s.Kind = cudakernel.KindLTOIR }},
		{"schema", func(s *cudakernel.ArtifactSpec) { s.Schema = schema }},
		{"resources", func(s *cudakernel.ArtifactSpec) {
			s.Resources = cudakernel.ResourceRequirements{StaticSharedMemory: 1, DynamicSharedMemory: 2, AllocatedRegisters: 3, PeakLiveGPR: 4, RegisterQuantum: 8, ExecutionGroup: "g"}
		}},
		{"jit", func(s *cudakernel.ArtifactSpec) {
			s.JITOptions = cudakernel.JITOptions{MaxRegisters: 64, ThreadsPerBlock: 128, OptLevel: 3}
		}},
		{"jit explicit zero", func(s *cudakernel.ArtifactSpec) { s.JITOptions = cudakernel.JITOptions{OptLevelSet: true} }},
		{"toolchain", func(s *cudakernel.ArtifactSpec) { s.Toolchain = "nvrtc 12.8" }},
		{"compiler opts", func(s *cudakernel.ArtifactSpec) { s.CompilerOpts = []string{"-O3", "--std=c++17"} }},
		{"dependencies", func(s *cudakernel.ArtifactSpec) {
			s.Dependencies = []cudakernel.Dependency{{Path: "a.h", Digest: "d", Tracked: true}, {Path: "b.h", ErrorMsg: "missing"}}
		}},
		{"all", func(s *cudakernel.ArtifactSpec) {
			s.Schema = schema
			s.Resources = cudakernel.ResourceRequirements{StaticSharedMemory: 16, AllocatedRegisters: 32}
			s.JITOptions = cudakernel.JITOptions{OptLevel: 1}
			s.Toolchain = "nvcc"
			s.CompilerOpts = []string{"-lineinfo"}
			s.Dependencies = []cudakernel.Dependency{{Path: "c.h", Tracked: true}}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := base()
			tt.edit(&spec)
			a, err := cudakernel.NewArtifactFromSpec(spec)
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(a)
			if err != nil {
				t.Fatal(err)
			}
			s := a.Spec()
			want, err := json.Marshal(legacyArtifact{s.Kind, a.Digest(), s.Raw, s.EntryPoint, s.Schema, s.Target, s.Resources, s.JITOptions, s.Toolchain, s.CompilerOpts, s.Dependencies})
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != string(want) {
				t.Fatalf("MarshalJSON =\n%s\nwant\n%s", data, want)
			}
			got, err := cudakernel.ParseArtifact(want)
			if err != nil {
				t.Fatalf("ParseArtifact(legacy bytes): %v", err)
			}
			again, err := json.Marshal(got)
			if err != nil || string(again) != string(want) {
				t.Fatalf("legacy round trip = %s, %v; want %s", again, err, want)
			}
		})
	}
}

func TestArtifactRejectsNULEntry(t *testing.T) {
	_, data := parseFixture(t)
	data = strings.Replace(data, `"entry_point":"first"`, `"entry_point":"k\u0000x"`, 1)
	if got, err := cudakernel.ParseArtifact([]byte(data)); err == nil || got != nil {
		t.Fatalf("ParseArtifact = %v, %v; want rejection", got, err)
	}
	a, _ := parseFixture(t)
	before := *a
	if err := json.Unmarshal([]byte(data), a); err == nil || *a != before {
		t.Fatalf("Unmarshal = %v, changed %v; want transactional rejection", err, *a != before)
	}
	spec := a.Spec()
	spec.EntryPoint = "k\x00x"
	if got, err := cudakernel.NewArtifactFromSpec(spec); err == nil || got != nil {
		t.Fatalf("NewArtifactFromSpec = %v, %v; want rejection", got, err)
	}
}
