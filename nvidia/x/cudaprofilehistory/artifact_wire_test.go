package cudaprofilehistory

import (
	"bytes"
	"encoding/json"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestRecordArtifactSnapshotRoundTrip(t *testing.T) {
	r := fixture(t)
	s, _ := NewStore(limits())
	key, err := s.Append(r)
	if err != nil {
		t.Fatal(err)
	}
	var snapshot bytes.Buffer
	if err := s.WriteSnapshot(&snapshot); err != nil {
		t.Fatal(err)
	}
	restored, err := ReadSnapshot(&snapshot, limits())
	if err != nil {
		t.Fatal(err)
	}
	rows, err := restored.Records()
	if err != nil || len(rows) != 1 || rows[0].Key != key || rows[0].Artifact.Digest() != r.Artifact.Digest() || !bytes.Equal(rows[0].Artifact.Bytes(), r.Artifact.Bytes()) {
		t.Fatalf("restored rows = %+v, %v", rows, err)
	}
}

// TestDecodeRecordStrictArtifact checks that the nested artifact keeps the
// record's strict spelling, duplicate and unknown-field checks.
func TestDecodeRecordStrictArtifact(t *testing.T) {
	b, err := json.Marshal(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeRecord(b, len(b)); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, old, new string
	}{
		{"unknown field", `"entry_point":`, `"extra":1,"entry_point":`},
		{"duplicate field", `"entry_point":"kernel"`, `"entry_point":"kernel","entry_point":"kernel"`},
		{"case alias", `"entry_point":`, `"Entry_Point":`},
		{"nested unknown field", `"resources":{`, `"resources":{"extra":1,`},
		{"missing digest", `"digest":`, `"digest_":`},
		{"nested duplicate field", `"resources":{`, `"resources":{"peak_live_gpr":0,`},
		{"deep nesting", `"schema":{`, `"schema":{"params":` + strings.Repeat("[", 100) + strings.Repeat("]", 100) + `,`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := bytes.Replace(b, []byte(tc.old), []byte(tc.new), 1)
			if bytes.Equal(bad, b) {
				t.Fatalf("record lacks %s", tc.old)
			}
			if _, err := DecodeRecord(bad, len(bad)); err == nil {
				t.Fatal("hostile artifact JSON accepted")
			}
		})
	}
	for _, bad := range []string{string(b) + "{}", string(b) + " x", string(b[:len(b)-1])} {
		if _, err := DecodeRecord([]byte(bad), len(bad)); err == nil {
			t.Fatalf("record with bad end accepted: ...%s", bad[len(bad)-8:])
		}
	}
}

// TestDecodeRecordArtifactDigestMismatch checks that the received artifact
// digest is verified, not recomputed, when it disagrees with the image.
func TestDecodeRecordArtifactDigestMismatch(t *testing.T) {
	r := fixture(t)
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	other := strings.Repeat("ab", 32)
	bad := bytes.Replace(b, []byte(`"digest":"`+r.Artifact.Digest()), []byte(`"digest":"`+other), 1)
	if bytes.Equal(bad, b) {
		t.Fatal("record lacks artifact digest")
	}
	if _, err := DecodeRecord(bad, len(bad)); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("DecodeRecord with wrong artifact digest = %v", err)
	}
}

// legacyArtifact is the exported struct cudakernel.Artifact was before it
// was sealed.
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

// legacyRecord returns r as the Record type was declared before Artifact was
// sealed: the same fields and tags, with the former artifact struct.
func legacyRecord(r Record) reflect.Value {
	rt := reflect.TypeOf(r)
	fields := make([]reflect.StructField, rt.NumField())
	for i := range fields {
		fields[i] = rt.Field(i)
		if fields[i].Name == "Artifact" {
			fields[i].Type = reflect.TypeOf(&legacyArtifact{})
		}
	}
	v := reflect.New(reflect.StructOf(fields)).Elem()
	for i := range fields {
		if fields[i].Name != "Artifact" {
			v.Field(i).Set(reflect.ValueOf(r).Field(i))
		}
	}
	a, s := r.Artifact, r.Artifact.Spec()
	v.FieldByName("Artifact").Set(reflect.ValueOf(&legacyArtifact{s.Kind, a.Digest(), s.Raw, s.EntryPoint, s.Schema, s.Target, s.Resources, s.JITOptions, s.Toolchain, s.CompilerOpts, s.Dependencies}))
	return v
}

// metadataFixtures returns the record fixture with artifacts that exercise
// each optional artifact field.
func metadataFixtures(t *testing.T) map[string]Record {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4})
	if err != nil {
		t.Fatal(err)
	}
	edits := map[string]func(*cudakernel.ArtifactSpec){
		"plain":  func(*cudakernel.ArtifactSpec) {},
		"schema": func(s *cudakernel.ArtifactSpec) { s.Schema = schema },
		"resources": func(s *cudakernel.ArtifactSpec) {
			s.Resources = cudakernel.ResourceRequirements{StaticSharedMemory: 4, AllocatedRegisters: 32, ExecutionGroup: "g"}
		},
		"jit": func(s *cudakernel.ArtifactSpec) {
			s.JITOptions = cudakernel.JITOptions{OptLevelSet: true, MaxRegisters: 64}
		},
		"toolchain": func(s *cudakernel.ArtifactSpec) { s.Toolchain = "nvrtc 13.0" },
		"options":   func(s *cudakernel.ArtifactSpec) { s.CompilerOpts = []string{"--std=c++20"} },
		"dependencies": func(s *cudakernel.ArtifactSpec) {
			s.Dependencies = []cudakernel.Dependency{{Path: "a.h", Digest: "d", Tracked: true}}
		},
	}
	records := map[string]Record{}
	for name, edit := range edits {
		r := fixture(t)
		spec := r.Artifact.Spec()
		edit(&spec)
		a, err := cudakernel.NewArtifactFromSpec(spec)
		if err != nil {
			t.Fatal(err)
		}
		r.Artifact = a
		if err := r.Validate(); err != nil {
			t.Fatal(err)
		}
		records[name] = r
	}
	return records
}

// TestSnapshotAcceptsLegacyArtifactRow checks that a snapshot row written
// with the former artifact struct decodes, keeps its observation key, and
// re-encodes to the same bytes.
func TestSnapshotAcceptsLegacyArtifactRow(t *testing.T) {
	for name, r := range metadataFixtures(t) {
		t.Run(name, func(t *testing.T) {
			old := legacyRecord(r)
			keyless, err := json.Marshal(old.Interface())
			if err != nil {
				t.Fatal(err)
			}
			oldKey := digest(keyless)
			if key, err := ObservationKey(r); err != nil || key != oldKey {
				t.Fatalf("ObservationKey = %s, %v; legacy key %s", key, err, oldKey)
			}
			old.FieldByName("Key").SetString(oldKey)
			row, err := json.Marshal(old.Interface())
			if err != nil {
				t.Fatal(err)
			}
			header, _ := json.Marshal(snapshotHeader{Version: Version, Count: 1, RowsDigest: digest(append(row, '\n'))})
			snapshot := string(header) + "\n" + string(row) + "\n"
			s, err := ReadSnapshot(strings.NewReader(snapshot), limits())
			if err != nil {
				t.Fatalf("ReadSnapshot(legacy row): %v", err)
			}
			rows, err := s.Records()
			if err != nil || len(rows) != 1 || rows[0].Key != oldKey {
				t.Fatalf("Records = %+v, %v; want key %s", rows, err, oldKey)
			}
			var out bytes.Buffer
			if err := s.WriteSnapshot(&out); err != nil || out.String() != snapshot {
				t.Fatalf("WriteSnapshot = %q, %v; want %q", out.String(), err, snapshot)
			}
		})
	}
}

// TestBoundedValueArtifactExact checks that an artifact costs exactly what
// the former artifact struct cost: the image once, plus its metadata.
func TestBoundedValueArtifactExact(t *testing.T) {
	for name, r := range metadataFixtures(t) {
		t.Run(name, func(t *testing.T) {
			old := legacyRecord(r)
			n := 1
			for boundedValue(old, n) != nil {
				n++
			}
			if err := boundedValue(reflect.ValueOf(r), n); err != nil {
				t.Fatalf("boundedValue at legacy cost %d: %v", n, err)
			}
			if err := boundedValue(reflect.ValueOf(r), n-1); err == nil {
				t.Fatalf("boundedValue accepted %d, below legacy cost %d", n-1, n)
			}
		})
	}
}

// TestBoundedValueArtifactPreflight checks that an image over the budget is
// refused before the handle's metadata is copied.
func TestBoundedValueArtifactPreflight(t *testing.T) {
	r := fixture(t)
	spec := r.Artifact.Spec()
	spec.Raw = bytes.Repeat([]byte("// padding\n"), 1<<16)
	spec.Dependencies = make([]cudakernel.Dependency, 1<<17)
	for i := range spec.Dependencies {
		spec.Dependencies[i] = cudakernel.Dependency{Path: "a.h", Tracked: true}
	}
	a, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	r.Artifact = a
	v := reflect.ValueOf(r)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	err = boundedValue(v, a.Size()-1)
	runtime.ReadMemStats(&after)
	if err == nil {
		t.Fatal("boundedValue accepted image over budget")
	}
	if n := after.TotalAlloc - before.TotalAlloc; n > 1<<20 {
		t.Fatalf("boundedValue allocated %d bytes before refusing", n)
	}
}

// TestAppendBoundsArtifactImage checks that the artifact image counts
// toward the record allocation budget before encoding.
func TestAppendBoundsArtifactImage(t *testing.T) {
	r := fixture(t)
	spec := r.Artifact.Spec()
	spec.Raw = bytes.Repeat([]byte("// padding\n"), limits().MaxRecordBytes/10)
	big, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	r.Artifact = big
	r.Identity.ArtifactDigest = big.Digest()
	r.Facts = nil
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	s, _ := NewStore(limits())
	if _, err := s.Append(r); err == nil || !strings.Contains(err.Error(), "allocation budget") {
		t.Fatalf("Append of oversized image = %v", err)
	}
}

// TestRecordArtifactFieldOrder pins the encoded artifact field order that
// observation keys hash.
func TestRecordArtifactFieldOrder(t *testing.T) {
	r := fixture(t)
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	want := `"artifact":{"kind":1,"digest":"` + r.Artifact.Digest() + `","raw":`
	if !bytes.Contains(b, []byte(want)) {
		t.Fatalf("record encoding lacks %s:\n%s", want, b)
	}
}
