package cudablock

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func retainedTopKFixture(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	schema, err := TopKSchema()
	if err != nil {
		t.Fatal(err)
	}
	raw := []byte("host admission fixture; not executable PTX")
	a, err := cudakernel.NewArtifactFromSpec(cudakernel.ArtifactSpec{Kind: cudakernel.KindPTX, Raw: raw, EntryPoint: "topk_fp32", Schema: schema, Target: "sm_121", CompilerOpts: []string{"--std=c++17"}, Dependencies: []cudakernel.Dependency{{Path: "cudablock_topk.cu", Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(TopKSource))), Tracked: true}}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestLoadTopKRetainedAdmission(t *testing.T) {
	for _, tc := range []struct {
		name  string
		alter func(*cudakernel.ArtifactSpec)
		valid bool
	}{
		{"valid", func(a *cudakernel.ArtifactSpec) {}, true},
		{"bf16-entry", func(a *cudakernel.ArtifactSpec) { a.EntryPoint = "topk_bf16" }, false},
		{"changed-schema", func(a *cudakernel.ArtifactSpec) { a.Schema.Params[0].Name = "other" }, false},
		{"wrong-kind", func(a *cudakernel.ArtifactSpec) { a.Kind = cudakernel.KindLTOIR }, false},
		{"missing-source", func(a *cudakernel.ArtifactSpec) { a.Dependencies = nil }, false},
		{"untracked-source", func(a *cudakernel.ArtifactSpec) { a.Dependencies[0].Tracked = false }, false},
		{"wrong-source", func(a *cudakernel.ArtifactSpec) { a.Dependencies[0].Digest = fmt.Sprintf("%064x", 1) }, false},
		{"failed-source", func(a *cudakernel.ArtifactSpec) { a.Dependencies[0].ErrorMsg = "unread" }, false},
		{"duplicate-source", func(a *cudakernel.ArtifactSpec) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }, false},
		{"invalid-target", func(a *cudakernel.ArtifactSpec) { a.Target = "unknown" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := retainedTopKFixture(t)
			spec := a.Spec()
			tc.alter(&spec)
			a, err := cudakernel.NewArtifactFromSpec(spec)
			if err != nil {
				if tc.name != "invalid-target" {
					t.Fatalf("metadata control construction: %v", err)
				}
				return
			}
			if tc.name == "invalid-target" {
				t.Fatal("invalid target accepted by constructor")
			}
			if _, err := topkRetainedArtifact(a); (err == nil) != tc.valid {
				t.Fatalf("valid=%t want%t: %v", err == nil, tc.valid, err)
			}
			// Public admission must reject defects before any device operation.
			if !tc.valid {
				if owner, err := LoadTopK(nil, TopKConfig{Rows: 1, Width: 1, RowStride: 1, K: 1}, a); err == nil || owner != nil {
					t.Fatalf("public admission: %v %v", owner, err)
				}
			}
		})
	}
	if _, err := topkRetainedArtifact(nil); err == nil {
		t.Fatal("nil artifact accepted")
	}
}

func TestLoadTopKCorruptImage(t *testing.T) {
	a := retainedTopKFixture(t)
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	raw := a.Bytes()
	raw[0] ^= 1
	wire["raw"], err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, a); err == nil {
		t.Fatal("corrupt image digest accepted")
	}
	if err := a.Validate(); err != nil {
		t.Fatalf("failed decode changed prior artifact: %v", err)
	}
}

func TestLoadTopKDetachesDescriptor(t *testing.T) {
	a := retainedTopKFixture(t)
	owned, err := topkRetainedArtifact(a)
	if err != nil {
		t.Fatal(err)
	}
	spec := a.Spec()
	spec.Raw[0] ^= 1
	spec.Schema.Params[0].Name = "changed"
	spec.CompilerOpts[0] = "changed"
	spec.Dependencies[0].Digest = "changed"
	replacement, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	*a = *replacement
	if a.Schema().Params[0].Name != "changed" {
		t.Fatal("source handle was not replaced")
	}

	if err := owned.Validate(); err != nil {
		t.Fatal(err)
	}
	if owned.Schema().Params[0].Name == "changed" || owned.CompilerOptions()[0] == "changed" || owned.Dependencies()[0].Digest == "changed" {
		t.Fatal("descriptor aliases caller storage")
	}
}

func ExampleLoadTopK() {
	_, err := LoadTopK(nil, TopKConfig{Rows: 1, Width: 1, RowStride: 1, K: 1}, nil)
	fmt.Println(err != nil)
	// Output: true
}
