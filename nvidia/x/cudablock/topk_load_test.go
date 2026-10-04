package cudablock

import (
	"crypto/sha256"
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
	return &cudakernel.Artifact{Kind: cudakernel.KindPTX, Raw: raw, Digest: fmt.Sprintf("%x", sha256.Sum256(raw)), EntryPoint: "topk_fp32", Schema: schema, Target: "sm_121", CompilerOpts: []string{"--std=c++17"}, Dependencies: []cudakernel.Dependency{{Path: "cudablock_topk.cu", Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(TopKSource))), Tracked: true}}}
}

func TestLoadTopKRetainedAdmission(t *testing.T) {
	for _, tc := range []struct {
		name  string
		alter func(*cudakernel.Artifact)
		valid bool
	}{
		{"valid", func(a *cudakernel.Artifact) {}, true},
		{"corrupt-image", func(a *cudakernel.Artifact) { a.Raw[0] ^= 1 }, false},
		{"bf16-entry", func(a *cudakernel.Artifact) { a.EntryPoint = "topk_bf16" }, false},
		{"changed-schema", func(a *cudakernel.Artifact) { a.Schema.Params[0].Name = "other" }, false},
		{"wrong-kind", func(a *cudakernel.Artifact) { a.Kind = cudakernel.KindLTOIR }, false},
		{"missing-source", func(a *cudakernel.Artifact) { a.Dependencies = nil }, false},
		{"untracked-source", func(a *cudakernel.Artifact) { a.Dependencies[0].Tracked = false }, false},
		{"wrong-source", func(a *cudakernel.Artifact) { a.Dependencies[0].Digest = fmt.Sprintf("%064x", 1) }, false},
		{"failed-source", func(a *cudakernel.Artifact) { a.Dependencies[0].ErrorMsg = "unread" }, false},
		{"duplicate-source", func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }, false},
		{"invalid-target", func(a *cudakernel.Artifact) { a.Target = "unknown" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := retainedTopKFixture(t)
			tc.alter(a)
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

func TestLoadTopKDetachesDescriptor(t *testing.T) {
	a := retainedTopKFixture(t)
	owned, err := topkRetainedArtifact(a)
	if err != nil {
		t.Fatal(err)
	}
	a.Raw[0] ^= 1
	a.Schema.Params[0].Name = "changed"
	a.CompilerOpts[0] = "changed"
	a.Dependencies[0].Digest = "changed"
	if err := owned.Validate(); err != nil {
		t.Fatal(err)
	}
	if owned.Schema.Params[0].Name == "changed" || owned.CompilerOpts[0] == "changed" || owned.Dependencies[0].Digest == "changed" {
		t.Fatal("descriptor aliases caller storage")
	}
}

func ExampleLoadTopK() {
	_, err := LoadTopK(nil, TopKConfig{Rows: 1, Width: 1, RowStride: 1, K: 1}, nil)
	fmt.Println(err != nil)
	// Output: true
}
