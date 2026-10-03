package cudaprof_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func evidenceArtifact(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_70\n.address_size 64\n.entry kernel() { ret; }"), "kernel", schema, "sm_70")
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestEvidenceIdentityPreservesProducerKeys(t *testing.T) {
	a := evidenceArtifact(t)
	w := cudacache.WorkloadIdentity{ProgramID: "fixture", ShapeDims: []int{3, 5}, NumericalMode: "fp32"}
	c := cudacache.CompilerIdentity{NVRTCVersion: "13.4", DriverSONAME: "libcuda.so.1", TargetArch: "sm_70", OutputKind: nvrtcx.OutputPTX}
	key := cudacache.CompilerCacheKey("original-source-and-dependencies-key", c)
	id, err := cudaprof.BindEvidence(a, w, key)
	if err != nil {
		t.Fatal(err)
	}
	if id.ArtifactDigest != a.Digest || id.CompilerKey != key || id.WorkloadKey != w.Key() || id.Entry != a.EntryPoint || id.Target != a.Target {
		t.Fatalf("lost producer identity: %+v", id)
	}
	if id.CompilerKey == cudacache.CompilerCacheKey(a.Digest, c) {
		t.Fatal("rehash substituted final image as compiler input")
	}
	for _, tc := range []struct {
		name   string
		change func(*cudakernel.Artifact, *cudacache.WorkloadIdentity, *string)
	}{
		{"corrupt image", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { a.Raw = []byte("changed") }},
		{"unknown kind", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { a.Kind = 255 }},
		{"missing entry", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { a.EntryPoint = "" }},
		{"unknown key", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { *k = "" }},
		{"missing workload", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { w.ProgramID = "" }},
		{"missing mode", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { w.NumericalMode = "" }},
		{"negative extent", func(a *cudakernel.Artifact, w *cudacache.WorkloadIdentity, k *string) { w.ShapeDims = []int{-1} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			badA := *a
			badW := w
			badKey := key
			tc.change(&badA, &badW, &badKey)
			if _, err := cudaprof.BindEvidence(&badA, badW, badKey); err == nil {
				t.Fatal("accepted bad identity")
			}
		})
	}
	if _, err := cudaprof.BindEvidence(nil, w, key); err == nil {
		t.Fatal("accepted nil artifact")
	}
}

func ExampleBindEvidence() {
	_, err := cudaprof.BindEvidence(nil, cudacache.WorkloadIdentity{}, "")
	fmt.Println(err)
	// Output: bind evidence: nil artifact
}
