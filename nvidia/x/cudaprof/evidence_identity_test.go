package cudaprof_test

import (
	"encoding/json"
	"fmt"
	"strings"
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
	if id.ArtifactDigest != a.Digest() || id.CompilerKey != key || id.WorkloadKey != w.Key() || id.Entry != a.EntryPoint() || id.Target != a.Target() {
		t.Fatalf("lost producer identity: %+v", id)
	}
	if id.CompilerKey == cudacache.CompilerCacheKey(a.Digest(), c) {
		t.Fatal("rehash substituted final image as compiler input")
	}
	unknownKind := a.Spec()
	unknownKind.Kind = 255
	for _, tc := range []struct {
		name   string
		spec   *cudakernel.ArtifactSpec // nil keeps a
		change func(*cudacache.WorkloadIdentity, *string)
	}{
		{"unknown kind", &unknownKind, func(w *cudacache.WorkloadIdentity, k *string) {}},
		{"unknown key", nil, func(w *cudacache.WorkloadIdentity, k *string) { *k = "" }},
		{"missing workload", nil, func(w *cudacache.WorkloadIdentity, k *string) { w.ProgramID = "" }},
		{"missing mode", nil, func(w *cudacache.WorkloadIdentity, k *string) { w.NumericalMode = "" }},
		{"negative extent", nil, func(w *cudacache.WorkloadIdentity, k *string) { w.ShapeDims = []int{-1} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			badA := a
			if tc.spec != nil {
				var err error
				if badA, err = cudakernel.NewArtifactFromSpec(*tc.spec); err != nil {
					t.Fatal(err)
				}
			}
			badW := w
			badKey := key
			tc.change(&badW, &badKey)
			if _, err := cudaprof.BindEvidence(badA, badW, badKey); err == nil {
				t.Fatal("accepted bad identity")
			}
		})
	}
	// A corrupt image or missing entry cannot be constructed; BindEvidence
	// still rejects the invalid zero handle.
	if _, err := cudaprof.BindEvidence(&cudakernel.Artifact{}, w, key); err == nil {
		t.Fatal("accepted invalid artifact")
	}
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	for _, edit := range [][2]string{{`"entry_point":"kernel"`, `"entry_point":""`}, {`"raw":"`, `"raw":"AAAA`}} {
		bad := strings.Replace(string(data), edit[0], edit[1], 1)
		if bad == string(data) {
			t.Fatalf("fixture lacks %s", edit[0])
		}
		if _, err := cudakernel.ParseArtifact([]byte(bad)); err == nil {
			t.Fatalf("parsed artifact with %s", edit[1])
		}
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
