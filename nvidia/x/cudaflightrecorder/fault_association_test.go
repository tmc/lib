package cudaflightrecorder

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestFaultCandidateAssociation(t *testing.T) {
	r, err := New(Config{Capacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a := CandidateAssociation{Version: 1, Kind: CandidateReferenceMismatch, IdentityProvenance: CallerAssertedCandidateIdentity, ExecutableID: "opaque", WorkloadKey: strings.Repeat("a", 64), InputKey: strings.Repeat("b", 64), VerificationKey: strings.Repeat("c", 64), Generation: 3}
	var out bytes.Buffer
	if _, err := r.FaultSnapshot(context.Background(), &out, FaultInfo{Candidate: &a, ErrorName: CandidateReferenceMismatch}); err != nil {
		t.Fatal(err)
	}
	b, err := ReadBundle(&out)
	if err != nil {
		t.Fatal(err)
	}
	f, c, err := b.DecodeFaultEvidence()
	if err != nil {
		t.Fatal(err)
	}
	if f.Candidate == nil || c.Candidate == nil || *f.Candidate != a || *c.Candidate != a || f.ModuleDigest != "" || c.CandidateDigest != "" {
		t.Fatalf("association or provenance lost %+v %+v", f, c)
	}
	if strings.Contains(c.Reason, "withheld") || strings.Contains(c.Reason, "gpu_fault:") || !strings.Contains(c.Reason, "caller-asserted") {
		t.Fatalf("wrong provenance: %s", c.Reason)
	}
	for _, change := range []string{"future", "kind", "input", "extent", "utf8", "generation"} {
		t.Run(change, func(t *testing.T) {
			x := a
			switch change {
			case "future":
				x.Version++
			case "kind":
				x.Kind = "gpu-fault"
			case "input":
				x.InputKey = ""
			case "extent":
				x.ExecutableID = strings.Repeat("x", 4097)
			case "utf8":
				x.ExecutableID = "\xff"
			case "generation":
				x.Generation = -1
			}
			if x.Validate() == nil {
				t.Fatal("invalid association admitted")
			}
			if _, err := json.Marshal(FaultInfo{Candidate: &x, ErrorName: CandidateReferenceMismatch}); err == nil {
				t.Fatal("writer admitted invalid association")
			}
		})
	}
	for _, fault := range []FaultInfo{{Candidate: &a, ErrorName: "CUDA_ERROR_ILLEGAL_ADDRESS"}, {Candidate: &a, ErrorName: CandidateReferenceMismatch, ErrorCode: 700}} {
		var out bytes.Buffer
		if _, err := r.FaultSnapshot(context.Background(), &out, fault); err == nil || out.Len() != 0 {
			t.Fatal("non-comparison fault association exported")
		}
	}
}
