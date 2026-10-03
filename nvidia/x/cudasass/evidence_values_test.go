package cudasass_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func evidenceSource() cudaevidence.Source {
	d := strings.Repeat("1", 64)
	return cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: d, Entry: "kernel", Target: "sm_90", CompilerKey: d, WorkloadKey: d}, Producer: "compiled fixture", Locator: "kernel record"}
}
func TestEvidenceAdapterClassificationAndIdentity(t *testing.T) {
	rows, err := cudasass.ParseLifeRangeCount(strings.NewReader("Function : kernel\n/*0000*/ MOV R1, R2; // | 2 | 0 | 0 | 0 |\n/*0010*/ EXIT; // | 0 | 0 | 0 | 0 |\n"))
	if err != nil {
		t.Fatal(err)
	}
	call := func(s cudaevidence.Source, known bool) (cudaevidence.Value, error) {
		if !known {
			return cudasass.PeakGPRValue(nil, s)
		}
		return cudasass.PeakGPRValue(rows, s)
	}
	source := evidenceSource()
	value, err := call(source, true)
	if err != nil {
		t.Fatal(err)
	}
	if r := value.Record(); r.Number == nil || *r.Number != 2 {
		t.Fatalf("parsed peak = %+v, want 2", r)
	}
	if err := value.CheckClaim(cudaevidence.Derived, source.Identity); err != nil {
		t.Fatal(err)
	}
	if err := value.CheckClaim(cudaevidence.Measured, source.Identity); err == nil {
		t.Fatal("derived promoted to measured")
	}
	other := source.Identity
	other.Target = "sm_80"
	if value.CheckClaim(cudaevidence.Derived, other) == nil {
		t.Fatal("target mismatch accepted")
	}
	other = source.Identity
	other.ArtifactDigest = strings.Repeat("2", 64)
	if value.CheckClaim(cudaevidence.Derived, other) == nil {
		t.Fatal("artifact mismatch accepted")
	}
	absent, err := call(source, false)
	if err != nil {
		t.Fatal(err)
	}
	r := absent.Record()
	if r.Class != cudaevidence.Unread || r.Number != nil || r.Reason == "" {
		t.Fatalf("missing metadata=%+v", r)
	}
	for _, field := range []string{"digest", "producer", "locator"} {
		bad := source
		switch field {
		case "digest":
			bad.Identity.ArtifactDigest = "bad"
		case "producer":
			bad.Producer = ""
		case "locator":
			bad.Locator = ""
		}
		if _, err := call(bad, false); err == nil {
			t.Errorf("invalid %s accepted", field)
		}
	}
}
func TestPressureEvidenceZeroAndAbsentAreDistinct(t *testing.T) {
	source := evidenceSource()
	rows := map[string]map[uint64]cudasass.RegisterPressure{"kernel": {0: {GPR: 0}}}
	v, err := cudasass.PeakGPRValue(rows, source)
	if err != nil {
		t.Fatal(err)
	}
	if r := v.Record(); r.Number == nil || *r.Number != 0 || r.Class != cudaevidence.Derived {
		t.Fatalf("zero=%+v", r)
	}
	source.Identity.Entry = "other"
	v, err = cudasass.PeakGPRValue(rows, source)
	if err != nil {
		t.Fatal(err)
	}
	if v.Record().Number != nil {
		t.Fatal("other entry silently joined")
	}
	source = evidenceSource()
	rows["kernel"][0] = cudasass.RegisterPressure{GPR: -1}
	if _, err = cudasass.PeakGPRValue(rows, source); err == nil {
		t.Fatal("negative pressure accepted")
	}
}
