package cudaoccupancy_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

func evidenceSource() cudaevidence.Source {
	d := strings.Repeat("1", 64)
	return cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: d, Entry: "kernel", Target: "sm_90", CompilerKey: d, WorkloadKey: d}, Producer: "compiled fixture", Locator: "kernel record"}
}
func TestEvidenceAdapterClassificationAndIdentity(t *testing.T) {
	k := cudaoccupancy.KernelResource{Name: "kernel", Registers: 32}
	call := func(s cudaevidence.Source, known bool) (cudaevidence.Value, error) {
		return k.OccupancyValue(s, 128, 0, known)
	}
	source := evidenceSource()
	value, err := call(source, true)
	if err != nil {
		t.Fatal(err)
	}
	report, err := k.Calculate(source.Identity.Target, 128, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r := value.Record(); r.Number == nil || *r.Number != report.Occupancy || *r.Number <= 0 {
		t.Fatalf("occupancy = %+v, want %g", r, report.Occupancy)
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
func TestOccupancyEvidenceRejectsWrongEntryAndNegativeResources(t *testing.T) {
	source := evidenceSource()
	for _, k := range []cudaoccupancy.KernelResource{{Name: "other", Registers: 32}, {Name: "kernel", Registers: -1}, {Name: "kernel", StaticSharedMem: -1}} {
		if _, err := k.OccupancyValue(source, 128, 0, true); err == nil {
			t.Errorf("invalid resources accepted: %+v", k)
		}
	}
	k := cudaoccupancy.KernelResource{Name: "kernel", Registers: 32}
	source.Identity.Target = "sm_invalid"
	if _, err := k.OccupancyValue(source, 128, 0, true); err == nil {
		t.Fatal("invalid target accepted")
	}
}

func TestOccupancyEvidenceRejectsOverflowAndExcessiveThreads(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		registers, shared, dynamic, threads int
	}{
		{"register multiplication", math.MaxInt, 0, 0, 128},
		{"register rounding", math.MaxInt / 32, 0, 0, 128},
		{"block register multiplication", math.MaxInt / 64, 0, 0, 128},
		{"shared addition", 32, math.MaxInt, 1, 128},
		{"shared rounding", 32, math.MaxInt, 0, 128},
		{"dynamic rounding", 32, 0, math.MaxInt, 128},
		{"excessive threads", 32, 0, 0, 1025},
		{"maximum threads", 32, 0, 0, math.MaxInt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := cudaoccupancy.KernelResource{Name: "kernel", Registers: tc.registers, StaticSharedMem: tc.shared}
			for _, known := range []bool{false, true} {
				if _, err := k.OccupancyValue(evidenceSource(), tc.threads, tc.dynamic, known); err == nil {
					t.Fatalf("unsafe arithmetic accepted known=%t", known)
				}
			}
		})
	}
	k := cudaoccupancy.KernelResource{Name: "kernel"}
	value, err := k.OccupancyValue(evidenceSource(), 128, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if r := value.Record(); r.Number == nil || r.Class != cudaevidence.Derived {
		t.Fatalf("known zero resources = %+v", r)
	}
}
