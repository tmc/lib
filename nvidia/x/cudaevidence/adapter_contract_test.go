package cudaevidence_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Supplied ordinary host records exercise adapters, not GPU source authenticity.
func TestOccupancyAndPressureRemainClassified(t *testing.T) {
	source := cudaevidence.Source{Identity: identity(), Producer: "supplied host records", Locator: "original83/adapter"}
	resource := cudaoccupancy.KernelResource{Name: "kernel", Registers: 32}
	occupied, err := resource.OccupancyValue(source, 128, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := cudasass.ParseLifeRangeCount(strings.NewReader("Function : kernel\n/*0000*/ MOV R1, R2; // | 7 | 0 | 0 | 0 |\n/*0010*/ EXIT; // | 0 | 0 | 0 | 0 |\n"))
	if err != nil {
		t.Fatal(err)
	}
	pressure, err := cudasass.PeakGPRValue(rows, source)
	if err != nil {
		t.Fatal(err)
	}
	unknownOccupancy, err := resource.OccupancyValue(source, 128, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	unknownPressure, err := cudasass.PeakGPRValue(nil, source)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name  string
		value cudaevidence.Value
		known bool
	}{
		{"occupancy", occupied, true}, {"pressure", pressure, true}, {"missing occupancy", unknownOccupancy, false}, {"missing pressure", unknownPressure, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.value.Record()
			if tt.known {
				if r.Class != cudaevidence.Derived || r.Number == nil || r.Source != source || r.Basis == "" {
					t.Fatalf("adapter=%+v", r)
				}
				if err := tt.value.CheckClaim(cudaevidence.Derived, source.Identity); err != nil {
					t.Fatal(err)
				}
				if tt.value.CheckClaim(cudaevidence.Measured, source.Identity) == nil {
					t.Fatal("adapter promoted to measured")
				}
				other := source.Identity
				other.ArtifactDigest = strings.Repeat("f", 64)
				if tt.value.CheckClaim(cudaevidence.Derived, other) == nil {
					t.Fatal("adapter accepted foreign artifact")
				}
			} else if r.Class != cudaevidence.Unread || r.Number != nil || r.Reason == "" {
				t.Fatalf("unavailable=%+v", r)
			}
			data, err := json.Marshal(tt.value)
			if err != nil {
				t.Fatal(err)
			}
			round, err := cudaevidence.Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(round.Record(), r) {
				t.Fatalf("JSON lost typed fields: %s", data)
			}
		})
	}
	if !strings.Contains(occupied.Record().Basis, "not runtime occupancy") || !strings.Contains(pressure.Record().Basis, "not allocated registers or spill proof") {
		t.Fatal("adapter scope caveats absent")
	}
	if *pressure.Record().Number != 7 {
		t.Fatal("pressure did not use parsed exact entry")
	}
	item := advise.Item{Location: advise.Location{Kernel: "kernel"}, EvidenceLevel: advise.EvidenceSuspected, ShareOfProfile: 25}
	item, err = item.WithShareEvidence(source, true, "supplied device duration divided by supplied profile total")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(item)
	if err != nil {
		t.Fatal(err)
	}
	var round advise.Item
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	if round.ClassifiedShare == nil || round.ClassifiedShare.Record().Class != cudaevidence.Inferred || round.CheckEvidenceClaim(cudaevidence.Inferred, source.Identity) != nil || round.CheckEvidenceClaim(cudaevidence.Measured, source.Identity) == nil {
		t.Fatalf("profile classification lost: %s", data)
	}
}
