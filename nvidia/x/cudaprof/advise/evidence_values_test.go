package advise

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func evidenceSource() cudaevidence.Source {
	return cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}, Producer: "profile", Locator: "profile/1"}
}

func TestProfileBottleneckRemainsClassified(t *testing.T) {
	source := evidenceSource()
	for _, tc := range []struct {
		name  string
		level EvidenceLevel
		known bool
		class cudaevidence.Class
	}{
		{"observed share", EvidenceObserved, true, cudaevidence.Derived},
		{"suspected cause", EvidenceSuspected, true, cudaevidence.Inferred},
		{"unmeasured", EvidenceUnmeasured, true, cudaevidence.Unread},
		{"missing", EvidenceObserved, false, cudaevidence.Unread},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := Item{Location: Location{Kernel: "kernel"}, EvidenceLevel: tc.level, ShareOfProfile: 0}
			got, err := original.WithShareEvidence(source, tc.known, "retained device-time numerator divided by profile denominator")
			if err != nil {
				t.Fatal(err)
			}
			if original.ClassifiedShare != nil {
				t.Fatal("adapter changed original")
			}
			record := got.ClassifiedShare.Record()
			if record.Class != tc.class {
				t.Fatalf("class=%q want %q", record.Class, tc.class)
			}
			if tc.class == cudaevidence.Unread && record.Number != nil {
				t.Fatal("missing share became zero")
			}
			if tc.class != cudaevidence.Unread && (record.Number == nil || *record.Number != 0) {
				t.Fatal("lost observed zero")
			}
			if got.ClassifiedShare.CheckClaim(cudaevidence.Measured, source.Identity) == nil {
				t.Fatal("promoted bottleneck to measured")
			}
			data, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var round Item
			if err := json.Unmarshal(data, &round); err != nil {
				t.Fatal(err)
			}
			if round.ClassifiedShare == nil || round.ClassifiedShare.Record().Class != tc.class {
				t.Fatal("lost JSON class")
			}
		})
	}
}

func TestProfileEvidenceRefusals(t *testing.T) {
	item := Item{Location: Location{Kernel: "kernel"}, EvidenceLevel: EvidenceObserved, ShareOfProfile: 50}
	if _, err := item.WithShareEvidence(evidenceSource(), true, ""); err == nil {
		t.Fatal("accepted missing denominator basis")
	}
	for _, tc := range []struct {
		name   string
		change func(*Item, *cudaevidence.Source)
	}{
		{"wrong entry", func(i *Item, s *cudaevidence.Source) { i.Location.Kernel = "other" }},
		{"missing artifact", func(i *Item, s *cudaevidence.Source) { s.Identity.ArtifactDigest = "" }},
		{"missing producer", func(i *Item, s *cudaevidence.Source) { s.Producer = "" }},
		{"missing locator", func(i *Item, s *cudaevidence.Source) { s.Locator = "" }},
		{"nonfinite", func(i *Item, s *cudaevidence.Source) { i.ShareOfProfile = math.NaN() }},
		{"bad percent", func(i *Item, s *cudaevidence.Source) { i.ShareOfProfile = 101 }},
		{"unknown level", func(i *Item, s *cudaevidence.Source) { i.EvidenceLevel = "other" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			i := Item{Location: Location{Kernel: "kernel"}, EvidenceLevel: EvidenceObserved, ShareOfProfile: 50}
			s := evidenceSource()
			tc.change(&i, &s)
			if _, err := i.WithShareEvidence(s, true, "retained device-time ratio"); err == nil {
				t.Fatal("accepted invalid evidence")
			}
		})
	}
}
