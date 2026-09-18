package cudapipeline

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestEveryProbedTargetIsReachable is a regression test for a hand-written
// map shadowing the evidence. knownProfiles listed five targets while the
// feature table recorded eleven, so sm_103a and sm_110a were measured,
// checked in, and then unreachable here: ArchProfileFor called them unknown
// architectures. Nothing failed, because nothing compared the two.
func TestEveryProbedTargetIsReachable(t *testing.T) {
	targets := cudadevicetable.Targets()
	if len(targets) == 0 {
		t.Fatal("no probed targets; this test would pass vacuously")
	}
	for _, name := range targets {
		p, err := ProbedArchProfileFor(name)
		if err != nil {
			t.Errorf("probed target %s is not reachable: %v", name, err)
			continue
		}
		if p.FeatureSource == "" {
			t.Errorf("%s is reachable but carries no feature source", name)
		}
		if p.Estimated {
			t.Errorf("%s is probed but marked Estimated", name)
		}
	}
}

// TestUnmeasuredIsNotUnsupported checks that silence reports as silence.
// Saying "does not support" for a target nobody probed is how a table's gap
// becomes a hardware claim, and it is the same error the wgmma literals made.
func TestUnmeasuredIsNotUnsupported(t *testing.T) {
	unmeasured, err := ArchProfileFor("sm_130a")
	if err != nil {
		t.Fatalf("ArchProfileFor(sm_130a): %v", err)
	}
	if !unmeasured.Estimated {
		t.Fatal("a target outside both tables is not marked Estimated")
	}
	err = unmeasured.ValidateFeature(FeatureWGMMA)
	if err == nil {
		t.Fatal("ValidateFeature succeeded for an unmeasured target")
	}
	if !strings.Contains(err.Error(), "cannot say") {
		t.Errorf("unmeasured target reported as a rejection: %v", err)
	}
	if !strings.Contains(err.Error(), "cmd/cudadevicetable") {
		t.Errorf("error does not name the way out: %v", err)
	}

	// The contrast that gives the assertion meaning: a measured rejection
	// must still read as a rejection, and must quote the assembler.
	measured, err := ProbedArchProfileFor(ArchSM121a)
	if err != nil {
		t.Fatalf("sm_121a: %v", err)
	}
	err = measured.ValidateFeature(FeatureWGMMA)
	if err == nil {
		t.Fatal("sm_121a accepted wgmma")
	}
	if strings.Contains(err.Error(), "cannot say") {
		t.Errorf("measured rejection reported as unmeasured: %v", err)
	}
	if !strings.Contains(err.Error(), "wgmma.fence") {
		t.Errorf("rejection does not quote the assembler: %v", err)
	}
	if _, err := ProbedArchProfileFor("sm_130a"); err == nil {
		t.Error("ProbedArchProfileFor accepted an unmeasured target")
	}
}

// TestUnknownArchitectureIsStillRejected keeps the new fallback from
// accepting typos. "sm_130a" is a plausible future part; "hopper" is a
// mistake, and synthesizing a profile for a mistake helps nobody.
func TestUnknownArchitectureIsStillRejected(t *testing.T) {
	for _, bad := range []string{"hopper", "compute_90", "sm_", "", "90a"} {
		if _, err := ArchProfileFor(bad); err == nil {
			t.Errorf("ArchProfileFor(%q) succeeded", bad)
		}
	}
}

// TestSeededProfilesAgreeWithTheTable checks the derivation itself rather
// than its output: every profile's flags must equal what the table says, so a
// broken init cannot leave a plausible set of falses behind.
func TestSeededProfilesAgreeWithTheTable(t *testing.T) {
	checked := 0
	for name, p := range knownProfiles {
		for feature, want := range map[string]bool{
			"wgmma":   p.SupportsWGMMA,
			"tcgen05": p.SupportsTCGen05,
			"tma":     p.SupportsTMA,
		} {
			a := cudadevicetable.Ask(name, feature)
			if !a.Measured {
				continue
			}
			checked++
			if a.Supported != want {
				t.Errorf("%s %s: profile says %v, table says %v", name, feature, want, a.Supported)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no profile field was compared against the table")
	}
}
