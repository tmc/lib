package cudakernel

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestUnknownTargetGeometryIsMarked covers the defect this file was written
// for: TargetSM answered a target nobody has measured with a full set of
// plausible numbers -- 48 KB of shared memory, 16 blocks per SM -- and no way
// for a caller to tell them from recorded ones.
func TestUnknownTargetGeometryIsMarked(t *testing.T) {
	info, err := TargetSM("sm_130a")
	if err != nil {
		t.Fatalf("TargetSM(sm_130a): %v", err)
	}
	if !info.Estimated {
		t.Fatal("geometry for an unrecorded target is not marked Estimated")
	}
	if info.Major != 13 || info.Suffix != "a" {
		t.Errorf("identity fields wrong: major %d suffix %q", info.Major, info.Suffix)
	}
	if _, err := ProbedTargetSM("sm_130a"); err == nil {
		t.Error("ProbedTargetSM accepted an unrecorded target")
	} else if !strings.Contains(err.Error(), "cmd/cudadevicetable") {
		t.Errorf("error does not name the way out: %v", err)
	}
}

// TestRecordedTargetsAreNotEstimated is the other half. If Estimated were set
// everywhere, ProbedTargetSM would refuse everything and the marker would
// carry no information -- a gate that rejects all input is not a gate.
func TestRecordedTargetsAreNotEstimated(t *testing.T) {
	if len(knownTargets) == 0 {
		t.Fatal("no recorded targets")
	}
	for name := range knownTargets {
		info, err := ProbedTargetSM(name)
		if err != nil {
			t.Errorf("ProbedTargetSM(%s): %v", name, err)
			continue
		}
		if info.Estimated {
			t.Errorf("%s is recorded but marked Estimated", name)
		}
		if info.MaxSharedMemoryBytes == 0 {
			t.Errorf("%s has no shared memory recorded", name)
		}
	}
}

// TestEstimatedLimitsAreNotEnforced checks the bug the marker fixed rather
// than the marker itself. The estimated 48 KB was used as a validation
// ceiling, so a kernel that a newer GPU runs would be rejected by a limit
// nobody measured -- while the same artifact passes on a recorded target
// whose real limit is higher.
func TestEstimatedLimitsAreNotEnforced(t *testing.T) {
	recorded, err := ProbedTargetSM("sm_90a")
	if err != nil {
		t.Fatalf("sm_90a: %v", err)
	}
	estimated, err := TargetSM("sm_130a")
	if err != nil {
		t.Fatalf("sm_130a: %v", err)
	}
	// The control: this test says nothing unless the estimate is actually
	// below the recorded limit.
	if estimated.MaxSharedMemoryBytes >= recorded.MaxSharedMemoryBytes {
		t.Skipf("estimate (%d B) is not below the recorded sm_90a limit (%d B); nothing to falsify",
			estimated.MaxSharedMemoryBytes, recorded.MaxSharedMemoryBytes)
	}
	want := estimated.MaxSharedMemoryBytes + 1024

	build := func(target string) *Artifact {
		t.Helper()
		// Built through the constructor, not a struct literal: Validate
		// checks the digest first, so a hand-made Artifact fails before it
		// ever reaches the geometry check and the test proves nothing. An
		// earlier version of this test did exactly that -- reverting the fix
		// it guards left it passing.
		a, err := NewArtifact(KindPTX, []byte(".version 9.0\n.target "+target+"\n.visible .entry k(){ ret; }\n"), "k", Schema{}, target,
			WithResources(ResourceRequirements{StaticSharedMemory: want}))
		if err != nil {
			t.Fatalf("NewArtifact(%s): %v", target, err)
		}
		return a
	}

	// The control: the same request against a target whose recorded limit is
	// genuinely below it must still be rejected, or "not enforced" is
	// indistinguishable from "never checked".
	low := build("sm_90")
	lowInfo, err := ProbedTargetSM("sm_90")
	if err != nil {
		t.Fatalf("sm_90: %v", err)
	}
	if want > lowInfo.MaxSharedMemoryBytes {
		if err := low.Validate(); err == nil || !strings.Contains(err.Error(), "shared memory") {
			t.Fatalf("recorded sm_90 limit (%d B) did not reject %d B: %v", lowInfo.MaxSharedMemoryBytes, want, err)
		}
	}

	if err := build("sm_130a").Validate(); err != nil && strings.Contains(err.Error(), "shared memory") {
		t.Errorf("an invented limit rejected a kernel: %v", err)
	}
	if err := build("sm_90a").Validate(); err != nil && strings.Contains(err.Error(), "shared memory") {
		t.Errorf("recorded sm_90a limit (%d B) rejected %d B: %v", recorded.MaxSharedMemoryBytes, want, err)
	}
}

// TestUnmeasuredFeaturesDoNotReadAsAbsent pins the tri-state at this
// package's boundary: an unprobed target has no FeatureSource, and callers
// must not read its false Supports fields as hardware facts.
func TestUnmeasuredFeaturesDoNotReadAsAbsent(t *testing.T) {
	info, err := TargetSM("sm_130a")
	if err != nil {
		t.Fatalf("TargetSM: %v", err)
	}
	if info.FeatureSource != "" {
		t.Errorf("unprobed target names a feature source %q", info.FeatureSource)
	}
	if a := cudadevicetable.Ask("sm_130a", "wgmma"); a.Measured {
		t.Error("table claims a measurement for sm_130a")
	}
	probed, err := ProbedTargetSM("sm_90a")
	if err != nil {
		t.Fatalf("sm_90a: %v", err)
	}
	if probed.FeatureSource == "" {
		t.Error("a recorded target carries no feature source")
	}
}
