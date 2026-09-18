package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// provenanceCapture ships one cubin and a profile whose mapping claims
// buildID for it.
func provenanceCapture(cubin []byte, buildID string) *cudacapture.Capture {
	return &cudacapture.Capture{
		Code:  cudacapture.CodeFiles{CUBIN: "kernels.cubin"},
		Files: map[string][]byte{"kernels.cubin": cubin},
		Profile: &profile.Profile{
			Mapping: []*profile.Mapping{{File: "/build/kernels.cubin", BuildID: buildID}},
		},
	}
}

func digest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

// TestProvenanceMismatchIsReported is the control the request names: a profile
// captured from a different build must be reported as such. Without the second
// case the check would pass vacuously by reporting everything.
func TestProvenanceMismatchIsReported(t *testing.T) {
	current := []byte("cubin of the code in the tree")
	stale := []byte("cubin of a build from before the D2R branch landed")

	var matched, mismatched bytes.Buffer
	reportProvenance(&matched, provenanceCapture(current, digest(current)))
	reportProvenance(&mismatched, provenanceCapture(current, digest(stale)))

	if strings.Contains(matched.String(), "did not come from") {
		t.Errorf("a matching build ID was reported as a mismatch: %q", matched.String())
	}
	got := mismatched.String()
	if !strings.Contains(got, "did not come from") {
		t.Fatalf("a stale profile was accepted silently: %q", got)
	}
	// Both IDs must appear: a reader cannot tell which artifact to re-capture
	// from a message that names only that they differ.
	for _, want := range []string{digest(stale), digest(current), "kernels.cubin"} {
		if !strings.Contains(got, want) {
			t.Errorf("report does not name %s: %q", want, got)
		}
	}
}

// TestUnrecordedBuildIDIsNotAgreement pins the third state. A capture whose
// profile never recorded a build ID cannot be checked, and passing it silently
// would make the check vacuous on every profile written before the ID was
// recorded -- which is every profile the two committed fixtures contain.
func TestUnrecordedBuildIDIsNotAgreement(t *testing.T) {
	var buf bytes.Buffer
	reportProvenance(&buf, provenanceCapture([]byte("cubin"), ""))
	got := buf.String()
	if !strings.Contains(got, "provenance unverified") {
		t.Fatalf("an unrecorded build ID passed silently: %q", got)
	}
	if strings.Contains(got, "did not come from") {
		t.Errorf("absence was reported as a mismatch: %q", got)
	}
	if !strings.Contains(got, "kernels.cubin") {
		t.Errorf("the unchecked artifact is not named: %q", got)
	}
}

// TestProvenanceIgnoresImagesTheCaptureDoesNotCarry keeps the check from
// crying wolf on the host binary and its shared libraries, whose build IDs the
// capture has no copy to compare against.
func TestProvenanceIgnoresImagesTheCaptureDoesNotCarry(t *testing.T) {
	cap := provenanceCapture([]byte("cubin"), digest([]byte("cubin")))
	cap.Profile.Mapping = append(cap.Profile.Mapping,
		&profile.Mapping{File: "/usr/lib/libcuda.so.1", BuildID: "0123456789abcdef"})
	var buf bytes.Buffer
	reportProvenance(&buf, cap)
	if strings.Contains(buf.String(), "did not come from") {
		t.Errorf("an unrelated image was reported as a provenance failure: %q", buf.String())
	}
}

// TestProvenancePrintsTheProfilesAge makes the staleness visible without
// arithmetic, which is the subtraction nobody performs on a profile they
// believe they just captured.
func TestProvenancePrintsTheProfilesAge(t *testing.T) {
	cap := provenanceCapture([]byte("cubin"), digest([]byte("cubin")))
	cap.Profile.TimeNanos = 1e18 // 2001-09-09T01:46:40Z
	var buf bytes.Buffer
	reportProvenance(&buf, cap)
	// Rendered in the reader's zone, so the expectation is built the same way
	// rather than pinned to a UTC date that is a day off west of Greenwich.
	want := time.Unix(0, cap.Profile.TimeNanos).Format(time.RFC3339)
	if !strings.Contains(buf.String(), want) {
		t.Errorf("capture time not printed: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "ago") {
		t.Errorf("age not printed: %q", buf.String())
	}
}
