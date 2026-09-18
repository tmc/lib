package cudadevicetable

import (
	"strings"
	"testing"
)

// TestResolverAnswersATargetTheTableLacks runs a real assembler against a
// target that is deliberately absent from devices.json. It is the property
// that keeps the checked-in table from acting as an allowlist: a target
// nobody recorded still gets a measured answer, with provenance.
func TestResolverAnswersATargetTheTableLacks(t *testing.T) {
	ptxas, err := FindPTXAS()
	if err != nil {
		t.Skipf("no ptxas available to resolve with: %v", err)
	}
	// sm_86 is old enough to be uncontroversial and is not in the table.
	const target = "sm_86"
	if _, ok := LookupTarget(target); ok {
		t.Skipf("%s is now in the table; pick a target that is not", target)
	}

	r := NewResolver(ptxas, "test")
	a := r.Ask(target, "tma")
	if !a.Measured {
		t.Fatalf("Ask(%s, tma) came back unmeasured: %s", target, a.Why())
	}
	if a.Source == "" {
		t.Error("resolved answer carries no source")
	}
	if !strings.Contains(a.Source, "ptxas") {
		t.Errorf("source %q does not name the assembler", a.Source)
	}
	// TMA is a Hopper construct; an Ampere target must reject it. If this
	// ever passes, the fixture stopped testing what it claims to.
	if a.Supported {
		t.Errorf("%s accepted tma: %s", target, a.Why())
	}
	if a.Diagnostic == "" {
		t.Error("a rejection with no diagnostic is not a measurement")
	}

	// The second Ask must not launch another assembler. Resolved() is the
	// observable proxy: one target resolved, not two.
	r.Ask(target, "mbarrier")
	if got := r.Resolved(); len(got) != 1 {
		t.Errorf("Resolved() has %d targets after two Asks, want 1", len(got))
	}
}

// TestResolverDegradesToUnmeasured checks the failure direction. A resolver
// that cannot run anything must produce silence; producing a negative would
// be worse than having no resolver at all.
func TestResolverDegradesToUnmeasured(t *testing.T) {
	r := NewResolver("/nonexistent/ptxas", "test")
	a := r.Ask("sm_130a", "wgmma")
	if a.Measured {
		t.Fatalf("a resolver with no assembler claims a measurement: %s", a.Why())
	}
	if a.Supported {
		t.Error("unmeasured answer is supported")
	}
	if got := a.Or(AssumeSupported); !got {
		t.Error("policy not applied to a failed resolution")
	}
	// A target already in the table must still answer without the assembler.
	if a := r.Ask("sm_90a", "wgmma"); !a.Measured || !a.Supported {
		t.Errorf("table fast path broken when the assembler is missing: %s", a.Why())
	}
}

// TestResolverRejectsNonTargets keeps the probe from shelling out for a typo.
func TestResolverRejectsNonTargets(t *testing.T) {
	r := NewResolver("/nonexistent/ptxas", "test")
	for _, bad := range []string{"hopper", "compute_90", "sm_", ""} {
		if _, err := r.Resolve(bad); err == nil {
			t.Errorf("Resolve(%q) succeeded", bad)
		}
	}
}
