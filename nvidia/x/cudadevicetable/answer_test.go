package cudadevicetable

import "testing"

// TestAskSeparatesThreeStates is the point of the type. A bool has room for
// two of these and the third is the one that invents hardware facts.
func TestAskSeparatesThreeStates(t *testing.T) {
	tests := []struct {
		name             string
		target, feature  string
		wantMeasured     bool
		wantSupported    bool
		wantDiagnostic   bool
		wantOrOptimistic bool
	}{
		{"measured yes", "sm_90a", "wgmma", true, true, false, true},
		{"measured no", "sm_121a", "wgmma", true, false, true, false},
		{"target never probed", "sm_130a", "wgmma", false, false, false, true},
		{"feature never probed", "sm_90a", "no_such_feature", false, false, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := Ask(tt.target, tt.feature)
			if a.Measured != tt.wantMeasured || a.Supported != tt.wantSupported {
				t.Fatalf("Ask(%s, %s) = measured %v supported %v, want %v %v",
					tt.target, tt.feature, a.Measured, a.Supported, tt.wantMeasured, tt.wantSupported)
			}
			if got := a.Diagnostic != ""; got != tt.wantDiagnostic {
				t.Errorf("diagnostic present = %v, want %v (%q)", got, tt.wantDiagnostic, a.Diagnostic)
			}
			if got := a.Measured && a.Source == ""; got {
				t.Errorf("measured answer carries no source")
			}
			if got := a.Or(AssumeSupported); got != tt.wantOrOptimistic {
				t.Errorf("Or(AssumeSupported) = %v, want %v", got, tt.wantOrOptimistic)
			}
			// A measured answer must ignore the policy entirely; an
			// unmeasured one must take it. If Or ever stopped consulting
			// Measured, both policies would agree and nothing here would
			// notice, so check that they disagree exactly when they should.
			differ := a.Or(AssumeSupported) != a.Or(AssumeAbsent)
			if differ != !a.Measured {
				t.Errorf("policies differ = %v, want %v (measured %v)", differ, !a.Measured, a.Measured)
			}
			if a.Why() == "" {
				t.Error("Why() is empty")
			}
		})
	}
}

// TestAskNeverInventsATarget guards the property the whole design rests on:
// a lookup miss produces silence, not a negative.
func TestAskNeverInventsATarget(t *testing.T) {
	for _, target := range []string{"sm_130a", "sm_999", "", "hopper"} {
		for _, feature := range FeatureNames() {
			if a := Ask(target, feature); a.Measured {
				t.Errorf("Ask(%q, %q) claims to be measured", target, feature)
			}
		}
	}
}

// TestFixtureVersionIsSingleValued checks that the stamp a probe run records
// is true of every fixture that run assembled.
func TestFixtureVersionIsSingleValued(t *testing.T) {
	v, err := FixtureVersion()
	if err != nil {
		t.Fatalf("FixtureVersion: %v", err)
	}
	if v == "" {
		t.Fatal("FixtureVersion returned an empty version")
	}
	for name, tgt := range table.Targets {
		if tgt.Probe != nil && tgt.Probe.PTXVersion != v {
			t.Errorf("%s recorded PTX version %q, fixtures declare %q", name, tgt.Probe.PTXVersion, v)
		}
	}
}
