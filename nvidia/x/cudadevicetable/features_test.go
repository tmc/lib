package cudadevicetable

import (
	"strings"
	"testing"
)

// probeRecord is a valid feature probe for tests that vary one field.
func probeRecord() *FeatureProbe {
	return &FeatureProbe{
		Tool:        "ptxas",
		ToolVersion: "13.0.88",
		PTXVersion:  "8.8",
		Host:        "gpu-host",
		ProbedAt:    "2026-09-17",
	}
}

func targetTable(t Target) Table {
	return Table{Schema: 1, Targets: map[string]Target{t.Target: t}}
}

// TestUnsupportedEverywhereIsRejected is the positive-control rule. A feature
// that no target accepts is a fixture that never assembled, and recording its
// negatives would publish a broken probe as a row of hardware facts.
func TestUnsupportedEverywhereIsRejected(t *testing.T) {
	tbl := Table{Schema: 1, Targets: map[string]Target{
		"sm_90a": {Target: "sm_90a", Probe: probeRecord(), Features: map[string]Feature{
			"ghost": {Source: Probed, Instrument: "ptxas -arch=sm_90a ghost.ptx", Fixture: "ptxprobe/ghost.ptx", Diagnostic: "error: no"},
		}},
		"sm_121a": {Target: "sm_121a", Probe: probeRecord(), Features: map[string]Feature{
			"ghost": {Source: Probed, Instrument: "ptxas -arch=sm_121a ghost.ptx", Fixture: "ptxprobe/ghost.ptx", Diagnostic: "error: no"},
		}},
	}}
	err := tbl.Validate()
	if err == nil {
		t.Fatal("Validate accepted a feature unsupported at every target")
	}
	if !strings.Contains(err.Error(), "assemble nowhere") {
		t.Errorf("Validate error = %v, want it to name the missing positive control", err)
	}
	if !strings.Contains(err.Error(), "ghost") {
		t.Errorf("Validate error = %v, want it to name the offending feature", err)
	}

	// Every offender is named, in sorted order. Map iteration would otherwise
	// blame a different one on each run, and a blame that moves between runs
	// reads as a fact about the feature it lands on.
	for _, tgt := range tbl.Targets {
		tgt.Features["apparition"] = Feature{Source: Probed, Instrument: "ptxas -arch=" + tgt.Target + " apparition.ptx", Fixture: "ptxprobe/apparition.ptx", Diagnostic: "error: no"}
	}
	err = tbl.Validate()
	if err == nil {
		t.Fatal("Validate accepted two features unsupported at every target")
	}
	if want := "feature(s) apparition, ghost unsupported"; !strings.Contains(err.Error(), want) {
		t.Errorf("Validate error = %v, want it to contain %q", err, want)
	}
	for _, tgt := range tbl.Targets {
		delete(tgt.Features, "apparition")
	}

	// One acceptance anywhere is what makes the negatives meaningful.
	sm90a := tbl.Targets["sm_90a"]
	sm90a.Features["ghost"] = Feature{Supported: true, Source: Probed, Instrument: "ptxas -arch=sm_90a ghost.ptx", Fixture: "ptxprobe/ghost.ptx"}
	if err := tbl.Validate(); err != nil {
		t.Errorf("Validate rejected a feature with a firing positive control: %v", err)
	}
}

func TestProbedFeatureMustNameItsInstrumentAndFixture(t *testing.T) {
	base := Feature{Supported: true, Source: Probed, Instrument: "ptxas -arch=sm_90a wgmma.ptx", Fixture: "ptxprobe/wgmma.ptx"}
	tests := []struct {
		name string
		mut  func(*Feature)
		want string
	}{
		{"no instrument", func(f *Feature) { f.Instrument = "" }, "names no instrument"},
		{"not ptxas", func(f *Feature) { f.Instrument = "the datasheet" }, "want a ptxas invocation"},
		{"other target", func(f *Feature) { f.Instrument = "ptxas -arch=sm_100a wgmma.ptx" }, "does not name this target"},
		{"no fixture", func(f *Feature) { f.Fixture = "" }, "names no fixture"},
		{"unsupported without diagnostic", func(f *Feature) { f.Supported = false }, "records no diagnostic"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := base
			tt.mut(&f)
			tbl := targetTable(Target{Target: "sm_90a", Probe: probeRecord(), Features: map[string]Feature{"wgmma": f}})
			err := tbl.Validate()
			if err == nil {
				t.Fatalf("Validate accepted %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Validate error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
	tbl := targetTable(Target{Target: "sm_90a", Probe: probeRecord(), Features: map[string]Feature{"wgmma": base}})
	if err := tbl.Validate(); err != nil {
		t.Errorf("Validate rejected a well-formed record: %v", err)
	}
}

func TestUnverifiedFeatureNeedsANoteAndNoInstrument(t *testing.T) {
	good := Target{Target: "sm_90a", Features: map[string]Feature{
		"wgmma": {Source: Unverified, Note: "no ptxas on this host"},
	}}
	if err := targetTable(good).Validate(); err != nil {
		t.Errorf("Validate rejected a noted unverified feature: %v", err)
	}
	bad := Target{Target: "sm_90a", Features: map[string]Feature{
		"wgmma": {Source: Unverified},
	}}
	if err := targetTable(bad).Validate(); err == nil {
		t.Error("Validate accepted an unverified feature with no note")
	}
	claimed := Target{Target: "sm_90a", Features: map[string]Feature{
		"wgmma": {Source: Unverified, Note: "n/a", Instrument: "ptxas -arch=sm_90a wgmma.ptx"},
	}}
	if err := targetTable(claimed).Validate(); err == nil {
		t.Error("Validate accepted an unverified feature naming an instrument")
	}
}

// TestSupportsSeparatesUnprobedFromUnsupported keeps the two negatives apart.
// A bool cannot hold both "the assembler said no" and "nobody asked", and
// collapsing them is how an unmeasured feature becomes a confident false.
func TestSupportsSeparatesUnprobedFromUnsupported(t *testing.T) {
	tgt := Target{Target: "sm_121a", Probe: probeRecord(), Features: map[string]Feature{
		"wgmma":   {Supported: false, Source: Probed, Instrument: "ptxas -arch=sm_121a wgmma.ptx", Fixture: "ptxprobe/wgmma.ptx", Diagnostic: "error: not supported"},
		"tcgen05": {Source: Unverified, Note: "fixture crashed ptxas; no fact recorded"},
	}}
	got, err := tgt.Supports("wgmma")
	if err != nil || got {
		t.Errorf("Supports(wgmma) = %v, %v; want false, nil", got, err)
	}
	if _, err := tgt.Supports("tcgen05"); err == nil {
		t.Error("Supports returned a bool for an unverified feature; an absent measurement must not read as a negative one")
	}
	if _, err := tgt.Supports("nosuch"); err == nil {
		t.Error("Supports returned a bool for an unrecorded feature")
	}
}

// TestFeatureSuffixIsLoadBearing proves the decision to key features by exact
// target rather than by normalized architecture is not cosmetic: sm_90 and
// sm_90a are the same silicon and give different answers, so a table that
// dropped the suffix would have to record one of them wrongly.
func TestFeatureSuffixIsLoadBearing(t *testing.T) {
	base, ok := LookupTarget("sm_90")
	if !ok {
		t.Skip("sm_90 not probed")
	}
	suffixed, ok := LookupTarget("sm_90a")
	if !ok {
		t.Skip("sm_90a not probed")
	}
	if Normalize("sm_90") != Normalize("sm_90a") {
		t.Fatal("Normalize distinguishes sm_90 from sm_90a; this test assumes it does not")
	}
	differs := false
	for _, name := range FeatureNames() {
		a, errA := base.Supports(name)
		b, errB := suffixed.Supports(name)
		if errA != nil || errB != nil {
			continue
		}
		if a != b {
			differs = true
			t.Logf("%s: sm_90=%v sm_90a=%v", name, a, b)
		}
	}
	if !differs {
		t.Error("no probed feature distinguishes sm_90 from sm_90a; the exact-target keyspace would be unnecessary")
	}
}

// TestMBarrierSplitsAtHopper checks that the mbarrier probe asks for the sm_80
// form and mbarriertx for the expect_tx and try_wait forms TMA needs. A single
// fixture once used expect_tx and so recorded Ampere and Ada as having no
// mbarrier at all.
func TestMBarrierSplitsAtHopper(t *testing.T) {
	tests := []struct {
		target       string
		mbarrier, tx bool
	}{
		{"sm_75", false, false},
		{"sm_80", true, false},
		{"sm_89", true, false},
		{"sm_90", true, true},
		{"sm_103a", true, true},
		{"sm_121a", true, true},
	}
	for _, tt := range tests {
		tgt, ok := LookupTarget(tt.target)
		if !ok {
			t.Errorf("%s not probed", tt.target)
			continue
		}
		for feature, want := range map[string]bool{"mbarrier": tt.mbarrier, "mbarriertx": tt.tx} {
			got, err := tgt.Supports(feature)
			if err != nil {
				t.Errorf("%s: %v", tt.target, err)
				continue
			}
			if got != want {
				t.Errorf("%s supports %s = %v, want %v", tt.target, feature, got, want)
			}
		}
	}
}

// TestRecordedFixturesExist keeps a recorded claim re-runnable. A fixture path
// that no longer resolves makes the fact it backs unreproducible.
func TestRecordedFixturesExist(t *testing.T) {
	fixtures := Fixtures()
	if len(fixtures) == 0 {
		t.Fatal("no embedded fixtures")
	}
	for _, name := range Targets() {
		tgt, _ := LookupTarget(name)
		for feature, f := range tgt.Features {
			if f.Source != Probed {
				continue
			}
			want := "ptxprobe/" + feature + ".ptx"
			if f.Fixture != want {
				t.Errorf("%s %s: fixture %q, want %q", name, feature, f.Fixture, want)
			}
			if _, ok := fixtures[feature]; !ok {
				t.Errorf("%s %s: fixture %q is not embedded", name, feature, f.Fixture)
			}
		}
	}
}

// TestEveryFixtureDeclaresAPlaceholder catches a fixture that pins its own
// target, which would silently probe the wrong one for every other row.
func TestEveryFixtureDeclaresAPlaceholder(t *testing.T) {
	for name, body := range Fixtures() {
		if !strings.Contains(body, ".target "+targetPlaceholder) {
			t.Errorf("%s.ptx does not declare .target %s", name, targetPlaceholder)
		}
	}
}

// TestOutcomeClassification is the guard on the distinction the whole probe
// rests on. ptxas exits 255 on a compile error, not 1, and dies on a signal for
// some inputs; reading any non-zero status as "unsupported" manufactures
// hardware facts out of assembler bugs.
func TestOutcomeClassification(t *testing.T) {
	tests := []struct {
		name string
		code int
		want bool
	}{
		{"ptxas compile error", 255, false},
		{"generic failure", 1, false},
		{"segfault via shell", 139, true},
		{"abort via shell", 134, true},
		{"go signalled child", -1, true},
		{"success", 0, false},
	}
	for _, tt := range tests {
		if got := signaled(tt.code); got != tt.want {
			t.Errorf("signaled(%d) = %v, want %v (%s)", tt.code, got, tt.want, tt.name)
		}
	}

	rejections := []string{
		"ptxas wgmma.ptx, line 22; error   : Instruction 'wgmma.fence' not supported on .target 'sm_100a'",
		"error   : Feature '.cta_group::1' not supported",
	}
	for _, s := range rejections {
		if !rejectionPattern.MatchString(s) {
			t.Errorf("rejectionPattern did not match a target rejection: %q", s)
		}
	}
	notRejections := []string{
		"ptxas t.ptx, line 2; error   : PTX .version 8.7 does not support .target sm_121a",
		"ptxas t.ptx, line 15; error   : Arguments mismatch for instruction 'wgmma.mma_async'",
		"ptxas fatal   : Cannot open input file",
	}
	for _, s := range notRejections {
		if rejectionPattern.MatchString(s) {
			t.Errorf("rejectionPattern matched a fixture or toolchain fault as a target rejection: %q", s)
		}
	}
}
