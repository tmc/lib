package cudaschedule_test

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestReplayPerturbRoundTrip(t *testing.T) {
	for _, intensity := range []float64{0, 0.1, 1, 2} {
		for seed := uint64(0); seed < 50; seed++ {
			opts := cudaschedule.Options{Seed: seed, Intensity: intensity, MaxSleepNs: 71, Focus: "atom"}
			want, err := cudaschedule.Perturb(PERTURB_PTX, opts)
			if err != nil {
				t.Fatal(err)
			}
			// Serialization must not depend on BlockIndex pointer identity.
			body, err := json.Marshal(want.Report.Decisions)
			if err != nil {
				t.Fatal(err)
			}
			var decisions []cudaschedule.InjectionDecision
			if err := json.Unmarshal(body, &decisions); err != nil {
				t.Fatal(err)
			}
			got, err := cudaschedule.Replay(PERTURB_PTX, opts, decisions)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("seed %d intensity %g: replay differs", seed, intensity)
			}
		}
	}
}

func TestReplaySubsetAndOrder(t *testing.T) {
	opts := cudaschedule.Options{Seed: 17, Intensity: 2}
	original, err := cudaschedule.Perturb(PERTURB_PTX, opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudaschedule.Replay(PERTURB_PTX, opts, nil)
	if err != nil || got.PTX != PERTURB_PTX || got.Report.SitesInjected != 0 {
		t.Fatalf("empty: %+v %v", got, err)
	}
	decisions := []cudaschedule.InjectionDecision{original.Report.Decisions[3], original.Report.Decisions[0]}
	got, err = cudaschedule.Replay(PERTURB_PTX, opts, decisions)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Report.Decisions, decisions) || got.Report.SitesTotal != len(original.Report.Decisions) || got.Report.SitesInjected != 2 {
		t.Fatalf("subset report: %+v", got.Report)
	}
	if got.PTX == original.PTX || strings.Count(got.PTX, "nanosleep.u32") != 2 {
		t.Fatalf("subset did not remove decisions: %s", got.PTX)
	}
	if _, err := cudaptx.ParseString(got.PTX); err != nil {
		t.Fatal(err)
	}
	// A returned report must not alias the caller's block pointer or slice.
	last := original.Report.Decisions[len(original.Report.Decisions)-1]
	if last.Site.BlockIndex == nil {
		t.Fatal("fixture needs a backedge")
	}
	got, err = cudaschedule.Replay(PERTURB_PTX, opts, []cudaschedule.InjectionDecision{last})
	if err != nil {
		t.Fatal(err)
	}
	*last.Site.BlockIndex = 999
	if *got.Report.Decisions[0].Site.BlockIndex == 999 {
		t.Fatal("report aliases caller block")
	}
}

func TestReplayRefusesCounterfeitSites(t *testing.T) {
	sites, err := cudaschedule.Analyze(PERTURB_PTX)
	if err != nil {
		t.Fatal(err)
	}
	opts := cudaschedule.Options{Intensity: 1, MaxSleepNs: 20}
	base := cudaschedule.InjectionDecision{Site: sites[0], BeforeNs: 10}
	cases := []struct {
		name   string
		change func(*cudaschedule.InjectionDecision)
	}{
		{"ordinal", func(d *cudaschedule.InjectionDecision) { d.Site.Ordinal = -1 }},
		{"unknown", func(d *cudaschedule.InjectionDecision) { d.Site.Ordinal = len(sites) }},
		{"callable", func(d *cudaschedule.InjectionDecision) { d.Site.Callable += "x" }},
		{"kind", func(d *cudaschedule.InjectionDecision) { d.Site.Kind = cudaschedule.SiteFence }},
		{"start", func(d *cudaschedule.InjectionDecision) { d.Site.Span.Start-- }},
		{"end", func(d *cudaschedule.InjectionDecision) { d.Site.Span.End++ }},
		{"head", func(d *cudaschedule.InjectionDecision) { d.Site.Head += "x" }},
		{"text", func(d *cudaschedule.InjectionDecision) { d.Site.Text += "x" }},
		{"predicate", func(d *cudaschedule.InjectionDecision) { d.Site.Predicate = "%fake" }},
		{"block", func(d *cudaschedule.InjectionDecision) { n := 1; d.Site.BlockIndex = &n }},
		{"before ceiling", func(d *cudaschedule.InjectionDecision) { d.BeforeNs = 21 }},
		{"after ceiling", func(d *cudaschedule.InjectionDecision) { d.AfterNs = 21 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := base
			tc.change(&d)
			if _, err := cudaschedule.Replay(PERTURB_PTX, opts, []cudaschedule.InjectionDecision{d}); err == nil {
				t.Fatal("accepted counterfeit")
			}
		})
	}
	if _, err := cudaschedule.Replay(PERTURB_PTX, opts, []cudaschedule.InjectionDecision{base, base}); err == nil {
		t.Fatal("accepted duplicate")
	}
	back := cudaschedule.InjectionDecision{Site: sites[len(sites)-1], AfterNs: 1}
	if _, err := cudaschedule.Replay(PERTURB_PTX, opts, []cudaschedule.InjectionDecision{back}); err == nil {
		t.Fatal("accepted backedge after")
	}
	for _, intensity := range []float64{math.NaN(), math.Inf(1), -1, 0} {
		opts.Intensity = intensity
		if _, err := cudaschedule.Replay(PERTURB_PTX, opts, []cudaschedule.InjectionDecision{base}); err == nil {
			t.Fatal("accepted invalid intensity/delay")
		}
	}
	if _, err := cudaschedule.Replay(strings.Replace(PERTURB_PTX, "atom.global.add", "atom.global.exch", 1), cudaschedule.Options{Intensity: 1}, []cudaschedule.InjectionDecision{base}); err == nil {
		t.Fatal("accepted changed source site")
	}
}

func TestReplaySameLineAndInlinePlacement(t *testing.T) {
	const src = `.version 8.7
.target sm_90
.address_size 64
.visible .entry replay() {
.reg .b32 %r;
.reg .b64 %rd;
membar.gl; bar.sync 0; mov.u32 %r, 1;
// begin inline asm
membar.gl; bar.sync 0;
// end inline asm
ret;
}`
	sites, err := cudaschedule.Analyze(src)
	if err != nil {
		t.Fatal(err)
	}
	decisions := make([]cudaschedule.InjectionDecision, len(sites))
	for i, s := range sites {
		decisions[i] = cudaschedule.InjectionDecision{Site: s, BeforeNs: 3, AfterNs: 7}
	}
	got, err := cudaschedule.Replay(src, cudaschedule.Options{Intensity: 1}, decisions)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(got.PTX, "nanosleep.u32 3;") != len(sites) || strings.Count(got.PTX, "nanosleep.u32 7;") != len(sites) {
		t.Fatalf("recorded delays not emitted: %s", got.PTX)
	}
	mod, err := cudaptx.ParseString(got.PTX)
	if err != nil {
		t.Fatal(err)
	}
	if len(mod.Kernels) != 1 || !strings.Contains(got.PTX, "mov.u32 %r, 1;") {
		t.Fatal("lost same-line instruction")
	}
	begin, end := strings.Index(got.PTX, "// begin inline asm"), strings.Index(got.PTX, "// end inline asm")
	if strings.Contains(got.PTX[begin:end], "nanosleep") {
		t.Fatal("injected inside inline block")
	}
}
