package advise

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func findElectedBulk(items []Item) *Item {
	for i := range items {
		if items[i].rule == "elected-bulk-copy" {
			return &items[i]
		}
		if found := findElectedBulk(items[i].Patterns); found != nil {
			return found
		}
	}
	return nil
}

func assertNoRuntimeAttribution(t *testing.T, item Item) {
	t.Helper()
	if !item.NoRuntimeAttribution || item.DeviceTimeNanos != 0 || item.HostTimeNanos != 0 || item.ShareOfProfile != 0 || item.Gap != nil {
		t.Fatalf("finding acquired runtime attribution: %+v", item)
	}
	if item.ClassifiedShare == nil {
		t.Fatal("missing explicit unread share")
	}
	r := item.ClassifiedShare.Record()
	if r.Class != cudaevidence.Unread || r.Number != nil || r.Reason != noRuntimeAttributionReason {
		t.Fatalf("runtime share was admitted: %+v", r)
	}
}

// A positive timed public input must not turn a static occurrence into this
// finding's elapsed cost. The existing context population is not its cause.
func TestElectedBulkTimedPublicEnvelopeNoRuntimeAttribution(t *testing.T) {
	cap := electedBulkEnvelope(t, electedBulkWitness)
	cap.Profile.SampleType = []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}
	cap.Profile.Sample = []*profile.Sample{{Value: []int64{1_000_000}, Label: map[string][]string{"kernel": {"k"}, "subtype": {"kernel"}}}}
	r, err := Analyze(cap)
	if err != nil {
		t.Fatal(err)
	}
	item := findElectedBulk(r.Items)
	if item == nil {
		t.Fatal("positive bulk control did not fire")
	}
	assertNoRuntimeAttribution(t, *item)
	if strings.Contains(r.Markdown(), "100.00% of device time") {
		t.Fatal("static lead rendered as all measured device time")
	}
	r.RankAgainst(testBaseline{})
	item = findElectedBulk(r.Items)
	if item == nil {
		t.Fatal("lost static lead while ranking")
	}
	assertNoRuntimeAttribution(t, *item)
	if gapSection(*item) != gapNoTerm {
		t.Fatal("static lead entered the gap-explaining ranking")
	}
}

// Even a complete supplied CPU journal/profile population must remain unread
// for this finding. A normal finding in that same input is the positive
// control proving that global capture rejection did not make this test pass.
func TestElectedBulkCompletePopulationStillNoAttribution(t *testing.T) {
	cap, source := consumerCapture(1_000_000)
	source.Identity.Target = "sm_90a"
	cap.Identity.ComputeCap = source.Identity.Target
	cap.Metadata = json.RawMessage(`{"compute_capability":"sm_90a","sm_count":132}`)
	cap.Code.SASS = "kernel.sass"
	cap.Files["kernel.sass"] = []byte(strings.ReplaceAll(electedBulkWitness, "Function : k", "Function : kernel"))
	r, err := AnalyzeWithEvidence(cap, source)
	if err != nil {
		t.Fatal(err)
	}
	item := findElectedBulk(r.Items)
	if item == nil {
		t.Fatal("positive bulk control did not fire")
	}
	assertNoRuntimeAttribution(t, *item)
	var admittedControl bool
	for _, other := range findings(r) {
		if other.rule != "elected-bulk-copy" && other.ClassifiedShare != nil && other.ClassifiedShare.Record().Class == cudaevidence.Derived {
			admittedControl = true
		}
	}
	if !admittedControl {
		t.Fatal("normal evidence control was not admitted; refusal would be vacuous")
	}
}

func TestNoAttributionEvidenceAndJSONCannotReacquireNumbers(t *testing.T) {
	source := evidenceSource()
	original := Item{NoRuntimeAttribution: true, EvidenceLevel: EvidenceSuspected, Location: Location{Kernel: source.Identity.Entry},
		DeviceTimeNanos: 1_000_000, HostTimeNanos: 2_000_000, ShareOfProfile: 100, Gap: &GapBound{BoundNanos: 3_000_000}, terms: []string{"idle"}, magnitude: "stale cost"}
	got, err := original.WithShareEvidence(source, true, "a complete supplied population")
	if err != nil {
		t.Fatal(err)
	}
	assertNoRuntimeAttribution(t, got)
	if original.DeviceTimeNanos != 1_000_000 || original.ShareOfProfile != 100 || original.ClassifiedShare != nil {
		t.Fatal("value adapter mutated caller")
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var round Item
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	if !round.NoRuntimeAttribution {
		t.Fatal("JSON lost attribution disposition")
	}
	// Simulate stale numeric fields imported beside the preserved disposition.
	round.DeviceTimeNanos, round.HostTimeNanos, round.ShareOfProfile = 9_000_000, 8_000_000, 100
	round.Gap = &GapBound{BoundNanos: 7_000_000}
	r := &Report{Items: []Item{round}}
	r.RankAgainst(testBaseline{})
	assertNoRuntimeAttribution(t, r.Items[0])
}

func TestNoAttributionDoesNotFoldOrGroupWithMeasuredPatterns(t *testing.T) {
	static := Item{NoRuntimeAttribution: true, rule: "imad-wide", Severity: SeverityContext, Category: CategoryArchitecture, Location: Location{Kernel: "k"}, Finding: "static"}
	a := Item{rule: "imad-wide", Severity: SeverityContext, Category: CategoryArchitecture, Location: Location{Kernel: "k"}, Finding: "normal k"}
	b := Item{rule: "imad-wide", Severity: SeverityContext, Category: CategoryArchitecture, Location: Location{Kernel: "other"}, Finding: "normal other"}
	r := &Report{Items: []Item{static, a, b}}
	finalizeAndRankFindingsWithTiming(r, map[string]int64{"k": 1000, "other": 2000}, 3000, true)
	if len(r.Items) != 2 {
		t.Fatalf("got %d items; static finding folded or grouped", len(r.Items))
	}
	var seenStatic, seenMeasured bool
	for _, item := range r.Items {
		if item.NoRuntimeAttribution {
			assertNoRuntimeAttribution(t, item)
			seenStatic = true
			continue
		}
		if item.DeviceTimeNanos != 3000 || len(item.Patterns) != 2 {
			t.Fatalf("normal group changed: %+v", item)
		}
		for _, child := range item.Patterns {
			if child.NoRuntimeAttribution {
				t.Fatal("static finding entered measured group")
			}
		}
		seenMeasured = true
	}
	if !seenStatic || !seenMeasured {
		t.Fatal("missing independent controls")
	}
}

func TestNoAttributionMovesNoGapTerm(t *testing.T) {
	static := Item{NoRuntimeAttribution: true, Severity: SeverityContext, Category: CategoryArchitecture, Location: Location{Kernel: "k"}}
	ordinary := static
	ordinary.NoRuntimeAttribution = false
	if got := gapBound(ordinary, testBaseline{}); got == nil || got.BoundNanos <= 0 {
		t.Fatal("ordinary gap control was not admitted")
	}
	if got := gapBound(static, testBaseline{}); got != nil {
		t.Fatalf("static occurrence acquired gap bound: %+v", got)
	}
}

func TestElectedBulkPCSamplesDoNotBecomePatternShare(t *testing.T) {
	cap := electedBulkEnvelope(t, electedBulkWitness)
	cap.Profile.SampleType = []*profile.ValueType{{Type: "pc_samples", Unit: "count"}}
	cap.Profile.Sample = []*profile.Sample{{Value: []int64{100}, Label: map[string][]string{"kernel": {"k"}, "subtype": {"kernel"}}}}
	r, err := Analyze(cap)
	if err != nil {
		t.Fatal(err)
	}
	item := findElectedBulk(r.Items)
	if item == nil {
		t.Fatal("positive bulk control did not fire")
	}
	assertNoRuntimeAttribution(t, *item)
	if !r.SampleOnly {
		t.Fatal("PC-sample instrument control lost")
	}
	if strings.Contains(r.Markdown(), "100.00% of observed samples") {
		t.Fatal("static occurrence acquired PC sample attribution")
	}
}
