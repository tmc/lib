package advise

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func TestAdviseEvidenceAnalyzeJSONRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		inferred, missingSource bool
		class                   cudaevidence.Class
	}{
		{name: "observed", class: cudaevidence.Derived},
		{name: "suspected", inferred: true, class: cudaevidence.Inferred},
		{name: "missing_source", missingSource: true, class: cudaevidence.Unread},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cap, source := consumerCapture(25)
			if tc.inferred {
				cap.Files["kernels.sass"] = []byte(strings.ReplaceAll(sampleHighPressureSASS, "compute", "kernel"))
			}
			var report *Report
			var err error
			if tc.missingSource {
				report, err = Analyze(cap)
			} else {
				report, err = AnalyzeWithEvidence(cap, source)
			}
			if err != nil {
				t.Fatal(err)
			}
			found := false
			var attach func([]Item)
			attach = func(items []Item) {
				for i := range items {
					item := &items[i]
					if item.Location.Kernel == source.Identity.Entry && item.ClassifiedShare != nil && item.ClassifiedShare.Record().Class == tc.class {
						before := *item
						attached, err := item.WithEvidence(*item.ClassifiedShare)
						if err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(before, attached) {
							t.Fatal("attachment changed analyzed finding")
						}
						*item = attached
						found = true
					}
					attach(item.Patterns)
				}
			}
			attach(report.Items)
			if !found {
				t.Fatal("actual Analyze finding absent")
			}
			encoded, err := report.JSON()
			if err != nil {
				t.Fatal(err)
			}
			var decoded Report
			if err := json.Unmarshal([]byte(encoded), &decoded); err != nil {
				t.Fatal(err)
			}
			reencoded, err := decoded.JSON()
			if err != nil {
				t.Fatal(err)
			}
			if encoded != reencoded {
				t.Fatal("JSON changed analyzed findings")
			}
			var check func([]Item, []Item)
			check = func(before, after []Item) {
				if len(before) != len(after) {
					t.Fatal("JSON changed finding count")
				}
				for i := range before {
					a, b := before[i], after[i]
					if a.Location != b.Location || a.EvidenceLevel != b.EvidenceLevel || !reflect.DeepEqual(a.ClassifiedShare.Record(), b.ClassifiedShare.Record()) {
						t.Fatal("JSON changed source-bound evidence")
					}
					rec := b.ClassifiedShare.Record()
					if rec.Number != nil {
						if err := b.CheckEvidenceClaim(rec.Class, rec.Source.Identity); err != nil {
							t.Fatal(err)
						}
						if b.CheckEvidenceClaim(cudaevidence.Measured, rec.Source.Identity) == nil {
							t.Fatal("share promoted to measured")
						}
						other := rec.Source.Identity
						other.WorkloadKey = strings.Repeat("4", 64)
						if b.CheckEvidenceClaim(rec.Class, other) == nil {
							t.Fatal("mismatched workload claim accepted")
						}
					} else if b.CheckEvidenceClaim(cudaevidence.Derived, source.Identity) == nil {
						t.Fatal("withheld evidence claimed as derived")
					}
					check(a.Patterns, b.Patterns)
				}
			}
			check(report.Items, decoded.Items)
		})
	}
}

func TestAdviseEvidenceQuantityAndPopulation(t *testing.T) {
	cap, source := consumerCapture(25)
	item := consumerFinding(t, cap, &source)
	for _, tc := range []struct {
		name, unit string
		class      cudaevidence.Class
		number     float64
	}{
		{"duration", "nanoseconds", cudaevidence.Measured, 25000},
		{"derived_duration", "nanoseconds", cudaevidence.Derived, 20},
		{"occupancy", "ratio", cudaevidence.Modeled, .5},
		{"modeled_percent", "percent", cudaevidence.Modeled, 20},
		{"inferred_observation", "percent", cudaevidence.Inferred, 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := item.ClassifiedShare.Record()
			r.Unit, r.Class, r.Number = tc.unit, tc.class, &tc.number
			v, err := cudaevidence.New(r)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := item.WithEvidence(v); err == nil {
				t.Fatal("different quantity/class attached as profile share")
			}
			altered := item
			altered.ClassifiedShare = &v
			if _, err := (&Report{Items: []Item{altered}}).JSON(); err == nil {
				t.Fatal("different quantity/class serialized as profile share")
			}
			// Construct the wire form without the validating Item marshaler.
			raw := itemJSON(altered)
			payload, err := json.Marshal(raw)
			if err != nil {
				t.Fatal(err)
			}
			var decoded Item
			if err := json.Unmarshal(payload, &decoded); err == nil {
				t.Fatal("different quantity/class decoded as profile share")
			}
		})
	}
	for _, tc := range []struct {
		name   string
		change func(*cudaevidence.Record)
	}{
		{"number", func(r *cudaevidence.Record) { n := 21.; r.Number = &n }},
		{"denominator", func(r *cudaevidence.Record) { r.Basis = "different population" }},
		{"source", func(r *cudaevidence.Record) { r.Source.Locator += "/different" }},
		{"artifact", func(r *cudaevidence.Record) { r.Source.Identity.ArtifactDigest = strings.Repeat("4", 64) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := item.ClassifiedShare.Record()
			tc.change(&r)
			v, err := cudaevidence.New(r)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := item.WithEvidence(v); err == nil {
				t.Fatal("attachment changed existing share evidence")
			}
		})
	}
	withheld := consumerFinding(t, cap, nil)
	if _, err := withheld.WithEvidence(*item.ClassifiedShare); err == nil {
		t.Fatal("missing-source population upgraded by setter")
	}
	cap.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"partial","dropped_records":0}}`)
	partial := consumerFinding(t, cap, &source)
	if _, err := partial.WithEvidence(*item.ClassifiedShare); err == nil {
		t.Fatal("partial population upgraded by setter")
	}
}

func TestAdviseEvidenceWithholdsUnmeasured(t *testing.T) {
	source := evidenceSource()
	cap, capSource := consumerCapture(25)

	// 1. Analyze without explicit source must withhold numeric share in JSON.
	unauthenticated, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	unauthJSON, err := unauthenticated.JSON()
	if err != nil {
		t.Fatalf("Analyze JSON: %v", err)
	}
	var decodedUnauth Report
	if err := json.Unmarshal([]byte(unauthJSON), &decodedUnauth); err != nil {
		t.Fatalf("Unmarshal Analyze JSON: %v", err)
	}
	for i, item := range decodedUnauth.Items {
		if item.ClassifiedShare == nil {
			t.Fatalf("item %d missing ClassifiedShare", i)
		}
		rec := item.ClassifiedShare.Record()
		if rec.Class != cudaevidence.Unread || rec.Number != nil {
			t.Fatalf("item %d leaked numeric provenance without source: %+v", i, rec)
		}
	}

	// 2. AnalyzeWithEvidence preserves Derived share on observed kernel finding
	// while withholding numeric values on unmeasured findings.
	authenticated, err := AnalyzeWithEvidence(cap, capSource)
	if err != nil {
		t.Fatalf("AnalyzeWithEvidence: %v", err)
	}
	unmeasuredItem, err := (Item{
		Severity:                 SeverityContext,
		Category:                 CategoryArchitecture,
		EvidenceLevel:            EvidenceUnmeasured,
		Location:                 Location{Kernel: capSource.Identity.Entry},
		Finding:                  "Unmeasured counter audit",
		ActionableRecommendation: "Capture with hardware counters",
	}).WithShareEvidence(capSource, false, "")
	if err != nil {
		t.Fatalf("WithShareEvidence on unmeasured item: %v", err)
	}
	authenticated.Items = append(authenticated.Items, unmeasuredItem)

	authJSON, err := authenticated.JSON()
	if err != nil {
		t.Fatalf("authenticated JSON: %v", err)
	}
	var decodedAuth Report
	if err := json.Unmarshal([]byte(authJSON), &decodedAuth); err != nil {
		t.Fatalf("Unmarshal authenticated JSON: %v", err)
	}
	var foundObserved *Item
	for i := range decodedAuth.Items {
		if decodedAuth.Items[i].Finding == "consumer finding" {
			foundObserved = &decodedAuth.Items[i]
			break
		}
	}
	if foundObserved == nil || foundObserved.ClassifiedShare == nil {
		t.Fatalf("consumer finding missing from decoded report: %+v", decodedAuth.Items)
	}
	gotObserved := foundObserved.ClassifiedShare.Record()
	if gotObserved.Class != cudaevidence.Derived || gotObserved.Number == nil || *gotObserved.Number != 20 {
		t.Fatalf("observed item lost derived share: %+v", gotObserved)
	}
	gotUnmeasured := decodedAuth.Items[len(decodedAuth.Items)-1].ClassifiedShare.Record()
	if gotUnmeasured.Class != cudaevidence.Unread || gotUnmeasured.Number != nil {
		t.Fatalf("unmeasured item did not withhold number: %+v", gotUnmeasured)
	}

	// 3. Any attempt to attach or decode a numeric class on an EvidenceUnmeasured
	// finding must be refused.
	numVal := 20.0
	for _, class := range []cudaevidence.Class{
		cudaevidence.Measured,
		cudaevidence.Derived,
		cudaevidence.Inferred,
		cudaevidence.Modeled,
	} {
		t.Run("refuse_"+string(class)+"_on_unmeasured", func(t *testing.T) {
			unit := "percent"
			v, err := cudaevidence.New(cudaevidence.Record{
				Class:  class,
				Number: &numVal,
				Unit:   unit,
				Basis:  "basis",
				Source: source,
			})
			if err != nil {
				t.Fatal(err)
			}
			item := Item{
				Severity:                 SeverityContext,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceUnmeasured,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Unmeasured finding",
				ActionableRecommendation: "Recapture",
			}
			if _, err := item.WithEvidence(v); err == nil {
				t.Fatalf("WithEvidence accepted %q on EvidenceUnmeasured item", class)
			}
			item.ClassifiedShare = &v
			if _, err := (&Report{Items: []Item{item}}).JSON(); err == nil {
				t.Fatalf("Report.JSON accepted %q on EvidenceUnmeasured item", class)
			}
			rawJSON := `{
				"severity": "Context",
				"category": "Architecture",
				"evidence_level": "unmeasured",
				"location": {"kernel": "kernel"},
				"finding": "Unmeasured finding",
				"actionable_recommendation": "Recapture",
				"classified_share": {
					"class": "` + string(class) + `",
					"number": 20,
					"unit": "` + unit + `",
					"basis": "basis",
					"source": {
						"identity": {
							"artifact_digest": "` + source.Identity.ArtifactDigest + `",
							"entry": "kernel",
							"target": "sm_70",
							"compiler_key": "` + source.Identity.CompilerKey + `",
							"workload_key": "` + source.Identity.WorkloadKey + `"
						},
						"producer": "profile",
						"locator": "profile/1"
					}
				}
			}`
			var decoded Item
			if err := json.Unmarshal([]byte(rawJSON), &decoded); err == nil {
				t.Fatalf("json.Unmarshal accepted %q on EvidenceUnmeasured item", class)
			}
		})
	}
}

func TestAdviseEvidenceRefusesFabricatedMeasured(t *testing.T) {
	source := evidenceSource()
	n := 50.0
	measuredShare, err := cudaevidence.New(cudaevidence.Record{
		Class:  cudaevidence.Measured,
		Number: &n,
		Unit:   "percent",
		Basis:  "fabricated measured share ratio",
		Source: source,
	})
	if err != nil {
		t.Fatal(err)
	}
	derivedShare, err := cudaevidence.New(cudaevidence.Record{
		Class:  cudaevidence.Derived,
		Number: &n,
		Unit:   "percent",
		Basis:  "derived share ratio",
		Source: source,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		item    Item
		value   cudaevidence.Value
		wantErr string
	}{
		{
			name: "suspected_finding_promoted_to_measured",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryRegisterPressure,
				EvidenceLevel:            EvidenceSuspected,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Heuristic occupancy cliff",
				ActionableRecommendation: "A/B test",
			},
			value:   measuredShare,
			wantErr: "cannot claim measured evidence",
		},
		{
			name: "unmeasured_finding_promoted_to_measured",
			item: Item{
				Severity:                 SeverityContext,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceUnmeasured,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Skipped geometry audit",
				ActionableRecommendation: "Record SM count",
			},
			value:   measuredShare,
			wantErr: "unmeasured finding must withhold numeric value",
		},
		{
			name: "missing_evidence_level_promoted_to_measured",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryCompute,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Unclassified finding",
				ActionableRecommendation: "Inspect",
			},
			value:   measuredShare,
			wantErr: "unsupported evidence level",
		},
		{
			name: "profile_share_ratio_promoted_to_measured",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryMemory,
				EvidenceLevel:            EvidenceObserved,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Observed remark",
				ActionableRecommendation: "Align tensor",
			},
			value:   measuredShare,
			wantErr: "profile share ratio cannot be measured",
		},
		{
			name: "suspected_profile_share_promoted_to_derived",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceSuspected,
				Location:                 Location{Kernel: source.Identity.Entry},
				Finding:                  "Suspected fallback",
				ActionableRecommendation: "Vectorize",
			},
			value:   derivedShare,
			wantErr: "suspected profile share cannot be derived",
		},
		{
			name: "measured_kernel_identity_mismatch",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryMemory,
				EvidenceLevel:            EvidenceObserved,
				Location:                 Location{Kernel: "other_kernel"},
				Finding:                  "Observed remark",
				ActionableRecommendation: "Align tensor",
			},
			value:   measuredShare,
			wantErr: "kernel identity mismatch",
		},
		{
			name: "measured_ambiguous_module_location",
			item: Item{
				Severity:                 SeverityCost,
				Category:                 CategoryMemory,
				EvidenceLevel:            EvidenceObserved,
				Location:                 Location{Kernel: source.Identity.Entry, Module: "mod_b.cubin"},
				Finding:                  "Observed remark",
				ActionableRecommendation: "Align tensor",
			},
			value:   measuredShare,
			wantErr: "kernel identity mismatch",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.item.WithEvidence(tc.value); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("WithEvidence err = %v, want substring %q", err, tc.wantErr)
			}

			mutated := tc.item
			v := tc.value
			mutated.ClassifiedShare = &v
			if _, err := (&Report{Items: []Item{mutated}}).JSON(); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Report.JSON err = %v, want substring %q", err, tc.wantErr)
			}

			valJSON, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatal(err)
			}
			locJSON, err := json.Marshal(tc.item.Location)
			if err != nil {
				t.Fatal(err)
			}
			payload := `{
				"severity": "` + string(tc.item.Severity) + `",
				"category": "` + string(tc.item.Category) + `",
				"evidence_level": "` + string(tc.item.EvidenceLevel) + `",
				"location": ` + string(locJSON) + `,
				"finding": "` + tc.item.Finding + `",
				"actionable_recommendation": "` + tc.item.ActionableRecommendation + `",
				"classified_share": ` + string(valJSON) + `
			}`
			var decoded Report
			if err := json.Unmarshal([]byte(`{"items":[`+payload+`]}`), &decoded); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("json.Unmarshal err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}
