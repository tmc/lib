package advise

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func addressFindings(items []Item) []Item {
	var found []Item
	for _, item := range items {
		if item.rule == "imad-wide" || item.rule == "ptx-mad-wide" || strings.HasPrefix(item.Finding, "Wide address arithmetic") || strings.HasPrefix(item.Finding, "Scalar address arithmetic fallback") {
			found = append(found, item)
		}
		found = append(found, addressFindings(item.Patterns)...)
	}
	return found
}

func TestWideAddressNormalMeasuredControl(t *testing.T) {
	cap, source := consumerCapture(1_000_000)
	item := consumerFinding(t, cap, &source)
	if item.NoRuntimeAttribution || item.ClassifiedShare.Record().Class != cudaevidence.Derived || item.DeviceTimeNanos == 0 {
		t.Fatalf("normal observed timing lost: %+v", item)
	}
}

func TestWideAddressPresenceHasNoCostAttribution(t *testing.T) {
	for _, tt := range []struct{ name, sass, ptx string }{
		{name: "unknown-memory-width", sass: "IMAD.WIDE R4, R1, 0x10, R2 ;"},
		{name: "already-vectorized-sass", sass: "IMAD.WIDE R4, R1, 0x10, R2 ;\n/*0010*/ LDG.E.128 R8, [R4] ;"},
		{name: "already-tma-sass", sass: "IMAD.WIDE R4, R1, 0x10, R2 ;\n/*0010*/ UTMALDG.2D [R2], [UR4] ;"},
		{name: "already-vectorized-ptx", ptx: "mad.wide.u32 %rd1, %r1, 16, %rd2;\nld.global.v4.u32 {%r2,%r3,%r4,%r5}, [%rd1];"},
		{name: "unknown-ptx", ptx: "mad.wide.u32 %rd1, %r1, 4, %rd2;"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, channel := range []string{"kernel_time", "pc_samples"} {
				cap, source := consumerCapture(1_000_000)
				source.Identity.Target = "sm_90a"
				cap.Identity.ComputeCap = source.Identity.Target
				cap.Metadata = json.RawMessage(`{"compute_capability":"sm_90a","sm_count":132}`)
				if channel == "pc_samples" {
					cap.Profile.SampleType = []*profile.ValueType{{Type: channel, Unit: "count"}}
				}
				if tt.sass != "" {
					cap.Code.SASS = "k.sass"
					cap.Files["k.sass"] = []byte("Function : kernel\n/*0000*/ " + tt.sass + "\n/*0020*/ EXIT ;\n")
				} else {
					cap.Code.PTX = "k.ptx"
					cap.Files["k.ptx"] = []byte(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry kernel() {\n" + tt.ptx + "\nret;\n}\n")
				}
				r, err := AnalyzeWithEvidence(cap, source)
				if err != nil {
					t.Fatal(err)
				}
				found := addressFindings(r.Items)
				if len(found) == 0 {
					t.Fatal("missing positive address occurrence")
				}
				for _, item := range found {
					assertNoRuntimeAttribution(t, item)
					if item.Severity != SeverityContext || item.EvidenceLevel != EvidenceSuspected {
						t.Fatalf("static occurrence promoted: %+v", item)
					}
					if strings.Contains(item.Finding, "fallback") || strings.Contains(item.ActionableRecommendation, "eliminate") {
						t.Fatalf("unproved mechanism: %+v", item)
					}
				}
				// A different rule on the same measured input must retain normal evidence.
				if channel == "kernel_time" && tt.sass != "" {
					admitted := false
					for _, other := range findings(r) {
						if !other.NoRuntimeAttribution && other.ClassifiedShare != nil && other.ClassifiedShare.Record().Class == cudaevidence.Derived {
							admitted = true
						}
					}
					if !admitted {
						t.Fatal("normal measured evidence control was not admitted")
					}
				}
				r.RankAgainst(testBaseline{})
				for _, item := range addressFindings(r.Items) {
					assertNoRuntimeAttribution(t, item)
					if gapSection(item) != gapNoTerm {
						t.Fatal("address presence acquired comparison term")
					}
				}
				raw, err := json.Marshal(r.Items)
				if err != nil {
					t.Fatal(err)
				}
				var round Report
				if err := json.Unmarshal(raw, &round.Items); err != nil {
					t.Fatal(err)
				}
				round.RankAgainst(testBaseline{})
				if len(addressFindings(round.Items)) == 0 {
					t.Fatal("serialized address finding missing")
				}
				for _, item := range addressFindings(round.Items) {
					assertNoRuntimeAttribution(t, item)
				}
			}
		})
	}
}
