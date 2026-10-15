package advise

import (
	"encoding/binary"
	"encoding/json"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"hash/crc32"
	"math"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

// These are supplied CPU records, not hardware-origin evidence. The explicit
// source asserts the producer's artifact/compiler association; no label can
// make either those declarations or this test a native qualification.
func consumerCapture(n int64) (*cudacapture.Capture, cudaevidence.Source) {
	source := evidenceSource()
	source.Producer, source.Locator = "supplied-cpu-records", "consumer/receipt"
	b := journaltest.New().Named("kernel", "kernel", 100, uint64(100+n), 1).
		Named("kernel", "other", 1000, 1100, 2)
	cap := journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize(), "remarks.yaml": []byte(`--- !Failure
Pass: Memory
Name: RemarkMemoryTmaLoadFailed
Function: kernel
Args:
  - Remark: consumer finding
...
`)}, journaltest.CompleteStatus())
	cap.Metadata = json.RawMessage(`{"compute_capability":"sm_70","sm_count":80}`)
	cap.Identity = &cudacapture.ComparisonIdentity{Version: "v1", Complete: true, Phase: "fixture", ComputeCap: source.Identity.Target, WorkloadHash: source.Identity.WorkloadKey}
	cap.Profile.Sample = []*profile.Sample{
		{Value: []int64{n}, Label: map[string][]string{"name": {"kernel"}, "subtype": {"kernel"}}},
		{Value: []int64{100}, Label: map[string][]string{"name": {"other"}, "subtype": {"kernel"}}},
	}
	return cap, source
}

func consumerFinding(t *testing.T, cap *cudacapture.Capture, source *cudaevidence.Source) Item {
	t.Helper()
	var report *Report
	var err error
	if source == nil {
		report, err = Analyze(cap)
	} else {
		report, err = AnalyzeWithEvidence(cap, *source)
	}
	if err != nil {
		t.Fatal(err)
	}
	text, err := report.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var decoded Report
	if err := json.Unmarshal([]byte(text), &decoded); err != nil {
		t.Fatal(err)
	}
	var find func([]Item) *Item
	find = func(items []Item) *Item {
		for i := range items {
			if items[i].Finding == "consumer finding" {
				return &items[i]
			}
			if found := find(items[i].Patterns); found != nil {
				return found
			}
		}
		return nil
	}
	item := find(decoded.Items)
	if item == nil {
		t.Fatalf("consumer finding absent: %s", text)
	}
	if item.ClassifiedShare == nil {
		t.Fatalf("Analyze -> Report.JSON omitted classified share: %s", text)
	}
	return *item
}

func TestAnalyzeEvidenceConsumer(t *testing.T) {
	for _, n := range []int64{0, 25, 100} {
		cap, source := consumerCapture(n)
		item := consumerFinding(t, cap, &source)
		r := item.ClassifiedShare.Record()
		want := float64(n) / float64(n+100) * 100
		if r.Class != cudaevidence.Derived || r.Number == nil || math.Abs(*r.Number-want) > 1e-12 || r.Source != source || !strings.Contains(r.Basis, "journal/profile per-kernel duration totals agree") || !strings.Contains(r.Basis, "not hardware authentication") {
			t.Fatalf("n=%d: typed share=%+v, want %g", n, r, want)
		}
		if item.ClassifiedShare.CheckClaim(cudaevidence.Measured, source.Identity) == nil {
			t.Fatal("derived share promoted to measured")
		}
		unknown := consumerFinding(t, cap, nil).ClassifiedShare.Record()
		if unknown.Class != cudaevidence.Unread || unknown.Number != nil || !strings.Contains(unknown.Reason, "source identity unavailable") {
			t.Fatalf("normal Analyze fabricated provenance: %+v", unknown)
		}
	}
}

func TestAnalyzeEvidenceConsumerRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudacapture.Capture, *cudaevidence.Source)
		reason string
	}{
		{"target", func(c *cudacapture.Capture, s *cudaevidence.Source) { s.Identity.Target = "sm_90" }, "target"},
		{"workload", func(c *cudacapture.Capture, s *cudaevidence.Source) { s.Identity.WorkloadKey = strings.Repeat("4", 64) }, "workload"},
		{"entry", func(c *cudacapture.Capture, s *cudaevidence.Source) { s.Identity.Entry = "other" }, "unique supplied kernel"},
		{"partial", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"partial","dropped_records":0}}`)
		}, "completeness"},
		{"loss", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete","dropped_records":1}}`)
		}, "loss"},
		{"unknown loss", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete"}}`)
		}, "loss"},
		{"null loss", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete","dropped_records":null}}`)
		}, "loss"},
		{"duplicate status", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete","dropped_records":1,"dropped_records":0}}`)
		}, "duplicate"},
		{"profile mismatch", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Profile.Sample[0].Value[0]++ }, "populations differ"},
		{"kernel absent", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Profile.Sample = c.Profile.Sample[1:]
			c.Files["capture.journal"] = journaltest.New().Named("kernel", "other", 1000, 1100, 2).Finalize()
		}, "population unavailable"},
		{"journal loss", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			data := c.Files["capture.journal"]
			// Dropped is cumulative in every CDW1 header; change the last record so
			// the unchanged decoder reports real wire loss despite a zero status claim.
			offset := 0
			for offset < len(data) {
				next := offset + wire.HeaderSize + int(binary.BigEndian.Uint32(data[offset+8:offset+12]))
				if next == len(data) {
					binary.BigEndian.PutUint64(data[offset+52:offset+60], 1)
					binary.BigEndian.PutUint32(data[offset+68:offset+72], crc32.ChecksumIEEE(data[offset:offset+68]))
					break
				}
				offset = next
			}
		}, "lost records"},
		{"journal missing", func(c *cudacapture.Capture, s *cudaevidence.Source) { delete(c.Files, "capture.journal") }, "journal population unavailable"},
		{"journal incomplete", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture.journal"] = journaltest.New().Named("kernel", "kernel", 100, 125, 1).Unfinalized()
		}, "journal population unavailable"},
		{"unit", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Profile.SampleType[0].Unit = "cycles" }, "unit"},
		{"sample only", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Profile.SampleType[0].Type = "pc_samples" }, "sample counts"},
		{"negative", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Profile.Sample[0].Value[0] = -1 }, "invalid"},
		{"overflow", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Profile.Sample[0].Value[0] = math.MaxInt64 }, "overflows"},
		{"ambiguous channel", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Profile.SampleType = append(c.Profile.SampleType, &profile.ValueType{Type: "gpu_time", Unit: "ns"})
			for _, sample := range c.Profile.Sample {
				sample.Value = append(sample.Value, 0)
			}
		}, "ambiguous"},
		{"ambiguous entry", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Profile.Sample[0].Label["kernel"] = []string{"conflict"}
		}, "ambiguous"},
		{"image mismatch", func(c *cudacapture.Capture, s *cudaevidence.Source) { c.Code.CUBIN = "missing" }, "retained image"},
		{"invalid manifest", func(c *cudacapture.Capture, s *cudaevidence.Source) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete","dropped_records":0,"capture_plan_manifest":null}}`)
		}, "plan identity"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cap, source := consumerCapture(25)
			tc.change(cap, &source)
			r := consumerFinding(t, cap, &source).ClassifiedShare.Record()
			if r.Class != cudaevidence.Unread || r.Number != nil || !strings.Contains(r.Reason, tc.reason) {
				t.Fatalf("refused typed value = %+v, want %q", r, tc.reason)
			}
		})
	}
}

func TestAnalyzeEvidenceConsumerSourceValidation(t *testing.T) {
	cap, source := consumerCapture(25)
	for _, change := range []func(*cudaevidence.Source){
		func(s *cudaevidence.Source) { s.Identity.CompilerKey = "" },
		func(s *cudaevidence.Source) { s.Producer = "" },
		func(s *cudaevidence.Source) { s.Locator = "" },
	} {
		bad := source
		change(&bad)
		if _, err := AnalyzeWithEvidence(cap, bad); err == nil {
			t.Fatal("invalid explicit source accepted")
		}
	}
}

func TestAnalyzeEvidenceConsumerMalformedProfile(t *testing.T) {
	for _, change := range []func(*cudacapture.Capture){
		func(c *cudacapture.Capture) { c.Profile.Sample = append(c.Profile.Sample, nil) },
		func(c *cudacapture.Capture) { c.Profile.Sample[0].Value = nil },
		func(c *cudacapture.Capture) { c.Profile.SampleType[0] = nil },
	} {
		cap, source := consumerCapture(25)
		change(cap)
		if _, err := AnalyzeWithEvidence(cap, source); err == nil {
			t.Fatal("malformed explicit profile accepted")
		}
	}
}

func TestAnalyzeEvidenceConsumerInferred(t *testing.T) {
	cap, source := consumerCapture(25)
	cap.Files["kernels.sass"] = []byte(strings.ReplaceAll(sampleHighPressureSASS, "compute", "kernel"))
	report, err := AnalyzeWithEvidence(cap, source)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := report.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var decoded Report
	if err := json.Unmarshal([]byte(encoded), &decoded); err != nil {
		t.Fatal(err)
	}
	found := false
	var check func([]Item)
	check = func(items []Item) {
		for _, item := range items {
			if item.EvidenceLevel == EvidenceSuspected && item.Location.Kernel == "kernel" {
				found = true
				if item.ClassifiedShare == nil || item.ClassifiedShare.Record().Class != cudaevidence.Inferred {
					t.Fatalf("suspected cause promoted or lost: %+v", item)
				}
			}
			check(item.Patterns)
		}
	}
	check(decoded.Items)
	if !found {
		t.Fatal("source-derived suspected bottleneck absent")
	}
}
