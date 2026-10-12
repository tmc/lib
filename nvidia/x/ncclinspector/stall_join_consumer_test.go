package ncclinspector_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
)

const fixtureDir = "testdata/two-rank-stall"

type fixtureManifest struct {
	SchemaVersion int                 `json:"schema_version"`
	Provenance    string              `json:"provenance"`
	Authentic     bool                `json:"authentic_capture"`
	Hash          string              `json:"communicator_hash"`
	Operation     string              `json:"operation"`
	Sequence      uint64              `json:"sequence"`
	Bytes         uint64              `json:"bytes"`
	Scope         fixtureScope        `json:"scope"`
	ExpectedRanks []int               `json:"expected_ranks"`
	RankFiles     map[string][]string `json:"rank_files"`
	Clock         fixtureClock        `json:"clock_control"`
	Completion    fixtureCompletion   `json:"completion_control"`
	Reports       map[string]struct {
		Diagnosis   string `json:"diagnosis"`
		Admitted    bool   `json:"admitted"`
		DelayedRank int    `json:"delayed_rank"`
	} `json:"expected_reports"`
	Hashes map[string]string `json:"sha256"`
}

type fixtureScope struct {
	Capture string `json:"capture"`
	Job     string `json:"job"`
	Epoch   string `json:"epoch"`
}

type fixtureClock struct {
	FactClass           string   `json:"fact_class"`
	SourceDomain        string   `json:"source_domain"`
	ReferenceDomain     string   `json:"reference_domain"`
	ReferenceEpoch      string   `json:"reference_epoch"`
	EvidenceRef         string   `json:"evidence_ref"`
	SourceBootEpochs    []string `json:"source_boot_epochs"`
	SourceProcessEpochs []string `json:"source_process_epochs"`
	OffsetNS            []int64  `json:"offset_ns"`
	UncertaintyNS       []int64  `json:"uncertainty_ns"`
	ValidFromNS         int64    `json:"valid_from_ns"`
	ValidUntilNS        int64    `json:"valid_until_ns"`
}

type fixtureCompletion struct {
	FactClass string `json:"fact_class"`
	Facts     []struct {
		Scenario string `json:"scenario"`
		Rank     int    `json:"rank"`
		Epoch    string `json:"epoch"`
		TimeNS   int64  `json:"completion_ns"`
	} `json:"facts"`
}

func loadManifest(t *testing.T) fixtureManifest {
	t.Helper()
	m, err := readManifest()
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range m.Hashes {
		b, err := os.ReadFile(filepath.Join(fixtureDir, name))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("fixture hash %s: got %s want %s", name, got, want)
		}
	}
	return m
}

func readManifest() (fixtureManifest, error) {
	b, err := os.ReadFile(filepath.Join(fixtureDir, "manifest.json"))
	if err != nil {
		return fixtureManifest{}, err
	}
	var m fixtureManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return fixtureManifest{}, fmt.Errorf("decode fixture manifest: %w", err)
	}
	if m.SchemaVersion != 1 || m.Provenance != "constructed_host_control" || m.Authentic || m.Clock.FactClass != "constructed_independent_control_fact" || m.Completion.FactClass != "constructed_independent_control_fact" {
		return fixtureManifest{}, errors.New("fixture provenance is not explicitly constructed")
	}
	return m, nil
}

func readCapture(t *testing.T, id, file string, rank int) nccljoin.Capture {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir, file))
	if err != nil {
		t.Fatal(err)
	}
	records, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
	if err != nil {
		t.Fatalf("read %s: %v", file, err)
	}
	if len(records) != 1 || records[0].Header.Rank != rank {
		t.Fatalf("%s is not the declared rank %d record: %+v", file, rank, records)
	}
	return nccljoin.Capture{ID: id, Records: records}
}

func capturesFor(t *testing.T, scenario string) []nccljoin.Capture {
	t.Helper()
	m := loadManifest(t)
	files, ok := m.RankFiles[scenario]
	if !ok || len(files) != 2 {
		t.Fatalf("missing two-rank file set for %q", scenario)
	}
	return []nccljoin.Capture{
		readCapture(t, scenario+"-rank0", files[0], 0),
		readCapture(t, scenario+"-rank1", files[1], 1),
	}
}

func joinCaptures(t *testing.T, m fixtureManifest, captures []nccljoin.Capture, epoch string) *nccljoin.Result {
	t.Helper()
	joined, err := nccljoin.Join(captures, nccljoin.Options{
		Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: epoch},
		Expected: map[string][]int{m.Hash: append([]int(nil), m.ExpectedRanks...)},
	})
	if err != nil {
		t.Fatal(err)
	}
	return joined
}

func clockFor(m fixtureManifest, rank int) ncclinspector.ClockMapping {
	return ncclinspector.ClockMapping{
		Domain:             ncclinspector.ClockDomain(m.Clock.SourceDomain),
		SourceBootEpoch:    m.Clock.SourceBootEpochs[rank],
		SourceProcessEpoch: m.Clock.SourceProcessEpochs[rank],
		CalibrationKnown:   true,
		UncertaintyKnown:   true,
		ValidityKnown:      true,
		ValidFromNS:        m.Clock.ValidFromNS,
		ValidUntilNS:       m.Clock.ValidUntilNS,
		OffsetNS:           m.Clock.OffsetNS[rank],
		UncertaintyNS:      m.Clock.UncertaintyNS[rank],
		ReferenceDomain:    ncclinspector.ClockDomain(m.Clock.ReferenceDomain),
		ReferenceEpoch:     m.Clock.ReferenceEpoch,
		EvidenceRef:        m.Clock.EvidenceRef,
	}
}

func completionFor(m fixtureManifest, scenario string, rank int, epoch string) (ncclinspector.BoundedTimestamp, bool) {
	for _, fact := range m.Completion.Facts {
		if fact.Scenario == scenario && fact.Rank == rank && fact.Epoch == epoch {
			return ncclinspector.BoundedTimestamp{
				Nanoseconds:   fact.TimeNS,
				UncertaintyNS: m.Clock.UncertaintyNS[rank],
				Domain:        ncclinspector.ClockDomain(m.Clock.ReferenceDomain),
				Epoch:         m.Clock.ReferenceEpoch,
			}, true
		}
	}
	return ncclinspector.BoundedTimestamp{}, false
}

func consume(t *testing.T, m fixtureManifest, scenario, epoch string, captures []nccljoin.Capture, complete bool, uncertainty []int64) (*nccljoin.Result, ncclinspector.TwoRankDelayReport, error) {
	t.Helper()
	joined := joinCaptures(t, m, captures, epoch)
	if len(joined.Collectives) != 1 {
		t.Fatalf("join produced %d collectives", len(joined.Collectives))
	}
	group := joined.Collectives[0]
	if group.Communicator != m.Hash || group.Sequence != m.Sequence || !reflect.DeepEqual(group.Expected, []int{0, 1}) || !reflect.DeepEqual(group.Present, []int{0, 1}) || len(group.Unobserved) != 0 {
		t.Fatalf("joined identity or membership differs: %+v", group)
	}
	if len(group.Observations) != 2 || group.Observations[0].Capture == group.Observations[1].Capture {
		t.Fatalf("join did not retain distinct rank captures: %+v", group.Observations)
	}
	a := ncclinspector.NewTwoRankAttributor()
	for _, joinedObservation := range group.Observations {
		record := ncclinspector.Record{
			Header:     joinedObservation.Header,
			Metadata:   joinedObservation.Metadata,
			Collective: &joinedObservation.Collective,
			Raw:        append([]byte(nil), joinedObservation.Raw...),
		}
		clock := clockFor(m, record.Header.Rank)
		if uncertainty != nil {
			clock.UncertaintyNS = uncertainty[record.Header.Rank]
		}
		obs, err := ncclinspector.ObservationFromRecord(record, clock)
		if err != nil {
			return joined, ncclinspector.TwoRankDelayReport{}, err
		}
		if complete {
			if completion, ok := completionFor(m, scenario, record.Header.Rank, epoch); ok {
				if uncertainty != nil {
					completion.UncertaintyNS = uncertainty[record.Header.Rank]
				}
				obs.Completion, obs.HasCompletion = completion, true
			} else {
				t.Fatalf("manifest lacks completion control for %s rank %d", scenario, record.Header.Rank)
			}
		}
		if err := a.AddObservation(obs); err != nil {
			return joined, ncclinspector.TwoRankDelayReport{}, err
		}
	}
	report, err := a.Attribute(group.Communicator, group.Sequence)
	return joined, report, err
}

func TestJoinedRankConsumerConstructedControls(t *testing.T) {
	m := loadManifest(t)
	for _, scenario := range []string{"normal", "delayed"} {
		t.Run(scenario, func(t *testing.T) {
			captures := capturesFor(t, scenario)
			joined, report, err := consume(t, m, scenario, m.Scope.Epoch, captures, true, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := m.Reports[scenario]
			if string(report.Diagnosis) != want.Diagnosis || report.Admitted != want.Admitted || report.DelayedRank != want.DelayedRank {
				t.Fatalf("report = %+v, want %+v", report, want)
			}
			for _, observation := range joined.Collectives[0].Observations {
				if len(observation.Raw) == 0 {
					t.Fatalf("raw rank record lost after join: %+v", observation)
				}
				original, err := os.ReadFile(filepath.Join(fixtureDir, m.RankFiles[scenario][observation.Header.Rank]))
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(observation.Raw, original) {
					t.Fatalf("raw rank record changed after join for rank %d", observation.Header.Rank)
				}
				parsed, err := ncclinspector.Read(bytes.NewReader(observation.Raw), ncclinspector.Limits{})
				if err != nil || len(parsed) != 1 || parsed[0].Header.Rank != observation.Header.Rank {
					t.Fatalf("retained raw record does not match joined rank %d: %v", observation.Header.Rank, err)
				}
			}
			if report.TransportObserved {
				t.Fatal("transport was reported without a transport observation")
			}
		})
	}
}

func TestJoinedRankConsumerCausalNegatives(t *testing.T) {
	m := loadManifest(t)
	t.Run("missing rank", func(t *testing.T) {
		captures := capturesFor(t, "normal")[:1]
		joined := joinCaptures(t, m, captures, m.Scope.Epoch)
		group := joined.Collectives[0]
		if !reflect.DeepEqual(group.Present, []int{0}) || !reflect.DeepEqual(group.Unobserved, []int{1}) {
			t.Fatalf("partial join membership: %+v", group)
		}
		a := ncclinspector.NewTwoRankAttributor()
		o := group.Observations[0]
		record := ncclinspector.Record{Header: o.Header, Metadata: o.Metadata, Collective: &o.Collective, Raw: o.Raw}
		obs, err := ncclinspector.ObservationFromRecord(record, clockFor(m, 0))
		if err != nil {
			t.Fatal(err)
		}
		if err := a.AddObservation(obs); err != nil {
			t.Fatal(err)
		}
		report, err := a.Attribute(group.Communicator, group.Sequence)
		if err != nil || report.Diagnosis != ncclinspector.DiagnosisPartialRank || report.Admitted || !reflect.DeepEqual(report.MissingRanks, []int{1}) {
			t.Fatalf("missing-rank report: %+v %v", report, err)
		}
	})
	t.Run("missing completion remains partial", func(t *testing.T) {
		_, report, err := consume(t, m, "normal", m.Scope.Epoch, capturesFor(t, "normal"), false, nil)
		if err != nil || report.Diagnosis != ncclinspector.DiagnosisPartialRank || report.Admitted {
			t.Fatalf("completion absence was promoted: %+v %v", report, err)
		}
		for _, capture := range capturesFor(t, "normal") {
			if capture.Records[0].Collective.EventTraceTS.KernelEvents[0].KernelRecordTS == 0 {
				t.Fatal("fixture unexpectedly lacks a kernel record timestamp")
			}
		}
	})
	t.Run("rank to file binding", func(t *testing.T) {
		if _, err := func() (nccljoin.Capture, error) {
			b, err := os.ReadFile(filepath.Join(fixtureDir, "rank1.normal.jsonl"))
			if err != nil {
				return nccljoin.Capture{}, err
			}
			records, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
			if err != nil {
				return nccljoin.Capture{}, err
			}
			if len(records) != 1 || records[0].Header.Rank != 0 {
				return nccljoin.Capture{}, errors.New("rank-to-file binding mismatch")
			}
			return nccljoin.Capture{ID: "swapped-rank1", Records: records}, nil
		}(); err == nil {
			t.Fatal("swapped rank file was admitted")
		}
	})
	t.Run("truncated communicator hash", func(t *testing.T) {
		captures := capturesFor(t, "normal")
		for i := range captures {
			captures[i].Records[0].Header.ID = "0x1234567890abcd"
			captures[i].Records[0].Raw = nil
		}
		joined, err := nccljoin.Join(captures, nccljoin.Options{
			Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: m.Scope.Epoch},
			Expected: map[string][]int{"0x1234567890abcd": {0, 1}},
		})
		if err != nil {
			t.Fatal(err)
		}
		observation := joined.Collectives[0].Observations[0]
		record := ncclinspector.Record{Header: observation.Header, Metadata: observation.Metadata, Collective: &observation.Collective}
		if _, err := ncclinspector.ObservationFromRecord(record, clockFor(m, 0)); !errors.Is(err, ncclinspector.ErrTruncatedHash) {
			t.Fatalf("truncated hash conversion error = %v", err)
		}
	})
	t.Run("wrong membership", func(t *testing.T) {
		captures := capturesFor(t, "normal")
		captures[0].Records[0].Header.Ranks = 3
		captures[0].Records[0].Raw = nil
		_, err := nccljoin.Join(captures, nccljoin.Options{
			Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: m.Scope.Epoch},
			Expected: map[string][]int{m.Hash: {0, 1}},
		})
		if err == nil || !bytes.Contains([]byte(err.Error()), []byte("expected membership differs")) {
			t.Fatalf("wrong membership error = %v", err)
		}
	})
	t.Run("clock domain mismatch", func(t *testing.T) {
		joined := joinCaptures(t, m, capturesFor(t, "normal"), m.Scope.Epoch)
		a := ncclinspector.NewTwoRankAttributor()
		for _, o := range joined.Collectives[0].Observations {
			clock := ncclinspector.ClockMapping{Domain: ncclinspector.ClockDomainHostRealtime, UncertaintyNS: 50}
			if o.Header.Rank == 1 {
				clock = ncclinspector.ClockMapping{Domain: ncclinspector.ClockDomainHostMonotonic, UncertaintyNS: 50}
			}
			record := ncclinspector.Record{Header: o.Header, Metadata: o.Metadata, Collective: &o.Collective}
			obs, err := ncclinspector.ObservationFromRecord(record, clock)
			if err != nil {
				t.Fatal(err)
			}
			if err := a.AddObservation(obs); err != nil {
				t.Fatal(err)
			}
		}
		report, err := a.Attribute(m.Hash, m.Sequence)
		if err != nil || report.Diagnosis != ncclinspector.DiagnosisIncompatibleClocks || report.Admitted {
			t.Fatalf("clock mismatch report: %+v %v", report, err)
		}
	})
	t.Run("clock uncertainty withholds order", func(t *testing.T) {
		_, report, err := consume(t, m, "delayed", m.Scope.Epoch, capturesFor(t, "delayed"), true, []int64{600, 600})
		if err != nil || report.Diagnosis != ncclinspector.DiagnosisOrderIndeterminate || report.Admitted || report.DelayedRank != -1 {
			t.Fatalf("uncertain order report: %+v %v", report, err)
		}
	})
	t.Run("rank restart is rejected", func(t *testing.T) {
		captures := capturesFor(t, "normal")
		restarted := captures[0]
		restarted.ID = "normal-rank0-restarted"
		restarted.Records = append([]ncclinspector.Record(nil), captures[0].Records...)
		restarted.Records[0].Metadata.PID++
		restarted.Records[0].Raw = nil
		_, err := nccljoin.Join(append(captures, restarted), nccljoin.Options{
			Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: m.Scope.Epoch},
			Expected: map[string][]int{m.Hash: {0, 1}},
		})
		if err == nil || !bytes.Contains([]byte(err.Error()), []byte("conflicting rank process identity")) {
			t.Fatalf("restart error = %v", err)
		}
	})
	t.Run("epoch reuse gets a fresh attributor", func(t *testing.T) {
		for _, epoch := range []string{m.Scope.Epoch, "constructed-communicator-epoch-18"} {
			_, report, err := consume(t, m, "normal", epoch, capturesFor(t, "normal"), true, nil)
			if err != nil || report.Diagnosis != ncclinspector.DiagnosisNormal || !report.Admitted {
				t.Fatalf("epoch %s reused stale state: %+v %v", epoch, report, err)
			}
		}
	})
	t.Run("same epoch duplicate is rejected", func(t *testing.T) {
		m := loadManifest(t)
		captures := capturesFor(t, "normal")
		joined := joinCaptures(t, m, captures, m.Scope.Epoch)
		a := ncclinspector.NewTwoRankAttributor()
		for _, o := range joined.Collectives[0].Observations {
			record := ncclinspector.Record{Header: o.Header, Metadata: o.Metadata, Collective: &o.Collective}
			obs, err := ncclinspector.ObservationFromRecord(record, clockFor(m, o.Header.Rank))
			if err != nil {
				t.Fatal(err)
			}
			if c, ok := completionFor(m, "normal", o.Header.Rank, m.Scope.Epoch); ok {
				obs.Completion, obs.HasCompletion = c, true
			}
			if err := a.AddObservation(obs); err != nil {
				t.Fatal(err)
			}
		}
		first := joined.Collectives[0].Observations[0]
		record := ncclinspector.Record{Header: first.Header, Metadata: first.Metadata, Collective: &first.Collective}
		obs, err := ncclinspector.ObservationFromRecord(record, clockFor(m, first.Header.Rank))
		if err != nil {
			t.Fatal(err)
		}
		if err := a.AddObservation(obs); !errors.Is(err, ncclinspector.ErrDuplicateSequence) {
			t.Fatalf("same-epoch duplicate error = %v", err)
		}
	})
}

func TestJoinedConsumerFixtureReportsAreDeterministic(t *testing.T) {
	m := loadManifest(t)
	_, a, err := consume(t, m, "delayed", m.Scope.Epoch, capturesFor(t, "delayed"), true, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, b, err := consume(t, m, "delayed", m.Scope.Epoch, capturesFor(t, "delayed"), true, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same static evidence produced different reports:\nfirst:  %+v\nsecond: %+v", a, b)
	}
}
