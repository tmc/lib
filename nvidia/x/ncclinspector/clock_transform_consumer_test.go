package ncclinspector_test

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// This uses the existing independently identified, constructed rank streams.
// The calibration is a supplied host model, not measured synchronization.
func affineRankSpec(rank int) ncclinspector.ClockTransformSpec {
	return ncclinspector.ClockTransformSpec{
		Source:     ncclinspector.ClockIdentity{Host: fmt.Sprintf("constructed-node-r%d", rank), PID: 2100 + rank, Domain: ncclinspector.ClockDomainHostRealtime, BootEpoch: fmt.Sprintf("constructed-boot-r%d", rank), ProcessEpoch: fmt.Sprintf("constructed-process-r%d", rank)},
		Reference:  ncclinspector.ClockIdentity{Host: "constructed-reference-node", Domain: ncclinspector.ClockDomainHostMonotonic, BootEpoch: "constructed-reference-boot", ProcessEpoch: "constructed-reference-epoch-17"},
		SourceUnit: "nanoseconds", ReferenceUnit: "nanoseconds", Anchors: []ncclinspector.ClockAnchor{{SourceNS: 1_000_000, ReferenceMinNS: 999_950, ReferenceMaxNS: 1_000_050}}, RateMinPPB: 1_000_000_000, RateMaxPPB: 1_000_000_000, DriftKnown: true, UncertaintyKnown: true, ValidFromNS: 0, ValidUntilNS: 1_000_000_000, WallStepKnown: true, Method: "constructed exchange interval", VerificationRef: fmt.Sprintf("constructed-calibration-r%d.json", rank),
	}
}

func attributeAffine(t *testing.T, scenario string, change func(int, *ncclinspector.ClockTransformSpec), mutate func(int, *ncclinspector.ClockMapping), complete bool, compositionUncertainty ...int64) (ncclinspector.TwoRankDelayReport, error) {
	t.Helper()
	m := loadManifest(t)
	captures := capturesFor(t, scenario) // calls the public strict Inspector reader
	originals := make([][]byte, len(captures))
	for i, c := range captures {
		originals[i] = append([]byte(nil), c.Records[0].Raw...)
	}
	joined := joinCaptures(t, m, captures, m.Scope.Epoch) // public bounded cross-rank Join
	if len(joined.Collectives) != 1 {
		t.Fatal("joined collective absent")
	}
	group := joined.Collectives[0]
	attr := ncclinspector.NewTwoRankAttributor()
	for _, r := range group.Observations {
		rank := r.Header.Rank
		spec := affineRankSpec(rank)
		if change != nil {
			change(rank, &spec)
		}
		transform, err := ncclinspector.NewClockTransform(spec)
		if err != nil {
			return ncclinspector.TwoRankDelayReport{}, err
		}
		if len(compositionUncertainty) != 0 {
			next := spec
			next.Source = spec.Reference
			next.Reference = ncclinspector.ClockIdentity{Host: "constructed-final-reference", Domain: ncclinspector.ClockDomainHostMonotonic, BootEpoch: "final-boot", ProcessEpoch: "final-generation"}
			next.Anchors = []ncclinspector.ClockAnchor{{SourceNS: 1_000_000, ReferenceMinNS: 3_000_000, ReferenceMaxNS: 3_000_000}}
			next.RateMinPPB, next.RateMaxPPB = 2_000_000_000, 2_000_000_000
			next.DriftPPB = 0
			next.UncertaintyNS = compositionUncertainty[0]
			next.VerificationRef = "constructed-second-calibration.json"
			second, err := ncclinspector.NewClockTransform(next)
			if err != nil {
				return ncclinspector.TwoRankDelayReport{}, err
			}
			transform, err = ncclinspector.ComposeClockTransforms(transform, second, "constructed-composition.json")
			if err != nil {
				return ncclinspector.TwoRankDelayReport{}, err
			}
		}
		mapping := transform.Mapping()
		if mutate != nil {
			mutate(rank, &mapping)
		}
		record := ncclinspector.Record{Header: r.Header, Metadata: r.Metadata, Collective: &r.Collective, Raw: r.Raw}
		obs, err := ncclinspector.ObservationFromRecord(record, mapping)
		if err != nil {
			return ncclinspector.TwoRankDelayReport{}, err
		}
		if complete {
			completion, ok := completionFor(m, scenario, rank, m.Scope.Epoch)
			if !ok {
				t.Fatal("independent constructed completion absent")
			}
			// The manifest uses unit-rate, zero-offset source/reference completion
			// controls. Bind the raw source generation explicitly before alignment.
			completion.Domain = spec.Source.Domain
			completion.Epoch = spec.Source.ProcessEpoch
			completion.BootEpoch = spec.Source.BootEpoch
			completion.UncertaintyNS = 0
			obs, err = obs.WithCompletion(completion)
			if err != nil {
				return ncclinspector.TwoRankDelayReport{}, err
			}
		}
		if err := attr.AddObservation(obs); err != nil {
			return ncclinspector.TwoRankDelayReport{}, err
		}
	}
	for i, c := range captures {
		if !reflect.DeepEqual(c.Records[0].Raw, originals[i]) {
			t.Fatal("normalization changed raw Inspector integers")
		}
	}
	return attr.Attribute(group.Communicator, group.Sequence)
}

func TestClockTransformPublicOrdering(t *testing.T) {
	for _, tc := range []struct {
		name      string
		change    func(int, *ncclinspector.ClockTransformSpec)
		diagnosis ncclinspector.DelayDiagnosis
		admitted  bool
		delay     int64
	}{
		{name: "separated", diagnosis: ncclinspector.DiagnosisHostEnqueueSkew, admitted: true, delay: 1000},
		{name: "overlap", change: func(_ int, s *ncclinspector.ClockTransformSpec) {
			s.Anchors[0].ReferenceMinNS = 999_000
			s.Anchors[0].ReferenceMaxNS = 1_001_000
		}, diagnosis: ncclinspector.DiagnosisOrderIndeterminate},
		{name: "asymmetric exchange", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.Anchors[0].ReferenceMinNS = 998_000
				s.Anchors[0].ReferenceMaxNS = 1_001_000
			}
		}, diagnosis: ncclinspector.DiagnosisOrderIndeterminate},
		{name: "drift", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.DriftPPB = 2_000_000_000
			}
		}, diagnosis: ncclinspector.DiagnosisOrderIndeterminate},
		{name: "rate uncertainty", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.RateMinPPB = 100_000_000
				s.RateMaxPPB = 1_900_000_000
			}
		}, diagnosis: ncclinspector.DiagnosisOrderIndeterminate},
		{name: "nonunit rate", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.RateMinPPB = 2_000_000_000
				s.RateMaxPPB = 2_000_000_000
			}
		}, diagnosis: ncclinspector.DiagnosisHostEnqueueSkew, admitted: true, delay: 2000},
		{name: "reference boot", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.Reference.BootEpoch = "reference-reboot"
			}
		}, diagnosis: ncclinspector.DiagnosisIncompatibleClocks},
		{name: "reference process", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.Reference.ProcessEpoch = "reference-restarted"
			}
		}, diagnosis: ncclinspector.DiagnosisIncompatibleClocks},
		{name: "reference host", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.Reference.Host = "another-reference"
			}
		}, diagnosis: ncclinspector.DiagnosisIncompatibleClocks},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := attributeAffine(t, "delayed", tc.change, nil, false)
			if err != nil || r.Admitted != tc.admitted || r.Diagnosis != tc.diagnosis || tc.admitted && r.DelayNS != tc.delay {
				t.Fatalf("report %+v err=%v", r, err)
			}
		})
	}
	normal, err := attributeAffine(t, "normal", nil, nil, true)
	if err != nil || !normal.Admitted || normal.Diagnosis != ncclinspector.DiagnosisNormal {
		t.Fatalf("explicit completion normal control: %+v %v", normal, err)
	}
	partial, err := attributeAffine(t, "normal", nil, nil, false)
	if err != nil || partial.Admitted || partial.Diagnosis != ncclinspector.DiagnosisPartialRank {
		t.Fatalf("record publication became completion: %+v %v", partial, err)
	}
}

func TestClockTransformPublicRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(int, *ncclinspector.ClockTransformSpec)
		mutate func(int, *ncclinspector.ClockMapping)
	}{
		{name: "units", change: func(_ int, s *ncclinspector.ClockTransformSpec) { s.SourceUnit = "microseconds" }},
		{name: "wall step", change: func(_ int, s *ncclinspector.ClockTransformSpec) { s.WallStepObserved = true }},
		{name: "wall step unknown", change: func(_ int, s *ncclinspector.ClockTransformSpec) { s.WallStepKnown = false }},
		{name: "anchors", change: func(_ int, s *ncclinspector.ClockTransformSpec) { s.Anchors = nil }},
		{name: "expired", change: func(rank int, s *ncclinspector.ClockTransformSpec) {
			if rank == 1 {
				s.ValidUntilNS = 1_300_999
			}
		}},
		{name: "source boot", mutate: func(_ int, m *ncclinspector.ClockMapping) { m.SourceBootEpoch = "rebooted" }},
		{name: "source process", mutate: func(_ int, m *ncclinspector.ClockMapping) { m.SourceProcessEpoch = "restarted" }},
		{name: "source domain", mutate: func(_ int, m *ncclinspector.ClockMapping) { m.Domain = ncclinspector.ClockDomainGPUDevice }},
		{name: "offset with affine", mutate: func(_ int, m *ncclinspector.ClockMapping) { m.OffsetNS = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := attributeAffine(t, "delayed", tc.change, tc.mutate, false)
			if !errors.Is(err, ncclinspector.ErrIncompatibleClocks) || r.Admitted {
				t.Fatalf("invalid mapping reached admitted consumer: %+v %v", r, err)
			}
		})
	}
}

func TestClockTransformPublicComposition(t *testing.T) {
	for _, tc := range []struct {
		name      string
		unc       int64
		admitted  bool
		diagnosis ncclinspector.DelayDiagnosis
	}{
		{"separated", 25, true, ncclinspector.DiagnosisHostEnqueueSkew},
		{"composed overlap", 1000, false, ncclinspector.DiagnosisOrderIndeterminate},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := attributeAffine(t, "delayed", nil, nil, false, tc.unc)
			if err != nil || r.Admitted != tc.admitted || r.Diagnosis != tc.diagnosis || r.DelayNS != 2000 {
				t.Fatalf("composition did not reach attribution: %+v %v", r, err)
			}
		})
	}
}

func TestClockTransformPublicRetainedLineage(t *testing.T) {
	for _, name := range []string{"PID mutation", "timestamp mutation", "unmapped completion", "wrong completion epoch"} {
		t.Run(name, func(t *testing.T) {
			m := loadManifest(t)
			joined := joinCaptures(t, m, capturesFor(t, "delayed"), m.Scope.Epoch)
			group := joined.Collectives[0]
			attr := ncclinspector.NewTwoRankAttributor()
			for _, r := range group.Observations {
				spec := affineRankSpec(r.Header.Rank)
				transform, err := ncclinspector.NewClockTransform(spec)
				if err != nil {
					t.Fatal(err)
				}
				record := ncclinspector.Record{Header: r.Header, Metadata: r.Metadata, Collective: &r.Collective, Raw: r.Raw}
				obs, err := ncclinspector.ObservationFromRecord(record, transform.Mapping())
				if err != nil {
					t.Fatal(err)
				}
				if r.Header.Rank == 1 {
					switch name {
					case "PID mutation":
						obs.HostPID++
					case "timestamp mutation":
						obs.DeviceStart.UncertaintyNS = 0
					case "unmapped completion":
						obs.HasCompletion = true
						obs.Completion = obs.DeviceEnd
					case "wrong completion epoch":
						_, err := obs.WithCompletion(ncclinspector.BoundedTimestamp{Nanoseconds: 1_400_000, Domain: spec.Source.Domain, Epoch: "restarted", BootEpoch: spec.Source.BootEpoch})
						if !errors.Is(err, ncclinspector.ErrIncompatibleClocks) {
							t.Fatal("completion epoch mismatch admitted")
						}
						return
					}
				}
				if err := attr.AddObservation(obs); err != nil {
					t.Fatal(err)
				}
			}
			report, err := attr.Attribute(group.Communicator, group.Sequence)
			if err != nil || report.Admitted || report.Diagnosis != ncclinspector.DiagnosisIncompatibleClocks {
				t.Fatalf("changed mapped lineage admitted: %+v %v", report, err)
			}
		})
	}
}

func TestClockTransformInspectorPIDBinding(t *testing.T) {
	m := loadManifest(t)
	captures := capturesFor(t, "delayed")
	joined := joinCaptures(t, m, captures, m.Scope.Epoch)
	if len(joined.Collectives) != 1 {
		t.Fatal("joined collective absent")
	}
	for _, r := range joined.Collectives[0].Observations {
		for _, tc := range []struct {
			name      string
			pid       int
			recordPID int
			wantError bool
		}{
			{"matched", r.Metadata.PID, r.Metadata.PID, false},
			{"unknown calibration PID", 0, r.Metadata.PID, true},
			{"different calibration PID", r.Metadata.PID + 1, r.Metadata.PID, true},
			{"raw PID restart", r.Metadata.PID, r.Metadata.PID + 1, true},
		} {
			t.Run(fmt.Sprintf("rank%d/%s", r.Header.Rank, tc.name), func(t *testing.T) {
				spec := affineRankSpec(r.Header.Rank)
				spec.Source.PID = tc.pid
				// Reference PID zero is an explicit generic provider, not a process.
				if spec.Reference.PID != 0 {
					t.Fatal("reference unexpectedly process-bound")
				}
				transform, err := ncclinspector.NewClockTransform(spec)
				if err != nil {
					t.Fatal(err)
				}
				record := ncclinspector.Record{Header: r.Header, Metadata: r.Metadata, Collective: &r.Collective, Raw: r.Raw}
				record.Metadata.PID = tc.recordPID
				obs, err := ncclinspector.ObservationFromRecord(record, transform.Mapping())
				if tc.wantError {
					if !errors.Is(err, ncclinspector.ErrIncompatibleClocks) {
						t.Fatalf("mismatched source PID accepted: observation=%+v error=%v", obs, err)
					}
					return
				}
				if err != nil || obs.HostPID != tc.recordPID {
					t.Fatalf("matching source PID rejected: observation=%+v error=%v", obs, err)
				}
				if got := transform.Steps()[0].Source; got.BootEpoch != spec.Source.BootEpoch || got.ProcessEpoch != spec.Source.ProcessEpoch {
					t.Fatal("PID binding changed independently supplied epochs")
				}
			})
		}
	}
}
