package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const accountingCanonicalStart int64 = 3600000000000
const accountingCanonicalEnd int64 = 10800000000000

func accountingCanonicalInput(t *testing.T) nvtelemetry.AccountingInput {
	t.Helper()
	in := accountingFixture(t)
	in.Energy = nil
	return in
}

func accountingCanonicalRecords(version int, signed bool, base uint64) (nvtelemetry.Manifest, []nvtelemetry.Sample) {
	m := nvtelemetry.Manifest{SchemaVersion: version, RunID: "canonical-energy-control", ClockMonotonicDomain: "process_local", ClockMonotonicEpochID: "fixture-epoch", HasClockCalibration: true, ClockMappingValidUntil: accountingCanonicalEnd, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-A"}}}
	var samples []nvtelemetry.Sample
	for i, at := range []int64{accountingCanonicalStart, accountingCanonicalEnd} {
		value := base + uint64(i)*200000
		r := nvtelemetry.Reading{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, DeviceUUID: "GPU-A", Backend: "synthetic-counter", Evidence: nvtelemetry.EvidenceMeasured, HasSourceTimestamp: true, SourceTimestamp: at, SourceTimestampDomain: "process_local", TimestampMonotonic: at}
		if version < nvtelemetry.SchemaVersionV3 {
			r.IsInt, r.IntValue = true, value
		} else {
			r.Observation = &nvtelemetry.Observation{Family: "energy", Kind: "counter", CounterEpoch: "counter-1", ClockEpoch: "fixture-epoch", SourceClass: "synthetic"}
			if version >= nvtelemetry.SchemaVersionV4 {
				r.Observation.Observer.SamplingCoverage = &nvtelemetry.SamplingCoverage{Status: "not_applicable"}
			}
			if signed {
				n := int64(value)
				r.Number = &nvtelemetry.Number{Int: &n}
			} else {
				r.Number = &nvtelemetry.Number{Uint: &value}
			}
		}
		samples = append(samples, nvtelemetry.Sample{Seq: uint64(i + 1), QueryStartNanos: at, QueryEndNanos: at, Readings: []nvtelemetry.Reading{r}})
	}
	return m, samples
}

func accountingCanonicalWire(t *testing.T, m nvtelemetry.Manifest, samples []nvtelemetry.Sample, complete bool) []byte {
	t.Helper()
	var wire bytes.Buffer
	w := nvtelemetry.NewWriter(&wire, nvtelemetry.CollectorConfig{})
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	for i := range samples {
		if err := w.WriteSample(&samples[i]); err != nil {
			t.Fatal(err)
		}
	}
	if complete {
		m.EndTimeMonotonic = accountingCanonicalEnd
		m.TotalSamples = uint64(len(samples))
		m.IsComplete = true
		if m.SchemaVersion >= nvtelemetry.SchemaVersionV3 {
			m.LostRecordsByFamily = w.FamilyLoss()
		}
		if err := w.WriteManifest(&m); err != nil {
			t.Fatal(err)
		}
	}
	return wire.Bytes()
}

func TestAccountingCanonicalStream(t *testing.T) {
	for _, version := range []int{2, 3, 4, 5} {
		for _, signed := range []bool{false, true} {
			if version == 2 && signed {
				continue
			}
			for _, base := range []uint64{0, 1<<53 + 1} {
				t.Run(fmt.Sprintf("v%d/signed=%t/base=%d", version, signed, base), func(t *testing.T) {
					m, samples := accountingCanonicalRecords(version, signed, base)
					raw := accountingCanonicalWire(t, m, samples, true)
					_, _, decoded, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 1<<20)
					if err != nil {
						t.Fatalf("strict reader: %v", err)
					}
					if version >= 3 {
						for i, s := range decoded {
							r := s.Readings[0]
							if r.IsInt || r.IntValue != 0 || r.Value != 0 {
								t.Fatal("canonical stream acquired legacy scalar")
							}
							want := base + uint64(i)*200000
							if signed {
								if r.Number.Int == nil || *r.Number.Int != int64(want) {
									t.Fatal("signed counter lost precision")
								}
							} else if r.Number.Uint == nil || *r.Number.Uint != want {
								t.Fatal("unsigned counter lost precision")
							}
						}
					}
					_, energy := nvtelemetry.AggregateWindow("measurement", accountingCanonicalStart, accountingCanonicalEnd, decoded, nvtelemetry.WindowOptions{})
					if len(energy) != 1 || !energy[0].Valid || energy[0].EnergyConsumedMJ != 200000 {
						t.Fatalf("window counterexample precondition: %+v", energy)
					}
					report, err := nvtelemetry.AccountTelemetry(bytes.NewReader(raw), accountingCanonicalInput(t), "measurement", accountingCanonicalStart, accountingCanonicalEnd, 1<<20)
					if err != nil {
						t.Fatalf("public canonical accounting: %v", err)
					}
					if report.PhysicalEnergyMillijoules != "200000" || report.ResidualEnergyMillijoules != "75000" || report.Shares[0].CostMajor != "3/2" {
						t.Fatalf("wrong accounting: %+v", report)
					}
					if len(report.Input.Energy) != 1 || report.Input.Energy[0].RawDigest != fmt.Sprintf("%x", sha256.Sum256(raw)) || report.Input.Energy[0].Summary.StartEnergyMJ != base || report.Input.Energy[0].Summary.EndEnergyMJ != base+200000 {
						t.Fatal("exact counter or source provenance lost")
					}
				})
			}
		}
	}
}

func TestAccountingCanonicalRefusals(t *testing.T) {
	cases := []struct {
		name   string
		change func(*nvtelemetry.Manifest, []nvtelemetry.Sample)
	}{
		{"unknown sensor time", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			s[0].Readings[0].HasSourceTimestamp = false
			s[0].Readings[0].SourceTimestamp = 0
			s[0].Readings[0].SourceTimestampDomain = ""
		}},
		{"foreign sensor clock", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			s[0].Readings[0].SourceTimestampDomain = "foreign"
		}},
		{"misaligned sensor time", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) { s[0].Readings[0].SourceTimestamp++ }},
		{"unknown calibration", func(m *nvtelemetry.Manifest, _ []nvtelemetry.Sample) { m.HasClockCalibration = false }},
		{"uncertain calibration", func(m *nvtelemetry.Manifest, _ []nvtelemetry.Sample) { m.ClockCalibrationBoundsNanos = 1 }},
		{"foreign manifest epoch", func(m *nvtelemetry.Manifest, _ []nvtelemetry.Sample) { m.ClockMonotonicEpochID = "other" }},
		{"counter epoch changed", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			s[1].Readings[0].Observation.CounterEpoch = "counter-2"
		}},
		{"counter epoch missing", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) { s[0].Readings[0].Observation.CounterEpoch = "" }},
		{"counter reset", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			s[1].Readings[0].Observation.CounterReset = true
		}},
		{"counter decrease", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			n := uint64(0)
			s[1].Readings[0].Number = &nvtelemetry.Number{Uint: &n}
		}},
		{"gauge", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) { s[0].Readings[0].Observation.Kind = "gauge" }},
		{"float integer", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			n := 100.0
			s[0].Readings[0].Number = &nvtelemetry.Number{Float: &n}
		}},
		{"fractional", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			n := 100.5
			s[0].Readings[0].Number = &nvtelemetry.Number{Float: &n}
		}},
		{"negative", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			n := int64(-1)
			s[0].Readings[0].Number = &nvtelemetry.Number{Int: &n}
		}},
		{"unavailable", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			r := &s[0].Readings[0]
			r.Number = nil
			r.Evidence = nvtelemetry.EvidenceUnavailable
			r.UnavailableReason = nvtelemetry.ReasonBlank
		}},
		{"histogram", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			r := &s[0].Readings[0]
			r.Number = nil
			r.Observation.Kind = "histogram"
			r.Observation.Histogram = &nvtelemetry.Histogram{Bounds: []string{"+Inf"}, Counts: []uint64{1}}
		}},
		{"missing start boundary", func(_ *nvtelemetry.Manifest, s []nvtelemetry.Sample) {
			s[0].QueryStartNanos++
			s[0].QueryEndNanos++
			s[0].Readings[0].TimestampMonotonic++
			s[0].Readings[0].SourceTimestamp++
		}},
	}
	for _, version := range []int{3, 4, 5} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("v%d/%s", version, tc.name), func(t *testing.T) {
				m, s := accountingCanonicalRecords(version, false, 100)
				tc.change(&m, s)
				if version == 5 && tc.name == "histogram" {
					h := s[0].Readings[0].Observation.Histogram
					count := uint64(1)
					h.Count = &count
					h.Temporality = "cumulative"
				}
				raw := accountingCanonicalWire(t, m, s, true)
				if _, err := nvtelemetry.AccountTelemetry(bytes.NewReader(raw), accountingCanonicalInput(t), "measurement", accountingCanonicalStart, accountingCanonicalEnd, 1<<20); err == nil {
					t.Fatal("invalid counter evidence admitted")
				}
			})
		}
	}
}

func TestAccountingCanonicalMalformedRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		number string
	}{
		{"unsigned overflow", `{"uint":18446744073709551616}`},
		{"signed overflow", `{"int":9223372036854775808}`},
		{"nonfinite overflow", `{"float":1e400}`},
		{"multiple numeric arms", `{"uint":100,"int":100}`},
		{"no numeric arm", `{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, s := accountingCanonicalRecords(3, false, 100)
			raw := accountingCanonicalWire(t, m, s, true)
			changed := strings.Replace(string(raw), `"number":{"uint":100}`, `"number":`+tc.number, 1)
			if changed == string(raw) {
				t.Fatal("mutation did not apply")
			}
			if _, err := nvtelemetry.AccountTelemetry(strings.NewReader(changed), accountingCanonicalInput(t), "measurement", accountingCanonicalStart, accountingCanonicalEnd, 1<<20); err == nil {
				t.Fatal("malformed counter admitted")
			}
		})
	}
	m, s := accountingCanonicalRecords(3, false, 100)
	raw := accountingCanonicalWire(t, m, s, false)
	if _, err := nvtelemetry.AccountTelemetry(bytes.NewReader(raw), accountingCanonicalInput(t), "measurement", accountingCanonicalStart, accountingCanonicalEnd, 1<<20); err == nil {
		t.Fatal("incomplete stream admitted")
	}
}

func TestAccountingCanonicalMaxUint(t *testing.T) {
	m, s := accountingCanonicalRecords(5, false, math.MaxUint64-200000)
	raw := accountingCanonicalWire(t, m, s, true)
	report, err := nvtelemetry.AccountTelemetry(bytes.NewReader(raw), accountingCanonicalInput(t), "measurement", accountingCanonicalStart, accountingCanonicalEnd, 1<<20)
	if err != nil || report.PhysicalEnergyMillijoules != "200000" {
		t.Fatalf("max uint counter: %+v, %v", report, err)
	}
}
