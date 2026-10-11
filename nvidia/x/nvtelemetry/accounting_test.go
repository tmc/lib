package nvtelemetry_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func accountingFixture(t *testing.T) nvtelemetry.AccountingInput {
	t.Helper()
	data, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var in nvtelemetry.AccountingInput
	if err = json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	return in
}
func accountingConsume(t *testing.T, in nvtelemetry.AccountingInput) (nvtelemetry.AccountingReport, error) {
	t.Helper()
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	return nvtelemetry.ReadAccounting(bytes.NewReader(data), 1<<20)
}
func TestAccountingPublicRoundTrip(t *testing.T) {
	in := accountingFixture(t)
	var raw bytes.Buffer
	if err := nvtelemetry.WriteAccounting(&raw, in); err != nil {
		t.Fatal(err)
	}
	report, err := nvtelemetry.ReadAccounting(&raw, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if report.PhysicalEnergyMillijoules != "200000" || report.ResidualEnergyMillijoules != "75000" || len(report.Shares) != 2 {
		t.Fatalf("report %+v", report)
	}
	a, b := report.Shares[0], report.Shares[1]
	if a.EnergyMillijoules != "100000" || a.AllocatedGPUHours != "1" || a.CostMajor != "3/2" || a.RoundedMinor != "150" || a.WorkPerAllocatedJoule != "1" {
		t.Fatalf("A %+v", a)
	}
	if b.EnergyMillijoules != "25000" || b.AllocatedGPUHours != "1/4" || b.CostMajor != "1/2" || b.RoundedMinor != "50" || b.EfficiencyUnavailable == "" {
		t.Fatalf("B %+v", b)
	}
	total := new(big.Rat)
	for _, s := range report.Shares {
		r, _ := new(big.Rat).SetString(s.EnergyMillijoules)
		total.Add(total, r)
	}
	residual, _ := new(big.Rat).SetString(report.ResidualEnergyMillijoules)
	total.Add(total, residual)
	if total.RatString() != report.PhysicalEnergyMillijoules {
		t.Fatal("energy not conserved")
	}
	var output bytes.Buffer
	if err := nvtelemetry.WriteAccountingReport(&output, report); err != nil {
		t.Fatal(err)
	}
	var decoded nvtelemetry.AccountingReport
	if err := json.Unmarshal(output.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Shares[0].CostMajor != a.CostMajor || decoded.Input.Energy[0].Lineage != in.Energy[0].Lineage {
		t.Fatal("report provenance/precision lost")
	}
}
func TestAccountingPublicMutations(t *testing.T) {
	mutations := []struct {
		name   string
		mutate func(*nvtelemetry.AccountingInput)
	}{
		{"double_lineage", func(i *nvtelemetry.AccountingInput) { i.Energy = append(i.Energy, i.Energy[0]) }},
		{"derived_physical", func(i *nvtelemetry.AccountingInput) { i.Energy[0].Evidence = nvtelemetry.EvidenceDerived }},
		{"energy_epoch", func(i *nvtelemetry.AccountingInput) { i.Energy[0].ClockEpoch = "other" }},
		{"reset", func(i *nvtelemetry.AccountingInput) { i.Energy[0].Summary.ResetDetected = true }},
		{"gap", func(i *nvtelemetry.AccountingInput) { i.Energy[0].Summary.CoveredDurationNanos-- }},
		{"float_only_energy", func(i *nvtelemetry.AccountingInput) { i.Energy[0].Summary.Method = "integrated_power" }},
		{"joule_copy", func(i *nvtelemetry.AccountingInput) { i.Energy[0].Summary.EnergyConsumedJoules = 1 }},
		{"unknown_sharing", func(i *nvtelemetry.AccountingInput) { i.Allocations[0].Assignment.Sharing = "unknown" }},
		{"overlap_capacity", func(i *nvtelemetry.AccountingInput) { i.Allocations[1].CapacityPermille = 1000 }},
		{"unknown_clock", func(i *nvtelemetry.AccountingInput) { i.Clock.UncertaintyKnown = false }},
		{"uncertain_clock", func(i *nvtelemetry.AccountingInput) { i.Clock.UncertaintyNanos = 1 }},
		{"epoch", func(i *nvtelemetry.AccountingInput) { i.Allocations[0].Assignment.ClockEpoch = "other" }},
		{"rate_gap", func(i *nvtelemetry.AccountingInput) { i.Rates = i.Rates[:1] }},
		{"currency", func(i *nvtelemetry.AccountingInput) { i.Rates[1].Currency = "EUR" }},
		{"invoice_scope", func(i *nvtelemetry.AccountingInput) { i.Rates[0].Scope = "entire_datacenter_invoice" }},
		{"no_discount_assumptions", func(i *nvtelemetry.AccountingInput) { i.Rates[0].CommitmentAssumption = "" }},
		{"unmatched_work", func(i *nvtelemetry.AccountingInput) { i.Work[0].AllocationID = "other" }},
		{"work_clock", func(i *nvtelemetry.AccountingInput) { i.Work[0].ClockEpoch = "other" }},
		{"work_policy", func(i *nvtelemetry.AccountingInput) { i.Work[0].Policy = "processed_tokens" }},
		{"duplicate_work", func(i *nvtelemetry.AccountingInput) { i.Work = append(i.Work, i.Work[0]) }},
	}
	for _, m := range mutations {
		t.Run(m.name, func(t *testing.T) {
			in := accountingFixture(t)
			m.mutate(&in)
			if _, err := accountingConsume(t, in); err == nil {
				t.Fatal("mutated input admitted through public reader")
			}
		})
	}
}
func TestAccountingExactLargeCounter(t *testing.T) {
	in := accountingFixture(t)
	s := &in.Energy[0].Summary
	s.StartEnergyMJ = 0
	s.EndEnergyMJ = math.MaxUint64
	s.EnergyConsumedMJ = math.MaxUint64
	s.EnergyConsumedJoules = float64(math.MaxUint64) / 1000
	r, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if r.PhysicalEnergyMillijoules != "18446744073709551615" || r.Shares[0].EnergyMillijoules != "18446744073709551615/2" {
		t.Fatalf("exact counter lost: %+v", r)
	}
}
func TestAccountingBaselineAndZeroWork(t *testing.T) {
	in := accountingFixture(t)
	b := in.Energy[0]
	s := &b.Summary
	s.StartMonoNanos = 0
	s.EndMonoNanos = 3600000000000
	s.TotalDurationNanos = s.EndMonoNanos
	s.CoveredDurationNanos = s.EndMonoNanos
	s.StartEnergyMJ = 0
	s.EndEnergyMJ = 20000
	s.EnergyConsumedMJ = 20000
	s.EnergyConsumedJoules = 20
	b.RawDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	in.Baselines = []nvtelemetry.AccountingEnergy{b}
	in.BaselinePolicy = "uniform_baseline_rate_v1"
	in.BaselineEvidence = "independent baseline window"
	in.Work[0].Units = 0
	r, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if r.BaselineAdjustedEnergyMillijoules != "160000" || r.Shares[0].EfficiencyUnavailable != "zero_work_or_allocated_energy" {
		t.Fatalf("report %+v", r)
	}
	in.Baselines[0].Summary.EndEnergyMJ = 200000
	in.Baselines[0].Summary.EnergyConsumedMJ = 200000
	in.Baselines[0].Summary.EnergyConsumedJoules = 200
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("negative baseline subtraction clamped/admitted")
	}
}
func TestAccountingMIGReconfiguration(t *testing.T) {
	in := accountingFixture(t)
	a := &in.Allocations[0].Assignment
	a.Sharing = "mig"
	a.Device.UUID = "MIG-A"
	a.Device.MIGParentUUID = "GPU-A"
	a.Device.MIGScope = nvtelemetry.MIGScopeMIGInstance
	a.MIGConfigurationEpoch = "mig-1"
	in.Energy[0].ConfigurationEpoch = "mig-1"
	if _, err := accountingConsume(t, in); err != nil {
		t.Fatal(err)
	}
	in.Energy[0].ConfigurationEpoch = "mig-2"
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("mig reconfiguration crossed")
	}
}

type accountingShortWriter struct{}

func (accountingShortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestAccountingParserAndMoney(t *testing.T) {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]byte{append(append([]byte(nil), raw...), raw...), []byte(`{"version":1,"unknown":true}`)} {
		if _, err := nvtelemetry.ReadAccounting(bytes.NewReader(data), 1<<20); err == nil {
			t.Fatal("invalid json admitted")
		}
	}
	if _, err := nvtelemetry.ReadAccounting(bytes.NewReader(raw), 16); err == nil {
		t.Fatal("budget ignored")
	}
	in := accountingFixture(t)
	if err := nvtelemetry.WriteAccounting(accountingShortWriter{}, in); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write %v", err)
	}
	for _, price := range []string{"-1", "NaN", "1e100000"} {
		in.Rates[0].PricePerGPUHour = price
		if _, err := accountingConsume(t, in); err == nil {
			t.Fatal("invalid price admitted")
		}
	}
}

func TestAccountingHalfEvenMoney(t *testing.T) {
	for _, tc := range []struct{ price, want string }{{"0.005", "0"}, {"0.015", "2"}} {
		in := accountingFixture(t)
		for i := range in.Rates {
			in.Rates[i].PricePerGPUHour = tc.price
		}
		r, err := accountingConsume(t, in)
		if err != nil {
			t.Fatal(err)
		}
		if r.Shares[0].RoundedMinor != tc.want {
			t.Fatalf("price %s got %s want %s", tc.price, r.Shares[0].RoundedMinor, tc.want)
		}
	}
	in := accountingFixture(t)
	r, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	r.PhysicalEnergyMillijoules = "999"
	if err := nvtelemetry.WriteAccountingReport(&bytes.Buffer{}, r); err == nil {
		t.Fatal("fabricated report written")
	}
}
func FuzzReadAccounting(f *testing.F) {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		f.Fatal(err)
	}
	f.Add(raw)
	f.Add([]byte(`{"version":9}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			return
		}
		r, err := nvtelemetry.ReadAccounting(bytes.NewReader(data), 1<<20)
		if err == nil {
			if err := nvtelemetry.WriteAccountingReport(&bytes.Buffer{}, r); err != nil {
				t.Fatalf("accepted report fails replay: %v", err)
			}
		}
	})
}

func accountingRawStream(t *testing.T, known bool) []byte {
	t.Helper()
	var raw bytes.Buffer
	w := nvtelemetry.NewWriter(&raw, nvtelemetry.CollectorConfig{})
	m := nvtelemetry.Manifest{SchemaVersion: 2, RunID: "accounting-native-contract", ClockMonotonicDomain: "process_local", ClockMonotonicEpochID: "fixture-epoch", HasClockCalibration: true, ClockMappingValidUntil: 10800000000000, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-A"}}}
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	for i, at := range []int64{3600000000000, 10800000000000} {
		s := nvtelemetry.Sample{Seq: uint64(i + 1), QueryStartNanos: at, QueryEndNanos: at, Readings: []nvtelemetry.Reading{{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, DeviceUUID: "GPU-A", Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: uint64(i) * 200000, HasSourceTimestamp: known, SourceTimestamp: at, SourceTimestampDomain: "process_local"}}}
		if err := w.WriteSample(&s); err != nil {
			t.Fatal(err)
		}
	}
	m.EndTimeMonotonic = 10800000000000
	m.TotalSamples = 2
	m.IsComplete = true
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	return raw.Bytes()
}
func TestAccountingRawV2EndToEnd(t *testing.T) {
	in := accountingFixture(t)
	in.Energy = nil
	r, err := nvtelemetry.AccountTelemetry(bytes.NewReader(accountingRawStream(t, true)), in, "measurement", 3600000000000, 10800000000000, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if r.PhysicalEnergyMillijoules != "200000" || r.Shares[0].CostMajor != "3/2" || len(r.Input.Energy[0].RawDigest) != 64 {
		t.Fatalf("raw report %+v", r)
	}
	if _, err := nvtelemetry.AccountTelemetry(bytes.NewReader(accountingRawStream(t, false)), in, "measurement", 3600000000000, 10800000000000, 1<<20); err == nil {
		t.Fatal("unknown sensor time upgraded via raw consumer")
	}
}

func TestAccountingStrictJSONAndBaselineCopies(t *testing.T) {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]byte{bytes.Replace(raw, []byte(`"version": 1`), []byte(`"version":999,"version":1`), 1), bytes.Replace(raw, []byte(`"version": 1`), []byte(`"VERSION":1`), 1), bytes.Replace(raw, []byte(`"energy_consumed_mj": 200000`), []byte(`"energy_consumed_mj":null`), 1), bytes.Replace(raw, []byte(`"minor_digits": 2,`), nil, 1)} {
		if _, err := nvtelemetry.ReadAccounting(bytes.NewReader(data), 1<<20); err == nil {
			t.Fatal("noncanonical/unknown numeric JSON admitted")
		}
	}
	in := accountingFixture(t)
	b := in.Energy[0]
	b.Summary.StartMonoNanos = 0
	b.Summary.EndMonoNanos = 3600000000000
	b.Summary.TotalDurationNanos = b.Summary.EndMonoNanos
	b.Summary.CoveredDurationNanos = b.Summary.EndMonoNanos
	b.Summary.StartEnergyMJ = 0
	b.Summary.EndEnergyMJ = 20000
	b.Summary.EnergyConsumedMJ = 20000
	in.Baselines = []nvtelemetry.AccountingEnergy{b}
	in.BaselinePolicy = "uniform_baseline_rate_v1"
	in.BaselineEvidence = "independent baseline"
	for _, copy := range []float64{1, math.NaN()} {
		in.Baselines[0].Summary.EnergyConsumedJoules = copy
		if _, err := nvtelemetry.AccountEnergy(in); err == nil {
			t.Fatal("contradictory baseline mirror admitted")
		}
	}
}

func TestAccountingAdjacentRatesKeepCurrenciesSeparate(t *testing.T) {
	in := accountingFixture(t)
	in.Allocations[0].Assignment.EndNanos = 7200000000000
	in.Work = nil
	in.Rates[1].Currency = "EUR"
	report, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if report.Shares[0].Currency != "USD" || report.Shares[1].Currency != "EUR" {
		t.Fatal("currency scope lost")
	}
	if report.Shares[0].CostMajor != "1/2" || report.Shares[1].CostMajor != "1/2" {
		t.Fatalf("boundary rates: %+v", report.Shares)
	}
}

func TestAccountingCostDoesNotRequireEnergy(t *testing.T) {
	in := accountingFixture(t)
	in.Energy = nil
	report, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if report.EnergyAvailable || report.PhysicalEnergyMillijoules != "" || report.ResidualEnergyMillijoules != "" || report.EnergyUnavailableReason == "" {
		t.Fatalf("missing energy became measured zero: %+v", report)
	}
	if report.Shares[0].AllocatedGPUHours != "1" || report.Shares[0].CostMajor != "3/2" || report.Shares[0].EnergyMillijoules != "" || report.Shares[0].WorkPerAllocatedJoule != "" || report.Shares[0].EfficiencyUnavailable != "physical_energy_unavailable" {
		t.Fatalf("cost/efficiency separation: %+v", report.Shares[0])
	}
	if err := nvtelemetry.WriteAccountingReport(&bytes.Buffer{}, report); err != nil {
		t.Fatal(err)
	}
	originalUntil := in.Clock.ValidUntil
	in.Clock.ValidUntil = in.Allocations[0].Assignment.StartNanos + 1
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("cost-only input outside mapping validity admitted")
	}
	in.Clock.ValidUntil = originalUntil
	in.Allocations[1].CapacityPermille = 1000
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("cost-only input bypassed capacity admission")
	}
}

func TestAccountingMeasuredZeroDiffersFromMissing(t *testing.T) {
	in := accountingFixture(t)
	s := &in.Energy[0].Summary
	s.EndEnergyMJ = s.StartEnergyMJ
	s.EnergyConsumedMJ = 0
	s.EnergyConsumedJoules = 0
	report, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if !report.EnergyAvailable || report.PhysicalEnergyMillijoules != "0" || report.Shares[0].EnergyMillijoules != "0" || report.Shares[0].EfficiencyUnavailable != "zero_work_or_allocated_energy" {
		t.Fatalf("measured zero misclassified: %+v", report)
	}
}

func TestAccountingNodeAndIdleScopes(t *testing.T) {
	in := accountingFixture(t)
	in.Allocations = nil
	in.Rates = nil
	in.Work = nil
	gpu := in.Energy[0]
	node := gpu
	node.Scope = "whole_node"
	node.Lineage = "node-counter"
	node.Summary.DeviceUUID = "node-1"
	node.Summary.EnergyConsumedMJ = 300000
	node.Summary.EndEnergyMJ = node.Summary.StartEnergyMJ + 300000
	node.Summary.EnergyConsumedJoules = 300
	in.Nodes = []nvtelemetry.AccountingNode{{NodeID: "node-1", Energy: node, DeviceUUIDs: []string{gpu.Summary.DeviceUUID}, MembershipEvidence: "synthetic inventory proof"}}
	in.Idle = []nvtelemetry.AccountingIdle{{DeviceUUID: gpu.Summary.DeviceUUID, ClockDomain: in.Clock.Domain, ClockEpoch: in.Clock.Epoch, StartNanos: gpu.Summary.StartMonoNanos, EndNanos: gpu.Summary.EndMonoNanos, CapacityPermille: 250, Authoritative: true, Evidence: "synthetic explicit idle capacity", Policy: "capacity_time_v1"}}
	r, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	if r.PhysicalEnergyMillijoules != "200000" || r.Nodes[0].NodeEnergyMillijoules != "300000" || r.Nodes[0].GPUComponentMillijoules != "200000" || r.Nodes[0].OtherEnergyMillijoules != "100000" || r.IdleEnergyMillijoules != "50000" || r.UnclassifiedResidualMillijoules != "150000" {
		t.Fatalf("scope conservation %+v", r)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*nvtelemetry.AccountingInput)
	}{
		{"missing proof", func(v *nvtelemetry.AccountingInput) { v.Nodes[0].MembershipEvidence = "" }},
		{"duplicate membership", func(v *nvtelemetry.AccountingInput) {
			v.Nodes[0].DeviceUUIDs = append(v.Nodes[0].DeviceUUIDs, v.Nodes[0].DeviceUUIDs[0])
		}},
		{"partial component", func(v *nvtelemetry.AccountingInput) {
			v.Energy[0].Summary.EndMonoNanos--
			v.Energy[0].Summary.TotalDurationNanos--
			v.Energy[0].Summary.CoveredDurationNanos--
		}},
		{"double lineage", func(v *nvtelemetry.AccountingInput) { v.Nodes[0].Energy.Lineage = v.Energy[0].Lineage }},
		{"negative component residual", func(v *nvtelemetry.AccountingInput) {
			s := &v.Nodes[0].Energy.Summary
			s.EnergyConsumedMJ = 1
			s.EndEnergyMJ = s.StartEnergyMJ + 1
			s.EnergyConsumedJoules = .001
		}},
		{"idle allocation conflict", func(v *nvtelemetry.AccountingInput) {
			v.Allocations = accountingFixture(t).Allocations
			v.Rates = accountingFixture(t).Rates
		}},
		{"unknown idle", func(v *nvtelemetry.AccountingInput) { v.Idle[0].Authoritative = false }},
		{"idle clock", func(v *nvtelemetry.AccountingInput) { v.Idle[0].ClockEpoch = "different" }},
		{"idle overlap", func(v *nvtelemetry.AccountingInput) { v.Idle = append(v.Idle, v.Idle[0]) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, _ := json.Marshal(in)
			var v nvtelemetry.AccountingInput
			json.Unmarshal(raw, &v)
			tc.mutate(&v)
			if _, err := accountingConsume(t, v); err == nil {
				t.Fatal("accepted contradictory scope")
			}
		})
	}
}

func TestAccountingNodeWithoutGPUCounters(t *testing.T) {
	in := accountingFixture(t)
	node := in.Energy[0]
	node.Scope = "whole_node"
	node.Lineage = "node-source"
	node.Summary.DeviceUUID = "node-1"
	in.Nodes = []nvtelemetry.AccountingNode{{NodeID: "node-1", Energy: node, DeviceUUIDs: []string{"gpu-unsupported"}, MembershipEvidence: "synthetic complete inventory"}}
	in.Energy = nil
	in.Allocations = nil
	in.Rates = nil
	in.Work = nil
	r, err := accountingConsume(t, in)
	if err != nil {
		t.Fatal(err)
	}
	n := r.Nodes[0]
	if n.NodeEnergyMillijoules != "200000" || n.GPUComponentsAvailable || n.GPUComponentsUnavailableReason != "gpu_counter_unavailable" || n.GPUComponentMillijoules != "" || n.OtherEnergyMillijoules != "" || r.EnergyAvailable || r.PhysicalEnergyMillijoules != "" {
		t.Fatalf("invented decomposition %+v", r)
	}
	var report bytes.Buffer
	if err := nvtelemetry.WriteAccountingReport(&report, r); err != nil {
		t.Fatal(err)
	}
	partial := node
	partial.Scope = "physical_device"
	partial.Lineage = "gpu-source"
	partial.Summary.DeviceUUID = "gpu-unsupported"
	partial.Summary.EndMonoNanos--
	partial.Summary.TotalDurationNanos--
	partial.Summary.CoveredDurationNanos--
	in.Energy = []nvtelemetry.AccountingEnergy{partial}
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("supplied partial component accepted")
	}
	in.Energy = nil
	in.Nodes[0].Energy.Summary.Valid = false
	if _, err := accountingConsume(t, in); err == nil {
		t.Fatal("bad supplied node counter accepted")
	}
}
