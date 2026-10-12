package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func registrationFixture(t testing.TB) (Registration, QualificationLedger, string) {
	t.Helper()
	p := DefaultProtocol(PhaseQualification)
	p.PropertyLimits = map[string]float64{"cost": 1}
	p.Tools = []ToolSpec{{Name: "offline-control", Version: "1", Configuration: "constructed", Lineage: "control", Support: "unknown"}}
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	closure := QualificationClosure{Candidate: strings.Repeat("a", 40), Tree: strings.Repeat("b", 40), Base: strings.Repeat("c", 40), SignatureMapSHA256: strings.Repeat("a", 64), BinarySHA256: strings.Repeat("b", 64), DependenciesSHA256: strings.Repeat("c", 64), InputsSHA256: strings.Repeat("d", 64), OutputsSHA256: strings.Repeat("e", 64), HostOS: "darwin", HostArch: "arm64", Device: "offline", Architecture: "offline", ProducerVersion: "control1", Libraries: map[string]string{}, Toolchain: "host-control", Topology: "offline", Workload: "constructed", Deployment: "offline", Observer: "off"}
	root := t.TempDir()
	if err := os.Chmod(root, 0700); err != nil {
		t.Fatal(err)
	}
	raw := []byte("constructed host output\n")
	ev := QualificationEvidence{CellID: "host", Gate: "host", Status: "passed", Closure: closure, RawPath: "raw.txt", RawSHA256: registrationSHA(raw), Command: []string{"declared-control-not-executed"}, Verifier: "constructed-verifier"}
	eb, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "raw.txt"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "receipt.json"), eb, 0600); err != nil {
		t.Fatal(err)
	}
	host := QualificationCell{ID: "host", CapabilityID: "offline", Capability: "supported", Gate: "host", Status: "qualified", SourcePaths: []string{"x/control.go"}, Consumers: []string{"x/control_test.go"}, Closure: closure, Attempts: []QualificationAttempt{{Path: "receipt.json", SHA256: registrationSHA(eb), Evidence: ev}}, Owner: "control", NextAction: "independent review"}
	native := QualificationCell{ID: "native", CapabilityID: "nvml", Capability: "unknown", Gate: "native", Status: "unavailable", SourcePaths: []string{"x/native.go"}, Consumers: []string{"x/native_test.go"}, Closure: closure, Owner: "prof", NextAction: "admit actual producer"}
	l := QualificationLedger{Version: 1, Cells: []QualificationCell{host, native}}
	var lb bytes.Buffer
	if err := WriteQualificationLedger(&lb, l); err != nil {
		t.Fatal(err)
	}
	cb, _ := json.Marshal(closure)
	f := RegistrationFact{State: "unknown", Reason: "no discovery run", Owner: "prof", NextAction: "supply actual facts"}

	r := Registration{Version: 1, ID: "control", Task: "offline contract property", Mode: "property", Owner: "root", Protocol: p, Source: closure, PrerequisiteLedgerSHA256: registrationSHA(lb.Bytes()), SelectedCells: []RegistrationCell{{Applicability: "selected", CellID: "host", CapabilityID: "offline", Gate: "host", ClosureSHA256: registrationSHA(cb), ReceiptSHA256: registrationSHA(eb), Owner: "root", NextAction: "verify actual evidence"}, {Applicability: "selected", CellID: "native", CapabilityID: "nvml", Gate: "native", ClosureSHA256: registrationSHA(cb), Reason: "no native receipt", Owner: "prof", NextAction: "admit native producer"}}, TargetRows: []RegistrationTarget{{ID: "native-unknown", Applicability: "not-selected", Reason: "native not acquired", Host: f, Device: f, Stack: f, Topology: f, Workload: f, Deployment: f, Observer: f, Owner: "prof", NextAction: "actual discovery"}}, Limits: RegistrationLimits{MaxDurationNanos: 1e9, MaxRecords: 100, MaxBytes: 1 << 20}}
	registrationDeclare(t, &r, root)
	return r, l, root
}

func frozenRegistration(t testing.TB) (Registration, QualificationLedger, string) {
	t.Helper()
	r, l, root := registrationFixture(t)
	r, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	return r, l, root
}

func TestRegistrationPublicRoundTrip(t *testing.T) {
	r, l, root := frozenRegistration(t)
	var b bytes.Buffer
	if err := WriteRegistration(&b, r); err != nil {
		t.Fatal(err)
	}
	got, err := ReadRegistration(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, r) {
		t.Fatal("round trip changed facts")
	}
	var next bytes.Buffer
	if err := WriteRegistration(&next, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(next.Bytes(), b.Bytes()) {
		t.Fatal("nondeterministic writer")
	}
	report, err := VerifyRegistration(context.Background(), got, l, root)
	if err != nil {
		t.Fatal(err)
	}
	if report.MeasurementAcceptance != "unverified" || len(report.Cells) != 2 || !report.Cells[0].ReceiptConsistency || report.Cells[0].MeasurementAcceptance != "unverified" || report.Cells[1].ReceiptConsistency || report.Cells[1].Status != "unavailable" || !report.Cells[1].Selected {
		t.Fatalf("receipt flags authenticated measurements: %+v", report)
	}
	if report.Targets[0].Device.State != "unknown" {
		t.Fatal("unknown native facts disappeared")
	}
	report.Targets[0].Device.Reason = "changed"
	if got.TargetRows[0].Device.Reason == "changed" {
		t.Fatal("report aliases registration")
	}
	if err := os.WriteFile(filepath.Join(root, "raw.txt"), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	report, err = VerifyRegistration(context.Background(), got, l, root)
	if err == nil {
		t.Fatal("corrupt raw output admitted", report)
	}
}

func TestRegistrationStrictReader(t *testing.T) {
	r, _, _ := frozenRegistration(t)
	b, _ := json.Marshal(r)
	for _, tc := range []struct {
		name string
		raw  []byte
	}{
		{"future", bytes.Replace(b, []byte(`"version":1`), []byte(`"version":2`), 1)},
		{"alias", bytes.Replace(b, []byte(`"task":`), []byte(`"Task":`), 1)},
		{"duplicate", bytes.Replace(b, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1)},
		{"unknown", bytes.Replace(b, []byte(`"version":1`), []byte(`"unknown":true,"version":1`), 1)},
		{"trailing", append(append([]byte(nil), b...), []byte(" {}")...)},
		{"nesting", []byte(strings.Repeat("[", 33) + "0" + strings.Repeat("]", 33))},
		{"large", bytes.Repeat([]byte(" "), registrationLimit+1)},
		{"scalar", bytes.Replace(b, []byte(`"id":"control"`), []byte(`"id":"`+strings.Repeat("x", 4097)+`"`), 1)},
		{"timestamp", bytes.Replace(b, []byte(`"registered_at":"`), []byte(`"Registered_At":"`), 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadRegistration(bytes.NewReader(tc.raw)); err == nil {
				t.Fatal("invalid registration accepted")
			}
		})
	}
	// Comparison is a pointer field, so exact-tag checking must descend into it.
	r, _, _ = registrationFixture(t)
	r.Mode = "comparison"
	r.Task = "pair"
	r.Observer.Body.Arm = "candidate"
	r.Observer.Body.ReferenceArm = "comparator"
	r.Protocol.Tools = []ToolSpec{{Name: "candidate", Version: "1", Configuration: "one", Lineage: "one", Support: "measured"}, {Name: "comparator", Version: "2", Configuration: "two", Lineage: "two", Support: "measured"}}
	r.Protocol.PropertyLimits = map[string]float64{"cost": 1}
	r.Protocol.Comparison = &ComparisonSpec{Task: "pair", Candidate: "candidate", Comparator: "comparator", Properties: map[string]MeasurementScope{"cost": {Unit: "ns", Denominator: 1}}}
	r.Protocol.Frozen = false
	r.Protocol.FrozenDigest = ""
	r.Protocol.FrozenTimestamp = time.Time{}
	if err := r.Protocol.Freeze(); err != nil {
		t.Fatal(err)
	}
	r, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	b, _ = json.Marshal(r)
	if _, err := ReadRegistration(bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
	b = bytes.Replace(b, []byte(`"candidate":"candidate"`), []byte(`"Candidate":"candidate"`), 1)
	if _, err := ReadRegistration(bytes.NewReader(b)); err == nil {
		t.Fatal("pointer case alias accepted")
	}
}

func TestRegistrationFreezeAndSuccessor(t *testing.T) {
	r, _, _ := registrationFixture(t)
	f, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	r.Protocol.Tools[0].Configuration = "mutated"
	r.Source.Libraries["invented"] = "mutated"
	r.TargetRows[0].Device.Reason = "mutated"
	if f.Protocol.Tools[0].Configuration == "mutated" || f.Source.Libraries["invented"] != "" || f.TargetRows[0].Device.Reason == "mutated" {
		t.Fatal("freeze aliases caller storage")
	}
	if err := f.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Freeze(time.Now().UTC()); err == nil {
		t.Fatal("double freeze accepted")
	}
	if err := CheckRegistrationSuccessor(f, f); err != nil {
		t.Fatal(err)
	}
	changed := f
	changed.Task = "changed"
	if err := changed.Validate(); err == nil {
		t.Fatal("frozen mutation accepted")
	}
	changed.Digest = ""
	changed.RegisteredAt = time.Time{}
	changed, err = changed.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckRegistrationSuccessor(f, changed); err == nil {
		t.Fatal("same ID rewrite accepted")
	}
	changed.Digest = ""
	changed.RegisteredAt = time.Time{}
	changed.ID = "control2"
	changed.ParentDigest = f.Digest
	changed.RevisionReason = "new task"
	changed, err = changed.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckRegistrationSuccessor(f, changed); err != nil {
		t.Fatal(err)
	}
}

func TestRegistrationBindings(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Registration)
	}{
		{"zero", func(r *Registration) { *r = Registration{} }},
		{"unfrozen protocol", func(r *Registration) { r.Protocol.Frozen = false }},
		{"guessed native", func(r *Registration) { r.TargetRows[0].Device.Value = "H100" }},
		{"upper digest", func(r *Registration) { r.PrerequisiteLedgerSHA256 = strings.Repeat("A", 64) }},
		{"traversal", func(r *Registration) {
			r.PilotArtifacts = []RegistrationArtifact{{Path: "../outside", SHA256: strings.Repeat("a", 64)}}
		}},
		{"overlap", func(r *Registration) {
			a := RegistrationArtifact{Path: "case.json", SHA256: strings.Repeat("a", 64)}
			r.PilotArtifacts = []RegistrationArtifact{a}
			r.HeldOutArtifacts = []RegistrationArtifact{a}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, _, _ := registrationFixture(t)
			tc.mutate(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid facts accepted")
			}
		})
	}
	for _, name := range []string{"ledger", "closure", "receipt", "missing cell"} {
		t.Run(name, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			switch name {
			case "ledger":
				r.PrerequisiteLedgerSHA256 = strings.Repeat("a", 64)
			case "closure":
				r.SelectedCells[0].ClosureSHA256 = strings.Repeat("a", 64)
			case "receipt":
				r.SelectedCells[0].ReceiptSHA256 = strings.Repeat("a", 64)
			case "missing cell":
				r.SelectedCells[0].CellID = "absent"
			}
			r, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := VerifyRegistration(context.Background(), r, l, root); err == nil {
				t.Fatal("scope mismatch accepted")
			}
		})
	}
}

type registrationShortWriter struct{}

func (registrationShortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestRegistrationIOAndCancellation(t *testing.T) {
	r, l, root := frozenRegistration(t)
	if err := WriteRegistration(registrationShortWriter{}, r); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if _, err := ReadRegistration(nil); err == nil {
		t.Fatal("nil reader accepted")
	}
	if err := WriteRegistration(nil, r); err == nil {
		t.Fatal("nil writer accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := VerifyRegistration(ctx, r, l, root); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestRegistrationSymlink(t *testing.T) {
	r, l, root := frozenRegistration(t)
	if err := os.Rename(filepath.Join(root, "raw.txt"), filepath.Join(root, "real.txt")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("real.txt", filepath.Join(root, "raw.txt")); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyRegistration(context.Background(), r, l, root); err == nil || !strings.Contains(err.Error(), "no-follow") {
		t.Fatal("symlink artifact accepted", err)
	}
}

func TestRegistrationMissingReceiptPreservesStatus(t *testing.T) {
	for _, status := range []string{"unavailable", "failed", "not-selected", "pending"} {
		t.Run(status, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			l.Cells[1].Status = status
			var b bytes.Buffer
			if err := WriteQualificationLedger(&b, l); err != nil {
				t.Fatal(err)
			}
			r.PrerequisiteLedgerSHA256 = registrationSHA(b.Bytes())
			r, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
			if err != nil {
				t.Fatal(err)
			}
			report, err := VerifyRegistration(context.Background(), r, l, root)
			if err != nil {
				t.Fatal(err)
			}
			if report.Cells[1].Status != status || report.Cells[1].ReceiptConsistency || report.Cells[1].MeasurementAcceptance != "unverified" {
				t.Fatalf("missing receipt changed explicit status: %+v", report.Cells[1])
			}
		})
	}
}

// Inert bytes support source-consistency controls only. No named method runs.
func registrationDeclare(t testing.TB, r *Registration, root string) {
	t.Helper()
	raw := []byte("inert pretrial method, calibration and independent expected input")
	ref := RegistrationArtifact{Path: "contract.txt", SHA256: registrationSHA(raw)}
	if err := os.WriteFile(filepath.Join(root, ref.Path), raw, 0600); err != nil {
		t.Fatal(err)
	}
	contract := RegistrationContract{State: "declared", ID: "constructed-contract", Owner: "control", NextAction: "independent actual acceptance"}
	na := RegistrationFact{State: "not-applicable", Reason: "explicit offline constructed task", Owner: "control", NextAction: "retain declared task scope"}
	r.Workload = RegistrationWorkload{Contract: contract, Body: &RegistrationWorkloadSpec{Question: "offline bytes consistent?", Mechanism: "constructed", Code: ref, Inputs: []RegistrationArtifact{ref}, CacheState: "not-applicable", PowerSettings: na, Contention: na, HardwareLease: na}}
	r.Oracle = RegistrationOracle{Contract: contract, Body: &RegistrationOracleSpec{Method: "exact", Implementation: ref, Expected: []RegistrationArtifact{ref}, Quality: []RegistrationQualitySpec{{Output: "control", Unit: "bytes", Nonfinite: "reject"}}}}
	r.Clock = RegistrationClock{Contract: contract, Body: &RegistrationClockSpec{Domains: []RegistrationClockDomain{{ID: "clock", Family: "host-monotonic", Epoch: "declared-epoch", Unit: "ns", Calibration: ref, ValidityStart: 0, ValidityEnd: 10}}}}
	r.Loss = RegistrationLoss{Contract: contract, Body: &RegistrationLossSpec{Families: []RegistrationLossFamily{{Name: "control", Required: true, DenominatorUnit: "declared events", ExpectedRecords: 1, Handling: "withhold"}}}}
	arm := "offline-control"
	reference := ""
	if r.Mode == "comparison" {
		arm = r.Protocol.Comparison.Candidate
		reference = r.Protocol.Comparison.Comparator
	}
	r.Observer = RegistrationObserver{Contract: contract, Body: &RegistrationObserverSpec{Mode: "off", SensorScope: []string{"offline bytes"}, Access: na, FreshnessNanos: 1, Arm: arm, ReferenceArm: reference}}
	seed := uint64(0)
	r.Analysis = RegistrationAnalysis{Contract: contract, Body: &RegistrationAnalysisSpec{Repetitions: 1, MinSamples: 1, Order: "randomized", Seed: &seed, Method: ref, Properties: []RegistrationPropertySpec{{Property: "cost", Unit: "ns", Denominator: 1, Direction: "lower", TimingFamily: "host-monotonic", ClockDomain: "clock", UncertaintyKind: "absolute-bound", UncertaintyMethod: ref, Overlap: "inconclusive", MaxValue: 1}}}}
	r.Stopping = RegistrationStopping{Contract: contract, Body: &RegistrationStoppingSpec{Rule: "fixed-samples", MaxSamples: 1, MaxDurationNanos: 1}}
	r.MissingData = RegistrationMissingData{Contract: contract, Body: &RegistrationMissingDataSpec{Handling: "withhold", Denominator: "declared-population"}}
}

func TestRegistrationTypedRefusals(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Registration)
	}{
		{"question", func(r *Registration) { r.Workload.Body.Question = "" }},
		{"mechanism", func(r *Registration) { r.Workload.Body.Mechanism = "" }},
		{"code", func(r *Registration) { r.Workload.Body.Code = RegistrationArtifact{} }},
		{"inputs", func(r *Registration) { r.Workload.Body.Inputs = nil }},
		{"cache", func(r *Registration) { r.Workload.Body.CacheState = "guess" }},
		{"oracle method", func(r *Registration) { r.Oracle.Body.Method = "guess" }},
		{"oracle implementation", func(r *Registration) { r.Oracle.Body.Implementation = RegistrationArtifact{} }},
		{"oracle expected", func(r *Registration) { r.Oracle.Body.Expected = nil }},
		{"quality", func(r *Registration) { r.Oracle.Body.Quality = nil }},
		{"tolerance", func(r *Registration) { r.Oracle.Body.Quality[0].AbsoluteTolerance = 1 }},
		{"nonfinite", func(r *Registration) { r.Oracle.Body.Quality[0].Nonfinite = "ignore" }},
		{"clock epoch", func(r *Registration) { r.Clock.Body.Domains[0].Epoch = "" }},
		{"clock family", func(r *Registration) { r.Clock.Body.Domains[0].Family = "wall-time-is-device-time" }},
		{"clock calibration", func(r *Registration) { r.Clock.Body.Domains[0].Calibration = RegistrationArtifact{} }},
		{"clock uncertainty", func(r *Registration) { r.Clock.Body.Domains[0].Uncertainty = -1 }},
		{"clock validity", func(r *Registration) { r.Clock.Body.Domains[0].ValidityEnd = -1 }},
		{"loss family", func(r *Registration) { r.Loss.Body.Families[0].Name = "" }},
		{"loss denominator", func(r *Registration) { r.Loss.Body.Families[0].DenominatorUnit = "" }},
		{"loss population", func(r *Registration) { r.Loss.Body.Families[0].ExpectedRecords = 0 }},
		{"loss budget", func(r *Registration) { r.Loss.Body.Families[0].MaxLost = 2 }},
		{"observer mode", func(r *Registration) { r.Observer.Body.Mode = "guess" }},
		{"observer scope", func(r *Registration) { r.Observer.Body.SensorScope = nil }},
		{"observer duty", func(r *Registration) { r.Observer.Body.DutyCycle = 2 }},
		{"observer freshness", func(r *Registration) { r.Observer.Body.FreshnessNanos = 0 }},
		{"observer access", func(r *Registration) { r.Observer.Body.Access = RegistrationFact{} }},
		{"observer arm", func(r *Registration) { r.Observer.Body.Arm = "absent" }},
		{"repetitions", func(r *Registration) { r.Analysis.Body.Repetitions = 0 }},
		{"minimum samples", func(r *Registration) { r.Analysis.Body.MinSamples = 2 }},
		{"order", func(r *Registration) { r.Analysis.Body.Order = "guess" }},
		{"seed", func(r *Registration) { r.Analysis.Body.Seed = nil }},
		{"fixed order", func(r *Registration) { r.Analysis.Body.Order = "fixed" }},
		{"method", func(r *Registration) { r.Analysis.Body.Method = RegistrationArtifact{} }},
		{"property", func(r *Registration) { r.Analysis.Body.Properties[0].Property = "latency-p99" }},
		{"property denominator", func(r *Registration) { r.Analysis.Body.Properties[0].Denominator = 0 }},
		{"property direction", func(r *Registration) { r.Analysis.Body.Properties[0].Direction = "higher" }},
		{"property family", func(r *Registration) { r.Analysis.Body.Properties[0].TimingFamily = "device-event" }},
		{"property clock", func(r *Registration) { r.Analysis.Body.Properties[0].ClockDomain = "absent" }},
		{"property unit", func(r *Registration) { r.Analysis.Body.Properties[0].Unit = "ticks" }},
		{"uncertainty kind", func(r *Registration) { r.Analysis.Body.Properties[0].UncertaintyKind = "95ci" }},
		{"uncertainty method", func(r *Registration) { r.Analysis.Body.Properties[0].UncertaintyMethod = RegistrationArtifact{} }},
		{"overlap", func(r *Registration) { r.Analysis.Body.Properties[0].Overlap = "equivalent" }},
		{"threshold", func(r *Registration) { r.Analysis.Body.Properties[0].MaxValue = 2 }},
		{"stopping", func(r *Registration) { r.Stopping.Body.Rule = "guess" }},
		{"stop samples", func(r *Registration) { r.Stopping.Body.MaxSamples = 2 }},
		{"stop duration", func(r *Registration) { r.Stopping.Body.MaxDurationNanos = 2e9 }},
		{"sequential method", func(r *Registration) { r.Stopping.Body.Rule = "bounded-sequential" }},
		{"missing denominator", func(r *Registration) { r.MissingData.Body.Denominator = "drop-withheld" }},
		{"missing budget", func(r *Registration) { r.MissingData.Body.MaxWithheld = 1 }},
		{"missing method", func(r *Registration) { r.MissingData.Body.Handling = "bounded" }},
		{"cell applicability", func(r *Registration) { r.SelectedCells[0].Applicability = "" }},
		{"source binary uppercase", func(r *Registration) { r.Source.BinarySHA256 = strings.Repeat("A", 64) }},
		{"source base uppercase", func(r *Registration) { r.Source.Base = strings.Repeat("A", 40) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, _, _ := registrationFixture(t)
			tc.change(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid typed contract accepted")
			}
		})
	}
}

func TestRegistrationUnknownContractsAndRequiredFacts(t *testing.T) {
	for _, name := range []string{"workload", "oracle", "clock", "loss", "observer", "analysis", "stopping", "missing", "access", "conflict", "lease", "power", "contention"} {
		t.Run(name, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			c := RegistrationContract{State: "unknown", Reason: "no accepted contract", Owner: "prof", NextAction: "supply real input"}
			f := RegistrationFact{State: "unknown", Reason: "no acquisition", Owner: "prof", NextAction: "supply real facts"}
			switch name {
			case "workload":
				r.Workload = RegistrationWorkload{Contract: c}
			case "oracle":
				r.Oracle = RegistrationOracle{Contract: c}
			case "clock":
				r.Clock = RegistrationClock{Contract: c}
			case "loss":
				r.Loss = RegistrationLoss{Contract: c}
			case "observer":
				r.Observer = RegistrationObserver{Contract: c}
			case "analysis":
				r.Analysis = RegistrationAnalysis{Contract: c}
			case "stopping":
				r.Stopping = RegistrationStopping{Contract: c}
			case "missing":
				r.MissingData = RegistrationMissingData{Contract: c}
			case "access":
				r.Observer.Body.Access = f
			case "conflict":
				r.Observer.Body.OwnerConflicts = []RegistrationFact{f}
			case "lease":
				r.Workload.Body.HardwareLease = f
			case "power":
				r.Workload.Body.PowerSettings = f
			case "contention":
				r.Workload.Body.Contention = f
			}
			frozen, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), frozen, l, root)
			if err != nil {
				t.Fatal(err)
			}
			if out.Cells[0].ReceiptConsistency || out.Cells[0].Status != "pending" {
				t.Fatal("unknown required fact accepted", out)
			}
		})
	}
}

func TestRegistrationPendingPrerequisitePropagation(t *testing.T) {
	r, l, root := registrationFixture(t)
	child := l.Cells[0]
	child.ID = "dependent"
	child.DependsOn = []string{"host"}
	child.Attempts = append([]QualificationAttempt(nil), child.Attempts...)
	ev := child.Attempts[0].Evidence
	ev.CellID = child.ID
	b, _ := json.Marshal(ev)
	child.Attempts[0] = QualificationAttempt{Path: "dependent.json", SHA256: registrationSHA(b), Evidence: ev}
	if err := os.WriteFile(filepath.Join(root, "dependent.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	l.Cells = append(l.Cells, child)
	var lb bytes.Buffer
	if err := WriteQualificationLedger(&lb, l); err != nil {
		t.Fatal(err)
	}
	r.PrerequisiteLedgerSHA256 = registrationSHA(lb.Bytes())
	sel := r.SelectedCells[0]
	sel.CellID = child.ID
	sel.ReceiptSHA256 = registrationSHA(b)
	r.SelectedCells = append(r.SelectedCells, sel)
	r.SelectedCells[0].ReceiptSHA256 = ""
	r.SelectedCells[0].Reason = "selected prerequisite not bound"
	r, err := r.Freeze(time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	out, err := VerifyRegistration(context.Background(), r, l, root)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out.Cells {
		if v.ReceiptConsistency {
			t.Fatal("dependent survived pending prerequisite", out)
		}
	}
}

func TestRegistrationArtifactConsistency(t *testing.T) {
	for _, name := range []string{"missing", "corrupt", "pilot same path", "different policy digest", "symlink", "fifo"} {
		t.Run(name, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			ref := r.Oracle.Body.Implementation
			switch name {
			case "missing":
				if err := os.Remove(filepath.Join(root, ref.Path)); err != nil {
					t.Fatal(err)
				}
			case "corrupt":
				if err := os.WriteFile(filepath.Join(root, ref.Path), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			case "pilot same path":
				r.PilotArtifacts = []RegistrationArtifact{ref}
				ref.SHA256 = strings.Repeat("c", 64)
				r.HeldOutArtifacts = []RegistrationArtifact{ref}
			case "different policy digest":
				r.Oracle.Body.Implementation.SHA256 = strings.Repeat("c", 64)
			case "symlink":
				if err := os.Rename(filepath.Join(root, ref.Path), filepath.Join(root, "real.txt")); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink("real.txt", filepath.Join(root, ref.Path)); err != nil {
					t.Fatal(err)
				}
			case "fifo":
				if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
					t.Skip("safe acquisition unavailable")
				}
				if err := os.Remove(filepath.Join(root, ref.Path)); err != nil {
					t.Fatal(err)
				}
				if err := exec.Command("mkfifo", "-m", "600", filepath.Join(root, ref.Path)).Run(); err != nil {
					t.Fatal(err)
				}
			}
			r, err := r.Freeze(time.Now().UTC())
			if name == "pilot same path" || name == "different policy digest" {
				if err == nil {
					t.Fatal("contradictory rooted identity accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), r, l, root)
			if err == nil {
				t.Fatal("bad required artifact admitted", out)
			}
		})
	}
}

func TestRegistrationKnownFactProvenance(t *testing.T) {
	r, l, root := registrationFixture(t)
	ref := r.Workload.Body.Code
	known := RegistrationFact{State: "known", Value: "declared host", SourceSHA256: ref.SHA256, Artifact: &ref, Field: "host.name", Owner: "control", NextAction: "independently decode actual field"}
	r.TargetRows[0].Applicability = "selected"
	na := r.Workload.Body.HardwareLease
	r.TargetRows[0].Host = known
	r.TargetRows[0].Device = na
	r.TargetRows[0].Stack = na
	r.TargetRows[0].Topology = na
	r.TargetRows[0].Workload = na
	r.TargetRows[0].Deployment = na
	r.TargetRows[0].Observer = na
	r.SourceLibraries = []RegistrationLibrary{{Name: "inert", Fact: known}}
	r.Source.Libraries = map[string]string{"inert": known.Value}
	// The registration source map is not the immutable ledger closure map.
	f, err := r.Freeze(time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	out, err := VerifyRegistration(context.Background(), f, l, root)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Cells[0].ReceiptConsistency || out.FactBinding != "artifact bytes checked; field/value binding declared" || len(out.SourceLibraries) != 1 {
		t.Fatal(out)
	}
	out.SourceLibraries[0].Fact.Artifact.Path = "report-only-mutation"
	if f.SourceLibraries[0].Fact.Artifact.Path == "report-only-mutation" {
		t.Fatal("report library provenance aliases frozen registration")
	}
	for _, name := range []string{"field", "artifact", "digest", "library"} {
		t.Run(name, func(t *testing.T) {
			x := f
			x.Digest = ""
			x.RegisteredAt = time.Time{}
			b, _ := json.Marshal(x)
			if err := json.Unmarshal(b, &x); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "field":
				x.TargetRows[0].Host.Field = ""
			case "artifact":
				x.TargetRows[0].Host.Artifact = nil
			case "digest":
				x.TargetRows[0].Host.SourceSHA256 = strings.Repeat("a", 64)
			case "library":
				x.Source.Libraries["inert"] = "mismatch"
			}
			if err := x.Validate(); err == nil {
				t.Fatal("unbound fact accepted")
			}
		})
	}
}

func TestRegistrationSnapshotIsolation(t *testing.T) {
	r, l, root := frozenRegistration(t)
	snapshot, cleanup, _, err := registrationSnapshot(context.Background(), r, l, root)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	info, err := os.Stat(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0700 {
		t.Fatal("snapshot mode", info.Mode())
	}
	for _, name := range []string{"raw.txt", "receipt.json", "contract.txt"} {
		before, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		copy, err := os.ReadFile(filepath.Join(snapshot, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, copy) {
			t.Fatal("snapshot changed bytes")
		}
		info, _ := os.Stat(filepath.Join(snapshot, name))
		if info.Mode().Perm() != 0600 {
			t.Fatal("snapshot file mode")
		}
	}
	if err := os.WriteFile(filepath.Join(root, "raw.txt"), []byte("swapped after acquisition"), 0600); err != nil {
		t.Fatal(err)
	}
	out, err := VerifyQualificationLedger(context.Background(), snapshot, l)
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Accepted {
		t.Fatal("immutable snapshot reread original", out)
	}
}

func TestRegistrationEveryFieldFrozen(t *testing.T) {
	r, _, _ := frozenRegistration(t)
	var paths []string
	var collect func(reflect.Value, string)
	collect = func(v reflect.Value, path string) {
		switch v.Kind() {
		case reflect.Pointer:
			if !v.IsNil() {
				collect(v.Elem(), path)
			}
		case reflect.Struct:
			if v.Type() == reflect.TypeFor[time.Time]() {
				paths = append(paths, path)
				return
			}
			for i := 0; i < v.NumField(); i++ {
				if v.Type().Field(i).IsExported() {
					collect(v.Field(i), path+"."+v.Type().Field(i).Name)
				}
			}
		case reflect.Slice:
			for i := 0; i < v.Len(); i++ {
				collect(v.Index(i), fmt.Sprintf("%s[%d]", path, i))
			}
		case reflect.Map:
			for _, key := range v.MapKeys() {
				collect(v.MapIndex(key), path+"["+key.String()+"]")
			}
		case reflect.String, reflect.Bool, reflect.Int, reflect.Int64, reflect.Uint64, reflect.Float64:
			paths = append(paths, path)
		}
	}
	// Mutate each primitive in an independently detached copy; every exact frozen
	// field, including zero seed/warmup/bounds, must affect validation/digest.
	var mutate func(reflect.Value, string, string) bool
	mutate = func(v reflect.Value, path, target string) bool {
		if path == target && v.Kind() != reflect.Pointer {
			switch v.Kind() {
			case reflect.Struct:
				if v.Type() == reflect.TypeFor[time.Time]() {
					v.Set(reflect.ValueOf(v.Interface().(time.Time).Add(time.Nanosecond)))
				}
			case reflect.String:
				v.SetString(v.String() + "x")
			case reflect.Bool:
				v.SetBool(!v.Bool())
			case reflect.Int, reflect.Int64:
				v.SetInt(v.Int() + 1)
			case reflect.Uint64:
				v.SetUint(v.Uint() + 1)
			case reflect.Float64:
				v.SetFloat(v.Float() + 1)
			}
			return true
		}
		switch v.Kind() {
		case reflect.Pointer:
			if !v.IsNil() {
				return mutate(v.Elem(), path, target)
			}
		case reflect.Struct:
			if v.Type() == reflect.TypeFor[time.Time]() {
				return false
			}
			for i := 0; i < v.NumField(); i++ {
				if v.Type().Field(i).IsExported() && mutate(v.Field(i), path+"."+v.Type().Field(i).Name, target) {
					return true
				}
			}
		case reflect.Slice:
			for i := 0; i < v.Len(); i++ {
				if mutate(v.Index(i), fmt.Sprintf("%s[%d]", path, i), target) {
					return true
				}
			}
		case reflect.Map:
			for _, key := range v.MapKeys() {
				x := reflect.New(v.Type().Elem()).Elem()
				x.Set(v.MapIndex(key))
				if mutate(x, path+"["+key.String()+"]", target) {
					v.SetMapIndex(key, x)
					return true
				}
			}
		}
		return false
	}
	collect(reflect.ValueOf(r), "registration")
	t.Logf("mutated %d concrete primitive/map/timestamp fields", len(paths))
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			b, _ := json.Marshal(r)
			var x Registration
			if err := json.Unmarshal(b, &x); err != nil {
				t.Fatal(err)
			}
			if !mutate(reflect.ValueOf(&x).Elem(), "registration", path) {
				t.Fatal("field not mutated")
			}
			if err := x.Validate(); err == nil {
				t.Fatal("semantic frozen field mutation admitted")
			}
		})
	}
}

func TestRegistrationNoFollowOpenRace(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("safe acquisition unavailable")
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "dir"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dir", "file"), []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "file"), []byte("redirected"), 0600); err != nil {
		t.Fatal(err)
	}
	opener, err := registrationOpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer opener.close()
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if os.Rename(filepath.Join(root, "dir"), filepath.Join(root, "held")) == nil {
				os.Symlink(outside, filepath.Join(root, "dir"))
				os.Remove(filepath.Join(root, "dir"))
				os.Rename(filepath.Join(root, "held"), filepath.Join(root, "dir"))
			}
		}
	}()
	defer func() { close(stop); wg.Wait() }()
	for i := 0; i < 200; i++ {
		f, err := opener.open(context.Background(), "dir/file")
		if err != nil {
			continue
		}
		b, err := io.ReadAll(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "original" {
			t.Fatal("race followed swapped symlink", string(b))
		}
	}
}

func TestRegistrationArtifactReadBounds(t *testing.T) {
	for _, name := range []string{"per-file", "aggregate", "cancel"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "file")
			if err := os.WriteFile(path, []byte("bounded"), 0600); err != nil {
				t.Fatal(err)
			}
			f, err := os.OpenFile(path, os.O_RDWR, 0600)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			remaining := int64(64 << 20)
			ctx := context.Background()
			switch name {
			case "per-file":
				if err := f.Truncate((16 << 20) + 1); err != nil {
					t.Fatal(err)
				}
			case "aggregate":
				remaining = 1
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			if _, err := registrationReadArtifact(ctx, f, &remaining); err == nil {
				t.Fatal("read bound bypassed")
			}
		})
	}
	r, _, _ := registrationFixture(t)
	a := r.Workload.Body.Code
	r.PilotArtifacts = make([]RegistrationArtifact, 4097)
	for i := range r.PilotArtifacts {
		r.PilotArtifacts[i] = RegistrationArtifact{fmt.Sprintf("%d", i), a.SHA256}
	}
	if err := r.Validate(); err == nil {
		t.Fatal("aggregate artifact count bypassed")
	}
}

func TestRegistrationOpaquePrototypeRefusal(t *testing.T) {
	r, _, _ := registrationFixture(t)
	b, _ := json.Marshal(r)
	b = bytes.Replace(b, []byte(`"workload":{"contract":`), []byte(`"workload":{"specification":"opaque","contract":`), 1)
	if _, err := ReadRegistration(bytes.NewReader(b)); err == nil {
		t.Fatal("old opaque prototype admitted")
	}
}

func TestRegistrationMetricTimingDistinction(t *testing.T) {
	for _, unit := range []string{"cpu-percent", "MiB", "errors/request"} {
		t.Run(unit, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			p := &r.Analysis.Body.Properties[0]
			p.Unit = unit
			p.TimingFamily = "not-applicable"
			p.ClockDomain = ""
			r, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), r, l, root)
			if err != nil {
				t.Fatal(err)
			}
			if !out.Cells[0].ReceiptConsistency {
				t.Fatal("non-temporal unit coerced to clock", out)
			}
		})
	}
	for _, name := range []string{"device-vs-host", "temporal-metric-unit", "non-temporal-domain"} {
		t.Run(name, func(t *testing.T) {
			r, _, _ := registrationFixture(t)
			p := &r.Analysis.Body.Properties[0]
			switch name {
			case "device-vs-host":
				p.TimingFamily = "device-event"
			case "temporal-metric-unit":
				p.Unit = "cpu-percent"
			case "non-temporal-domain":
				p.TimingFamily = "not-applicable"
			}
			if err := r.Validate(); err == nil {
				t.Fatal("timing mismatch admitted")
			}
		})
	}
}

func TestRegistrationApplicabilitySeparateFromCapability(t *testing.T) {
	for _, app := range []string{"not-selected", "unsupported"} {
		t.Run(app, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			r.SelectedCells[0].Applicability = app
			r.SelectedCells[0].Reason = "explicit task selection"
			f, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), f, l, root)
			if err != nil {
				t.Fatal(err)
			}
			host := out.Cells[0]
			if host.Applicability != app || host.Capability != "supported" || host.ReceiptConsistency || host.Selected {
				t.Fatal("selection changed discovery or admitted unselected evidence", out)
			}
		})
	}
}

func TestRegistrationSnapshotAggregateBudgetAndCleanup(t *testing.T) {
	r, l, root := registrationFixture(t)
	data := make([]byte, 16<<20)
	digest := registrationSHA(data)
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("budget-%d", i)
		f, err := os.OpenFile(filepath.Join(root, name), os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Truncate(16 << 20); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		r.Workload.Contract.Artifacts = append(r.Workload.Contract.Artifacts, RegistrationArtifact{name, digest})
	}
	temp := t.TempDir()
	t.Setenv("TMPDIR", temp)
	f, err := r.Freeze(time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyRegistration(context.Background(), f, l, root); err == nil || !strings.Contains(err.Error(), "aggregate budget") {
		t.Fatal("64 MiB aggregate was not enforced", err)
	}
	dirs, err := os.ReadDir(temp)
	if err != nil {
		t.Fatal(err)
	}
	if len(dirs) != 0 {
		t.Fatal("failed snapshot left private copies", dirs)
	}
}

func TestRegistrationRequiredFactDispositions(t *testing.T) {
	for _, state := range []string{"unknown", "unavailable", "denied", "conflicting", "not-applicable"} {
		t.Run(state, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			r.Observer.Body.Access = RegistrationFact{State: state, Reason: "explicit supplied task fact", Owner: "prof", NextAction: "retain disposition"}
			f, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), f, l, root)
			if err != nil {
				t.Fatal(err)
			}
			if out.Cells[0].ReceiptConsistency != (state == "not-applicable") {
				t.Fatal("required unresolved access acquired consistency", out)
			}
		})
	}
}

func TestRegistrationRecordedCommandsAreData(t *testing.T) {
	r, l, root := registrationFixture(t)
	sentinel := filepath.Join(root, "must-not-exist")
	ev := l.Cells[0].Attempts[0].Evidence
	ev.Command = []string{"sh", "-c", "touch " + sentinel}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "receipt.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	l.Cells[0].Attempts[0] = QualificationAttempt{Path: "receipt.json", SHA256: registrationSHA(b), Evidence: ev}
	r.SelectedCells[0].ReceiptSHA256 = registrationSHA(b)
	var lb bytes.Buffer
	if err := WriteQualificationLedger(&lb, l); err != nil {
		t.Fatal(err)
	}
	r.PrerequisiteLedgerSHA256 = registrationSHA(lb.Bytes())
	f, err := r.Freeze(time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	out, err := VerifyRegistration(context.Background(), f, l, root)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Cells[0].ReceiptConsistency {
		t.Fatal(out)
	}
	if _, err := os.Stat(sentinel); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("recorded command executed", err)
	}
}

func TestRegistrationComparisonSelectedPropertySubset(t *testing.T) {
	r, l, root := registrationFixture(t)
	r.Mode = "comparison"
	r.Task = "selected cost only"
	r.Protocol.Tools = []ToolSpec{{Name: "candidate", Version: "1", Configuration: "inert", Lineage: "one", Support: "measured"}, {Name: "comparator", Version: "2", Configuration: "inert", Lineage: "two", Support: "measured"}}
	r.Protocol.PropertyLimits = map[string]float64{"cost": 1, "attribution": 0.01, "diagnosis": 0.02, "reproducibility": 0.03}
	r.Protocol.Comparison = &ComparisonSpec{Task: r.Task, Candidate: "candidate", Comparator: "comparator", Properties: map[string]MeasurementScope{"cost": {Unit: "ns", Denominator: 1}}}
	r.Protocol.Frozen = false
	r.Protocol.FrozenDigest = ""
	r.Protocol.FrozenTimestamp = time.Time{}
	if err := r.Protocol.Freeze(); err != nil {
		t.Fatal(err)
	}
	r.Observer.Body.Arm = "candidate"
	r.Observer.Body.ReferenceArm = "comparator"
	f, err := r.Freeze(time.Now().UTC())
	if err != nil {
		t.Fatal("unselected protocol properties became mandatory", err)
	}
	out, err := VerifyRegistration(context.Background(), f, l, root)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Cells[0].ReceiptConsistency || out.MeasurementAcceptance != "unverified" {
		t.Fatal(out)
	}
}

func TestRegistrationOptionalUnselectedArtifacts(t *testing.T) {
	for _, kind := range []string{"missing", "corrupt", "contradictory"} {
		t.Run(kind, func(t *testing.T) {
			r, l, root := registrationFixture(t)
			r.SelectedCells = r.SelectedCells[:1]
			cell := l.Cells[0]
			cell.ID = "optional"
			cell.Attempts = append([]QualificationAttempt(nil), cell.Attempts...)
			ev := cell.Attempts[0].Evidence
			ev.CellID = cell.ID
			ev.RawPath = "optional.raw"
			data := []byte("old optional source")
			ev.RawSHA256 = registrationSHA(data)
			b, _ := json.Marshal(ev)
			cell.Attempts[0] = QualificationAttempt{Path: "optional.json", SHA256: registrationSHA(b), Evidence: ev}
			if err := os.WriteFile(filepath.Join(root, "optional.json"), b, 0600); err != nil {
				t.Fatal(err)
			}
			if kind == "corrupt" {
				if err := os.WriteFile(filepath.Join(root, "optional.raw"), []byte("changed old source"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if kind == "contradictory" {
				cell.References = []QualificationReference{{Path: "contract.txt", SHA256: strings.Repeat("c", 64), Scope: "historical", Reason: "different original digest"}}
			}
			l.Cells = append(l.Cells, cell)
			var lb bytes.Buffer
			if err := WriteQualificationLedger(&lb, l); err != nil {
				t.Fatal(err)
			}
			r.PrerequisiteLedgerSHA256 = registrationSHA(lb.Bytes())
			before := append([]byte(nil), lb.Bytes()...)
			f, err := r.Freeze(time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			out, err := VerifyRegistration(context.Background(), f, l, root)
			if kind == "contradictory" {
				if err == nil {
					t.Fatal("optional conflict redefined required identity")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var host, optional RegistrationResult
			for _, v := range out.Cells {
				if v.CellID == "host" {
					host = v
				}
				if v.CellID == cell.ID {
					optional = v
				}
			}
			if !host.ReceiptConsistency || optional.ReceiptConsistency || optional.Selected || optional.Status != "pending" || out.ReceiptConsistency {
				t.Fatal("optional outcome erased or promoted scope", out)
			}
			found := false
			for _, a := range out.Artifacts {
				if a.Artifact.Path == ev.RawPath {
					found = true
					if a.Required || a.Status != kind || a.Artifact.SHA256 != ev.RawSHA256 || a.Reason == "" {
						t.Fatal("optional expected identity/outcome changed", a)
					}
				}
			}
			if !found {
				t.Fatal("optional omission unrecorded")
			}
			lb.Reset()
			if err := WriteQualificationLedger(&lb, l); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, lb.Bytes()) {
				t.Fatal("ledger history rewritten")
			}
		})
	}
}

func TestRegistrationMandatorySelectedArtifacts(t *testing.T) {
	for _, kind := range []string{"missing", "corrupt"} {
		t.Run(kind, func(t *testing.T) {
			r, l, root := frozenRegistration(t)
			if kind == "missing" {
				if err := os.Remove(filepath.Join(root, "raw.txt")); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.WriteFile(filepath.Join(root, "raw.txt"), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if out, err := VerifyRegistration(context.Background(), r, l, root); err == nil || out.ReceiptConsistency {
				t.Fatal("required original omission admitted", out, err)
			}
		})
	}
}

func TestRegistrationExplicitZeroFields(t *testing.T) {
	r, _, _ := frozenRegistration(t)
	b, _ := json.Marshal(r)
	for _, field := range []string{`"uncertainty":0,`, `"warmup":0,`, `"max_missing":0,`, `"notes":"",`} {
		t.Run(field, func(t *testing.T) {
			changed := bytes.Replace(b, []byte(field), nil, 1)
			if bytes.Equal(changed, b) {
				t.Fatal("zero-field control did not omit a field")
			}
			if _, err := ReadRegistration(bytes.NewReader(changed)); err == nil {
				t.Fatal("missing semantic field silently became zero")
			}
		})
	}
}

func TestRegistrationNullValues(t *testing.T) {
	r, _, _ := frozenRegistration(t)
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{`"warmup":0`, `"uncertainty":0`, `"duty_cycle":0`, `"max_missing":0`, `"max_lost":0`, `"notes":""`, `"required":true`, `"seed":0`, `"validity_start":0`, `"max_withheld":0`} {
		t.Run(value, func(t *testing.T) {
			colon := strings.IndexByte(value, ':')
			changed := bytes.Replace(raw, []byte(value), []byte(value[:colon+1]+"null"), 1)
			if bytes.Equal(changed, raw) {
				t.Fatal("null mutation did not apply")
			}
			if _, err := ReadRegistration(bytes.NewReader(changed)); err == nil {
				t.Fatal("null admitted as a semantic value")
			}
		})
	}
	for _, field := range []string{"registered_at", "frozen_timestamp"} {
		t.Run(field, func(t *testing.T) {
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(raw, &wire); err != nil {
				t.Fatal(err)
			}
			if field == "registered_at" {
				wire[field] = json.RawMessage("null")
			} else {
				var protocol map[string]json.RawMessage
				if err := json.Unmarshal(wire["protocol"], &protocol); err != nil {
					t.Fatal(err)
				}
				protocol[field] = json.RawMessage("null")
				wire["protocol"], err = json.Marshal(protocol)
				if err != nil {
					t.Fatal(err)
				}
			}
			changed, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ReadRegistration(bytes.NewReader(changed)); err == nil {
				t.Fatal("null timestamp admitted")
			}
		})
	}
	// False booleans and zero strings/numbers must not accept null either.
	for _, typ := range []reflect.Type{reflect.TypeFor[bool](), reflect.TypeFor[string](), reflect.TypeFor[int](), reflect.TypeFor[uint64](), reflect.TypeFor[float64](), reflect.TypeFor[time.Time]()} {
		if err := registrationKeys(json.RawMessage("null"), typ); err == nil {
			t.Fatalf("null accepted for %v", typ)
		}
	}
	// Planned bodies, optional pointers and collections retain their null forms.
	planned, _, _ := registrationFixture(t)
	var b bytes.Buffer
	if err := WriteRegistration(&b, planned); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadRegistration(&b); err != nil {
		t.Fatal(err)
	}
	for _, typ := range []reflect.Type{reflect.TypeFor[*uint64](), reflect.TypeFor[[]RegistrationArtifact](), reflect.TypeFor[map[string]string]()} {
		if err := registrationKeys(json.RawMessage("null"), typ); err != nil {
			t.Fatalf("legal null for %v: %v", typ, err)
		}
	}
}
