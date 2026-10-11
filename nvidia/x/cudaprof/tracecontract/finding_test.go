package tracecontract

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func validReceiptFixture() Receipt {
	r := receiptFixture()
	r.Backends = []Backend{{Name: "native", Version: "12.6"}}
	r.ProbeCoverage = map[string]ProbeCoverage{
		"driver_api":     {Present: true, Ratio: 1.0},
		"cupti_activity": {Present: true, Ratio: 1.0},
	}
	r.LostRecords = map[string]Loss{
		"activity": {Dropped: 0, Total: 100, Unknown: false},
		"call":     {Dropped: 0, Total: 50, Unknown: false},
		"stack":    {Dropped: 0, Total: 200, Unknown: false},
	}
	r.CaptureBounds = &CaptureBounds{
		Start: 1000,
		End:   50000,
		Clock: "monotonic",
	}
	r.Correlation = &CorrelationOutcome{
		Matched:   50,
		Unmatched: 0,
		Ambiguous: 0,
		Missing:   0,
		Status:    "exact",
	}
	r.ClockDomain = "monotonic"
	r.ClockProvenance = "cupti_timestamp"
	unc := int64(10)
	r.ClockUncertaintyNS = &unc
	fUnc := 10.0
	r.Uncertainty = &fUnc
	return r
}

func TestPositiveFixtures(t *testing.T) {
	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{receipt.InputSHA256},
		Intervals:   []Interval{{Start: 2000, End: 3000, Clock: "monotonic"}},
		HasReturn:   true,
		ClockDomain: "monotonic",
	}

	// 1. Observed Launch Count finding
	reqsLaunch := []Requirement{
		{Name: "backend_native", Backend: "native", MinVersion: "12.0"},
		{Name: "activity_loss", Family: "activity", MaxLoss: 0},
		{Name: "activity_probe", Probe: "cupti_activity", MinProbeCoverage: 1.0},
	}
	val := 100.0
	launchFinding := Admit("f-launch-count", "launch_count", FindingObserved, "observed 100 launches", &val, "launches", reqsLaunch, ev)
	if !launchFinding.Admission.Admitted {
		t.Fatalf("launch finding withheld: %s (%s)", launchFinding.Admission.Reason, launchFinding.Admission.ReasonCode)
	}
	if launchFinding.Class != FindingObserved {
		t.Fatalf("launch finding class got %s, want %s", launchFinding.Class, FindingObserved)
	}
	if err := launchFinding.CheckClaim(FindingObserved); err != nil {
		t.Fatalf("CheckClaim failed: %v", err)
	}

	// 2. Associated correlation finding
	reqsCorr := []Requirement{
		{Name: "correlation_valid", RequireCorrelation: true},
		{Name: "clock_domain", ClockDomain: "monotonic"},
	}
	corrFinding := Admit("f-host-device-corr", "host_device_correlation", FindingAssociated, "associated submission to kernel", nil, "", reqsCorr, ev)
	if !corrFinding.Admission.Admitted {
		t.Fatalf("correlation finding withheld: %s (%s)", corrFinding.Admission.Reason, corrFinding.Admission.ReasonCode)
	}
	if corrFinding.Class != FindingAssociated {
		t.Fatalf("correlation finding class got %s, want %s", corrFinding.Class, FindingAssociated)
	}
	if err := corrFinding.CheckClaim(FindingAssociated); err != nil {
		t.Fatalf("CheckClaim failed: %v", err)
	}

	// 3. Experimentally confirmed bottleneck finding
	inv := InterventionReceipt{
		InterventionSHA256: strings.Repeat("1", 64),
		ControlSHA256:      strings.Repeat("2", 64),
		InterventionName:   "clock_lock_2100MHz",
		ControlName:        "clock_default",
		Outcome:            "bottleneck_shifted_to_memory",
		Verified:           true,
	}
	evConfirmed := ev
	evConfirmed.Intervention = &inv
	reqsConfirmed := []Requirement{
		{Name: "intervention_control", RequireControl: true},
	}
	confirmedFinding := Admit("f-bottleneck", "compute_bottleneck", FindingExperimentallyConfirmed, "experimentally confirmed compute bound", nil, "", reqsConfirmed, evConfirmed)
	if !confirmedFinding.Admission.Admitted {
		t.Fatalf("confirmed finding withheld: %s (%s)", confirmedFinding.Admission.Reason, confirmedFinding.Admission.ReasonCode)
	}
	if confirmedFinding.Class != FindingExperimentallyConfirmed {
		t.Fatalf("confirmed finding class got %s, want %s", confirmedFinding.Class, FindingExperimentallyConfirmed)
	}
	if err := confirmedFinding.CheckClaim(FindingExperimentallyConfirmed); err != nil {
		t.Fatalf("CheckClaim failed: %v", err)
	}
}

// Acceptance 1: Missing return blocks that call duration; lost stack records need not invalidate launch count.
func TestMissingReturnAndStackLossIsolation(t *testing.T) {
	receipt := validReceiptFixture()
	// Introduce stack loss into the receipt
	receipt.LostRecords["stack"] = Loss{Dropped: 50, Total: 100, Unknown: false}
	receipt.Partial = true

	// Call evidence: has NO return event (HasReturn: false)
	ev := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{receipt.InputSHA256},
		Intervals:   []Interval{{Start: 2000, End: 3000, Clock: "monotonic"}},
		HasReturn:   false,
		ClockDomain: "monotonic",
	}

	// Finding 1: Call duration requires return event
	reqsDuration := []Requirement{
		{Name: "call_loss", Family: "call", MaxLoss: 0},
		{Name: "call_return", RequireReturn: true},
	}
	durationFinding := Admit("f-call-duration", "call_duration", FindingObserved, "call duration 1000ns", nil, "ns", reqsDuration, ev)
	if durationFinding.Admission.Admitted {
		t.Fatal("call duration without return was admitted, want withheld")
	}
	if durationFinding.Class != FindingUnavailable {
		t.Fatalf("call duration class got %s, want %s", durationFinding.Class, FindingUnavailable)
	}
	if durationFinding.Admission.ReasonCode != ReasonMissingReturn {
		t.Fatalf("call duration reason code got %q, want %q", durationFinding.Admission.ReasonCode, ReasonMissingReturn)
	}

	// Finding 2: Launch count does NOT require call return, and does NOT require stack records
	reqsLaunch := []Requirement{
		{Name: "activity_loss", Family: "activity", MaxLoss: 0},
	}
	val := 100.0
	launchFinding := Admit("f-launch-count", "launch_count", FindingObserved, "observed 100 launches", &val, "launches", reqsLaunch, ev)
	if !launchFinding.Admission.Admitted {
		t.Fatalf("launch finding was unexpectedly withheld despite complete activity records: %s (%s)", launchFinding.Admission.Reason, launchFinding.Admission.ReasonCode)
	}
	if launchFinding.Class != FindingObserved {
		t.Fatalf("launch finding class got %s, want %s", launchFinding.Class, FindingObserved)
	}

	// Finding 3: Stack profile finding DOES require stack records; should be withheld due to stack loss
	reqsStack := []Requirement{
		{Name: "stack_loss", Family: "stack", MaxLoss: 0},
	}
	stackFinding := Admit("f-stack-profile", "stack_profile", FindingObserved, "stack profile", nil, "", reqsStack, ev)
	if stackFinding.Admission.Admitted {
		t.Fatal("stack profile admitted despite stack record loss")
	}
	if stackFinding.Admission.ReasonCode != ReasonLossExceeded {
		t.Fatalf("stack finding reason code got %q, want %q", stackFinding.Admission.ReasonCode, ReasonLossExceeded)
	}
}

// Acceptance 2: Changing clocks blocks correlation, but unrelated findings remain admissible.
func TestClockDomainMismatch(t *testing.T) {
	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{receipt.InputSHA256},
		Intervals:   []Interval{{Start: 2000, End: 3000, Clock: "wall_time"}},
		HasReturn:   true,
		ClockDomain: "wall_time", // Evidence clock differs from requirement "monotonic"
	}

	reqsCorr := []Requirement{
		{Name: "clock_domain", ClockDomain: "monotonic"},
		{Name: "correlation", RequireCorrelation: true},
	}
	corrFinding := Admit("f-corr", "correlation", FindingAssociated, "correlated events", nil, "", reqsCorr, ev)
	if corrFinding.Admission.Admitted {
		t.Fatal("correlation finding admitted with mismatched clock domain")
	}
	if corrFinding.Admission.ReasonCode != ReasonClockDomainMismatch {
		t.Fatalf("reason code got %q, want %q", corrFinding.Admission.ReasonCode, ReasonClockDomainMismatch)
	}

	// Unrelated finding that does not depend on clock domain remains admissible
	reqsCount := []Requirement{
		{Name: "activity_loss", Family: "activity", MaxLoss: 0},
	}
	val := 10.0
	countFinding := Admit("f-count", "launch_count", FindingObserved, "10 launches", &val, "launches", reqsCount, ev)
	if !countFinding.Admission.Admitted {
		t.Fatalf("count finding withheld: %s", countFinding.Admission.Reason)
	}
}

// Acceptance 3: Introducing unknown loss blocks loss-sensitive findings; independent families remain admissible.
func TestUnknownLossWithholding(t *testing.T) {
	receipt := validReceiptFixture()
	// Set activity loss to unknown
	receipt.LostRecords["activity"] = Loss{Dropped: 0, Total: 100, Unknown: true}

	ev := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		HasReturn: true,
	}

	// Finding requiring activity without unknown loss
	reqsActivity := []Requirement{
		{Name: "activity_loss", Family: "activity", MaxLoss: 0, AllowUnknownLoss: false},
	}
	actFinding := Admit("f-activity", "activity_count", FindingObserved, "activities", nil, "", reqsActivity, ev)
	if actFinding.Admission.Admitted {
		t.Fatal("finding admitted despite unknown activity loss")
	}
	if actFinding.Admission.ReasonCode != ReasonUnknownLoss {
		t.Fatalf("reason code got %q, want %q", actFinding.Admission.ReasonCode, ReasonUnknownLoss)
	}

	// Call family is complete with 0 loss; call finding remains admissible
	reqsCall := []Requirement{
		{Name: "call_loss", Family: "call", MaxLoss: 0},
		{Name: "call_return", RequireReturn: true},
	}
	val := 50.0
	callFinding := Admit("f-call", "call_count", FindingObserved, "50 calls", &val, "calls", reqsCall, ev)
	if !callFinding.Admission.Admitted {
		t.Fatalf("call finding withheld: %s", callFinding.Admission.Reason)
	}
}

// Acceptance 4: Truncating streams / intervals exceeding capture bounds.
func TestStreamTruncationAndCaptureBounds(t *testing.T) {
	receipt := validReceiptFixture()
	// Capture bounds are [1000, 50000]
	// Interval 1 is completely within bounds [2000, 10000]
	evInBounds := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		Intervals: []Interval{{Start: 2000, End: 10000, Clock: "monotonic"}},
		HasReturn: true,
	}
	// Interval 2 extends past capture bounds [40000, 60000]
	evOutOfBounds := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		Intervals: []Interval{{Start: 40000, End: 60000, Clock: "monotonic"}},
		HasReturn: true,
	}

	reqs := []Requirement{
		{Name: "activity_loss", Family: "activity", MaxLoss: 0},
	}

	findingIn := Admit("f-in", "workload_span", FindingObserved, "in-bounds", nil, "", reqs, evInBounds)
	if !findingIn.Admission.Admitted {
		t.Fatalf("in-bounds finding withheld: %s", findingIn.Admission.Reason)
	}

	findingOut := Admit("f-out", "workload_span", FindingObserved, "out-of-bounds", nil, "", reqs, evOutOfBounds)
	if findingOut.Admission.Admitted {
		t.Fatal("out-of-bounds finding admitted")
	}
	if findingOut.Admission.ReasonCode != ReasonCaptureBoundsExceeded {
		t.Fatalf("reason code got %q, want %q", findingOut.Admission.ReasonCode, ReasonCaptureBoundsExceeded)
	}
}

// Acceptance 5: Mismatching identities.
func TestIdentityMismatch(t *testing.T) {
	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		HasReturn: true,
	}

	expectedDigest := strings.Repeat("f", 64)
	reqs := []Requirement{
		{Name: "artifact_identity", ArtifactDigest: expectedDigest},
	}
	finding := Admit("f-id", "kernel_profile", FindingObserved, "kernel profile", nil, "", reqs, ev)
	if finding.Admission.Admitted {
		t.Fatal("finding admitted with mismatched artifact identity")
	}
	if finding.Admission.ReasonCode != ReasonIdentityMismatch {
		t.Fatalf("reason code got %q, want %q", finding.Admission.ReasonCode, ReasonIdentityMismatch)
	}
}

// Acceptance 6: Changing required backend support.
func TestBackendSupportRequirement(t *testing.T) {
	receipt := validReceiptFixture() // has backend "native" version "12.6"
	ev := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		HasReturn: true,
	}

	// 1. Missing backend
	reqsMissing := []Requirement{
		{Name: "trace_processor_backend", Backend: "trace_processor_v58.2"},
	}
	fMissing := Admit("f-backend-missing", "sql_profile", FindingObserved, "sql", nil, "", reqsMissing, ev)
	if fMissing.Admission.Admitted {
		t.Fatal("finding admitted without required backend")
	}
	if fMissing.Admission.ReasonCode != ReasonBackendUnsupported {
		t.Fatalf("reason code got %q, want %q", fMissing.Admission.ReasonCode, ReasonBackendUnsupported)
	}

	// 2. Version requirement mismatch
	reqsVer := []Requirement{
		{Name: "native_backend_ver", Backend: "native", MinVersion: "13.0"},
	}
	fVer := Admit("f-backend-ver", "native_profile", FindingObserved, "profile", nil, "", reqsVer, ev)
	if fVer.Admission.Admitted {
		t.Fatal("finding admitted with insufficient backend version")
	}
	if fVer.Admission.ReasonCode != ReasonBackendVersionMismatch {
		t.Fatalf("reason code got %q, want %q", fVer.Admission.ReasonCode, ReasonBackendVersionMismatch)
	}
}

// Acceptance 7: Experimental confirmation additionally references intervention/control receipt.
func TestExperimentalConfirmationControlVerification(t *testing.T) {
	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:   receipt,
		Artifacts: []string{receipt.InputSHA256},
		HasReturn: true,
	}

	reqs := []Requirement{
		{Name: "experimental_control", RequireControl: true},
	}

	// Case 1: No intervention provided
	fNoInv := Admit("f-exp1", "speedup_claim", FindingExperimentallyConfirmed, "speedup", nil, "", reqs, ev)
	if fNoInv.Admission.Admitted {
		t.Fatal("confirmed finding admitted without intervention")
	}
	if fNoInv.Admission.ReasonCode != ReasonInterventionMissing {
		t.Fatalf("reason code got %q, want %q", fNoInv.Admission.ReasonCode, ReasonInterventionMissing)
	}

	// Case 2: Intervention present but control hash missing
	evMissingControl := ev
	evMissingControl.Intervention = &InterventionReceipt{
		InterventionSHA256: strings.Repeat("1", 64),
		ControlSHA256:      "",
		Verified:           true,
	}
	fMissingControl := Admit("f-exp2", "speedup_claim", FindingExperimentallyConfirmed, "speedup", nil, "", reqs, evMissingControl)
	if fMissingControl.Admission.Admitted {
		t.Fatal("confirmed finding admitted with empty control hash")
	}
	if fMissingControl.Admission.ReasonCode != ReasonControlMissing {
		t.Fatalf("reason code got %q, want %q", fMissingControl.Admission.ReasonCode, ReasonControlMissing)
	}

	// Case 3: Intervention present and control present, but verification failed
	evUnverified := ev
	evUnverified.Intervention = &InterventionReceipt{
		InterventionSHA256: strings.Repeat("1", 64),
		ControlSHA256:      strings.Repeat("2", 64),
		Verified:           false,
	}
	fUnverified := Admit("f-exp3", "speedup_claim", FindingExperimentallyConfirmed, "speedup", nil, "", reqs, evUnverified)
	if fUnverified.Admission.Admitted {
		t.Fatal("confirmed finding admitted with unverified intervention")
	}
	if fUnverified.Admission.ReasonCode != ReasonControlVerificationFailed {
		t.Fatalf("reason code got %q, want %q", fUnverified.Admission.ReasonCode, ReasonControlVerificationFailed)
	}
}

// Acceptance 8: Replay reproduces findings independent of record input order where order is semantically irrelevant.
func TestReplayOrderIndependence(t *testing.T) {
	receipt := validReceiptFixture()
	iv1 := Interval{Start: 1000, End: 2000, Clock: "monotonic"}
	iv2 := Interval{Start: 3000, End: 4000, Clock: "monotonic"}
	iv3 := Interval{Start: 5000, End: 6000, Clock: "monotonic"}

	art1 := strings.Repeat("a", 64)
	art2 := strings.Repeat("b", 64)
	art3 := strings.Repeat("c", 64)

	req1 := Requirement{Name: "req_backend", Backend: "native"}
	req2 := Requirement{Name: "req_activity", Family: "activity", MaxLoss: 0}
	req3 := Requirement{Name: "req_call", Family: "call", MaxLoss: 0}

	evOrderA := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{art1, art2, art3},
		Intervals:   []Interval{iv1, iv2, iv3},
		HasReturn:   true,
		ClockDomain: "monotonic",
	}

	evOrderB := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{art3, art1, art2},
		Intervals:   []Interval{iv3, iv1, iv2},
		HasReturn:   true,
		ClockDomain: "monotonic",
	}

	val := 42.0
	fA := Admit("f-replay", "deterministic_rule", FindingObserved, "replay test", &val, "units", []Requirement{req1, req2, req3}, evOrderA)
	fB := Admit("f-replay", "deterministic_rule", FindingObserved, "replay test", &val, "units", []Requirement{req3, req1, req2}, evOrderB)

	jsonA, err := fA.JSON()
	if err != nil {
		t.Fatal(err)
	}
	jsonB, err := fB.JSON()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(jsonA, jsonB) {
		t.Fatalf("replay produced non-identical JSON:\nA: %s\nB: %s", jsonA, jsonB)
	}
}

// Acceptance 9: Downstream prose/LLM summaries cannot upgrade evidence.
func TestDownstreamSummaryIntegrity(t *testing.T) {
	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{receipt.InputSHA256},
		HasReturn:   false, // Missing return!
		ClockDomain: "monotonic",
	}

	reqs := []Requirement{
		{Name: "call_return", RequireReturn: true},
	}
	finding := Admit("f-summary", "call_duration", FindingObserved, "duration", nil, "ns", reqs, ev)

	// Finding is unavailable
	if finding.Class != FindingUnavailable {
		t.Fatalf("expected unavailable, got %s", finding.Class)
	}

	// Attempt 1: Asserting observed directly via CheckClaim fails
	if err := finding.CheckClaim(FindingObserved); err == nil {
		t.Fatal("CheckClaim allowed claiming observed on withheld finding")
	}

	// Attempt 2: Manually forging Class = FindingObserved without Admitted true fails JSON validation
	forged := finding
	forged.Class = FindingObserved
	if _, err := forged.JSON(); err == nil {
		t.Fatal("forged unadmitted finding encoded without error")
	}

	// Attempt 3: Forging confirmed without verified control fails CheckClaim
	confirmed := finding
	confirmed.Admission.Admitted = true
	confirmed.Class = FindingExperimentallyConfirmed
	confirmed.Intervention = nil
	if err := confirmed.CheckClaim(FindingExperimentallyConfirmed); err == nil {
		t.Fatal("CheckClaim allowed confirmed class without intervention receipt")
	}
}

// Acceptance 10: Backward compatibility with v1 receipts without extended fields.
func TestBackwardReading(t *testing.T) {
	// A minimal v1 receipt JSON without probe_coverage, lost_records_by_family, etc.
	v1JSON := `{
		"version": 1,
		"command": "trace analyze",
		"options": {},
		"input": "capture.pprof",
		"input_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
		"detected_type": "capture",
		"backends": [{"name": "native"}],
		"gates": {
			"collector": {"disposition": "passed", "reason": "ok", "provenance": "test"},
			"import": {"disposition": "unchecked", "reason": "ok", "provenance": "test"},
			"workload": {"disposition": "unchecked", "reason": "ok", "provenance": "test"},
			"stationarity": {"disposition": "unchecked", "reason": "ok", "provenance": "test"},
			"shape": {"disposition": "unchecked", "reason": "ok", "provenance": "test"},
			"correctness": {"disposition": "unchecked", "reason": "ok", "provenance": "test"}
		},
		"warnings": [],
		"limits": {},
		"partial": false
	}`

	var r Receipt
	if err := json.Unmarshal([]byte(v1JSON), &r); err != nil {
		t.Fatalf("failed to decode v1 receipt: %v", err)
	}

	// Extended fields are nil / unassessed
	if _, ok := r.LossFor("activity"); ok {
		t.Fatal("expected LossFor to report false for unassessed receipt")
	}
	if _, ok := r.ProbeCoverageFor("driver_api"); ok {
		t.Fatal("expected ProbeCoverageFor to report false for unassessed receipt")
	}
	if r.CaptureBounds != nil {
		t.Fatal("expected nil CaptureBounds")
	}

	// Reserializing v1 receipt succeeds
	data, err := r.JSON()
	if err != nil {
		t.Fatalf("v1 receipt reserialization failed: %v", err)
	}
	if !strings.Contains(string(data), `"version":1`) {
		t.Fatal("reserialized JSON missing version")
	}
}

// Test validation invariants on extended receipt fields.
func TestReceiptExtendedInvariants(t *testing.T) {
	r := validReceiptFixture()

	// Inverted capture bounds
	r.CaptureBounds = &CaptureBounds{Start: 5000, End: 1000}
	if _, err := r.JSON(); err == nil {
		t.Fatal("accepted inverted capture bounds")
	}

	// Invalid probe coverage (> 1.0)
	r = validReceiptFixture()
	r.ProbeCoverage["driver_api"] = ProbeCoverage{Present: true, Ratio: 1.5}
	if _, err := r.JSON(); err == nil {
		t.Fatal("accepted probe coverage > 1.0")
	}

	// Invalid probe coverage (NaN)
	r = validReceiptFixture()
	r.ProbeCoverage["driver_api"] = ProbeCoverage{Present: true, Ratio: math.NaN()}
	if _, err := r.JSON(); err == nil {
		t.Fatal("accepted NaN probe coverage")
	}

	// Dropped > Total when not Unknown
	r = validReceiptFixture()
	r.LostRecords["activity"] = Loss{Dropped: 150, Total: 100, Unknown: false}
	if _, err := r.JSON(); err == nil {
		t.Fatal("accepted dropped > total for known loss")
	}

	// Negative uncertainty
	r = validReceiptFixture()
	neg := -5.0
	r.Uncertainty = &neg
	if _, err := r.JSON(); err == nil {
		t.Fatal("accepted negative uncertainty")
	}
}

// Test Writing findings to Perfetto writer.
func TestWriteFindingsPerfetto(t *testing.T) {
	var buf bytes.Buffer
	writer := perfetto.NewWriter(&buf)

	receipt := validReceiptFixture()
	ev := EvidenceContext{
		Receipt:     receipt,
		Artifacts:   []string{receipt.InputSHA256},
		HasReturn:   true,
		ClockDomain: "monotonic",
	}

	reqs := []Requirement{
		{Name: "activity_loss", Family: "activity", MaxLoss: 0},
	}
	val := 42.0
	finding := Admit("f-perfetto", "launch_count", FindingObserved, "42 launches", &val, "launches", reqs, ev)

	writer.WriteTrackDescriptor(10, "diagnostic findings", 0)
	if err := finding.Write(writer, 10, 100); err != nil {
		t.Fatalf("finding.Write failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("perfetto writer emitted 0 bytes")
	}
}
