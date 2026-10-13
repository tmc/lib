package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type syntheticAcceptance func(context.Context, EvidenceRequest) (EvidenceDecision, error)

func (f syntheticAcceptance) Check(ctx context.Context, r EvidenceRequest) (EvidenceDecision, error) {
	return f(ctx, r)
}

type ownedSyntheticRecord struct {
	*bytes.Reader
	closes, reads       atomic.Int32
	closeErr, errorRead error
}

func (r *ownedSyntheticRecord) Read(b []byte) (int, error) {
	r.reads.Add(1)
	if r.errorRead != nil {
		return 0, r.errorRead
	}
	return r.Reader.Read(b)
}
func (r *ownedSyntheticRecord) Close() error { r.closes.Add(1); return r.closeErr }

func syntheticDecision(r EvidenceRequest) (EvidenceDecision, *ownedSyntheticRecord) {
	scopes := []ScopeDecision{}
	for _, scope := range r.Roles {
		scopes = append(scopes, ScopeDecision{Scope: scope, Disposition: "accepted"})
	}
	// A constructed policy uses the exact bound attestation inventory. It is
	// deliberately synthetic and cannot establish a genuine first-pair receipt.
	return EvidenceDecision{Version: 1, RequestDigest: r.RequestDigest, AuthorityID: r.Authority.AuthorityID, AuthorityPolicyDigest: r.Authority.AuthorityPolicyDigest, AuthorityConfigDigest: r.Authority.AuthorityConfigDigest, Disposition: "accepted", ScopedRoleDecisions: scopes}, nil
}
func syntheticProvider(c RegisteredComparison, record **ownedSyntheticRecord, calls *atomic.Int32, mutate func(*EvidenceDecision)) EvidenceAcceptance {
	return syntheticAcceptance(func(ctx context.Context, r EvidenceRequest) (EvidenceDecision, error) {
		calls.Add(1)
		preimage := pairCopy(r)
		preimage.RequestDigest = ""
		if pairDigest(preimage) != r.RequestDigest {
			return EvidenceDecision{}, errors.New("synthetic request preimage mismatch")
		}
		if r.PairDigest != c.state.identity.PairDigest {
			return EvidenceDecision{}, errors.New("synthetic pair request mismatch")
		}
		d, _ := syntheticDecision(r)
		for i := range d.ScopedRoleDecisions {
			d.ScopedRoleDecisions[i].AttestationArtifactDigest = c.state.attestations[d.ScopedRoleDecisions[i].Scope.ScopeID]
		}
		if mutate != nil {
			mutate(&d)
		}
		body := AcceptanceRecord{Version: d.Version, RequestDigest: d.RequestDigest, AuthorityID: d.AuthorityID, AuthorityPolicyDigest: d.AuthorityPolicyDigest, AuthorityConfigDigest: d.AuthorityConfigDigest, Disposition: d.Disposition, ScopedRoleDecisions: d.ScopedRoleDecisions, Reason: d.Reason, Owner: d.Owner, NextAction: d.NextAction}
		raw, _ := json.Marshal(body)
		reader := &ownedSyntheticRecord{Reader: bytes.NewReader(raw)}
		*record = reader
		d.Record = reader
		d.AcceptanceRecordDigest = registrationSHA(raw)
		return d, nil
	})
}
func syntheticComparison(t testing.TB, property string) (RegisteredComparison, ValidatedRegisteredPair, EvidenceAuthority) {
	t.Helper()
	r, l, c, b, root, a := syntheticPairFixture(t, property)
	p, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
	if err != nil {
		t.Fatal(err)
	}
	comparison, err := CompareRegisteredPair(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	return comparison, p, a
}

func TestSyntheticClaimPositive(t *testing.T) {
	c, _, a := syntheticComparison(t, "reproducibility")
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	claim, err := FormulateComparisonClaim(context.Background(), c, a, syntheticProvider(c, &record, &calls, nil))
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 || record.closes.Load() != 1 || claim.Task != "synthetic bounded rehearsal" || !strings.Contains(strings.Join(claim.Limits, ";"), "synthetic constructed provider rehearsal") {
		t.Fatal("synthetic claim scope", claim, record, calls.Load())
	}
}

func TestSyntheticClaimSharedLatch(t *testing.T) {
	for _, first := range []string{"missing-provider", "canceled", "success", "invalid-authority"} {
		t.Run(first, func(t *testing.T) {
			c, p, a := syntheticComparison(t, "reproducibility")
			copyC, err := CompareRegisteredPair(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			var record *ownedSyntheticRecord
			var calls atomic.Int32
			provider := syntheticProvider(c, &record, &calls, nil)
			ctx := context.Background()
			auth := a
			use := provider
			switch first {
			case "missing-provider":
				use = nil
			case "canceled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "invalid-authority":
				auth.AuthorityPolicyDigest = ""
			}
			_, firstErr := FormulateComparisonClaim(ctx, c, auth, use)
			if first == "success" && firstErr != nil {
				t.Fatal(firstErr)
			}
			if first != "success" && firstErr == nil {
				t.Fatal("failed attempt accepted")
			}
			if _, err := FormulateComparisonClaim(context.Background(), copyC, a, provider); err == nil {
				t.Fatal("repeated Compare minted a new attempt")
			}
			want := int32(0)
			if first == "success" {
				want = 1
			}
			if calls.Load() != want {
				t.Fatal("provider invoked after consumed attempt", calls.Load())
			}
		})
	}
	c, p, a := syntheticComparison(t, "reproducibility")
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	provider := syntheticProvider(c, &record, &calls, nil)
	copies := []RegisteredComparison{c, c}
	for i := 0; i < 8; i++ {
		v, err := CompareRegisteredPair(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		copies = append(copies, v)
	}
	var wg sync.WaitGroup
	var successes atomic.Int32
	for _, v := range copies {
		wg.Add(1)
		go func(v RegisteredComparison) {
			defer wg.Done()
			if _, err := FormulateComparisonClaim(context.Background(), v, a, provider); err == nil {
				successes.Add(1)
			}
		}(v)
	}
	wg.Wait()
	if successes.Load() != 1 || calls.Load() != 1 || record.closes.Load() != 1 {
		t.Fatal("shared latch failed", successes.Load(), calls.Load(), record.closes.Load())
	}
}

func TestSyntheticProviderOwnership(t *testing.T) {
	for _, kind := range []string{"provider-error", "read-error", "close-error", "refused", "canceled-after-provider", "wrong-hash", "noncanonical-record", "missing-record", "typednil-record"} {
		t.Run(kind, func(t *testing.T) {
			c, _, a := syntheticComparison(t, "reproducibility")
			var record *ownedSyntheticRecord
			var calls atomic.Int32
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			base := syntheticProvider(c, &record, &calls, nil)
			provider := syntheticAcceptance(func(ctx context.Context, r EvidenceRequest) (EvidenceDecision, error) {
				d, err := base.Check(ctx, r)
				switch kind {
				case "provider-error":
					return d, errors.New("synthetic provider error")
				case "read-error":
					record.errorRead = errors.New("synthetic read error")
				case "close-error":
					record.closeErr = errors.New("synthetic close error")
				case "refused":
					d.Disposition = "refused"
				case "canceled-after-provider":
					cancel()
				case "wrong-hash":
					d.AcceptanceRecordDigest = strings.Repeat("a", 64)
				case "noncanonical-record":
					raw := append(record.ReaderBytes(), byte('\n'))
					record.Reader = bytes.NewReader(raw)
					d.AcceptanceRecordDigest = registrationSHA(raw)
				case "missing-record":
					d.Record = nil
				case "typednil-record":
					d.Record = (*ownedSyntheticRecord)(nil)
				}
				return d, err
			})
			if claim, err := FormulateComparisonClaim(ctx, c, a, provider); err == nil || claim != nil {
				t.Fatal("provider fault admitted", kind, claim, err)
			}
			if kind == "missing-record" || kind == "typednil-record" {
				if record.closes.Load() != 0 {
					t.Fatal("library closed provider-owned unreturned record")
				}
			} else if record.closes.Load() != 1 {
				t.Fatal("record close not once", record.closes.Load())
			}
			if (kind == "provider-error" || kind == "refused" || kind == "canceled-after-provider") && record.reads.Load() != 0 {
				t.Fatal("record read on failed provider", record.reads.Load())
			}
		})
	}
}
func (r *ownedSyntheticRecord) ReaderBytes() []byte {
	v := make([]byte, r.Reader.Size())
	_, _ = r.Reader.ReadAt(v, 0)
	return v
}

func TestSyntheticProviderScopeRefusal(t *testing.T) {
	for _, mutation := range []struct {
		name string
		f    func(*EvidenceDecision)
	}{{"request", func(d *EvidenceDecision) { d.RequestDigest = strings.Repeat("a", 64) }}, {"policy", func(d *EvidenceDecision) { d.AuthorityPolicyDigest = strings.Repeat("a", 64) }}, {"config", func(d *EvidenceDecision) { d.AuthorityConfigDigest = strings.Repeat("a", 64) }}, {"missing-scope", func(d *EvidenceDecision) { d.ScopedRoleDecisions = d.ScopedRoleDecisions[1:] }}, {"extra-scope", func(d *EvidenceDecision) {
		d.ScopedRoleDecisions = append(d.ScopedRoleDecisions, d.ScopedRoleDecisions[0])
	}}, {"unknown", func(d *EvidenceDecision) { d.ScopedRoleDecisions[0].Disposition = "unknown" }}, {"one-bound-both", func(d *EvidenceDecision) { d.ScopedRoleDecisions[0].Scope.Arm = "both" }}} {
		t.Run(mutation.name, func(t *testing.T) {
			c, _, a := syntheticComparison(t, "reproducibility")
			var record *ownedSyntheticRecord
			var calls atomic.Int32
			if claim, err := FormulateComparisonClaim(context.Background(), c, a, syntheticProvider(c, &record, &calls, mutation.f)); err == nil || claim != nil {
				t.Fatal("bad accepted metadata admitted")
			}
			if record.closes.Load() != 1 {
				t.Fatal("close")
			}
		})
	}
}

func TestSyntheticFailedFactsCannotBeOverridden(t *testing.T) {
	for _, kind := range []string{"false-quality", "false-chronology", "unknown-attestation", "unknown-population", "missing-role", "wrong-order", "wrong-uncertainty", "wrong-clock", "cost-without-phases"} {
		t.Run(kind, func(t *testing.T) {
			property := "reproducibility"
			if kind == "cost-without-phases" {
				property = "cost"
			}
			r, l, c, b, root, a := syntheticPairFixture(t, property)
			switch kind {
			case "unknown-population":
				c.Population[0].Unknown = true
			case "missing-role":
				c.EvidenceBindings = c.EvidenceBindings[1:]
			case "cost-without-phases":
				out := []EvidenceBinding{}
				for _, v := range c.EvidenceBindings {
					if v.Scope.ExperimentPhase == "task" {
						out = append(out, v)
					}
				}
				c.EvidenceBindings = out
			default:
				for i := range c.EvidenceBindings {
					binding := &c.EvidenceBindings[i]
					if binding.DecoderProfile != "external-attestation-v1" {
						continue
					}
					raw, _ := os.ReadFile(filepath.Join(root, binding.Artifacts[0].Path))
					var att ExternalAttestation
					if err := json.Unmarshal(raw, &att); err != nil {
						t.Fatal(err)
					}
					changed := false
					switch kind {
					case "false-quality":
						if att.Quality != nil {
							att.Quality.Passed = false
							changed = true
						}
					case "false-chronology":
						if att.Preregistration != nil {
							att.Preregistration.RegistrationSeenBeforeAcquisition = false
							changed = true
						}
					case "unknown-attestation":
						att.Disposition = "unknown"
						changed = true
					case "wrong-order":
						if att.Analysis != nil {
							att.Analysis.Warmup++
							changed = true
						}
					case "wrong-uncertainty":
						if att.Uncertainty != nil {
							att.Uncertainty.AbsoluteBound = 0
							changed = true
						}
					case "wrong-clock":
						if att.Environment != nil {
							att.Environment.ClockBounds[0].Epoch = "wrong"
							changed = true
						}
					}
					if changed {
						raw, _ = json.Marshal(att)
						binding.Artifacts[0] = syntheticArtifact(t, root, binding.Artifacts[0].Path, raw)
						break
					}
				}
			}
			p, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
			if err != nil {
				return
			}
			comparison, err := CompareRegisteredPair(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			var record *ownedSyntheticRecord
			var calls atomic.Int32
			if claim, err := FormulateComparisonClaim(context.Background(), comparison, a, syntheticProvider(comparison, &record, &calls, nil)); err == nil || claim != nil {
				t.Fatal("failed fact upgraded by provider", kind)
			}
			if calls.Load() != 0 {
				t.Fatal("provider called on blocked factual role")
			}
		})
	}
}

func TestSyntheticAbsoluteThreshold(t *testing.T) {
	r, l, c, b, root, a := syntheticPairFixture(t, "reproducibility")
	raw, _ := os.ReadFile(filepath.Join(root, b.ToolReceipt.Path))
	raw = bytes.Replace(raw, []byte(`"value":5`), []byte(`"value":50`), 1)
	updateSyntheticRaw(t, root, &b, raw)
	p, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
	if err != nil {
		t.Fatal(err)
	}
	comparison, err := CompareRegisteredPair(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Results()[0].Status != "candidate_lower" {
		t.Fatal("factual difference lost")
	}
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	if _, err := FormulateComparisonClaim(context.Background(), comparison, a, syntheticProvider(comparison, &record, &calls, nil)); err == nil {
		t.Fatal("over-budget comparator admitted")
	}
	if calls.Load() != 0 {
		t.Fatal("provider after failed threshold")
	}
}

func TestSyntheticRecordBudget(t *testing.T) {
	c, _, a := syntheticComparison(t, "reproducibility")
	c.state.lease.remaining = 1
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	if _, err := FormulateComparisonClaim(context.Background(), c, a, syntheticProvider(c, &record, &calls, nil)); !errors.Is(err, registrationBudgetError) {
		t.Fatal("record budget reset", err)
	}
	if record.closes.Load() != 1 {
		t.Fatal("record leaked")
	}
}

func TestDiagnosticSourcePartitions(t *testing.T) {
	valid := DiagnosticReceipt{HeldOutTotal: 4, HeldOutCorrect: 1, HeldOutIncorrect: 1, HeldOutWithheld: 1, HeldOutMissing: 1, NoFaultTotal: 4, NoFaultPassed: 1, NoFaultWithheld: 1, NoFaultMissing: 1, CompetingTotal: 4, CompetingDiverted: 1, CompetingWithheld: 1, CompetingMissing: 1, ResilienceTotal: 4, ResiliencePassed: 1, ResilienceWithheld: 1, ResilienceMissing: 1}
	if err := diagnosticPartitions(valid); err != nil {
		t.Fatal(err)
	}
	for _, f := range []func(*DiagnosticReceipt){func(d *DiagnosticReceipt) { d.HeldOutMissing++ }, func(d *DiagnosticReceipt) { d.NoFaultWithheld = 5 }, func(d *DiagnosticReceipt) { d.CompetingDiverted = -1 }, func(d *DiagnosticReceipt) { d.ResiliencePassed = 5 }} {
		changed := valid
		f(&changed)
		if err := diagnosticPartitions(changed); err == nil {
			t.Fatal("bad partition admitted")
		}
	}
	raw, _ := json.Marshal(valid)
	var got DiagnosticReceipt
	raw = bytes.Replace(raw, []byte(`"no_fault_withheld":1,`), nil, 1)
	if err := pairedDecode(raw, &got, false); err == nil {
		t.Fatal("historical missing withholding became zero")
	}
	if reflect.DeepEqual(valid, got) {
		t.Fatal("negative did not change")
	}
}

var _ io.ReadCloser = (*ownedSyntheticRecord)(nil)

func TestSyntheticCostPhaseIdentityUnproven(t *testing.T) {
	c, _, a := syntheticComparison(t, "cost")
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	_, err := FormulateComparisonClaim(context.Background(), c, a, syntheticProvider(c, &record, &calls, nil))
	if err == nil || !strings.Contains(err.Error(), "per-arm observer-off/on configuration identity unavailable") {
		t.Fatal("phase labels supplied missing actual configuration", err)
	}
	if calls.Load() != 0 {
		t.Fatal("provider tried to override missing phase identity")
	}
}

// These public controls use constructed artifacts and a synthetic policy. They
// establish report admission behavior, not genuine diagnostic measurements.
func TestDiagnosticReportProtocolBinding(t *testing.T) {
	for _, test := range []struct {
		name      string
		passed    int
		change    func(*Protocol)
		wire      func([]byte) []byte
		wantClaim bool
		wantError string
	}{
		{name: "registered-positive", passed: 2, wantClaim: true},
		{name: "registered-no-fault-failure", passed: 1, wantError: "no-fault"},
		{name: "changed-minimum-same-digest", passed: 1, change: func(p *Protocol) { p.Budgets.MinNoFaultRejectionPct = 1 }, wantError: "report protocol"},
		{name: "unfrozen-lowered-same-digest", passed: 1, change: func(p *Protocol) { p.Frozen = false; p.Budgets.MinNoFaultRejectionPct = 1 }, wantError: "report protocol"},
		{name: "unfrozen-intact", passed: 2, change: func(p *Protocol) { p.Frozen = false }, wantError: "report protocol"},
		{name: "changed-name", passed: 2, change: func(p *Protocol) { p.Name += "-changed" }, wantError: "report protocol"},
		{name: "changed-hardware", passed: 2, change: func(p *Protocol) { p.Hardware.DriverVersion += "-changed" }, wantError: "report protocol"},
		{name: "changed-workload", passed: 2, change: func(p *Protocol) { p.Workloads[0].TestIterations++ }, wantError: "report protocol"},
		{name: "changed-sampling", passed: 2, change: func(p *Protocol) { p.Sampling.RateHz++ }, wantError: "report protocol"},
		{name: "changed-order", passed: 2, change: func(p *Protocol) { p.ExecutionOrder += "-changed" }, wantError: "report protocol"},
		{name: "changed-uncertainty", passed: 2, change: func(p *Protocol) { p.UncertaintyMethod += "-changed" }, wantError: "report protocol"},
		{name: "changed-stopping", passed: 2, change: func(p *Protocol) { p.StoppingRules[0] += "-changed" }, wantError: "report protocol"},
		{name: "changed-exclusion", passed: 2, change: func(p *Protocol) { p.ExclusionRules[0] += "-changed" }, wantError: "report protocol"},
		{name: "nil-comparison", passed: 2, change: func(p *Protocol) { p.Comparison = nil }, wantError: "report protocol"},
		{name: "nil-tools", passed: 2, change: func(p *Protocol) { p.Tools = nil }, wantError: "report protocol"},
		{name: "nil-limits", passed: 2, change: func(p *Protocol) { p.PropertyLimits = nil }, wantError: "report protocol"},
		{name: "nil-workloads", passed: 2, change: func(p *Protocol) { p.Workloads = nil }, wantError: "report protocol"},
		{name: "null-protocol", passed: 2, wire: func(b []byte) []byte {
			return diagnosticWireProtocol(t, b, func(m map[string]json.RawMessage) { m["protocol"] = json.RawMessage("null") })
		}},
		{name: "missing-protocol", passed: 2, wire: func(b []byte) []byte {
			return diagnosticWireProtocol(t, b, func(m map[string]json.RawMessage) { delete(m, "protocol") })
		}},
		{name: "null-minimum", passed: 2, wire: func(b []byte) []byte {
			return bytes.Replace(b, []byte(`"min_no_fault_rejection_pct":100`), []byte(`"min_no_fault_rejection_pct":null`), 1)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, l, c, b, root, a := syntheticPairFixture(t, "diagnosis")
			report := EvaluationReport{Protocol: pairCopy(r.Protocol), Passed: true, Diagnostic: DiagnosticReceipt{HeldOutTotal: 1, HeldOutCorrect: 1, NoFaultTotal: 2, NoFaultPassed: test.passed, CompetingTotal: 1, CompetingDiverted: 1, ResilienceTotal: 1, ResiliencePassed: 1}}
			if test.change != nil {
				test.change(&report.Protocol)
			}
			if report.Protocol.FrozenDigest != r.Protocol.FrozenDigest {
				t.Fatal("control lost matching asserted digest")
			}
			if test.name == "unfrozen-lowered-same-digest" && report.Protocol.Validate() != nil {
				t.Fatal("control must isolate full equality from validation")
			}
			raw, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			if test.wire != nil {
				changed := test.wire(raw)
				if bytes.Equal(raw, changed) {
					t.Fatal("mutation did not apply")
				}
				raw = changed
			}
			artifact := syntheticArtifact(t, root, "synthetic-diagnostic.json", raw)
			for _, arm := range []*RegisteredReceipt{&c, &b} {
				for i := range arm.EvidenceBindings {
					binding := &arm.EvidenceBindings[i]
					if binding.Role != "diagnosis" && binding.Role != "no-fault" {
						continue
					}
					binding.DecoderProfile = "evaluation-diagnostic-root52"
					binding.Scope.DecoderProfile = binding.DecoderProfile
					binding.Scope.ArtifactDigests = []string{artifact.SHA256}
					binding.Scope.MethodArtifactDigest = ""
					binding.Scope.ScopeID = scopeDigest(binding.Scope)
					binding.Artifacts = []RegistrationArtifact{artifact}
				}
			}
			a.AllowedProfiles = append(a.AllowedProfiles, "evaluation-diagnostic-root52")
			sort.Strings(a.AllowedProfiles)
			pair, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
			if err != nil {
				t.Fatal(err)
			}
			comparison, err := CompareRegisteredPair(context.Background(), pair)
			if err != nil {
				t.Fatal(err)
			}
			var record *ownedSyntheticRecord
			var calls atomic.Int32
			claim, err := FormulateComparisonClaim(context.Background(), comparison, a, syntheticProvider(comparison, &record, &calls, nil))
			if test.wantClaim {
				if err != nil || claim == nil || calls.Load() != 1 || record == nil || record.closes.Load() != 1 {
					t.Fatal("legitimate constructed report refused", claim, err, calls.Load())
				}
			} else {
				if err == nil || claim != nil || calls.Load() != 0 || record != nil {
					t.Fatal("invalid report reached provider admission", claim, err, calls.Load())
				}
				if test.wantError != "" && !strings.Contains(err.Error(), test.wantError) {
					t.Fatal("wrong refusal", err)
				}
			}
		})
	}
}

func diagnosticWireProtocol(t testing.TB, raw []byte, change func(map[string]json.RawMessage)) []byte {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	change(m)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// These providers examine only public request and pair-identity values. Their
// policies are synthetic rehearsals, not independent acquisition acceptance.
func publicSyntheticDecision(request EvidenceRequest, identity PairIdentity) (EvidenceDecision, []byte, error) {
	preimage := request
	preimage.RequestDigest = ""
	if pairDigest(preimage) != request.RequestDigest || request.PairDigest != identity.PairDigest {
		return EvidenceDecision{}, nil, errors.New("synthetic public request identity mismatch")
	}
	pairPreimage := identity
	pairPreimage.PairDigest = ""
	if pairDigest(pairPreimage) != identity.PairDigest {
		return EvidenceDecision{}, nil, errors.New("synthetic public pair preimage mismatch")
	}
	attestations := map[string]string{}
	for _, binding := range identity.CheckedBindings {
		if binding.Membership == "enclosing-attestation" {
			if old, ok := attestations[binding.ScopeID]; ok && old != binding.ArtifactDigest {
				return EvidenceDecision{}, nil, errors.New("synthetic public attestation conflict")
			}
			attestations[binding.ScopeID] = binding.ArtifactDigest
		}
	}
	decision := EvidenceDecision{Version: 1, RequestDigest: request.RequestDigest, AuthorityID: request.Authority.AuthorityID, AuthorityPolicyDigest: request.Authority.AuthorityPolicyDigest, AuthorityConfigDigest: request.Authority.AuthorityConfigDigest, Disposition: "accepted"}
	for _, scope := range request.Roles {
		if scopeDigest(scope) != scope.ScopeID {
			return EvidenceDecision{}, nil, errors.New("synthetic public scope preimage mismatch")
		}
		decision.ScopedRoleDecisions = append(decision.ScopedRoleDecisions, ScopeDecision{Scope: scope, Disposition: "accepted", AttestationArtifactDigest: attestations[scope.ScopeID]})
	}
	body := AcceptanceRecord{Version: decision.Version, RequestDigest: decision.RequestDigest, AuthorityID: decision.AuthorityID, AuthorityPolicyDigest: decision.AuthorityPolicyDigest, AuthorityConfigDigest: decision.AuthorityConfigDigest, Disposition: decision.Disposition, ScopedRoleDecisions: decision.ScopedRoleDecisions}
	raw, err := json.Marshal(body)
	if err != nil {
		return EvidenceDecision{}, nil, err
	}
	decision.AcceptanceRecordDigest = registrationSHA(raw)
	return decision, raw, nil
}

// Rebuild the enclosing bytes and reference after every changed scope or body.
// This does not put the enclosing attestation digest into its own preimage.
func publicSyntheticRewriteAttestation(t testing.TB, root string, binding *EvidenceBinding, edit func(*ExternalAttestation)) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, binding.Artifacts[0].Path))
	if err != nil {
		t.Fatal(err)
	}
	var attestation ExternalAttestation
	if err := json.Unmarshal(raw, &attestation); err != nil {
		t.Fatal(err)
	}
	edit(&attestation)
	binding.Scope = attestation.Scope
	binding.Scope.ScopeID = scopeDigest(binding.Scope)
	attestation.Scope = binding.Scope
	raw, err = json.Marshal(attestation)
	if err != nil {
		t.Fatal(err)
	}
	binding.Artifacts = []RegistrationArtifact{syntheticArtifact(t, root, binding.Artifacts[0].Path, raw)}
	if hasDigest(binding.Scope.ArtifactDigests, binding.Artifacts[0].SHA256) {
		t.Fatal("enclosing attestation entered its scope")
	}
}

func publicSyntheticDiagnosticBindings(t testing.TB, root string, registration Registration, candidate, comparator *RegisteredReceipt, authority *EvidenceAuthority) string {
	t.Helper()
	report := EvaluationReport{Protocol: pairCopy(registration.Protocol), Passed: true, Diagnostic: DiagnosticReceipt{HeldOutTotal: 1, HeldOutCorrect: 1, NoFaultTotal: 1, NoFaultPassed: 1, CompetingTotal: 1, CompetingDiverted: 1, ResilienceTotal: 1, ResiliencePassed: 1}}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	artifact := syntheticArtifact(t, root, "public-synthetic-diagnostic.json", raw)
	for _, arm := range []*RegisteredReceipt{candidate, comparator} {
		for i := range arm.EvidenceBindings {
			binding := &arm.EvidenceBindings[i]
			if binding.Role != "diagnosis" && binding.Role != "no-fault" {
				continue
			}
			binding.DecoderProfile = "evaluation-diagnostic-root52"
			binding.Scope.DecoderProfile = binding.DecoderProfile
			binding.Scope.ArtifactDigests = []string{artifact.SHA256}
			binding.Scope.MethodArtifactDigest = ""
			binding.Scope.ScopeID = scopeDigest(binding.Scope)
			binding.Artifacts = []RegistrationArtifact{artifact}
		}
	}
	authority.AllowedProfiles = append(authority.AllowedProfiles, "evaluation-diagnostic-root52")
	sort.Strings(authority.AllowedProfiles)
	return artifact.SHA256
}
