package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func publicSyntheticReadDeclaration(t testing.TB, root string, ref RegistrationArtifact) ExternalAttestation {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, ref.Path))
	if err != nil || registrationSHA(raw) != ref.SHA256 {
		t.Fatal("synthetic declaration bytes/hash mismatch", err)
	}
	var declaration ExternalAttestation
	if err := json.Unmarshal(raw, &declaration); err != nil {
		t.Fatal(err)
	}
	canonical, err := json.Marshal(declaration)
	if err != nil || !bytes.Equal(canonical, raw) || scopeDigest(declaration.Scope) != declaration.Scope.ScopeID {
		t.Fatal("synthetic declaration canonical/preimage mismatch", err)
	}
	return declaration
}

func publicSyntheticPolicy(t testing.TB, root string, candidate, comparator *RegisteredReceipt, authority *EvidenceAuthority, semantic bool) {
	t.Helper()
	name := "synthetic-permissive-public-inventory"
	if semantic {
		name = "synthetic-provisioned-semantic-policy"
	}
	authority.AuthorityID = name
	authority.AuthorityPolicyDigest = registrationSHA([]byte(name + ": policy"))
	authority.AuthorityConfigDigest = registrationSHA([]byte(name + ": configuration"))
	for _, arm := range []*RegisteredReceipt{candidate, comparator} {
		for i := range arm.EvidenceBindings {
			binding := &arm.EvidenceBindings[i]
			if binding.DecoderProfile == "external-attestation-v1" {
				publicSyntheticReadDeclaration(t, root, binding.Artifacts[0])
				publicSyntheticRewriteAttestation(t, root, binding, func(a *ExternalAttestation) {
					a.AuthorityID = authority.AuthorityID
					a.AuthorityPolicyDigest = authority.AuthorityPolicyDigest
				})
			}
		}
	}
}

func publicSyntheticIndependentResults(t testing.TB, root string, r Registration, ledger QualificationLedger, arm *RegisteredReceipt) {
	t.Helper()
	var selected SelectedCellIdentity
	for _, cell := range selectedIdentities(r, ledger) {
		if cell.CellID == arm.CellID {
			selected = cell
		}
	}
	if selected.CellID == "" {
		t.Fatal("selected original cell absent")
	}
	for i := range arm.EvidenceBindings {
		binding := &arm.EvidenceBindings[i]
		if binding.Role != "analysis" && binding.Role != "order" && binding.Role != "stopping" {
			continue
		}
		declaration := publicSyntheticReadDeclaration(t, root, binding.Artifacts[0])
		scope, err := analysisResultScope(r, *arm, declaration.Scope)
		if err != nil {
			t.Fatal(err)
		}
		output := syntheticArtifact(t, root, "independent-output-"+arm.Arm+"-"+binding.Role+".json", []byte("synthetic independent analysis output; no method execution"))
		result := AnalysisResult{Version: 1, Profile: "analysis-result-v1", RegistrationDigest: r.Digest, ProtocolDigest: r.Protocol.FrozenDigest, Arm: arm.Arm, SelectedCell: selected, OriginalToolReceipt: arm.ToolReceipt, MethodArtifact: r.Analysis.Body.Method, Population: arm.Population, Scope: scope, OutputArtifacts: []RegistrationArtifact{output}}
		raw, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		ref := syntheticArtifact(t, root, "independent-result-"+arm.Arm+"-"+binding.Role+".json", raw)
		arm.ProducerArtifacts = append(arm.ProducerArtifacts, output, ref)
		publicSyntheticRewriteAttestation(t, root, binding, func(a *ExternalAttestation) {
			a.Analysis.ResultArtifact = ref
			a.EvidenceArtifacts = append(a.EvidenceArtifacts, output, ref)
			a.Scope.ArtifactDigests = uniqueSyntheticHashes(a.EvidenceArtifacts)
		})
	}
}

func publicSyntheticResultProvider(t testing.TB, root string, identity PairIdentity, semantic bool, calls *int, record **ownedSyntheticRecord) EvidenceAcceptance {
	t.Helper()
	// Explicit preprovisioning is possible through the public Identity view.
	inventory := map[string]PairIdentity{identity.PairDigest: identity}
	recovered := map[string]ExternalAttestation{}
	for _, binding := range identity.CheckedBindings {
		if binding.Membership == "enclosing-attestation" {
			ref := RegistrationArtifact{Path: binding.Path, SHA256: binding.ArtifactDigest}
			recovered[binding.ArtifactDigest] = publicSyntheticReadDeclaration(t, root, ref)
		}
	}
	return syntheticAcceptance(func(_ context.Context, request EvidenceRequest) (EvidenceDecision, error) {
		*calls = *calls + 1
		known, ok := inventory[request.PairDigest]
		if !ok {
			return EvidenceDecision{}, errors.New("synthetic public pair not provisioned")
		}
		decision, raw, err := publicSyntheticDecision(request, known)
		if err != nil {
			return EvidenceDecision{}, err
		}
		if semantic {
			for _, scope := range request.Roles {
				for _, binding := range known.CheckedBindings {
					if binding.ScopeID != scope.ScopeID {
						continue
					}
					if binding.Membership == "underlying" && !hasDigest(scope.ArtifactDigests, binding.ArtifactDigest) {
						return EvidenceDecision{}, errors.New("synthetic semantic policy refuses omitted source")
					}
					if binding.Membership != "enclosing-attestation" {
						continue
					}
					declaration, ok := recovered[binding.ArtifactDigest]
					if !ok || declaration.Scope.ScopeID != scope.ScopeID {
						return EvidenceDecision{}, errors.New("synthetic semantic declaration unavailable")
					}
					if declaration.Analysis == nil {
						continue
					}
					original := request.CandidateRawDigest
					if scope.Arm == "comparator" {
						original = request.ComparatorRawDigest
					}
					if declaration.Analysis.ResultArtifact.SHA256 == original {
						continue
					}
					ref := declaration.Analysis.ResultArtifact
					data, err := os.ReadFile(filepath.Join(root, ref.Path))
					if err != nil || registrationSHA(data) != ref.SHA256 {
						return EvidenceDecision{}, errors.New("synthetic semantic result hash mismatch")
					}
					var result AnalysisResult
					if err := json.Unmarshal(data, &result); err != nil {
						return EvidenceDecision{}, err
					}
					canonical, err := json.Marshal(result)
					if err != nil || !bytes.Equal(canonical, data) || result.Profile != "analysis-result-v1" || result.Arm != scope.Arm || result.OriginalToolReceipt.SHA256 != original || result.RegistrationDigest != request.RegistrationDigest || result.ProtocolDigest != request.ProtocolDigest {
						return EvidenceDecision{}, errors.New("synthetic semantic policy refuses original arm relation")
					}
				}
			}
		}
		*record = &ownedSyntheticRecord{Reader: bytes.NewReader(raw)}
		decision.Record = *record
		return decision, nil
	})
}

func TestSyntheticAnalysisResultProfilesPositive(t *testing.T) {
	for _, independent := range []bool{false, true} {
		for _, semantic := range []bool{false, true} {
			name := fmt.Sprintf("independent=%t/semantic=%t", independent, semantic)
			t.Run(name, func(t *testing.T) {
				r, ledger, candidate, comparator, root, authority := syntheticPairFixture(t, "reproducibility")
				if independent {
					publicSyntheticIndependentResults(t, root, r, ledger, &candidate)
					publicSyntheticIndependentResults(t, root, r, ledger, &comparator)
				}
				publicSyntheticPolicy(t, root, &candidate, &comparator, &authority, semantic)
				pair, err := ValidateRegisteredPair(context.Background(), r, ledger, candidate, comparator, root)
				if err != nil {
					t.Fatal(err)
				}
				comparison, err := CompareRegisteredPair(context.Background(), pair)
				if err != nil {
					t.Fatal(err)
				}
				calls := 0
				var record *ownedSyntheticRecord
				claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, publicSyntheticResultProvider(t, root, pair.Identity(), semantic, &calls, &record))
				if err != nil || claim == nil || calls != 1 || record == nil || record.closes.Load() != 1 {
					t.Fatal("synthetic result positive", claim, err, calls, record)
				}
				if !strings.Contains(strings.Join(claim.Limits, ";"), "synthetic constructed provider rehearsal") {
					t.Fatal("synthetic boundary missing")
				}
			})
		}
	}
}

func publicSyntheticRewriteResult(t testing.TB, root string, arm *RegisteredReceipt, mutate func(*AnalysisResult), wire func([]byte) []byte) {
	t.Helper()
	changed := false
	for i := range arm.EvidenceBindings {
		binding := &arm.EvidenceBindings[i]
		if binding.Role != "analysis" {
			continue
		}
		declaration := publicSyntheticReadDeclaration(t, root, binding.Artifacts[0])
		old := declaration.Analysis.ResultArtifact
		raw, err := os.ReadFile(filepath.Join(root, old.Path))
		if err != nil || registrationSHA(raw) != old.SHA256 {
			t.Fatal("result precondition bytes/hash", err)
		}
		var result AnalysisResult
		if err := json.Unmarshal(raw, &result); err != nil {
			t.Fatal(err)
		}
		if mutate != nil {
			mutate(&result)
		}
		nextRaw, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if wire != nil {
			nextRaw = wire(nextRaw)
		}
		if bytes.Equal(nextRaw, raw) {
			t.Fatal("result mutation did not apply")
		}
		next := syntheticArtifact(t, root, old.Path, nextRaw)
		for j := range arm.ProducerArtifacts {
			if arm.ProducerArtifacts[j] == old {
				arm.ProducerArtifacts[j] = next
			}
		}
		publicSyntheticRewriteAttestation(t, root, binding, func(a *ExternalAttestation) {
			a.Analysis.ResultArtifact = next
			for j := range a.EvidenceArtifacts {
				if a.EvidenceArtifacts[j] == old {
					a.EvidenceArtifacts[j] = next
				}
			}
			a.Scope.ArtifactDigests = uniqueSyntheticHashes(a.EvidenceArtifacts)
		})
		changed = true
	}
	if !changed {
		t.Fatal("analysis role mutation absent")
	}
}

func TestSyntheticAnalysisResultRefusals(t *testing.T) {
	for _, kind := range []string{"wrong-arm", "wrong-cell", "wrong-capability", "wrong-gate", "wrong-latest", "wrong-original", "wrong-method", "wrong-population", "wrong-property", "wrong-unit", "wrong-denominator", "wrong-clock", "wrong-phase", "wrong-target", "wrong-registration", "wrong-protocol", "version", "profile", "self-reference", "enclosing-reference", "duplicate-output", "conflicting-output", "unchecked-output", "missing-output", "missing-identity", "unknown-field", "noncanonical", "wrong-raw-result"} {
		t.Run(kind, func(t *testing.T) {
			r, ledger, candidate, comparator, root, authority := syntheticPairFixture(t, "reproducibility")
			publicSyntheticIndependentResults(t, root, r, ledger, &candidate)
			publicSyntheticIndependentResults(t, root, r, ledger, &comparator)
			publicSyntheticPolicy(t, root, &candidate, &comparator, &authority, true)
			resultPath, enclosingPath := "", ""
			for _, binding := range candidate.EvidenceBindings {
				if binding.Role == "analysis" {
					declaration := publicSyntheticReadDeclaration(t, root, binding.Artifacts[0])
					resultPath = declaration.Analysis.ResultArtifact.Path
					enclosingPath = binding.Artifacts[0].Path
				}
			}
			if resultPath == "" || enclosingPath == "" {
				t.Fatal("result mutation identities absent")
			}
			want := "analysis result:"
			var wire func([]byte) []byte
			mutate := func(result *AnalysisResult) {
				switch kind {
				case "wrong-arm":
					result.Arm = "comparator"
				case "wrong-cell":
					result.SelectedCell.CellID = "different-cell"
				case "wrong-capability":
					result.SelectedCell.CapabilityID = "different-capability"
				case "wrong-gate":
					result.SelectedCell.Gate = "diagnostic"
				case "wrong-latest":
					result.SelectedCell.LatestReceiptDigest = strings.Repeat("a", 64)
				case "wrong-original":
					result.OriginalToolReceipt = comparator.ToolReceipt
				case "wrong-method":
					result.MethodArtifact = *r.Analysis.Body.OrderArtifact
				case "wrong-population":
					result.Population[0].Declared++
				case "wrong-property":
					result.Scope.Properties[0].Property = "cost"
				case "wrong-unit":
					result.Scope.Properties[0].Unit = "seconds"
				case "wrong-denominator":
					result.Scope.Properties[0].Denominator++
				case "wrong-clock":
					result.Scope.ClockEpoch = "different-epoch"
				case "wrong-phase":
					result.Scope.ExperimentPhase = "observer-on"
				case "wrong-target":
					result.Scope.SelectedTargets = []string{"different-target"}
				case "wrong-registration":
					result.RegistrationDigest = strings.Repeat("a", 64)
				case "wrong-protocol":
					result.ProtocolDigest = strings.Repeat("a", 64)
				case "version":
					result.Version = 2
				case "profile":
					result.Profile = "arbitrary-method"
				case "self-reference":
					result.OutputArtifacts[0] = RegistrationArtifact{Path: resultPath, SHA256: r.Analysis.Body.Method.SHA256}
				case "enclosing-reference":
					result.OutputArtifacts[0] = RegistrationArtifact{Path: enclosingPath, SHA256: r.Analysis.Body.Method.SHA256}
				case "duplicate-output":
					result.OutputArtifacts = append(result.OutputArtifacts, result.OutputArtifacts[0])
				case "conflicting-output":
					other := result.OutputArtifacts[0]
					other.SHA256 = strings.Repeat("a", 64)
					result.OutputArtifacts = append(result.OutputArtifacts, other)
				case "unchecked-output":
					result.OutputArtifacts[0].Path = "not-acquired.json"
				case "missing-output":
					result.OutputArtifacts = nil
				}
			}
			switch kind {
			case "wrong-arm", "wrong-registration", "wrong-protocol":
				want += " original arm or registration mismatch"
			case "wrong-cell", "wrong-capability", "wrong-gate", "wrong-latest":
				want += " selected cell identity mismatch"
			case "wrong-original", "wrong-method":
				want += " original receipt or method mismatch"
			case "wrong-population":
				want += " original population mismatch"
			case "wrong-property", "wrong-unit", "wrong-denominator", "wrong-clock", "wrong-phase", "wrong-target":
				want += " registered semantic scope mismatch"
			case "version", "profile":
				want += " unsupported profile or version"
			case "self-reference", "enclosing-reference":
				want += " self or enclosing reference"
			case "duplicate-output", "conflicting-output":
				want += " duplicate or conflicting output reference"
			case "unchecked-output":
				want += " unchecked or out-of-scope reference"
			case "missing-output":
				want += " independent output references missing"
			case "missing-identity":
				wire = func(raw []byte) []byte {
					var object map[string]json.RawMessage
					if err := json.Unmarshal(raw, &object); err != nil {
						t.Fatal(err)
					}
					delete(object, "original_tool_receipt")
					next, err := json.Marshal(object)
					if err != nil {
						t.Fatal(err)
					}
					return next
				}
				want = "missing field original_tool_receipt"
			case "unknown-field":
				wire = func(raw []byte) []byte {
					return append(append([]byte{}, raw[:len(raw)-1]...), []byte(`,"pair_digest":"forbidden-recursive-identity"}`)...)
				}
				want = "unknown field pair_digest"
			case "noncanonical":
				wire = func(raw []byte) []byte { return append(raw, '\n') }
				want = "noncanonical bytes"
			}
			if kind == "wrong-raw-result" {
				for i := range candidate.EvidenceBindings {
					binding := &candidate.EvidenceBindings[i]
					if binding.Role != "analysis" {
						continue
					}
					publicSyntheticRewriteAttestation(t, root, binding, func(a *ExternalAttestation) {
						old := a.Analysis.ResultArtifact
						a.Analysis.ResultArtifact = comparator.ToolReceipt
						for j := range a.EvidenceArtifacts {
							if a.EvidenceArtifacts[j] == old {
								a.EvidenceArtifacts[j] = comparator.ToolReceipt
							}
						}
						a.Scope.ArtifactDigests = uniqueSyntheticHashes(a.EvidenceArtifacts)
					})
				}
			} else {
				publicSyntheticRewriteResult(t, root, &candidate, mutate, wire)
			}
			pair, err := ValidateRegisteredPair(context.Background(), r, ledger, candidate, comparator, root)
			if err != nil {
				t.Fatalf("unexpected earlier acquisition refusal: %v", err)
			}
			comparison, err := CompareRegisteredPair(context.Background(), pair)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			var record *ownedSyntheticRecord
			provider := publicSyntheticResultProvider(t, root, pair.Identity(), true, &calls, &record)
			claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, provider)
			if err == nil || claim != nil || calls != 0 || record != nil || !strings.Contains(err.Error(), want) {
				t.Fatal("wrong local result refusal gate", kind, claim, err, calls, record, want)
			}
		})
	}
}

func TestSyntheticDecodedRoleMembership(t *testing.T) {
	for _, role := range []string{"measurement", "selected-prerequisite", "diagnosis", "no-fault"} {
		for _, omit := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/omit=%t", role, omit), func(t *testing.T) {
				r, ledger, candidate, comparator, root, authority := syntheticPairFixture(t, "diagnosis")
				publicSyntheticDiagnosticBindings(t, root, r, &candidate, &comparator, &authority)
				publicSyntheticPolicy(t, root, &candidate, &comparator, &authority, true)
				changed := false
				if omit {
					for i := range candidate.EvidenceBindings {
						binding := &candidate.EvidenceBindings[i]
						if binding.Role == role {
							if !hasDigest(binding.Scope.ArtifactDigests, binding.Artifacts[0].SHA256) {
								t.Fatal("decoded membership precondition missing")
							}
							binding.Scope.ArtifactDigests = nil
							binding.Scope.ScopeID = scopeDigest(binding.Scope)
							changed = true
						}
					}
					if !changed {
						t.Fatal("decoded membership mutation absent")
					}
				}
				calls := 0
				pair, err := ValidateRegisteredPair(context.Background(), r, ledger, candidate, comparator, root)
				if omit {
					if err == nil {
						comparison, compareErr := CompareRegisteredPair(context.Background(), pair)
						if compareErr != nil {
							t.Fatal("unexpected omission arithmetic refusal", compareErr)
						}
						var record *ownedSyntheticRecord
						claim, claimErr := FormulateComparisonClaim(context.Background(), comparison, authority, publicSyntheticResultProvider(t, root, pair.Identity(), false, &calls, &record))
						t.Fatal("decoded source omission minted handle", claim, claimErr, calls, record)
					}
					if !strings.Contains(err.Error(), "decoded role artifact missing from scope") || calls != 0 {
						t.Fatal("decoded source omission wrong gate", err, calls)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				comparison, err := CompareRegisteredPair(context.Background(), pair)
				if err != nil {
					t.Fatal(err)
				}
				var record *ownedSyntheticRecord
				claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, publicSyntheticResultProvider(t, root, pair.Identity(), true, &calls, &record))
				if claim == nil || err != nil || calls != 1 || record == nil || record.closes.Load() != 1 {
					t.Fatal("decoded membership positive", claim, err, calls, record)
				}
			})
		}
	}
}

func ExampleAnalysisResult() {
	// This prints the profile identity, not a complete admissible result.
	result := AnalysisResult{Version: 1, Profile: "analysis-result-v1", Arm: "candidate"}
	fmt.Println(result.Profile, result.Arm)
	// Output: analysis-result-v1 candidate
}

func ExampleAnalysisResultScope() {
	scope := AnalysisResultScope{Properties: []RegistrationPropertySpec{{Property: "reproducibility", Unit: "bytes", Denominator: 1}}}
	fmt.Println(scope.Properties[0].Unit, scope.Properties[0].Denominator)
	// Output: bytes 1
}

// These constructed controls use coherent result, declaration and pair hashes.
// No provider call is an independent acquisition or statistical measurement.
func TestSyntheticAnalysisDeclaredScopeRefusal(t *testing.T) {
	for _, independent := range []bool{false, true} {
		for _, field := range []string{"unit", "denominator", "timing", "clock", "property", "epoch", "range", "combined"} {
			t.Run(fmt.Sprintf("independent=%t/%s", independent, field), func(t *testing.T) {
				r, ledger, candidate, comparator, root, authority := syntheticPairFixture(t, "reproducibility")
				if independent {
					publicSyntheticIndependentResults(t, root, r, ledger, &candidate)
					publicSyntheticIndependentResults(t, root, r, ledger, &comparator)
				}
				publicSyntheticPolicy(t, root, &candidate, &comparator, &authority, false)
				mutate := func(scope *AnalysisResultScope) {
					if field == "unit" || field == "combined" {
						scope.Unit = "seconds"
					}
					if field == "denominator" || field == "combined" {
						scope.Denominator = 99
					}
					if field == "timing" || field == "combined" {
						scope.TimingFamily = "device-event"
					}
					if field == "clock" {
						scope.ClockDomain = "clock"
					}
					if field == "property" {
						scope.Property = "cost"
					}
					if field == "epoch" || field == "combined" {
						scope.ClockEpoch = "invented-epoch"
					}
					if field == "range" || field == "combined" {
						scope.RangeStart = 20
						scope.RangeEnd = 10
					}
				}
				if independent {
					publicSyntheticRewriteResult(t, root, &candidate, func(result *AnalysisResult) { mutate(&result.Scope) }, nil)
				}
				for i := range candidate.EvidenceBindings {
					binding := &candidate.EvidenceBindings[i]
					if binding.Role != "analysis" {
						continue
					}
					publicSyntheticReadDeclaration(t, root, binding.Artifacts[0])
					publicSyntheticRewriteAttestation(t, root, binding, func(a *ExternalAttestation) {
						scope := AnalysisResultScope{}
						mutate(&scope)
						a.Scope.Property = scope.Property
						a.Scope.ClockDomain = scope.ClockDomain
						a.Scope.Unit = scope.Unit
						a.Scope.Denominator = scope.Denominator
						a.Scope.TimingFamily = scope.TimingFamily
						a.Scope.ClockEpoch = scope.ClockEpoch
						a.Scope.RangeStart = scope.RangeStart
						a.Scope.RangeEnd = scope.RangeEnd
					})
				}
				pair, err := ValidateRegisteredPair(context.Background(), r, ledger, candidate, comparator, root)
				if err != nil {
					t.Fatal(err)
				}
				comparison, err := CompareRegisteredPair(context.Background(), pair)
				if err != nil {
					t.Fatal(err)
				}
				calls := 0
				var record *ownedSyntheticRecord
				provider := publicSyntheticResultProvider(t, root, pair.Identity(), false, &calls, &record)
				claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, provider)
				want := "analysis result: declared semantic scope mismatch"
				if field == "clock" || field == "property" {
					want = "analysis result: unregistered role selector"
				}
				if claim != nil || err == nil || !strings.Contains(err.Error(), want) || calls != 0 || record != nil {
					t.Fatalf("coherent declaration/result refusal: claim=%v err=%v calls=%d record=%v", claim, err, calls, record)
				}
			})
		}
	}
}

func TestSyntheticAnalysisResultByteLimit(t *testing.T) {
	for _, size := range []int{registrationLimit, registrationLimit + 1} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			r, ledger, candidate, comparator, root, authority := syntheticPairFixture(t, "reproducibility")
			publicSyntheticIndependentResults(t, root, r, ledger, &candidate)
			publicSyntheticIndependentResults(t, root, r, ledger, &comparator)
			publicSyntheticPolicy(t, root, &candidate, &comparator, &authority, false)
			// Canonical, typed JSON with an overlong profile deliberately supplies
			// a later string/profile refusal. The size guard must precede it.
			publicSyntheticRewriteResult(t, root, &candidate, nil, func(raw []byte) []byte {
				var result AnalysisResult
				if err := json.Unmarshal(raw, &result); err != nil {
					t.Fatal(err)
				}
				result.Profile = strings.Repeat("x", size-len(raw)+len(result.Profile))
				next, err := json.Marshal(result)
				if err != nil || len(next) != size {
					t.Fatalf("exact size fixture: len=%d want=%d err=%v", len(next), size, err)
				}
				return next
			})
			pair, err := ValidateRegisteredPair(context.Background(), r, ledger, candidate, comparator, root)
			if err != nil {
				t.Fatal(err)
			}
			comparison, err := CompareRegisteredPair(context.Background(), pair)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			var record *ownedSyntheticRecord
			provider := publicSyntheticResultProvider(t, root, pair.Identity(), false, &calls, &record)
			claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, provider)
			if claim != nil || err == nil || calls != 0 || record != nil {
				t.Fatalf("byte refusal: claim=%v err=%v calls=%d", claim, err, calls)
			}
			limited := strings.Contains(err.Error(), "analysis result: record exceeds 4 MiB limit")
			if limited != (size > registrationLimit) {
				t.Fatalf("byte guard ordering at %d: %v", size, err)
			}
		})
	}
}

func TestAnalysisResultRegisteredScopePreservesOptionalUnknown(t *testing.T) {
	r, _, candidate, _, _, _ := syntheticPairFixture(t, "reproducibility")
	scope := EvidenceScope{Role: "analysis", PopulationFamily: "control", ExperimentPhase: "task"}
	got, err := analysisResultScope(r, candidate, scope)
	if err != nil {
		t.Fatal(err)
	}
	if got.Unit != "" || got.Denominator != 0 || got.TimingFamily != "" || got.ClockDomain != "" || got.ClockEpoch != "" || got.RangeStart != 0 || got.RangeEnd != 0 || !reflect.DeepEqual(got.Properties, r.Analysis.Body.Properties) {
		t.Fatalf("aggregate registered facts changed: %+v", got)
	}
	// Absent aggregate target/clock selectors remain explicit empty fields;
	// they do not invent a scalar epoch, range or denominator. Registered
	// property limits and uncertainty methods remain unchanged, not unknown.
	if got.TargetID != "" || got.ClockDomain != "" || got.ClockEpoch != "" {
		t.Fatalf("aggregate optional selectors: %+v", got)
	}
}
