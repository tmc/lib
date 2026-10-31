package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// These are actual task-private CPU configuration files read by a tiny host
// observer. Their acquired settings are not GPU/vendor acquisition evidence.
func observerPhaseFixture(t *testing.T) (Registration, QualificationLedger, RegisteredReceipt, RegisteredReceipt, string, EvidenceAuthority) {
	t.Helper()
	r, l, c, b, root, authority := syntheticPairFixture(t, "cost")
	plan := &ObserverPhasePlan{Version: 1}
	records := map[[2]string]RegistrationArtifact{}
	configs := map[[2]string]RegistrationArtifact{}
	for i, arm := range []string{"candidate", "comparator"} {
		for _, phase := range []string{"observer-off", "observer-on"} {
			settings := ObserverPhaseSettings{Mode: "off", SensorScope: []string{"host-counter"}, Access: r.Observer.Body.Access, FreshnessNanos: 1}
			if phase == "observer-on" {
				settings.Mode = "host"
				settings.DutyCycle = 1
				settings.SamplingRateHz = 1
			}
			raw, err := json.Marshal(settings)
			if err != nil {
				t.Fatal(err)
			}
			config := syntheticArtifact(t, root, arm+"-"+phase+"-config.json", raw)
			expected := ObserverPhaseConfiguration{Version: 1, Arm: arm, Phase: phase, Tool: r.Protocol.Tools[i], Settings: settings, Configuration: config}
			plan.Configurations = append(plan.Configurations, expected)
			// The host producer reads the original settings and actually applies the
			// observer switch to a fixed loop. It never copies expected.Settings.
			input, err := os.ReadFile(filepath.Join(root, config.Path))
			if err != nil {
				t.Fatal(err)
			}
			var acquired ObserverPhaseSettings
			if err := json.Unmarshal(input, &acquired); err != nil {
				t.Fatal(err)
			}
			count := 0
			sum := 0
			for n := 0; n < 16; n++ {
				sum += n
				if acquired.Mode != "off" {
					count++
				}
			}
			want := 0
			if phase == "observer-on" {
				want = 16
			}
			if sum != 120 || count != want {
				t.Fatal("host observer did not apply settings", sum, count, want)
			}
			actual := ObserverPhaseConfiguration{Version: 1, Arm: arm, Phase: phase, Tool: r.Protocol.Tools[i], Settings: acquired, Configuration: config}
			recordBytes, err := json.Marshal(actual)
			if err != nil {
				t.Fatal(err)
			}
			key := [2]string{arm, phase}
			records[key] = syntheticArtifact(t, root, arm+"-"+phase+"-acquired.json", recordBytes)
			configs[key] = config
		}
	}
	r.Observer.Body.PhaseConfigurations = plan
	r.Digest = ""
	r.RegisteredAt = time.Time{}
	var err error
	r, err = r.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	for _, arm := range []*RegisteredReceipt{&c, &b} {
		arm.RegistrationDigest = r.Digest
		for key, ref := range records {
			if key[0] == arm.Arm {
				arm.ProducerArtifacts = append(arm.ProducerArtifacts, ref, configs[key])
			}
		}
		for i := range arm.EvidenceBindings {
			binding := &arm.EvidenceBindings[i]
			if binding.DecoderProfile != "external-attestation-v1" {
				continue
			}
			old, err := os.ReadFile(filepath.Join(root, binding.Artifacts[0].Path))
			if err != nil {
				t.Fatal(err)
			}
			var a ExternalAttestation
			if err := json.Unmarshal(old, &a); err != nil {
				t.Fatal(err)
			}
			a.RegistrationDigest = r.Digest
			if a.Analysis != nil && binding.Scope.Property == "cost" {
				binding.Scope.Unit = "bytes"
				binding.Scope.Denominator = 1
				binding.Scope.TimingFamily = "not-applicable"
				binding.Scope.ScopeID = scopeDigest(binding.Scope)
			}
			if a.Environment != nil {
				a.Environment.ObserverContractDigest = pairDigest(r.Observer.Body)
				if binding.Scope.Property == "cost" && binding.Scope.ExperimentPhase != "task" {
					key := [2]string{arm.Arm, binding.Scope.ExperimentPhase}
					a.Environment.ObserverPhase = &ObserverPhaseAcquisition{Version: 1, Record: records[key]}
					a.EvidenceArtifacts = append(a.EvidenceArtifacts, records[key], configs[key])
					binding.Scope.ArtifactDigests = uniqueSyntheticHashes(a.EvidenceArtifacts)
					binding.Scope.ScopeID = scopeDigest(binding.Scope)
				}
			}
			a.Scope = binding.Scope
			raw, err := json.Marshal(a)
			if err != nil {
				t.Fatal(err)
			}
			binding.Artifacts[0] = syntheticArtifact(t, root, binding.Artifacts[0].Path, raw)
		}
	}
	return r, l, c, b, root, authority
}

func observerPhaseComparison(t *testing.T, r Registration, l QualificationLedger, c, b RegisteredReceipt, root string) (RegisteredComparison, error) {
	t.Helper()
	pair, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
	if err != nil {
		return RegisteredComparison{}, err
	}
	return CompareRegisteredPair(context.Background(), pair)
}

func observerPhaseClaim(t *testing.T, r Registration, l QualificationLedger, c, b RegisteredReceipt, root string, a EvidenceAuthority) (error, int32) {
	t.Helper()
	comparison, err := observerPhaseComparison(t, r, l, c, b, root)
	if err != nil {
		return err, 0
	}
	var record *ownedSyntheticRecord
	var calls atomic.Int32
	claim, err := FormulateComparisonClaim(context.Background(), comparison, a, syntheticProvider(comparison, &record, &calls, nil))
	if err == nil && (claim == nil || record == nil || record.closes.Load() != 1 || !strings.Contains(strings.Join(claim.Limits, ";"), "synthetic constructed provider rehearsal")) {
		t.Fatal("constructed acceptance escaped scope or cleanup", claim, record)
	}
	return err, calls.Load()
}

func mutateObserverPhase(t *testing.T, root string, arm *RegisteredReceipt, mutate func(*ExternalAttestation, *RegisteredReceipt)) {
	t.Helper()
	for i := range arm.EvidenceBindings {
		binding := &arm.EvidenceBindings[i]
		if binding.Scope.Role != "acquisition-environment" || binding.Scope.ExperimentPhase != "observer-off" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(root, binding.Artifacts[0].Path))
		if err != nil {
			t.Fatal(err)
		}
		var att ExternalAttestation
		if err := json.Unmarshal(raw, &att); err != nil {
			t.Fatal(err)
		}
		mutate(&att, arm)
		binding.Scope.ArtifactDigests = uniqueSyntheticHashes(att.EvidenceArtifacts)
		binding.Scope.ScopeID = scopeDigest(binding.Scope)
		att.Scope = binding.Scope
		raw, err = json.Marshal(att)
		if err != nil {
			t.Fatal(err)
		}
		binding.Artifacts[0] = syntheticArtifact(t, root, binding.Artifacts[0].Path, raw)
		return
	}
	t.Fatal("phase role not found")
}

func TestObserverPhaseHostConfigurations(t *testing.T) {
	r, l, c, b, root, a := observerPhaseFixture(t)
	if err, calls := observerPhaseClaim(t, r, l, c, b, root, a); err != nil || calls != 1 {
		t.Fatal("complete host configuration binding", err, calls)
	}
	// Equivalent off settings in the two arms are legitimate. Their original
	// config bytes match even though the scoped acquired records differ.
	x, _ := r.observerPhase("candidate", "observer-off")
	y, _ := r.observerPhase("comparator", "observer-off")
	if x.Configuration.SHA256 != y.Configuration.SHA256 {
		t.Fatal("host off control differs")
	}
}

func TestObserverPhaseAcquisitionRefusals(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*testing.T, string, *ExternalAttestation, *RegisteredReceipt)
	}{
		{"absent", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			a.Environment.ObserverPhase = nil
		}},
		{"future-version", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			a.Environment.ObserverPhase.Version = 2
		}},
		{"wrong-arm", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) { actual.Arm = "comparator" })
		}},
		{"wrong-phase", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) { actual.Phase = "observer-on" })
		}},
		{"on-labelled-off", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) { actual.Settings.Mode = "host"; actual.Settings.DutyCycle = 1 })
		}},
		{"tool-config", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) { actual.Tool.Configuration = "changed" })
		}},
		{"settings", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) { actual.Settings.SensorScope = []string{"different"} })
		}},
		{"native-config", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			rewriteObserverRecord(t, root, a, c, func(actual *ObserverPhaseConfiguration) {
				actual.Configuration = syntheticArtifact(t, root, "wrong-config.json", []byte("other settings"))
			})
		}},
		{"missing-underlying", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			a.EvidenceArtifacts = a.EvidenceArtifacts[:1]
		}},
		{"enclosing-record", func(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt) {
			a.Environment.ObserverPhase.Record = c.EvidenceBindings[0].Artifacts[0]
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, l, c, b, root, a := observerPhaseFixture(t)
			mutateObserverPhase(t, root, &c, func(att *ExternalAttestation, c *RegisteredReceipt) { test.mutate(t, root, att, c) })
			err, calls := observerPhaseClaim(t, r, l, c, b, root, a)
			if err == nil || calls != 0 {
				t.Fatal("configuration mismatch accepted", err, calls)
			}
		})
	}
}

func rewriteObserverRecord(t *testing.T, root string, a *ExternalAttestation, c *RegisteredReceipt, mutate func(*ObserverPhaseConfiguration)) {
	t.Helper()
	ref := a.Environment.ObserverPhase.Record
	raw, err := os.ReadFile(filepath.Join(root, ref.Path))
	if err != nil {
		t.Fatal(err)
	}
	var record ObserverPhaseConfiguration
	if err := json.Unmarshal(raw, &record); err != nil {
		t.Fatal(err)
	}
	mutate(&record)
	raw, err = json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	next := syntheticArtifact(t, root, ref.Path, raw)
	a.Environment.ObserverPhase.Record = next
	for i, x := range a.EvidenceArtifacts {
		if x == ref {
			a.EvidenceArtifacts[i] = next
		}
	}
	for i, x := range c.ProducerArtifacts {
		if x == ref {
			c.ProducerArtifacts[i] = next
		}
	}
}

func TestObserverPhaseRegistrationRefusals(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*Registration)
	}{
		{"missing", func(r *Registration) {
			r.Observer.Body.PhaseConfigurations.Configurations = r.Observer.Body.PhaseConfigurations.Configurations[:3]
		}},
		{"duplicate", func(r *Registration) {
			p := r.Observer.Body.PhaseConfigurations
			p.Configurations[3] = p.Configurations[0]
		}},
		{"version", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Version = 2 }},
		{"record-version", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Configurations[0].Version = 2 }},
		{"unknown-arm", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Configurations[0].Arm = "both" }},
		{"on-as-off", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Configurations[0].Settings.Mode = "host" }},
		{"off-sampling", func(r *Registration) {
			r.Observer.Body.PhaseConfigurations.Configurations[0].Settings.SamplingRateHz = 1
		}},
		{"on-disabled", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Configurations[1].Settings.Mode = "off" }},
		{"tool", func(r *Registration) { r.Observer.Body.PhaseConfigurations.Configurations[0].Tool.Version = "other" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, _, _, _, _, _ := observerPhaseFixture(t)
			r.Digest = ""
			r.RegisteredAt = time.Time{}
			test.mutate(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid plan accepted")
			}
		})
	}
}

func TestObserverPhaseFrozenAndMissingArtifact(t *testing.T) {
	t.Run("frozen", func(t *testing.T) {
		r, _, _, _, _, _ := observerPhaseFixture(t)
		r.Observer.Body.PhaseConfigurations.Configurations[0].Settings.FreshnessNanos++
		if err := r.Validate(); err == nil {
			t.Fatal("phase drift accepted")
		}
	})
	t.Run("raw-changed", func(t *testing.T) {
		r, l, c, b, root, _ := observerPhaseFixture(t)
		ref := r.Observer.Body.PhaseConfigurations.Configurations[0].Configuration
		if err := os.WriteFile(filepath.Join(root, ref.Path), []byte("changed"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil {
			t.Fatal("changed raw config accepted")
		}
	})
	t.Run("raw-missing", func(t *testing.T) {
		r, l, c, b, root, _ := observerPhaseFixture(t)
		ref := r.Observer.Body.PhaseConfigurations.Configurations[0].Configuration
		if err := os.Remove(filepath.Join(root, ref.Path)); err != nil {
			t.Fatal(err)
		}
		if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil {
			t.Fatal("missing raw config accepted")
		}
	})
	t.Run("detached", func(t *testing.T) {
		r, l, c, b, root, a := observerPhaseFixture(t)
		pair, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
		if err != nil {
			t.Fatal(err)
		}
		r.Observer.Body.PhaseConfigurations.Configurations[0].Settings.Mode = "host"
		comparison, err := CompareRegisteredPair(context.Background(), pair)
		if err != nil {
			t.Fatal(err)
		}
		var record *ownedSyntheticRecord
		var calls atomic.Int32
		if _, err := FormulateComparisonClaim(context.Background(), comparison, a, syntheticProvider(comparison, &record, &calls, nil)); err != nil {
			t.Fatal("caller mutation changed acquired snapshot", err)
		}
	})
}

func TestObserverPhaseLegacySerialization(t *testing.T) {
	for _, property := range []string{"cost", "reproducibility"} {
		t.Run(property, func(t *testing.T) {
			r, l, c, b, root, a := syntheticPairFixture(t, property)
			before, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(before, []byte("phase_configurations")) {
				t.Fatal("legacy bytes expanded")
			}
			var buf bytes.Buffer
			if err := WriteRegistration(&buf, r); err != nil {
				t.Fatal(err)
			}
			got, err := ReadRegistration(&buf)
			if err != nil {
				t.Fatal(err)
			}
			after, err := json.Marshal(got)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatal("legacy bytes changed", err)
			}
			err, calls := observerPhaseClaim(t, r, l, c, b, root, a)
			if property == "cost" {
				if err == nil || !strings.Contains(err.Error(), "per-arm observer-off/on configuration identity unavailable") || calls != 0 {
					t.Fatal("legacy cost enabled", err, calls)
				}
			} else if err != nil || calls != 1 {
				t.Fatal("legacy noncost changed", err, calls)
			}
		})
	}
	env := EnvironmentCheck{}
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("observer_phase")) {
		t.Fatal("legacy environment bytes expanded")
	}
}

func TestObserverPhaseStrictReader(t *testing.T) {
	r, _, _, _, _, _ := observerPhaseFixture(t)
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ReadRegistration(bytes.NewReader(raw))
	if err != nil || !reflect.DeepEqual(got, r) {
		t.Fatal("phase roundtrip", err)
	}
	for _, test := range []struct{ name, from, to string }{
		{"missing-required", "\"warmup\":0,", ""},
		{"unknown-phase", "\"phase\":\"observer-off\"", "\"phase\":\"observer-off\",\"extra\":0"},
		{"missing-settings", "\"freshness_nanos\":1,", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := bytes.Replace(raw, []byte(test.from), []byte(test.to), 1)
			if bytes.Equal(changed, raw) {
				t.Fatal("test mutation did not match", test.from)
			}
			if _, err := ReadRegistration(bytes.NewReader(changed)); err == nil {
				t.Fatal("malformed registration accepted")
			}
		})
	}
	t.Run("missing-phase-zero", func(t *testing.T) {
		var value map[string]any
		if err := json.Unmarshal(raw, &value); err != nil {
			t.Fatal(err)
		}
		observer := value["observer"].(map[string]any)["body"].(map[string]any)
		plan := observer["phase_configurations"].(map[string]any)
		first := plan["configurations"].([]any)[0].(map[string]any)
		delete(first["settings"].(map[string]any), "sampling_rate_hz")
		changed, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ReadRegistration(bytes.NewReader(changed)); err == nil {
			t.Fatal("missing explicit phase zero accepted")
		}
	})
}

func TestObserverPhaseNoAuthority(t *testing.T) {
	r, l, c, b, root, a := observerPhaseFixture(t)
	comparison, err := observerPhaseComparison(t, r, l, c, b, root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FormulateComparisonClaim(context.Background(), comparison, a, nil); err == nil {
		t.Fatal("configuration consistency manufactured authority")
	}
}

func TestObserverPhaseUnresolvedAccess(t *testing.T) {
	r, l, c, b, root, a := observerPhaseFixture(t)
	r.Digest = ""
	r.RegisteredAt = time.Time{}
	r.Observer.Body.PhaseConfigurations.Configurations[0].Settings.Access = RegistrationFact{State: "unknown", Reason: "not observed", Owner: "fixture", NextAction: "acquire"}
	frozen, err := r.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if !frozen.pendingContracts() {
		t.Fatal("unknown phase access became ready")
	}
	// Changing registration does not silently rebind an earlier acquisition.
	if err, calls := observerPhaseClaim(t, frozen, l, c, b, root, a); err == nil || calls != 0 {
		t.Fatal("old receipt accepted changed registration", err, calls)
	}
}
