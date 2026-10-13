package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// Everything produced here is synthetic constructed data. A measured wire
// status exercises syntax; it is not an executed tool or authentic acquisition.
func syntheticPairFixture(t testing.TB, property string) (Registration, QualificationLedger, RegisteredReceipt, RegisteredReceipt, string, EvidenceAuthority) {
	t.Helper()
	r, l, root := registrationFixture(t)
	r.SelectedCells = r.SelectedCells[:1]
	r.ID = "synthetic-pair"
	r.Task = "synthetic bounded rehearsal"
	r.Mode = "comparison"
	r.Protocol.Frozen = false
	r.Protocol.FrozenDigest = ""
	r.Protocol.PropertyLimits = map[string]float64{property: 10}
	r.Protocol.Tools = []ToolSpec{{Name: "fixture-candidate", Version: "fixture-v1", Configuration: "synthetic constructed data; no tool execution", Lineage: "fixture-a", Support: "measured"}, {Name: "fixture-comparator", Version: "fixture-v1", Configuration: "synthetic constructed data; no tool execution", Lineage: "fixture-b", Support: "measured"}}
	r.Protocol.Comparison = &ComparisonSpec{Task: r.Task, Candidate: r.Protocol.Tools[0].Name, Comparator: r.Protocol.Tools[1].Name, Properties: map[string]MeasurementScope{property: {Unit: "bytes", Denominator: 1}}}
	if err := r.Protocol.Freeze(); err != nil {
		t.Fatal(err)
	}
	method := r.Analysis.Body.Method
	order := syntheticArtifact(t, root, "order.json", []byte("synthetic registered order; not executed"))
	r.Analysis.Body.OrderArtifact = &order
	r.Analysis.Body.Properties = []RegistrationPropertySpec{{Property: property, Unit: "bytes", Denominator: 1, Direction: "lower", TimingFamily: "not-applicable", UncertaintyKind: "absolute-bound", UncertaintyMethod: method, Overlap: "inconclusive", MaxValue: 10}}
	r.Observer.Body.Arm = r.Protocol.Comparison.Candidate
	r.Observer.Body.ReferenceArm = r.Protocol.Comparison.Comparator
	var err error
	r, err = r.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	authority := EvidenceAuthority{AuthorityID: "synthetic-rehearsal-only", AuthorityPolicyDigest: registrationSHA([]byte("synthetic policy")), AuthorityConfigDigest: registrationSHA([]byte("synthetic config")), AllowedProfiles: []string{"external-attestation-v1", "qualification-evidence-v1", "tool-receipt-v1"}}
	output := syntheticArtifact(t, root, "output.json", []byte("synthetic independent expected output"))
	makeArm := func(i int) RegisteredReceipt {
		u := 0.1
		tool := ToolReceipt{Task: r.Task, Tool: r.Protocol.Tools[i], ProtocolDigest: r.Protocol.FrozenDigest, Measurements: []PropertyMeasurement{{Property: property, Status: "measured", Value: float64(2 + i*3), Unit: "bytes", Denominator: 1, Uncertainty: &u}}}
		raw, _ := json.Marshal(tool)
		ref := syntheticArtifact(t, root, fmt.Sprintf("tool-%d.json", i), raw)
		z := 0.0
		arm := "candidate"
		if i == 1 {
			arm = "comparator"
		}
		return RegisteredReceipt{Version: 1, RegistrationDigest: r.Digest, CellID: "host", Arm: arm, ToolReceipt: ref, SourceClosure: r.Source, ProducerArtifacts: []RegistrationArtifact{method, order, output}, InputArtifacts: pairCopy(r.Workload.Body.Inputs), OutputArtifacts: []RegistrationArtifact{output}, Population: []Population{{Family: "control", DenominatorUnit: "declared events", Declared: 1, Observed: 1}}, AcquisitionBounds: []AcquisitionBounds{{Domain: "clock", Family: "host-monotonic", Epoch: "declared-epoch", Start: 0, End: 10, Uncertainty: &z, Unit: "ns"}}}
	}
	c, b := makeArm(0), makeArm(1)
	stub := &registeredPair{registration: r, ledger: l, candidate: c, comparator: b}
	required := requiredRoleKeys(stub)
	keys := []roleKey{}
	for key := range required {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keyName(keys[i]) < keyName(keys[j]) })
	for _, key := range keys {
		arm := &c
		if key.arm == "comparator" {
			arm = &b
		}
		scope := EvidenceScope{Role: key.role, Arm: key.arm, Property: key.property, PopulationFamily: key.family, ExperimentPhase: key.phase, TargetID: key.target, CellID: key.cell, ClockDomain: key.clock, DecodeStatus: "supported"}
		scope.ClosureDigest = pairDigest(arm.SourceClosure)
		profile := "external-attestation-v1"
		var refs []RegistrationArtifact
		var att ExternalAttestation
		switch key.role {
		case "measurement":
			profile = "tool-receipt-v1"
			scope.Unit = "bytes"
			scope.Denominator = 1
			refs = []RegistrationArtifact{arm.ToolReceipt}
			scope.ArtifactDigests = []string{arm.ToolReceipt.SHA256}
		case "selected-prerequisite":
			profile = "qualification-evidence-v1"
			id := selectedIdentities(r, l)[0]
			scope.CapabilityID = id.CapabilityID
			scope.Gate = id.Gate
			scope.ClosureDigest = id.ClosureDigest
			scope.LatestReceiptDigest = id.LatestReceiptDigest
			refs = []RegistrationArtifact{{Path: "receipt.json", SHA256: id.LatestReceiptDigest}}
			scope.ArtifactDigests = []string{id.LatestReceiptDigest}
		default:
			underlying := []RegistrationArtifact{method}
			selectedMethod := method
			att = ExternalAttestation{Version: 1, RegistrationDigest: r.Digest, ProtocolDigest: r.Protocol.FrozenDigest, AuthorityID: authority.AuthorityID, AuthorityPolicyDigest: authority.AuthorityPolicyDigest, Disposition: "declared"}
			switch key.role {
			case "preregistration":
				att.Preregistration = &PreregistrationCheck{ChronologyArtifact: order, RegistrationSeenBeforeAcquisition: true, RegisteredAt: r.RegisteredAt, FirstAcquisitionAt: r.RegisteredAt.Add(time.Minute)}
				underlying = append(underlying, order)
			case "output-quality", "oracle":
				selectedMethod = r.Oracle.Body.Implementation
				names := []string{}
				for _, q := range r.Oracle.Body.Quality {
					names = append(names, q.Output)
				}
				sort.Strings(names)
				att.Quality = &QualityCheck{ExpectedArtifacts: r.Oracle.Body.Expected, ObservedArtifacts: arm.OutputArtifacts, QualityContractDigest: pairDigest(r.Oracle.Body.Quality), OracleMethodDigest: r.Oracle.Body.Implementation.SHA256, OutputNames: names, Passed: true}
				underlying = append(underlying, r.Oracle.Body.Expected...)
				underlying = append(underlying, arm.OutputArtifacts...)
			case "analysis", "order", "stopping":
				att.Analysis = &AnalysisCheck{MethodArtifact: method, OrderArtifact: order, Seed: r.Analysis.Body.Seed, Repetitions: r.Analysis.Body.Repetitions, Warmup: r.Analysis.Body.Warmup, Order: r.Analysis.Body.Order, Population: arm.Population, ResultArtifact: arm.ToolReceipt, StoppingContractDigest: pairDigest(r.Stopping.Body), MissingDataContractDigest: pairDigest(r.MissingData.Body)}
				underlying = append(underlying, order, arm.ToolReceipt)
			case "uncertainty":
				scope.Unit = "bytes"
				scope.Denominator = 1
				scope.TimingFamily = "not-applicable"
				att.Uncertainty = &UncertaintyCheck{MethodArtifact: method, Property: property, Unit: "bytes", AbsoluteBound: 0.1, PopulationFamily: "control", SampleCount: 1}
			case "acquisition-environment":
				selectedMethod = r.Workload.Body.Code
				att.Environment = &EnvironmentCheck{SourceClosure: arm.SourceClosure, ClockBounds: arm.AcquisitionBounds, ObserverContractDigest: pairDigest(r.Observer.Body), WorkloadContractDigest: pairDigest(r.Workload.Body)}
				if key.clock != "" {
					scope.ClockEpoch = "declared-epoch"
					scope.TimingFamily = "host-monotonic"
					scope.RangeStart = 0
					scope.RangeEnd = 10
				}
			}
			scope.MethodArtifactDigest = selectedMethod.SHA256
			scope.ArtifactDigests = uniqueSyntheticHashes(underlying)
			att.MethodArtifact = selectedMethod
			att.EvidenceArtifacts = underlying
		}
		scope.DecoderProfile = profile
		scope.ScopeID = scopeDigest(scope)
		if profile == "external-attestation-v1" {
			att.Scope = scope
			raw, _ := json.Marshal(att)
			refs = []RegistrationArtifact{syntheticArtifact(t, root, fmt.Sprintf("att-%d.json", len(c.EvidenceBindings)+len(b.EvidenceBindings)), raw)}
		}
		arm.EvidenceBindings = append(arm.EvidenceBindings, EvidenceBinding{Role: key.role, DecoderProfile: profile, Artifacts: refs, Scope: scope, State: "present"})
	}
	return r, l, c, b, root, authority
}
func syntheticArtifact(t testing.TB, root, path string, b []byte) RegistrationArtifact {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, path), b, 0600); err != nil {
		t.Fatal(err)
	}
	return RegistrationArtifact{Path: path, SHA256: registrationSHA(b)}
}
func uniqueSyntheticHashes(as []RegistrationArtifact) []string {
	m := map[string]bool{}
	for _, a := range as {
		m[a.SHA256] = true
	}
	out := []string{}
	for h := range m {
		out = append(out, h)
	}
	sort.Strings(out)
	return out
}

func TestRegisteredPairSyntheticPublic(t *testing.T) {
	r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
	var wire bytes.Buffer
	if err := WriteRegisteredReceipt(&wire, c); err != nil {
		t.Fatal(err)
	}
	read, err := ReadRegisteredReceipt(&wire)
	if err != nil {
		t.Fatal(err)
	}
	pair, err := ValidateRegisteredPair(context.Background(), r, l, read, b, root)
	if err != nil {
		t.Fatal(err)
	}
	comparison, err := CompareRegisteredPair(context.Background(), pair)
	if err != nil {
		t.Fatal(err)
	}
	if got := comparison.Results(); len(got) != 1 || got[0].Effect != 3 || got[0].Status != "candidate_lower" {
		t.Fatal(got)
	}
	id := pair.Identity()
	digest := id.PairDigest
	id.PairDigest = ""
	if pairDigest(id) != digest {
		t.Fatal("pair preimage")
	}
	c.EvidenceBindings[0].Scope.Role = "caller mutated"
	r.Protocol.Comparison.Properties["reproducibility"] = MeasurementScope{Unit: "caller mutation", Denominator: 99}
	r.Analysis.Body.Properties[0].Unit = "caller mutation"
	l.Cells[0].ID = "changed"
	if err := os.WriteFile(filepath.Join(root, b.ToolReceipt.Path), []byte("late mutation"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := CompareRegisteredPair(context.Background(), pair)
	if err != nil || !reflect.DeepEqual(after.Results(), comparison.Results()) {
		t.Fatal("handle alias", err)
	}
	id.SelectedCells[0].CellID = "view mutation"
	if pair.Identity().SelectedCells[0].CellID == "view mutation" {
		t.Fatal("identity alias")
	}
}

func TestRegisteredReceiptWire(t *testing.T) {
	_, _, c, _, _, _ := syntheticPairFixture(t, "reproducibility")
	raw, _ := json.Marshal(c)
	for _, wire := range [][]byte{bytes.Replace(raw, []byte(`"version":1`), []byte(`"version":null`), 1), bytes.Replace(raw, []byte(`"arm":`), []byte(`"ARM":`), 1), append(append([]byte{}, raw...), []byte("\n")...), bytes.Replace(raw, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1)} {
		if _, err := ReadRegisteredReceipt(bytes.NewReader(wire)); err == nil {
			t.Fatal("bad wire accepted")
		}
	}
	if err := WriteRegisteredReceipt(registrationShortWriter{}, c); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
}

func TestRegisteredOriginalPrimitivePresence(t *testing.T) {
	for _, mutation := range []struct{ name, from, to string }{{"nullvalue", `"value":2`, `"value":null`}, {"missingvalue", `,"value":2`, ``}, {"nullsynthetic", `"synthetic":false`, `"synthetic":null`}, {"missingsynthetic", `,"synthetic":false`, ``}, {"missingtoolconfig", `"configuration":"synthetic constructed data; no tool execution",`, ``}, {"nulluncertainty", `"uncertainty":0.1`, `"uncertainty":null`}, {"missingdenominator", `"denominator":1,`, ``}, {"nullmeasurements", `"measurements":[`, `"measurements":null,"bad":[`}} {
		t.Run(mutation.name, func(t *testing.T) {
			r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
			raw, err := os.ReadFile(filepath.Join(root, c.ToolReceipt.Path))
			if err != nil {
				t.Fatal(err)
			}
			changed := bytes.Replace(raw, []byte(mutation.from), []byte(mutation.to), 1)
			if bytes.Equal(changed, raw) {
				t.Fatal("mutation did not apply")
			}
			updateSyntheticRaw(t, root, &c, changed)
			if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil {
				t.Fatal("primitive omission admitted")
			}
		})
	}
}

// Keep every independently hashed binding coherent, so raw decoder negatives
// fail on their intended semantics rather than an earlier stale artifact hash.
func updateSyntheticRaw(t testing.TB, root string, v *RegisteredReceipt, raw []byte) {
	t.Helper()
	old := v.ToolReceipt
	next := syntheticArtifact(t, root, old.Path, raw)
	v.ToolReceipt = next
	for i := range v.EvidenceBindings {
		b := &v.EvidenceBindings[i]
		for j := range b.Scope.ArtifactDigests {
			if b.Scope.ArtifactDigests[j] == old.SHA256 {
				b.Scope.ArtifactDigests[j] = next.SHA256
			}
		}
		sort.Strings(b.Scope.ArtifactDigests)
		b.Scope.ScopeID = scopeDigest(b.Scope)
		if b.DecoderProfile == "tool-receipt-v1" {
			b.Artifacts = []RegistrationArtifact{next}
			continue
		}
		if b.DecoderProfile != "external-attestation-v1" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, b.Artifacts[0].Path))
		if err != nil {
			t.Fatal(err)
		}
		data = bytes.ReplaceAll(data, []byte(old.SHA256), []byte(next.SHA256))
		var att ExternalAttestation
		if err := json.Unmarshal(data, &att); err != nil {
			t.Fatal(err)
		}
		att.Scope = b.Scope
		data, err = json.Marshal(att)
		if err != nil {
			t.Fatal(err)
		}
		b.Artifacts[0] = syntheticArtifact(t, root, b.Artifacts[0].Path, data)
	}
}

func TestRegisteredPairSharedAcquisitionBudget(t *testing.T) {
	for _, extra := range []int{0, 1} {
		t.Run(fmt.Sprintf("extra-%d", extra), func(t *testing.T) {
			r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
			for i := 0; i < 4; i++ {
				n := 16 << 20
				if i == 3 {
					n = 0
				}
				ref := syntheticArtifact(t, root, fmt.Sprintf("padding-%d.bin", i), make([]byte, n))
				c.ProducerArtifacts = append(c.ProducerArtifacts, ref)
			}
			size := syntheticOperationBytes(t, r, l, c, b, root)
			need := (64 << 20) - size + extra
			if need < 0 || need > 16<<20 {
				t.Fatal("budget fixture size", size, need)
			}
			c.ProducerArtifacts[len(c.ProducerArtifacts)-1] = syntheticArtifact(t, root, "padding-3.bin", make([]byte, need))
			pair, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
			if extra == 0 {
				if err != nil {
					t.Fatal("exact budget refused", err)
				}
				if pair.state.lease.remaining != 0 {
					t.Fatal("charged bytes mismatch", pair.state.lease.remaining)
				}
			} else if !errors.Is(err, registrationBudgetError) {
				t.Fatal("aggregate overrun accepted", err)
			}
		})
	}
}
func syntheticOperationBytes(t testing.TB, r Registration, l QualificationLedger, c, b RegisteredReceipt, root string) int {
	t.Helper()
	n := 0
	for _, v := range []any{r, l, c, b} {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		n += len(raw)
	}
	refs, err := r.artifacts()
	if err != nil {
		t.Fatal(err)
	}
	refs = append(refs, c.artifacts()...)
	refs = append(refs, b.artifacts()...)
	for _, cell := range l.Cells {
		for _, a := range cell.Attempts {
			refs = append(refs, RegistrationArtifact{a.Path, a.SHA256}, RegistrationArtifact{a.Evidence.RawPath, a.Evidence.RawSHA256})
		}
		for _, a := range cell.References {
			if a.SHA256 != "" {
				refs = append(refs, RegistrationArtifact{a.Path, a.SHA256})
			}
		}
	}
	seen := map[string]bool{}
	for _, ref := range refs {
		if seen[ref.Path] {
			continue
		}
		seen[ref.Path] = true
		info, err := os.Stat(filepath.Join(root, ref.Path))
		if err != nil {
			t.Fatal(err)
		}
		n += int(info.Size())
	}
	return n
}

func TestRegisteredPairPathAndCancellation(t *testing.T) {
	for _, kind := range []string{"symlink", "missing", "corrupt", "cancel", "conflicting-path", "selected-receipt"} {
		t.Run(kind, func(t *testing.T) {
			r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
			ctx := context.Background()
			switch kind {
			case "symlink":
				path := filepath.Join(root, c.ToolReceipt.Path)
				raw, _ := os.ReadFile(path)
				if err := os.WriteFile(filepath.Join(root, "mutable-original.json"), raw, 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink("mutable-original.json", path); err != nil {
					t.Fatal(err)
				}
			case "missing":
				if err := os.Remove(filepath.Join(root, c.ToolReceipt.Path)); err != nil {
					t.Fatal(err)
				}
			case "corrupt":
				if err := os.WriteFile(filepath.Join(root, c.ToolReceipt.Path), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "conflicting-path":
				c.ProducerArtifacts = append(c.ProducerArtifacts, RegistrationArtifact{Path: b.ToolReceipt.Path, SHA256: registrationSHA([]byte("different"))})
			case "selected-receipt":
				r.SelectedCells[0].ReceiptSHA256 = ""
				r.SelectedCells[0].Reason = "missing"
				r.Digest = ""
				r.RegisteredAt = time.Time{}
				var err error
				r, err = r.Freeze(time.Date(2026, 10, 12, 0, 0, 0, 0, time.UTC))
				if err != nil {
					t.Fatal(err)
				}
				c.RegistrationDigest = r.Digest
				b.RegistrationDigest = r.Digest
			}
			if _, err := ValidateRegisteredPair(ctx, r, l, c, b, root); err == nil {
				t.Fatal("unsafe admission", kind)
			}
		})
	}
}

func TestRegisteredPairIdentityInventory(t *testing.T) {
	r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
	p, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root)
	if err != nil {
		t.Fatal(err)
	}
	id := p.Identity()
	if id.CandidateEnvelopeDigest != pairDigest(c) || id.ComparatorEnvelopeDigest != pairDigest(b) || id.RegistrationDigest != r.Digest || id.ProtocolDigest != r.Protocol.FrozenDigest || id.LedgerDigest != r.PrerequisiteLedgerSHA256 {
		t.Fatal("identity omitted enclosing input")
	}
	for _, v := range []RegisteredReceipt{c, b} {
		for _, binding := range v.EvidenceBindings {
			found := false
			for _, item := range id.CheckedBindings {
				if item.Path == binding.Artifacts[0].Path && item.ArtifactDigest == binding.Artifacts[0].SHA256 && item.ScopeID == binding.Scope.ScopeID && item.DecoderProfile == binding.DecoderProfile {
					found = true
				}
			}
			if !found {
				t.Fatal("checked membership omitted", binding.Role)
			}
			if scopeDigest(binding.Scope) != binding.Scope.ScopeID {
				t.Fatal("scope preimage")
			}
		}
	}
}

func TestRegisteredEnclosingAttestationIsNotUnderlying(t *testing.T) {
	r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
	var other string
	for _, binding := range b.EvidenceBindings {
		if binding.DecoderProfile == "external-attestation-v1" {
			other = binding.Artifacts[0].SHA256
			break
		}
	}
	if other == "" {
		t.Fatal("no enclosing control")
	}
	for i := range c.EvidenceBindings {
		binding := &c.EvidenceBindings[i]
		if binding.DecoderProfile != "external-attestation-v1" {
			continue
		}
		binding.Scope.ArtifactDigests = append(binding.Scope.ArtifactDigests, other)
		sort.Strings(binding.Scope.ArtifactDigests)
		binding.Scope.ScopeID = scopeDigest(binding.Scope)
		raw, _ := os.ReadFile(filepath.Join(root, binding.Artifacts[0].Path))
		var att ExternalAttestation
		if err := json.Unmarshal(raw, &att); err != nil {
			t.Fatal(err)
		}
		att.Scope = binding.Scope
		raw, _ = json.Marshal(att)
		binding.Artifacts[0] = syntheticArtifact(t, root, binding.Artifacts[0].Path, raw)
		break
	}
	if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil || !strings.Contains(err.Error(), "attestation self-reference or enclosing evidence") {
		t.Fatal("enclosing content entered scope preimage", err)
	}
}

func TestRegisteredRegistrationReplay(t *testing.T) {
	r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
	c.RegistrationDigest = registrationSHA([]byte("another registration"))
	if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil {
		t.Fatal("envelope replay under another registration")
	}
}
func TestRegisteredScopeIdentity(t *testing.T) {
	r, l, c, b, root, _ := syntheticPairFixture(t, "reproducibility")
	c.EvidenceBindings[0].Scope.ScopeID = registrationSHA([]byte("another scope"))
	if _, err := ValidateRegisteredPair(context.Background(), r, l, c, b, root); err == nil {
		t.Fatal("scope digest was not validated")
	}
}

// invalidBoundaryRecord deliberately violates io.Reader at the short read-slice
// boundary. It is a private contract control, not authentic admission evidence.
type invalidBoundaryRecord struct {
	prefix, consumed, lastSize, reads, closes int
}

func (r *invalidBoundaryRecord) Read(p []byte) (int, error) {
	r.reads++
	if r.prefix > 0 {
		n := len(p)
		if n > r.prefix {
			n = r.prefix
		}
		clear(p[:n])
		r.prefix -= n
		r.consumed += n
		return n, nil
	}
	r.lastSize = len(p)
	return len(p) + 1, nil
}

func (r *invalidBoundaryRecord) Close() error {
	r.closes++
	return nil
}

func TestPairedReaderInvalidBoundaryCount(t *testing.T) {
	for _, kind := range []string{"aggregate", "record"} {
		t.Run(kind, func(t *testing.T) {
			remaining := int64(1)
			prefix := 0
			if kind == "record" {
				remaining = 64 << 20
				prefix = registrationLimit - 1
			}
			reader := &invalidBoundaryRecord{prefix: prefix}
			lease := &pairLease{remaining: remaining}
			_, err := pairedReadRecord(context.Background(), reader, lease)
			if err == nil || !strings.Contains(err.Error(), "invalid reader count") {
				t.Fatalf("invalid short-slice count did not reach its own guard: %v", err)
			}
			if reader.lastSize != 2 || lease.remaining != remaining-int64(reader.consumed) {
				t.Fatal("invalid count charged or wrong boundary", reader.lastSize, lease.remaining, reader.consumed)
			}
			if reader.closes != 0 {
				t.Fatal("private reader helper took caller close ownership")
			}
		})
	}
}

func TestSyntheticInvalidBoundaryRecordCloseOnce(t *testing.T) {
	for _, kind := range []string{"aggregate", "record"} {
		t.Run(kind, func(t *testing.T) {
			comparison, pair, authority := syntheticComparison(t, "reproducibility")
			reader := &invalidBoundaryRecord{}
			remaining := int64(1)
			if kind == "record" {
				remaining = 64 << 20
				reader.prefix = registrationLimit - 1
			}
			// Direct boundary injection is private test setup. Public admission
			// behavior is exercised below, but this is not a public budget proof.
			comparison.state.lease.mu.Lock()
			comparison.state.lease.remaining = remaining
			comparison.state.lease.mu.Unlock()
			identity := pair.Identity()
			calls := 0
			provider := syntheticAcceptance(func(_ context.Context, request EvidenceRequest) (EvidenceDecision, error) {
				calls++
				decision, _, err := publicSyntheticDecision(request, identity)
				decision.Record = reader
				return decision, err
			})
			claim, err := FormulateComparisonClaim(context.Background(), comparison, authority, provider)
			if claim != nil || err == nil || !strings.Contains(err.Error(), "invalid reader count") || calls != 1 || reader.closes != 1 || reader.lastSize != 2 {
				t.Fatal("invalid record/close boundary", claim, err, calls, reader.closes, reader.lastSize)
			}
			if comparison.state.lease.remaining != remaining-int64(reader.consumed) {
				t.Fatal("invalid returned count charged")
			}
			if _, err := FormulateComparisonClaim(context.Background(), comparison, authority, provider); err == nil || calls != 1 || reader.closes != 1 {
				t.Fatal("reader failure restored admission attempt", err, calls, reader.closes)
			}
		})
	}
}
