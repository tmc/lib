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
	"sync"
	"time"
)

// RegisteredReceipt retains explicit bounded evidence identities, not acquisition authority.
type RegisteredReceipt struct {
	Version            int                    `json:"version"`
	RegistrationDigest string                 `json:"registration_digest"`
	CellID             string                 `json:"cell_id"`
	Arm                string                 `json:"arm"`
	ToolReceipt        RegistrationArtifact   `json:"tool_receipt"`
	SourceClosure      QualificationClosure   `json:"source_closure"`
	ProducerArtifacts  []RegistrationArtifact `json:"producer_artifacts"`
	InputArtifacts     []RegistrationArtifact `json:"input_artifacts"`
	OutputArtifacts    []RegistrationArtifact `json:"output_artifacts"`
	EvidenceBindings   []EvidenceBinding      `json:"evidence_bindings"`
	Population         []Population           `json:"population"`
	AcquisitionBounds  []AcquisitionBounds    `json:"acquisition_bounds"`
}

// EvidenceBinding retains explicit bounded evidence identities, not acquisition authority.
type EvidenceBinding struct {
	Role           string                 `json:"role"`
	DecoderProfile string                 `json:"decoder_profile"`
	Artifacts      []RegistrationArtifact `json:"artifacts"`
	Scope          EvidenceScope          `json:"scope"`
	State          string                 `json:"state"`
	Reason         string                 `json:"reason"`
	Owner          string                 `json:"owner"`
	NextAction     string                 `json:"next_action"`
}

// Population retains explicit bounded evidence identities, not acquisition authority.
type Population struct {
	Family          string `json:"family"`
	DenominatorUnit string `json:"denominator_unit"`
	Declared        uint64 `json:"declared"`
	Observed        uint64 `json:"observed"`
	Missing         uint64 `json:"missing"`
	Lost            uint64 `json:"lost"`
	Withheld        uint64 `json:"withheld"`
	Unknown         bool   `json:"unknown"`
}

// AcquisitionBounds retains explicit bounded evidence identities, not acquisition authority.
type AcquisitionBounds struct {
	Domain      string   `json:"domain"`
	Family      string   `json:"family"`
	Epoch       string   `json:"epoch"`
	Start       int64    `json:"start"`
	End         int64    `json:"end"`
	Uncertainty *float64 `json:"uncertainty"`
	Unit        string   `json:"unit"`
}

// EvidenceScope retains explicit bounded evidence identities, not acquisition authority.
type EvidenceScope struct {
	ScopeID              string   `json:"scope_id"`
	Role                 string   `json:"role"`
	DecoderProfile       string   `json:"decoder_profile"`
	Arm                  string   `json:"arm"`
	TargetID             string   `json:"target_id"`
	CellID               string   `json:"cell_id"`
	CapabilityID         string   `json:"capability_id"`
	Gate                 string   `json:"gate"`
	ClosureDigest        string   `json:"closure_digest"`
	LatestReceiptDigest  string   `json:"latest_receipt_digest"`
	Property             string   `json:"property"`
	PopulationFamily     string   `json:"population_family"`
	Unit                 string   `json:"unit"`
	Denominator          uint64   `json:"denominator"`
	TimingFamily         string   `json:"timing_family"`
	ClockDomain          string   `json:"clock_domain"`
	ClockEpoch           string   `json:"clock_epoch"`
	RangeStart           int64    `json:"range_start"`
	RangeEnd             int64    `json:"range_end"`
	MethodArtifactDigest string   `json:"method_artifact_digest"`
	ArtifactDigests      []string `json:"artifact_digests"`
	DecodeStatus         string   `json:"decode_status"`
	ExperimentPhase      string   `json:"experiment_phase"`
}

// EvidenceAuthority retains explicit bounded evidence identities, not acquisition authority.
type EvidenceAuthority struct {
	AuthorityID           string   `json:"authority_id"`
	AuthorityPolicyDigest string   `json:"authority_policy_digest"`
	AuthorityConfigDigest string   `json:"authority_config_digest"`
	AllowedProfiles       []string `json:"allowed_profiles"`
}

// SelectedCellIdentity retains explicit bounded evidence identities, not acquisition authority.
type SelectedCellIdentity struct {
	CellID                 string                 `json:"cell_id"`
	CapabilityID           string                 `json:"capability_id"`
	Gate                   string                 `json:"gate"`
	ClosureDigest          string                 `json:"closure_digest"`
	LatestReceiptDigest    string                 `json:"latest_receipt_digest"`
	TransitiveDependencies []SelectedCellIdentity `json:"transitive_dependencies"`
}

// ArmValue retains explicit bounded evidence identities, not acquisition authority.
type ArmValue struct {
	Property                string  `json:"property"`
	Unit                    string  `json:"unit"`
	Denominator             uint64  `json:"denominator"`
	CandidateValue          float64 `json:"candidate_value"`
	CandidateAbsoluteBound  float64 `json:"candidate_absolute_bound"`
	ComparatorValue         float64 `json:"comparator_value"`
	ComparatorAbsoluteBound float64 `json:"comparator_absolute_bound"`
	MaxValue                float64 `json:"max_value"`
	ThresholdPolicy         string  `json:"threshold_policy"`
}

// EvidenceRequest retains explicit bounded evidence identities, not acquisition authority.
type EvidenceRequest struct {
	Version             int                    `json:"version"`
	RegistrationDigest  string                 `json:"registration_digest"`
	ProtocolDigest      string                 `json:"protocol_digest"`
	LedgerDigest        string                 `json:"ledger_digest"`
	PairDigest          string                 `json:"pair_digest"`
	CandidateRawDigest  string                 `json:"candidate_raw_digest"`
	ComparatorRawDigest string                 `json:"comparator_raw_digest"`
	Task                string                 `json:"task"`
	CandidateTool       ToolSpec               `json:"candidate_tool"`
	ComparatorTool      ToolSpec               `json:"comparator_tool"`
	Authority           EvidenceAuthority      `json:"authority"`
	SelectedCells       []SelectedCellIdentity `json:"selected_cells"`
	Roles               []EvidenceScope        `json:"roles"`
	ArmValues           []ArmValue             `json:"arm_values"`
	MatchedResults      []MatchedResult        `json:"matched_results"`
	RequestDigest       string                 `json:"request_digest"`
}

// ScopeDecision retains explicit bounded evidence identities, not acquisition authority.
type ScopeDecision struct {
	Scope                     EvidenceScope `json:"scope"`
	Disposition               string        `json:"disposition"`
	AttestationArtifactDigest string        `json:"attestation_artifact_digest"`
	Reason                    string        `json:"reason"`
	Owner                     string        `json:"owner"`
	NextAction                string        `json:"next_action"`
}

// EvidenceDecision retains explicit bounded evidence identities, not acquisition authority.
type EvidenceDecision struct {
	Version                int             `json:"version"`
	RequestDigest          string          `json:"request_digest"`
	AuthorityID            string          `json:"authority_id"`
	AuthorityPolicyDigest  string          `json:"authority_policy_digest"`
	AuthorityConfigDigest  string          `json:"authority_config_digest"`
	AcceptanceRecordDigest string          `json:"acceptance_record_digest"`
	Disposition            string          `json:"disposition"`
	ScopedRoleDecisions    []ScopeDecision `json:"scoped_role_decisions"`
	Reason                 string          `json:"reason"`
	Owner                  string          `json:"owner"`
	NextAction             string          `json:"next_action"`
	Record                 io.ReadCloser   `json:"-"`
}

// AcceptanceRecord retains explicit bounded evidence identities, not acquisition authority.
type AcceptanceRecord struct {
	Version               int             `json:"version"`
	RequestDigest         string          `json:"request_digest"`
	AuthorityID           string          `json:"authority_id"`
	AuthorityPolicyDigest string          `json:"authority_policy_digest"`
	AuthorityConfigDigest string          `json:"authority_config_digest"`
	Disposition           string          `json:"disposition"`
	ScopedRoleDecisions   []ScopeDecision `json:"scoped_role_decisions"`
	Reason                string          `json:"reason"`
	Owner                 string          `json:"owner"`
	NextAction            string          `json:"next_action"`
}

// ExternalAttestation retains explicit bounded evidence identities, not acquisition authority.
type ExternalAttestation struct {
	Version               int                    `json:"version"`
	RegistrationDigest    string                 `json:"registration_digest"`
	ProtocolDigest        string                 `json:"protocol_digest"`
	Scope                 EvidenceScope          `json:"scope"`
	AuthorityID           string                 `json:"authority_id"`
	AuthorityPolicyDigest string                 `json:"authority_policy_digest"`
	Disposition           string                 `json:"disposition"`
	MethodArtifact        RegistrationArtifact   `json:"method_artifact"`
	EvidenceArtifacts     []RegistrationArtifact `json:"evidence_artifacts"`
	Preregistration       *PreregistrationCheck  `json:"preregistration"`
	Quality               *QualityCheck          `json:"quality"`
	Analysis              *AnalysisCheck         `json:"analysis"`
	Uncertainty           *UncertaintyCheck      `json:"uncertainty"`
	Environment           *EnvironmentCheck      `json:"environment"`
	Reason                string                 `json:"reason"`
	Owner                 string                 `json:"owner"`
	NextAction            string                 `json:"next_action"`
}

// PreregistrationCheck retains explicit bounded evidence identities, not acquisition authority.
type PreregistrationCheck struct {
	ChronologyArtifact                RegistrationArtifact `json:"chronology_artifact"`
	RegistrationSeenBeforeAcquisition bool                 `json:"registration_seen_before_acquisition"`
	RegisteredAt                      time.Time            `json:"registered_at"`
	FirstAcquisitionAt                time.Time            `json:"first_acquisition_at"`
}

// QualityCheck retains explicit bounded evidence identities, not acquisition authority.
type QualityCheck struct {
	ExpectedArtifacts     []RegistrationArtifact `json:"expected_artifacts"`
	ObservedArtifacts     []RegistrationArtifact `json:"observed_artifacts"`
	QualityContractDigest string                 `json:"quality_contract_digest"`
	OracleMethodDigest    string                 `json:"oracle_method_digest"`
	OutputNames           []string               `json:"output_names"`
	Passed                bool                   `json:"passed"`
	Missing               uint64                 `json:"missing"`
	Nonfinite             uint64                 `json:"nonfinite"`
}

// AnalysisCheck retains explicit bounded evidence identities, not acquisition authority.
type AnalysisCheck struct {
	MethodArtifact            RegistrationArtifact `json:"method_artifact"`
	OrderArtifact             RegistrationArtifact `json:"order_artifact"`
	Seed                      *uint64              `json:"seed"`
	Repetitions               uint64               `json:"repetitions"`
	Warmup                    uint64               `json:"warmup"`
	Order                     string               `json:"order"`
	Population                []Population         `json:"population"`
	ResultArtifact            RegistrationArtifact `json:"result_artifact"`
	StoppingContractDigest    string               `json:"stopping_contract_digest"`
	MissingDataContractDigest string               `json:"missing_data_contract_digest"`
}

// UncertaintyCheck retains explicit bounded evidence identities, not acquisition authority.
type UncertaintyCheck struct {
	MethodArtifact   RegistrationArtifact `json:"method_artifact"`
	Property         string               `json:"property"`
	Unit             string               `json:"unit"`
	AbsoluteBound    float64              `json:"absolute_bound"`
	PopulationFamily string               `json:"population_family"`
	SampleCount      uint64               `json:"sample_count"`
}

// EnvironmentCheck retains explicit bounded evidence identities, not acquisition authority.
type EnvironmentCheck struct {
	SourceClosure          QualificationClosure `json:"source_closure"`
	LeaseArtifact          RegistrationArtifact `json:"lease_artifact"`
	ClockBounds            []AcquisitionBounds  `json:"clock_bounds"`
	ObserverContractDigest string               `json:"observer_contract_digest"`
	WorkloadContractDigest string               `json:"workload_contract_digest"`
}

// PairIdentity retains explicit bounded evidence identities, not acquisition authority.
type PairIdentity struct {
	Version                  int                    `json:"version"`
	RegistrationDigest       string                 `json:"registration_digest"`
	ProtocolDigest           string                 `json:"protocol_digest"`
	LedgerDigest             string                 `json:"ledger_digest"`
	CandidateEnvelopeDigest  string                 `json:"candidate_envelope_digest"`
	ComparatorEnvelopeDigest string                 `json:"comparator_envelope_digest"`
	CandidateRawDigest       string                 `json:"candidate_raw_digest"`
	ComparatorRawDigest      string                 `json:"comparator_raw_digest"`
	SelectedCells            []SelectedCellIdentity `json:"selected_cells"`
	RoleScopeIDs             []string               `json:"role_scope_ids"`
	CheckedBindings          []CheckedBinding       `json:"checked_bindings"`
	OptionalOutcomes         []OptionalOutcome      `json:"optional_outcomes"`
	PairDigest               string                 `json:"pair_digest"`
}

// CheckedBinding retains explicit bounded evidence identities, not acquisition authority.
type CheckedBinding struct {
	Path           string `json:"path"`
	ArtifactDigest string `json:"artifact_digest"`
	DecoderProfile string `json:"decoder_profile"`
	ScopeID        string `json:"scope_id"`
	Membership     string `json:"membership"`
}

// OptionalOutcome retains the original acquisition disposition of an unselected artifact.
type OptionalOutcome struct {
	Path           string `json:"path"`
	ExpectedDigest string `json:"expected_digest"`
	Status         string `json:"status"`
	Reason         string `json:"reason"`
	Required       bool   `json:"required"`
}

// EvidenceAcceptance is caller-configured independent acceptance policy.
// Receipt content cannot choose this provider. Implementations own their
// blocking behavior; returned Record ownership transfers even on an error.
type EvidenceAcceptance interface {
	Check(context.Context, EvidenceRequest) (EvidenceDecision, error)
}

// ValidatedRegisteredPair is an opaque, detached integrity result. Its zero
// value is invalid. It establishes no authentic acquisition or measurements.
type ValidatedRegisteredPair struct{ state *registeredPair }

// RegisteredComparison retains factual paired results. Copies and repeated
// comparisons share one claim-attempt lease; none establish authenticity.
type RegisteredComparison struct {
	state   *registeredPair
	results []MatchedResult
}

type pairLease struct {
	mu        sync.Mutex
	remaining int64
	used      bool
}
type registeredPair struct {
	registration                  Registration
	ledger                        QualificationLedger
	candidate, comparator         RegisteredReceipt
	candidateTool, comparatorTool ToolReceipt
	candidateRaw, comparatorRaw   []byte
	identity                      PairIdentity
	scopes                        []EvidenceScope
	attestations                  map[string]string
	policies                      map[string]EvidenceAuthority
	problems                      []string
	lease                         *pairLease
}

// Identity returns a detached identity view. A zero handle returns a zero view.
func (p ValidatedRegisteredPair) Identity() PairIdentity {
	if p.state == nil {
		return PairIdentity{}
	}
	return pairCopy(p.state.identity)
}

// Results returns detached factual results, never an accepted claim.
func (c RegisteredComparison) Results() []MatchedResult { return pairCopy(c.results) }

func pairCopy[T any](v T) T {
	var out T
	b, _ := json.Marshal(v)
	_ = json.Unmarshal(b, &out)
	return out
}

func validateEnvelope(v RegisteredReceipt) error {
	if err := registrationShapeBounds(reflect.ValueOf(v)); err != nil {
		return err
	}
	if v.Version != 1 || !registrationDigest(v.RegistrationDigest, 64) || v.CellID == "" || !ledgerOneOf(v.Arm, "candidate", "comparator") {
		return errors.New("registered receipt: invalid identity")
	}
	c := v.SourceClosure
	for _, h := range []string{c.Candidate, c.Tree, c.Base} {
		if !registrationDigest(h, 40) {
			return errors.New("registered receipt: noncanonical source revision")
		}
	}
	for _, h := range []string{c.SignatureMapSHA256, c.BinarySHA256, c.DependenciesSHA256, c.InputsSHA256, c.OutputsSHA256} {
		if !registrationDigest(h, 64) {
			return errors.New("registered receipt: noncanonical source digest")
		}
	}
	for _, h := range c.Libraries {
		if !registrationDigest(h, 64) {
			return errors.New("registered receipt: noncanonical library digest")
		}
	}
	for _, a := range v.artifacts() {
		if err := registrationArtifacts([]RegistrationArtifact{a}); err != nil {
			return err
		}
	}
	if len(v.EvidenceBindings) > 4096 {
		return errors.New("registered receipt: too many roles")
	}
	for _, b := range v.EvidenceBindings {
		if b.Role != b.Scope.Role || b.DecoderProfile != b.Scope.DecoderProfile || !ledgerOneOf(b.State, "present", "unknown", "unavailable") || len(b.Artifacts) == 0 {
			return errors.New("registered receipt: invalid role binding")
		}
		if b.Scope.ScopeID != scopeDigest(b.Scope) {
			return errors.New("registered receipt: scope digest mismatch")
		}
		if b.Scope.Arm != v.Arm && b.Scope.Arm != "prerequisite" && !(b.Scope.Arm == "both" && b.Role == "preregistration") {
			return errors.New("registered receipt: scope arm mismatch")
		}
		if err := sortedDigests(b.Scope.ArtifactDigests); err != nil {
			return err
		}
	}
	return nil
}

func (v RegisteredReceipt) artifacts() []RegistrationArtifact {
	a := []RegistrationArtifact{v.ToolReceipt}
	a = append(a, v.ProducerArtifacts...)
	a = append(a, v.InputArtifacts...)
	a = append(a, v.OutputArtifacts...)
	for _, b := range v.EvidenceBindings {
		a = append(a, b.Artifacts...)
	}
	return a
}

func scopeDigest(s EvidenceScope) string { s.ScopeID = ""; return pairDigest(s) }
func pairDigest(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return registrationSHA(b)
}
func sortedDigests(a []string) error {
	for i, s := range a {
		if !registrationDigest(s, 64) || i > 0 && a[i-1] >= s {
			return errors.New("paired receipt: noncanonical digest list")
		}
	}
	return nil
}

// ValidateRegisteredPair acquires both envelopes and their prerequisite and
// producer artifacts under one 64 MiB operation budget. Artifact root ancestors
// are caller trusted; its final component and relative artifacts use the existing
// no-follow opener on Darwin/Linux. Other hosts refuse safe filesystem admission.
// The opaque result verifies integrity, not authentic acquisition. Unknown or
// failed role facts remain ineligible for claims regardless of provider metadata.
func ValidateRegisteredPair(ctx context.Context, r Registration, l QualificationLedger, candidate, comparator RegisteredReceipt, root string) (out ValidatedRegisteredPair, resultErr error) {
	if ctx == nil {
		return out, errors.New("paired receipt: nil context")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if err := r.Validate(); err != nil {
		return out, err
	}
	if r.Digest == "" || r.Mode != "comparison" {
		return out, errors.New("paired receipt: frozen comparison registration required")
	}
	if err := l.Validate(); err != nil {
		return out, err
	}
	remaining := int64(64 << 20)
	for _, v := range []any{r, l, candidate, comparator} {
		if err := registrationShapeBounds(reflect.ValueOf(v)); err != nil {
			return out, err
		}
		b, err := json.Marshal(v)
		if err != nil {
			return out, err
		}
		if len(b) > registrationLimit {
			return out, registrationBudgetError
		}
		if err := pairedCharge(&remaining, len(b)); err != nil {
			return out, err
		}
	}
	if err := validateEnvelope(candidate); err != nil {
		return out, err
	}
	if err := validateEnvelope(comparator); err != nil {
		return out, err
	}
	if candidate.Arm != "candidate" || comparator.Arm != "comparator" || candidate.RegistrationDigest != r.Digest || comparator.RegistrationDigest != r.Digest {
		return out, errors.New("paired receipt: wrong arm or registration")
	}
	r = pairCopy(r)
	l = pairCopy(l)
	candidate = pairCopy(candidate)
	comparator = pairCopy(comparator)
	cells := map[string]QualificationCell{}
	for _, c := range l.Cells {
		cells[c.ID] = c
	}
	for _, v := range []RegisteredReceipt{candidate, comparator} {
		c, ok := cells[v.CellID]
		selected := false
		for _, s := range r.SelectedCells {
			if s.CellID == v.CellID && s.Applicability == "selected" {
				selected = true
			}
		}
		if !ok || !selected || !reflect.DeepEqual(c.Closure, v.SourceClosure) {
			return out, errors.New("paired receipt: producer closure not selected")
		}
	}
	extra := append(candidate.artifacts(), comparator.artifacts()...)
	snapshot, cleanup, artifacts, err := registrationSnapshotBudget(ctx, r, l, root, extra, &remaining)
	if err != nil {
		return out, err
	}
	defer func() {
		if err := cleanup(); err != nil {
			out = ValidatedRegisteredPair{}
			resultErr = errors.Join(resultErr, err)
		}
	}()
	report, err := verifyRegistrationSnapshot(ctx, r, l, func() (string, func() error, []RegistrationArtifactResult, error) {
		return snapshot, func() error { return nil }, artifacts, nil
	})
	if err != nil {
		return out, err
	}
	// Optional unselected failures remain visible, but every selected prerequisite
	// must be consistent. They cannot be waived by an external provider.
	for _, c := range report.Cells {
		if c.Selected && !c.ReceiptConsistency {
			return out, errors.New("paired receipt: selected prerequisite incomplete")
		}
	}
	data := map[string][]byte{}
	hashes := map[string]string{}
	for _, a := range artifacts {
		if a.Status != "consistent" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(snapshot, filepath.FromSlash(a.Artifact.Path)))
		if err != nil {
			return out, err
		}
		if registrationSHA(b) != a.Artifact.SHA256 {
			return out, errors.New("paired receipt: snapshot identity changed")
		}
		data[a.Artifact.Path] = b
		hashes[a.Artifact.Path] = a.Artifact.SHA256
	}
	state := &registeredPair{registration: r, ledger: l, candidate: candidate, comparator: comparator, attestations: map[string]string{}, policies: map[string]EvidenceAuthority{}}
	state.candidateRaw = data[candidate.ToolReceipt.Path]
	state.comparatorRaw = data[comparator.ToolReceipt.Path]
	if err := pairedDecode(state.candidateRaw, &state.candidateTool, false); err != nil {
		return out, fmt.Errorf("candidate raw: %w", err)
	}
	if err := pairedDecode(state.comparatorRaw, &state.comparatorTool, false); err != nil {
		return out, fmt.Errorf("comparator raw: %w", err)
	}
	if state.candidateTool.RawDigest != "" || state.comparatorTool.RawDigest != "" {
		return out, errors.New("paired receipt: embedded raw digest must be empty")
	}
	if state.candidateTool.Measurements == nil || state.comparatorTool.Measurements == nil {
		return out, errors.New("paired receipt: null measurements")
	}
	state.candidateTool.RawDigest = candidate.ToolReceipt.SHA256
	state.comparatorTool.RawDigest = comparator.ToolReceipt.SHA256
	// Each original arm is checked against the existing single-tool verifier.
	// Matched arithmetic is deferred to CompareRegisteredPair, using CompareTools.
	for _, arm := range []struct {
		tool ToolReceipt
		raw  []byte
		name string
	}{{state.candidateTool, state.candidateRaw, r.Protocol.Comparison.Candidate}, {state.comparatorTool, state.comparatorRaw, r.Protocol.Comparison.Comparator}} {
		if _, err := CompareTool(r.Protocol, arm.tool, bytes.NewReader(arm.raw)); err != nil {
			return out, err
		}
		if arm.tool.Synthetic || arm.tool.Task != r.Protocol.Comparison.Task || arm.tool.Tool.Name != arm.name {
			return out, errors.New("paired receipt: synthetic or wrong task/arm")
		}
		for _, m := range arm.tool.Measurements {
			scope, selected := r.Protocol.Comparison.Properties[m.Property]
			if selected && m.Status == "measured" && (m.Uncertainty == nil || m.Unit != scope.Unit || m.Denominator != scope.Denominator) {
				return out, errors.New("paired receipt: selected original uncertainty or scope missing")
			}
		}
	}
	identity := PairIdentity{Version: 1, RegistrationDigest: r.Digest, ProtocolDigest: r.Protocol.FrozenDigest, LedgerDigest: r.PrerequisiteLedgerSHA256, CandidateEnvelopeDigest: pairDigest(candidate), ComparatorEnvelopeDigest: pairDigest(comparator), CandidateRawDigest: candidate.ToolReceipt.SHA256, ComparatorRawDigest: comparator.ToolReceipt.SHA256}
	identity.SelectedCells = selectedIdentities(r, l)
	if len(identity.SelectedCells) == 0 {
		return out, errors.New("paired receipt: dependency identity count exceeded")
	}
	if err := state.bindRoles(data, hashes, &identity); err != nil {
		return out, err
	}
	for _, a := range artifacts {
		if !a.Required {
			identity.OptionalOutcomes = append(identity.OptionalOutcomes, OptionalOutcome{Path: a.Artifact.Path, ExpectedDigest: a.Artifact.SHA256, Status: a.Status, Reason: a.Reason})
		}
		if a.Status == "consistent" {
			identity.CheckedBindings = append(identity.CheckedBindings, CheckedBinding{Path: a.Artifact.Path, ArtifactDigest: a.Artifact.SHA256, Membership: "registration-policy"})
		}
	}
	sort.Slice(identity.CheckedBindings, func(i, j int) bool {
		return pairDigest(identity.CheckedBindings[i]) < pairDigest(identity.CheckedBindings[j])
	})
	sort.Strings(identity.RoleScopeIDs)
	if err := registrationShapeBounds(reflect.ValueOf(identity)); err != nil {
		return out, err
	}
	identity.PairDigest = pairDigest(identity)
	state.identity = identity
	state.lease = &pairLease{remaining: remaining}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	return ValidatedRegisteredPair{state: state}, nil
}

// Identities retain the complete selected graph without recursively expanding
// shared DAG subtrees. Nested entries identify immediate prerequisites; every
// reachable dependency has its own complete top-level identity.
func selectedIdentities(r Registration, l QualificationLedger) []SelectedCellIdentity {
	cells := map[string]QualificationCell{}
	for _, c := range l.Cells {
		cells[c.ID] = c
	}
	selected := map[string]bool{}
	var visit func(string)
	visit = func(id string) {
		if selected[id] {
			return
		}
		selected[id] = true
		for _, d := range cells[id].DependsOn {
			visit(d)
		}
	}
	for _, s := range r.SelectedCells {
		if s.Applicability == "selected" {
			visit(s.CellID)
		}
	}
	basic := func(id string) SelectedCellIdentity {
		c := cells[id]
		s := SelectedCellIdentity{CellID: id, CapabilityID: c.CapabilityID, Gate: c.Gate, ClosureDigest: pairDigest(c.Closure)}
		if len(c.Attempts) > 0 {
			s.LatestReceiptDigest = c.Attempts[len(c.Attempts)-1].SHA256
		}
		return s
	}
	var out []SelectedCellIdentity
	edges := 0
	for id := range selected {
		s := basic(id)
		for _, dep := range cells[id].DependsOn {
			edges++
			if edges > 4096 {
				return nil
			}
			s.TransitiveDependencies = append(s.TransitiveDependencies, basic(dep))
		}
		sort.Slice(s.TransitiveDependencies, func(i, j int) bool { return s.TransitiveDependencies[i].CellID < s.TransitiveDependencies[j].CellID })
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CellID < out[j].CellID })
	return out
}
