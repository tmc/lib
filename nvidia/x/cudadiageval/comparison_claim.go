package cudadiageval

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// CompareRegisteredPair returns factual matched results from the retained
// original raw bytes. It does not acquire artifacts or authenticate measurements.
// Every result and copy shares the pair's single-use claim lease.
func CompareRegisteredPair(ctx context.Context, p ValidatedRegisteredPair) (RegisteredComparison, error) {
	if ctx == nil {
		return RegisteredComparison{}, errors.New("paired comparison: nil context")
	}
	if err := ctx.Err(); err != nil {
		return RegisteredComparison{}, err
	}
	if p.state == nil || p.state.lease == nil {
		return RegisteredComparison{}, errors.New("paired comparison: invalid handle")
	}
	s := p.state
	results, err := CompareTools(s.registration.Protocol, s.candidateTool, s.comparatorTool, bytes.NewReader(s.candidateRaw), bytes.NewReader(s.comparatorRaw))
	if err != nil {
		return RegisteredComparison{}, err
	}
	if err := ctx.Err(); err != nil {
		return RegisteredComparison{}, err
	}
	return RegisteredComparison{state: s, results: results}, nil
}

type roleKey struct{ role, arm, property, family, phase, target, cell, clock string }

func bindingKey(s EvidenceScope) roleKey {
	return roleKey{s.Role, s.Arm, s.Property, s.PopulationFamily, s.ExperimentPhase, s.TargetID, s.CellID, s.ClockDomain}
}
func keyName(k roleKey) string {
	return fmt.Sprintf("%s/%s/%s/%s/%s/%s/%s/%s", k.role, k.arm, k.property, k.family, k.phase, k.target, k.cell, k.clock)
}

func requiredRoleKeys(s *registeredPair) map[roleKey]bool {
	out := map[roleKey]bool{}
	add := func(role, arm, property, family, phase, target, cell, clock string) {
		out[roleKey{role, arm, property, family, phase, target, cell, clock}] = true
	}
	for _, id := range selectedIdentities(s.registration, s.ledger) {
		add("selected-prerequisite", "prerequisite", "", "", "task", "", id.CellID, "")
	}
	add("preregistration", "both", "", "", "task", "", "", "")
	r := s.registration
	if r.Analysis.Body == nil || r.Observer.Body == nil || r.Loss.Body == nil || r.Clock.Body == nil {
		return out
	}
	families := map[string]bool{}
	for _, f := range r.Loss.Body.Families {
		if f.Required {
			families[f.Name] = true
		}
	}
	for _, arm := range []RegisteredReceipt{s.candidate, s.comparator} {
		for _, p := range r.Analysis.Body.Properties {
			add("measurement", arm.Arm, p.Property, "", "task", "", arm.CellID, "")
			family := ""
			for _, f := range r.Loss.Body.Families {
				if f.Required && f.ExpectedRecords == p.Denominator {
					if family != "" {
						family = "!ambiguous"
						break
					}
					family = f.Name
				}
			}
			add("uncertainty", arm.Arm, p.Property, family, "task", "", arm.CellID, p.ClockDomain)
		}
		for _, role := range []string{"output-quality", "oracle"} {
			add(role, arm.Arm, "", "", "task", "", arm.CellID, "")
		}
		for family := range families {
			for _, role := range []string{"analysis", "order", "stopping"} {
				add(role, arm.Arm, "", family, "task", "", arm.CellID, "")
			}
		}
		targets := []string{}
		for _, t := range r.TargetRows {
			if t.Applicability == "selected" {
				targets = append(targets, t.ID)
			}
		}
		if len(targets) == 0 {
			targets = []string{""}
		}
		for _, target := range targets {
			add("acquisition-environment", arm.Arm, "", "", "task", target, arm.CellID, "")
			for _, d := range r.Clock.Body.Domains {
				add("acquisition-environment", arm.Arm, "", "", "task", target, arm.CellID, d.ID)
			}
		}
		if _, cost := r.Protocol.Comparison.Properties["cost"]; cost {
			for _, phase := range []string{"observer-off", "observer-on"} {
				for family := range families {
					for _, role := range []string{"analysis", "order", "stopping"} {
						add(role, arm.Arm, "cost", family, phase, "", arm.CellID, "")
					}
				}
				for _, target := range targets {
					add("acquisition-environment", arm.Arm, "cost", "", phase, target, arm.CellID, "")
				}
			}
		}
		diagnostic := false
		if _, ok := r.Protocol.Comparison.Properties["diagnosis"]; ok {
			diagnostic = true
		}
		for _, id := range selectedIdentities(r, s.ledger) {
			if id.Gate == "diagnostic" {
				diagnostic = true
			}
		}
		if diagnostic {
			add("diagnosis", arm.Arm, "diagnosis", "", "task", "", arm.CellID, "")
			add("no-fault", arm.Arm, "diagnosis", "", "task", "", arm.CellID, "")
		}
	}
	return out
}

func profileRole(role, profile string) bool {
	switch role {
	case "measurement":
		return profile == "tool-receipt-v1"
	case "selected-prerequisite":
		return profile == "qualification-evidence-v1"
	case "diagnosis", "no-fault":
		return profile == "evaluation-diagnostic-root52"
	case "observation":
		return strings.HasPrefix(profile, "nvtelemetry-jsonl-v") && ledgerOneOf(strings.TrimPrefix(profile, "nvtelemetry-jsonl-v"), "2", "3", "4", "5") || profile == "tracecontract-receipt-v1-root52"
	case "host-observation":
		return profile == "host-evidence-root52" || profile == "tracecontract-receipt-v1-root52"
	case "preregistration", "output-quality", "oracle", "analysis", "order", "stopping", "acquisition-environment", "uncertainty":
		return profile == "external-attestation-v1"
	}
	return false
}

func (s *registeredPair) bindRoles(data map[string][]byte, hashes map[string]string, id *PairIdentity) error {
	required := requiredRoleKeys(s)
	if _, cost := s.registration.Protocol.Comparison.Properties["cost"]; cost {
		s.problems = append(s.problems, "registered per-arm observer-off/on configuration identity unavailable")
	}
	seen := map[roleKey]bool{}
	seenScope := map[string]bool{}
	enclosing := map[string]bool{}
	for _, v := range []RegisteredReceipt{s.candidate, s.comparator} {
		for _, b := range v.EvidenceBindings {
			if b.DecoderProfile == "external-attestation-v1" {
				for _, a := range b.Artifacts {
					enclosing[a.SHA256] = true
				}
			}
		}
	}
	for _, v := range []RegisteredReceipt{s.candidate, s.comparator} {
		if err := s.validatePopulation(v); err != nil {
			s.problems = append(s.problems, err.Error())
		}
		if s.registration.Workload.Body == nil || !reflect.DeepEqual(v.InputArtifacts, s.registration.Workload.Body.Inputs) {
			s.problems = append(s.problems, "arm workload inputs do not match registration")
		}
		for _, b := range v.EvidenceBindings {
			key := bindingKey(b.Scope)
			if seenScope[b.Scope.ScopeID] {
				return errors.New("paired receipt: duplicate role scope")
			}
			seenScope[b.Scope.ScopeID] = true
			if seen[key] {
				return errors.New("paired receipt: duplicate role key")
			}
			seen[key] = true
			if !profileRole(b.Role, b.DecoderProfile) {
				s.problems = append(s.problems, "unsupported role profile")
			}
			if !required[key] && b.Role != "observation" && b.Role != "host-observation" {
				s.problems = append(s.problems, "role not derived from frozen registration: "+keyName(key))
			}
			if b.State != "present" || b.Scope.DecodeStatus != "supported" {
				s.problems = append(s.problems, "required role facts unknown or unavailable")
			}
			for _, h := range b.Scope.ArtifactDigests {
				if enclosing[h] {
					return errors.New("paired receipt: attestation self-reference or enclosing evidence")
				}
				found := false
				for _, d := range hashes {
					if d == h {
						found = true
					}
				}
				if !found {
					return errors.New("paired receipt: unchecked scope artifact")
				}
			}
			for _, a := range b.Artifacts {
				if hashes[a.Path] != a.SHA256 {
					return errors.New("paired receipt: unchecked role artifact")
				}
				if b.DecoderProfile != "external-attestation-v1" && profileRole(b.Role, b.DecoderProfile) && !hasDigest(b.Scope.ArtifactDigests, a.SHA256) {
					return errors.New("paired receipt: decoded role artifact missing from scope")
				}
				membership := "underlying"
				if b.DecoderProfile == "external-attestation-v1" {
					membership = "enclosing-attestation"
				}
				id.CheckedBindings = append(id.CheckedBindings, CheckedBinding{Path: a.Path, ArtifactDigest: a.SHA256, DecoderProfile: b.DecoderProfile, ScopeID: b.Scope.ScopeID, Membership: membership})
			}
			if err := s.decodeRole(v, b, data); err != nil {
				s.problems = append(s.problems, fmt.Sprintf("%s: %v", keyName(key), err))
			}
			s.scopes = append(s.scopes, b.Scope)
			id.RoleScopeIDs = append(id.RoleScopeIDs, b.Scope.ScopeID)
		}
	}
	for key := range required {
		if !seen[key] {
			s.problems = append(s.problems, "missing derived role: "+keyName(key))
		}
	}
	sort.Slice(s.scopes, func(i, j int) bool { return s.scopes[i].ScopeID < s.scopes[j].ScopeID })
	sort.Strings(s.problems)
	return nil
}

func (s *registeredPair) validatePopulation(v RegisteredReceipt) error {
	r := s.registration
	if r.Loss.Body == nil || r.MissingData.Body == nil {
		return errors.New("population contracts unresolved")
	}
	found := map[string]Population{}
	for _, p := range v.Population {
		if p.Family == "" || p.Unknown {
			return errors.New("population unknown")
		}
		if _, ok := found[p.Family]; ok {
			return errors.New("population duplicate")
		}
		rest := p.Declared
		for _, n := range []uint64{p.Observed, p.Missing, p.Lost, p.Withheld} {
			if n > rest {
				return errors.New("population partition overflow or conflict")
			}
			rest -= n
		}
		if rest != 0 {
			return errors.New("population partition incomplete")
		}
		found[p.Family] = p
	}
	for _, f := range r.Loss.Body.Families {
		p, ok := found[f.Name]
		if !ok && f.Required {
			return errors.New("required population absent")
		}
		if !ok {
			continue
		}
		if p.Declared != f.ExpectedRecords || p.DenominatorUnit != f.DenominatorUnit || p.Missing > f.MaxMissing || p.Lost > f.MaxLost || p.Missing > r.MissingData.Body.MaxMissing || p.Withheld > r.MissingData.Body.MaxWithheld {
			return errors.New("population budget or denominator mismatch")
		}
		if f.Handling == "withhold" && (p.Missing > 0 || p.Lost > 0 || p.Withheld > 0) {
			return errors.New("required population withheld")
		}
		delete(found, f.Name)
	}
	if len(found) > 0 {
		return errors.New("unregistered population family")
	}
	return nil
}

func (s *registeredPair) decodeRole(v RegisteredReceipt, b EvidenceBinding, data map[string][]byte) error {
	if len(b.Artifacts) != 1 {
		return errors.New("profile requires one exact artifact")
	}
	raw := data[b.Artifacts[0].Path]
	scope := b.Scope
	r := s.registration
	if b.Role != "preregistration" && b.Role != "selected-prerequisite" {
		if scope.CellID != v.CellID || scope.ClosureDigest != pairDigest(v.SourceClosure) {
			return errors.New("role producer scope mismatch")
		}
	}
	switch b.DecoderProfile {
	case "tool-receipt-v1":
		if b.Artifacts[0] != v.ToolReceipt {
			return errors.New("measurement raw binding mismatch")
		}
		tool := s.candidateTool
		if v.Arm == "comparator" {
			tool = s.comparatorTool
		}
		p, ok := r.Protocol.Comparison.Properties[scope.Property]
		if !ok || scope.Unit != p.Unit || scope.Denominator != p.Denominator {
			return errors.New("measurement scope mismatch")
		}
		for _, m := range tool.Measurements {
			if m.Property == scope.Property && m.Status == "measured" && m.Uncertainty != nil {
				return nil
			}
		}
		return errors.New("measurement is unknown")
	case "qualification-evidence-v1":
		var ev QualificationEvidence
		if err := pairedDecode(raw, &ev, false); err != nil {
			return err
		}
		for _, id := range selectedIdentities(r, s.ledger) {
			if id.CellID == scope.CellID {
				if ev.CellID != id.CellID || ev.Gate != id.Gate || ev.Status != "passed" || scope.CapabilityID != id.CapabilityID || scope.Gate != id.Gate || scope.ClosureDigest != id.ClosureDigest || scope.LatestReceiptDigest != id.LatestReceiptDigest || b.Artifacts[0].SHA256 != id.LatestReceiptDigest {
					return errors.New("prerequisite role identity mismatch")
				}
				return nil
			}
		}
		return errors.New("unselected prerequisite role")
	case "external-attestation-v1":
		var a ExternalAttestation
		if err := pairedDecode(raw, &a, true); err != nil {
			return err
		}
		if a.Version != 1 || a.Disposition != "declared" || a.RegistrationDigest != r.Digest || a.ProtocolDigest != r.Protocol.FrozenDigest || !reflect.DeepEqual(a.Scope, scope) || a.AuthorityID == "" || !registrationDigest(a.AuthorityPolicyDigest, 64) {
			return errors.New("invalid external declaration")
		}
		for _, ref := range append(append([]RegistrationArtifact{}, a.EvidenceArtifacts...), a.MethodArtifact) {
			if ref.SHA256 == b.Artifacts[0].SHA256 || !hasDigest(scope.ArtifactDigests, ref.SHA256) || !checkedReference(ref, data) {
				return errors.New("attestation underlying membership mismatch")
			}
		}
		if a.MethodArtifact.SHA256 != scope.MethodArtifactDigest {
			return errors.New("attestation method mismatch")
		}
		if err := s.validateAttestation(v, a, data); err != nil {
			return err
		}
		s.attestations[scope.ScopeID] = b.Artifacts[0].SHA256
		s.policies[scope.ScopeID] = EvidenceAuthority{AuthorityID: a.AuthorityID, AuthorityPolicyDigest: a.AuthorityPolicyDigest}
		return nil
	case "evaluation-diagnostic-root52":
		var report EvaluationReport
		if err := pairedDecode(raw, &report, false); err != nil {
			return err
		}
		if err := report.Protocol.Validate(); err != nil {
			return fmt.Errorf("diagnostic/no-fault report protocol: %w", err)
		}
		if !reflect.DeepEqual(report.Protocol, r.Protocol) {
			return errors.New("diagnostic/no-fault report protocol differs from registration")
		}
		if report.Synthetic || !report.Passed {
			return errors.New("diagnostic report is unqualified")
		}
		if err := diagnosticPartitions(report.Diagnostic); err != nil {
			return err
		}
		d := report.Diagnostic
		for _, g := range []struct {
			good, missing, total int
			threshold            float64
		}{{d.HeldOutCorrect, d.HeldOutMissing, d.HeldOutTotal, r.Protocol.Budgets.MinHeldOutDiagnosisPct}, {d.CompetingDiverted, d.CompetingMissing, d.CompetingTotal, 100}, {d.ResiliencePassed, d.ResilienceMissing, d.ResilienceTotal, 100}} {
			if g.total == 0 {
				return errors.New("required diagnostic population absent")
			}
			disposition, _ := rateGate("registered diagnostic", g.good, g.missing, g.total, g.threshold)
			if disposition != "passed" {
				return errors.New("diagnostic population gate not passed")
			}
		}
		for _, g := range report.Gates {
			if g.Disposition != "passed" {
				return errors.New("diagnostic gate failed")
			}
		}
		return validateNoFaultClaimReceipt(&report)
	case "tracecontract-receipt-v1-root52":
		var receipt tracecontract.Receipt
		if err := pairedDecode(raw, &receipt, false); err != nil {
			return err
		}
		if _, err := receipt.JSON(); err != nil {
			return err
		}
		if receipt.Partial {
			return errors.New("trace population partial")
		}
		for _, g := range receipt.Gates {
			if g.Disposition != "passed" {
				return errors.New("trace gate not accepted")
			}
		}
		for _, f := range r.Loss.Body.Families {
			if !f.Required {
				continue
			}
			loss, ok := receipt.LossFor(f.Name)
			if !ok || loss.Unknown || loss.Total != f.ExpectedRecords || loss.Dropped > f.MaxLost {
				return errors.New("trace loss unknown or mismatched")
			}
		}
		for _, p := range v.Population {
			loss, ok := receipt.LossFor(p.Family)
			if !ok || loss.Unknown || loss.Total != p.Declared || loss.Dropped != p.Lost || p.Missing != 0 || p.Withheld != 0 {
				return errors.New("trace population categories unknown or incompatible")
			}
		}
		if scope.ClockDomain != "" && (receipt.CaptureBounds == nil || receipt.ClockUncertaintyNS == nil || receipt.ClockDomain != scope.ClockDomain) {
			return errors.New("trace clock unknown")
		}
		if scope.ClockEpoch != "" {
			return errors.New("trace profile has no typed clock epoch binding")
		}
		return nil
	case "host-evidence-root52":
		var ev cudadiag.HostEvidence
		if err := pairedDecode(raw, &ev, false); err != nil {
			return err
		}
		if ev.Records < 0 || ev.TID == 0 || !registrationDigest(ev.RawSHA256, 64) {
			return errors.New("host evidence identity invalid")
		}
		if ev.Receipt.Partial || ev.Scheduler.UnknownLoss {
			return errors.New("host received count is not full population")
		}
		if err := ev.Scheduler.Validate(ev.Process, ev.TID); err != nil {
			return err
		}
		return errors.New("host profile does not establish causal GPU or complete acquisition population")
	default:
		if strings.HasPrefix(b.DecoderProfile, "nvtelemetry-jsonl-v") {
			manifest, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), int64(len(raw)))
			if err != nil {
				return err
			}
			version, err := strconv.Atoi(strings.TrimPrefix(b.DecoderProfile, "nvtelemetry-jsonl-v"))
			if err != nil || manifest == nil || manifest.SchemaVersion != version || len(samples) > 4096 {
				return errors.New("telemetry profile version or count mismatch")
			}
			for _, pop := range v.Population {
				loss, ok := manifest.LostRecordsByFamily[pop.Family]
				if !ok || loss.Unknown || loss.Total != pop.Declared || loss.Dropped != pop.Lost || pop.Missing != 0 || pop.Withheld != 0 {
					return errors.New("telemetry population has unknown or mismatched categories")
				}
			}
			if scope.ClockDomain != "" {
				if !manifest.HasClockCalibration || manifest.ClockMonotonicDomain != scope.ClockDomain || manifest.ClockMonotonicEpochID != scope.ClockEpoch || manifest.ClockMappingValidFrom > scope.RangeStart || manifest.ClockMappingValidUntil < scope.RangeEnd {
					return errors.New("telemetry clock calibration unknown or incompatible")
				}
			}
			return nil
		}
	}
	return errors.New("unsupported decoder profile")
}

func hasDigest(a []string, s string) bool {
	for _, v := range a {
		if v == s {
			return true
		}
	}
	return false
}
func checkedReference(a RegistrationArtifact, data map[string][]byte) bool {
	b, ok := data[a.Path]
	return ok && registrationSHA(b) == a.SHA256
}

func diagnosticPartitions(d DiagnosticReceipt) error {
	partition := func(total int, parts ...int) bool {
		if total < 0 {
			return false
		}
		for _, n := range parts {
			if n < 0 || n > total {
				return false
			}
			total -= n
		}
		return total == 0
	}
	if !partition(d.HeldOutTotal, d.HeldOutCorrect, d.HeldOutIncorrect, d.HeldOutWithheld, d.HeldOutMissing) {
		return errors.New("held-out partition invalid")
	}
	if _, _, err := noFaultCounts(d.NoFaultPassed, d.NoFaultWithheld, d.NoFaultMissing, d.NoFaultTotal); err != nil {
		return err
	}
	for _, p := range [][4]int{{d.CompetingTotal, d.CompetingDiverted, d.CompetingWithheld, d.CompetingMissing}, {d.ResilienceTotal, d.ResiliencePassed, d.ResilienceWithheld, d.ResilienceMissing}} {
		rest := p[0]
		if rest < 0 {
			return errors.New("diagnostic total negative")
		}
		for _, n := range p[1:] {
			if n < 0 || n > rest {
				return errors.New("diagnostic partition overflow")
			}
			rest -= n
		}
	}
	return nil
}

func (s *registeredPair) validateAttestation(v RegisteredReceipt, a ExternalAttestation, data map[string][]byte) error {
	r := s.registration
	scope := a.Scope
	bodies := 0
	for _, present := range []bool{a.Preregistration != nil, a.Quality != nil, a.Analysis != nil, a.Uncertainty != nil, a.Environment != nil} {
		if present {
			bodies++
		}
	}
	if bodies != 1 {
		return errors.New("attestation requires exactly one role body")
	}
	refs := func(as []RegistrationArtifact) bool {
		if len(as) == 0 {
			return false
		}
		for _, ref := range as {
			if !checkedReference(ref, data) || !hasDigest(scope.ArtifactDigests, ref.SHA256) {
				return false
			}
		}
		return true
	}
	switch scope.Role {
	case "preregistration":
		c := a.Preregistration
		if c == nil || !c.RegistrationSeenBeforeAcquisition || !c.RegisteredAt.Equal(r.RegisteredAt) || c.FirstAcquisitionAt.IsZero() || c.FirstAcquisitionAt.Location() != time.UTC || !c.FirstAcquisitionAt.After(c.RegisteredAt) || !refs([]RegistrationArtifact{c.ChronologyArtifact}) || r.Analysis.Body == nil || a.MethodArtifact != r.Analysis.Body.Method {
			return errors.New("chronology unknown, false or unbound")
		}
		// An independent chronology policy must establish the time/domain evidence.
		// The caller's timestamps/true flag alone never authenticate preregistration.
	case "output-quality", "oracle":
		c := a.Quality
		if c == nil || r.Oracle.Body == nil || r.MissingData.Body == nil || !c.Passed || c.Missing > r.MissingData.Body.MaxMissing || c.QualityContractDigest != pairDigest(r.Oracle.Body.Quality) || c.OracleMethodDigest != r.Oracle.Body.Implementation.SHA256 || a.MethodArtifact != r.Oracle.Body.Implementation || !reflect.DeepEqual(c.ExpectedArtifacts, r.Oracle.Body.Expected) || !reflect.DeepEqual(c.ObservedArtifacts, v.OutputArtifacts) || !refs(c.ExpectedArtifacts) || !refs(c.ObservedArtifacts) {
			return errors.New("output quality or independent oracle mismatch")
		}
		outputs := []string{}
		allowNonfinite := true
		for _, q := range r.Oracle.Body.Quality {
			outputs = append(outputs, q.Output)
			if q.Nonfinite != "explicit-expected" {
				allowNonfinite = false
			}
		}
		sort.Strings(outputs)
		if !reflect.DeepEqual(outputs, c.OutputNames) || c.Nonfinite > 0 && !allowNonfinite {
			return errors.New("quality output or nonfinite rule failed")
		}
	case "analysis", "order", "stopping":
		c := a.Analysis
		if c == nil || r.Analysis.Body == nil || r.Stopping.Body == nil || r.MissingData.Body == nil {
			return errors.New("analysis contracts missing")
		}
		plan := r.Analysis.Body
		if c.MethodArtifact != plan.Method || a.MethodArtifact != plan.Method || !reflect.DeepEqual(c.Seed, plan.Seed) || c.Warmup != plan.Warmup || c.Repetitions != plan.Repetitions || c.Order != plan.Order || c.StoppingContractDigest != pairDigest(r.Stopping.Body) || c.MissingDataContractDigest != pairDigest(r.MissingData.Body) || !reflect.DeepEqual(c.Population, v.Population) || !refs([]RegistrationArtifact{c.MethodArtifact, c.OrderArtifact, c.ResultArtifact}) {
			return errors.New("analysis/order/stopping binding mismatch")
		}
		if plan.OrderArtifact == nil {
			return errors.New("registered actual order artifact unavailable")
		}
		if c.OrderArtifact != *plan.OrderArtifact {
			return errors.New("order artifact mismatch")
		}
		if !hasDigest(scope.ArtifactDigests, plan.Method.SHA256) {
			return errors.New("analysis method membership mismatch")
		}
		if err := s.validateAnalysisResult(v, a, data); err != nil {
			return err
		}
		found := false
		for _, p := range c.Population {
			if p.Family == scope.PopulationFamily && !p.Unknown {
				found = true
			}
		}
		if !found {
			return errors.New("analysis population family unknown")
		}
	case "uncertainty":
		c := a.Uncertainty
		if c == nil || scope.Arm == "both" || !registrationFinite(c.AbsoluteBound) || r.Analysis.Body == nil || c.PopulationFamily != scope.PopulationFamily {
			return errors.New("arm uncertainty unknown")
		}
		tool := s.candidateTool
		if v.Arm == "comparator" {
			tool = s.comparatorTool
		}
		found := false
		for _, p := range r.Analysis.Body.Properties {
			if p.Property == scope.Property {
				if c.Property != p.Property || c.Unit != p.Unit || scope.Unit != p.Unit || scope.Denominator != p.Denominator || scope.TimingFamily != p.TimingFamily || scope.ClockDomain != p.ClockDomain || a.MethodArtifact != p.UncertaintyMethod || c.MethodArtifact != p.UncertaintyMethod {
					return errors.New("uncertainty method or scope mismatch")
				}
				for _, m := range tool.Measurements {
					if m.Property == p.Property && m.Uncertainty != nil && *m.Uncertainty == c.AbsoluteBound {
						found = true
					}
				}
			}
		}
		sampleOK := false
		for _, p := range v.Population {
			if p.Family == c.PopulationFamily && !p.Unknown && c.SampleCount == p.Observed && r.Analysis.Body != nil && c.SampleCount >= r.Analysis.Body.MinSamples {
				sampleOK = true
			}
		}
		if !found || !sampleOK || !refs([]RegistrationArtifact{c.MethodArtifact}) {
			return errors.New("uncertainty arm bound or sample count mismatch")
		}
	case "acquisition-environment":
		c := a.Environment
		if c == nil || r.Workload.Body == nil || r.Observer.Body == nil || r.Clock.Body == nil || a.MethodArtifact != r.Workload.Body.Code || !reflect.DeepEqual(c.SourceClosure, v.SourceClosure) || c.ObserverContractDigest != pairDigest(r.Observer.Body) || c.WorkloadContractDigest != pairDigest(r.Workload.Body) || !reflect.DeepEqual(c.ClockBounds, v.AcquisitionBounds) {
			return errors.New("environment contract mismatch")
		}
		lease := r.Workload.Body.HardwareLease
		if lease.State == "known" {
			if lease.Artifact == nil || c.LeaseArtifact != *lease.Artifact || !refs([]RegistrationArtifact{c.LeaseArtifact}) {
				return errors.New("hardware lease mismatch")
			}
		} else if lease.State != "not-applicable" {
			return errors.New("hardware lease unresolved")
		}
		if scope.TargetID != "" {
			found := false
			for _, t := range r.TargetRows {
				if t.ID == scope.TargetID && t.Applicability == "selected" {
					found = true
					for _, f := range []RegistrationFact{t.Host, t.Device, t.Stack, t.Topology, t.Workload, t.Deployment, t.Observer} {
						if !ledgerOneOf(f.State, "known", "not-applicable") {
							return errors.New("selected target fact unresolved")
						}
					}
				}
			}
			if !found {
				return errors.New("target not selected")
			}
		}
		for _, d := range r.Clock.Body.Domains {
			found := false
			for _, b := range c.ClockBounds {
				if b.Domain == d.ID {
					if b.Family != d.Family || b.Epoch != d.Epoch || b.Unit != d.Unit || b.Uncertainty == nil || !registrationFinite(*b.Uncertainty) || *b.Uncertainty > d.Uncertainty || b.Start < d.ValidityStart || b.End > d.ValidityEnd || b.End < b.Start {
						return errors.New("acquisition clock epoch/range/bound incompatible")
					}
					found = true
					if scope.ClockDomain == d.ID && (scope.ClockEpoch != d.Epoch || scope.TimingFamily != d.Family || scope.RangeStart != b.Start || scope.RangeEnd != b.End) {
						return errors.New("role clock scope mismatch")
					}
				}
			}
			if !found {
				return errors.New("actual clock acquisition bounds absent")
			}
		}
	default:
		return errors.New("unknown external variant")
	}
	return nil
}

// FormulateComparisonClaim consumes the shared pair admission attempt before
// checking context, provider or scope. Every failure consumes it; retry requires
// a new explicit pair acquisition. Provider policy is an external trust boundary,
// never decoded from a receipt. Record ownership transfers on Check return, even
// with an error; close is called exactly once and its error prevents a Claim.
func FormulateComparisonClaim(ctx context.Context, c RegisteredComparison, authority EvidenceAuthority, provider EvidenceAcceptance) (claim *Claim, resultErr error) {
	if c.state == nil || c.state.lease == nil || len(c.results) == 0 {
		return nil, fmt.Errorf("%w: invalid comparison handle", ErrIncompleteClaim)
	}
	s := c.state
	lease := s.lease
	lease.mu.Lock()
	if lease.used {
		lease.mu.Unlock()
		return nil, fmt.Errorf("%w: pair admission already consumed", ErrIncompleteClaim)
	}
	lease.used = true
	lease.mu.Unlock()
	if ctx == nil {
		return nil, errors.New("paired claim: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if nilInterface(provider) {
		return nil, fmt.Errorf("%w: independent acceptance provider required", ErrIncompleteClaim)
	}
	if err := validateAuthority(authority); err != nil {
		return nil, err
	}
	if len(s.problems) > 0 {
		return nil, fmt.Errorf("%w: %s", ErrIncompleteClaim, strings.Join(s.problems, "; "))
	}
	values, err := claimArmValues(s, c.results)
	if err != nil {
		return nil, err
	}
	for _, scope := range s.scopes {
		if !hasDigest(authority.AllowedProfiles, scope.DecoderProfile) {
			return nil, errors.New("paired claim: decoder profile not authorized")
		}
	}
	if len(s.policies) > 0 {
		for _, a := range s.policies {
			if a.AuthorityID != authority.AuthorityID || a.AuthorityPolicyDigest != authority.AuthorityPolicyDigest {
				return nil, errors.New("paired claim: attestation authority mismatch")
			}
		}
	}
	request := EvidenceRequest{Version: 1, RegistrationDigest: s.registration.Digest, ProtocolDigest: s.registration.Protocol.FrozenDigest, LedgerDigest: s.registration.PrerequisiteLedgerSHA256, PairDigest: s.identity.PairDigest, CandidateRawDigest: s.candidateTool.RawDigest, ComparatorRawDigest: s.comparatorTool.RawDigest, Task: s.registration.Task, CandidateTool: s.candidateTool.Tool, ComparatorTool: s.comparatorTool.Tool, Authority: pairCopy(authority), SelectedCells: pairCopy(s.identity.SelectedCells), Roles: pairCopy(s.scopes), ArmValues: values, MatchedResults: pairCopy(c.results)}
	request.RequestDigest = pairDigest(request)
	decision, err := provider.Check(ctx, pairCopy(request))
	if !nilInterface(decision.Record) {
		defer func() {
			if closeErr := decision.Record.Close(); closeErr != nil {
				claim = nil
				resultErr = errors.Join(resultErr, closeErr)
			}
		}()
	}
	if err != nil {
		return nil, fmt.Errorf("paired claim provider: %w", err)
	}
	metadata := decision
	metadata.Record = nil
	if err := registrationShapeBounds(reflect.ValueOf(metadata)); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if nilInterface(decision.Record) {
		return nil, errors.New("paired claim: acceptance record missing")
	}
	if decision.Disposition != "accepted" {
		return nil, fmt.Errorf("%w: provider disposition %s", ErrIncompleteClaim, decision.Disposition)
	}
	recordRaw, err := pairedReadRecord(ctx, decision.Record, lease)
	if err != nil {
		return nil, err
	}
	if !registrationDigest(decision.AcceptanceRecordDigest, 64) || registrationSHA(recordRaw) != decision.AcceptanceRecordDigest {
		return nil, errors.New("paired claim: acceptance record hash mismatch")
	}
	var record AcceptanceRecord
	if err := pairedDecode(recordRaw, &record, true); err != nil {
		return nil, err
	}
	want := AcceptanceRecord{Version: decision.Version, RequestDigest: decision.RequestDigest, AuthorityID: decision.AuthorityID, AuthorityPolicyDigest: decision.AuthorityPolicyDigest, AuthorityConfigDigest: decision.AuthorityConfigDigest, Disposition: decision.Disposition, ScopedRoleDecisions: decision.ScopedRoleDecisions, Reason: decision.Reason, Owner: decision.Owner, NextAction: decision.NextAction}
	if !reflect.DeepEqual(record, want) || record.Version != 1 || record.RequestDigest != request.RequestDigest || record.AuthorityID != authority.AuthorityID || record.AuthorityPolicyDigest != authority.AuthorityPolicyDigest || record.AuthorityConfigDigest != authority.AuthorityConfigDigest {
		return nil, errors.New("paired claim: record request/policy/decision mismatch")
	}
	if len(record.ScopedRoleDecisions) != len(request.Roles) {
		return nil, errors.New("paired claim: incomplete accepted scope")
	}
	for i, d := range record.ScopedRoleDecisions {
		if d.Disposition != "accepted" || !reflect.DeepEqual(d.Scope, request.Roles[i]) || d.AttestationArtifactDigest != s.attestations[d.Scope.ScopeID] {
			return nil, errors.New("paired claim: accepted scope mismatch")
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	advantages := []string{}
	for _, v := range c.results {
		advantages = append(advantages, fmt.Sprintf("%s lower by %g %s per %d (absolute bound %g)", v.Property, v.Effect, v.Unit, v.Denominator, v.Uncertainty))
	}
	r := s.registration
	limits := []string{"only the frozen task, selected cells and original populations", "absolute error bounds are caller evidence, not inferred confidence intervals", "independent acquisition acceptance depends on the configured authority policy", "source field/value binding remains declared unless its decoder validates it", "analysis result identity is checked; method execution and statistical validity require independent policy acceptance"}
	for _, t := range r.TargetRows {
		if t.Applicability != "selected" {
			limits = append(limits, fmt.Sprintf("target %s: %s; %s; %s", t.ID, t.Applicability, t.Reason, t.NextAction))
		}
	}
	selected := map[string]bool{}
	for _, cell := range s.identity.SelectedCells {
		selected[cell.CellID] = true
	}
	for _, cell := range s.ledger.Cells {
		if !selected[cell.ID] {
			limits = append(limits, fmt.Sprintf("cell %s: not-selected/%s; capability %s; %s", cell.ID, cell.Status, cell.Capability, cell.NextAction))
		}
	}
	for _, o := range s.identity.OptionalOutcomes {
		if o.Status != "consistent" {
			limits = append(limits, fmt.Sprintf("unselected artifact %s %s: %s", o.Path, o.Status, o.Reason))
		}
	}
	if strings.Contains(strings.ToLower(authority.AuthorityID), "synthetic") {
		limits = append(limits, "synthetic constructed provider rehearsal; no authentic measurement acceptance")
	}
	claim = &Claim{Task: r.Task, SpecificAdvantage: strings.Join(advantages, "; "), ComparedWith: []string{fmt.Sprintf("%s %s (%s; lineage %s)", s.comparatorTool.Tool.Name, s.comparatorTool.Tool.Version, s.comparatorTool.Tool.Configuration, s.comparatorTool.Tool.Lineage)}, QualifiedOn: fmt.Sprintf("%s/%s; device %s architecture %s; source %s binary %s", r.Source.HostOS, r.Source.HostArch, r.Source.Device, r.Source.Architecture, r.Source.Candidate, r.Source.BinarySHA256), Evidence: fmt.Sprintf("registration %s; pair %s; candidate %s; comparator %s; request %s; acceptance %s", r.Digest, s.identity.PairDigest, s.candidateTool.RawDigest, s.comparatorTool.RawDigest, request.RequestDigest, decision.AcceptanceRecordDigest), Limits: limits}
	if err := claim.Validate(); err != nil {
		return nil, err
	}
	return claim, nil
}

func nilInterface(v any) bool {
	if v == nil {
		return true
	}
	r := reflect.ValueOf(v)
	switch r.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return r.IsNil()
	}
	return false
}
func validateAuthority(a EvidenceAuthority) error {
	if err := registrationShapeBounds(reflect.ValueOf(a)); err != nil {
		return err
	}
	if a.AuthorityID == "" || !registrationDigest(a.AuthorityPolicyDigest, 64) || !registrationDigest(a.AuthorityConfigDigest, 64) || len(a.AllowedProfiles) == 0 {
		return errors.New("paired claim: independent authority configuration missing")
	}
	for i, v := range a.AllowedProfiles {
		if i > 0 && a.AllowedProfiles[i-1] >= v {
			return errors.New("paired claim: noncanonical authority profiles")
		}
		known := false
		for _, role := range []string{"measurement", "selected-prerequisite", "diagnosis", "observation", "host-observation", "preregistration"} {
			if profileRole(role, v) {
				known = true
			}
		}
		if !known {
			return errors.New("paired claim: unknown authority profile")
		}
	}
	return nil
}
func claimArmValues(s *registeredPair, results []MatchedResult) ([]ArmValue, error) {
	r := s.registration
	if len(results) != len(r.Protocol.Comparison.Properties) {
		return nil, errors.New("paired claim: selected property omitted")
	}
	cm, bm := map[string]PropertyMeasurement{}, map[string]PropertyMeasurement{}
	for _, v := range s.candidateTool.Measurements {
		cm[v.Property] = v
	}
	for _, v := range s.comparatorTool.Measurements {
		bm[v.Property] = v
	}
	var out []ArmValue
	for _, result := range results {
		a, b := cm[result.Property], bm[result.Property]
		limit := r.Protocol.PropertyLimits[result.Property]
		if result.Status != "candidate_lower" || a.Status != "measured" || b.Status != "measured" || a.Uncertainty == nil || b.Uncertainty == nil || a.Value > limit || b.Value > limit || result.Effect <= result.Uncertainty || math.IsNaN(result.Effect) {
			return nil, fmt.Errorf("%w: selected property has no bounded qualified advantage", ErrIncompleteClaim)
		}
		out = append(out, ArmValue{Property: result.Property, Unit: result.Unit, Denominator: result.Denominator, CandidateValue: a.Value, CandidateAbsoluteBound: *a.Uncertainty, ComparatorValue: b.Value, ComparatorAbsoluteBound: *b.Uncertainty, MaxValue: limit, ThresholdPolicy: "both-arms"})
	}
	return out, nil
}
