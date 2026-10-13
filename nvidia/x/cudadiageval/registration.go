package cudadiageval

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"reflect"
	"sort"
	"strings"
	"time"
)

// RegistrationFact records a supplied fact, not a measurement acceptance.
// State is known, unknown, unavailable, denied, conflicting or not-applicable.
// Known artifact bytes are verified; their named field/value binding is declared.
// Unknown facts
// retain their reason, owner and next action; they have no guessed value.
type RegistrationFact struct {
	State        string                `json:"state"`
	Value        string                `json:"value"`
	SourceSHA256 string                `json:"source_sha256"`
	Artifact     *RegistrationArtifact `json:"artifact"`
	Field        string                `json:"field"`
	Reason       string                `json:"reason"`
	Owner        string                `json:"owner"`
	NextAction   string                `json:"next_action"`
}

// RegistrationArtifact identifies an immutable input. Path is relative to the
// caller's artifact root. Its digest does not establish acquisition authenticity.
type RegistrationArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// RegistrationCell selects a prerequisite from the frozen ledger snapshot.
// ReceiptSHA256 may be empty for a pending cell with a reason and next action.
// ClosureSHA256 hashes the JSON encoding of the selected QualificationClosure.
type RegistrationCell struct {
	Applicability string `json:"applicability"`
	CellID        string `json:"cell_id"`
	CapabilityID  string `json:"capability_id"`
	Gate          string `json:"gate"`
	ClosureSHA256 string `json:"closure_sha256"`
	ReceiptSHA256 string `json:"receipt_sha256"`
	Reason        string `json:"reason"`
	Owner         string `json:"owner"`
	NextAction    string `json:"next_action"`
}

// RegistrationTarget retains a selected or unavailable target row.
// Applicability is selected, not-selected or unsupported. It never qualifies
// hardware or implies that discovery or a test was executed.
type RegistrationTarget struct {
	ID            string           `json:"id"`
	Applicability string           `json:"applicability"`
	Host          RegistrationFact `json:"host"`
	Device        RegistrationFact `json:"device"`
	Stack         RegistrationFact `json:"stack"`
	Topology      RegistrationFact `json:"topology"`
	Workload      RegistrationFact `json:"workload"`
	Deployment    RegistrationFact `json:"deployment"`
	Observer      RegistrationFact `json:"observer"`
	Reason        string           `json:"reason"`
	Owner         string           `json:"owner"`
	NextAction    string           `json:"next_action"`
}

// RegistrationContract describes a planned contract, not an executed method.
// Declared contracts have a typed body. Unknown/unavailable bodies are nil.
// Notes are supplemental and have no admission meaning.
type RegistrationContract struct {
	State      string                 `json:"state"`
	ID         string                 `json:"id"`
	Notes      string                 `json:"notes"`
	Artifacts  []RegistrationArtifact `json:"artifacts"`
	Reason     string                 `json:"reason"`
	Owner      string                 `json:"owner"`
	NextAction string                 `json:"next_action"`
}

// RegistrationLibrary retains a supplied loaded-library fact and its source.
type RegistrationLibrary struct {
	Name string           `json:"name"`
	Fact RegistrationFact `json:"fact"`
}

// RegistrationWorkload pins the question, mechanism and pretrial inputs.
type RegistrationWorkload struct {
	Contract RegistrationContract      `json:"contract"`
	Body     *RegistrationWorkloadSpec `json:"body"`
}

// RegistrationWorkloadSpec contains proposed environment facts, not executions.
type RegistrationWorkloadSpec struct {
	Question        string                 `json:"question"`
	Mechanism       string                 `json:"mechanism"`
	Code            RegistrationArtifact   `json:"code"`
	Inputs          []RegistrationArtifact `json:"inputs"`
	ExpectedOutputs []RegistrationArtifact `json:"expected_outputs"`
	CacheState      string                 `json:"cache_state"`
	PowerSettings   RegistrationFact       `json:"power_settings"`
	Contention      RegistrationFact       `json:"contention"`
	HardwareLease   RegistrationFact       `json:"hardware_lease"`
}

// RegistrationOracle pins an independent expected result and quality rules.
type RegistrationOracle struct {
	Contract RegistrationContract    `json:"contract"`
	Body     *RegistrationOracleSpec `json:"body"`
}

// RegistrationOracleSpec names supplied oracle artifacts; none is executed.
type RegistrationOracleSpec struct {
	Method         string                    `json:"method"`
	Implementation RegistrationArtifact      `json:"implementation"`
	Expected       []RegistrationArtifact    `json:"expected"`
	Quality        []RegistrationQualitySpec `json:"quality"`
}

// RegistrationQualitySpec freezes a numerical rule for one output.
type RegistrationQualitySpec struct {
	Output            string  `json:"output"`
	Unit              string  `json:"unit"`
	AbsoluteTolerance float64 `json:"absolute_tolerance"`
	RelativeTolerance float64 `json:"relative_tolerance"`
	Nonfinite         string  `json:"nonfinite"`
}

// RegistrationClock retains distinct timing domains and their calibrations.
type RegistrationClock struct {
	Contract RegistrationContract   `json:"contract"`
	Body     *RegistrationClockSpec `json:"body"`
}

// RegistrationClockSpec does not infer equivalence between timing families.
type RegistrationClockSpec struct {
	Domains []RegistrationClockDomain `json:"domains"`
}

// RegistrationClockDomain records a proposed epoch, range and absolute bound.
type RegistrationClockDomain struct {
	ID            string               `json:"id"`
	Family        string               `json:"family"`
	Epoch         string               `json:"epoch"`
	Unit          string               `json:"unit"`
	Calibration   RegistrationArtifact `json:"calibration"`
	Uncertainty   float64              `json:"uncertainty"`
	ValidityStart int64                `json:"validity_start"`
	ValidityEnd   int64                `json:"validity_end"`
}

// RegistrationLoss pins completeness rules for each supplied record family.
type RegistrationLoss struct {
	Contract RegistrationContract  `json:"contract"`
	Body     *RegistrationLossSpec `json:"body"`
}

// RegistrationLossSpec retains all supplied families separately.
type RegistrationLossSpec struct {
	Families []RegistrationLossFamily `json:"families"`
}

// RegistrationLossFamily never subtracts missing records from the denominator.
type RegistrationLossFamily struct {
	Name            string `json:"name"`
	Required        bool   `json:"required"`
	DenominatorUnit string `json:"denominator_unit"`
	ExpectedRecords uint64 `json:"expected_records"`
	MaxMissing      uint64 `json:"max_missing"`
	MaxLost         uint64 `json:"max_lost"`
	Handling        string `json:"handling"`
}

// RegistrationObserver pins an arm's sensor access and perturbation proposals.
type RegistrationObserver struct {
	Contract RegistrationContract      `json:"contract"`
	Body     *RegistrationObserverSpec `json:"body"`
}

// RegistrationObserverSpec retains unresolved access and ownership conflicts.
type RegistrationObserverSpec struct {
	Mode           string             `json:"mode"`
	SensorScope    []string           `json:"sensor_scope"`
	Access         RegistrationFact   `json:"access"`
	FreshnessNanos int64              `json:"freshness_nanos"`
	DutyCycle      float64            `json:"duty_cycle"`
	SamplingRateHz float64            `json:"sampling_rate_hz"`
	OwnerConflicts []RegistrationFact `json:"owner_conflicts"`
	Arm            string             `json:"arm"`
	ReferenceArm   string             `json:"reference_arm"`
}

// RegistrationAnalysis freezes sample selection and the existing properties.
type RegistrationAnalysis struct {
	Contract RegistrationContract      `json:"contract"`
	Body     *RegistrationAnalysisSpec `json:"body"`
}

// RegistrationAnalysisSpec pins an externally supplied analysis method.
type RegistrationAnalysisSpec struct {
	Repetitions   uint64                     `json:"repetitions"`
	Warmup        uint64                     `json:"warmup"`
	MinSamples    uint64                     `json:"min_samples"`
	Order         string                     `json:"order"`
	Seed          *uint64                    `json:"seed"`
	OrderArtifact *RegistrationArtifact      `json:"order_artifact"`
	Method        RegistrationArtifact       `json:"method"`
	Properties    []RegistrationPropertySpec `json:"properties"`
	Exclusions    []RegistrationArtifact     `json:"exclusions"`
}

// RegistrationPropertySpec matches CompareTools' lower-is-better absolute bounds.
type RegistrationPropertySpec struct {
	Property          string               `json:"property"`
	Unit              string               `json:"unit"`
	Denominator       uint64               `json:"denominator"`
	Direction         string               `json:"direction"`
	TimingFamily      string               `json:"timing_family"`
	ClockDomain       string               `json:"clock_domain"`
	UncertaintyKind   string               `json:"uncertainty_kind"`
	UncertaintyMethod RegistrationArtifact `json:"uncertainty_method"`
	Overlap           string               `json:"overlap"`
	MaxValue          float64              `json:"max_value"`
}

// RegistrationStopping pins the sample and duration bounds without running them.
type RegistrationStopping struct {
	Contract RegistrationContract      `json:"contract"`
	Body     *RegistrationStoppingSpec `json:"body"`
}

// RegistrationStoppingSpec requires a method for sequential stopping.
type RegistrationStoppingSpec struct {
	Rule             string                `json:"rule"`
	MaxSamples       uint64                `json:"max_samples"`
	MaxDurationNanos int64                 `json:"max_duration_nanos"`
	Method           *RegistrationArtifact `json:"method"`
}

// RegistrationMissingData keeps withheld cases in the declared population.
type RegistrationMissingData struct {
	Contract RegistrationContract         `json:"contract"`
	Body     *RegistrationMissingDataSpec `json:"body"`
}

// RegistrationMissingDataSpec pins proposed budgets, not observed loss counts.
type RegistrationMissingDataSpec struct {
	Handling    string                `json:"handling"`
	Denominator string                `json:"denominator"`
	MaxMissing  uint64                `json:"max_missing"`
	MaxWithheld uint64                `json:"max_withheld"`
	Method      *RegistrationArtifact `json:"method"`
}

// RegistrationLimits bounds the registered experiment. These are caller
// proposals, not measured resource use or permission to launch that experiment.
type RegistrationLimits struct {
	MaxDurationNanos int64  `json:"max_duration_nanos"`
	MaxRecords       uint64 `json:"max_records"`
	MaxBytes         uint64 `json:"max_bytes"`
}

// Registration is a versioned, immutable plan beside the existing Protocol.
// Protocol must already be frozen. PrerequisiteLedgerSHA256 hashes the bytes
// emitted by WriteQualificationLedger; it does not bind future trial results.
// RegisteredAt is caller supplied and does not prove premeasurement registration.
// The zero value is invalid. Reading, freezing and verification execute no tests.
type Registration struct {
	Version                  int                     `json:"version"`
	ID                       string                  `json:"id"`
	ParentDigest             string                  `json:"parent_digest"`
	RevisionReason           string                  `json:"revision_reason"`
	Task                     string                  `json:"task"`
	Mode                     string                  `json:"mode"`
	Owner                    string                  `json:"owner"`
	Protocol                 Protocol                `json:"protocol"`
	SourceLibraries          []RegistrationLibrary   `json:"source_libraries"`
	Source                   QualificationClosure    `json:"source"`
	PrerequisiteLedgerSHA256 string                  `json:"prerequisite_ledger_sha256"`
	SelectedCells            []RegistrationCell      `json:"selected_cells"`
	TargetRows               []RegistrationTarget    `json:"target_rows"`
	Workload                 RegistrationWorkload    `json:"workload"`
	Oracle                   RegistrationOracle      `json:"oracle"`
	Clock                    RegistrationClock       `json:"clock"`
	Loss                     RegistrationLoss        `json:"loss"`
	Observer                 RegistrationObserver    `json:"observer"`
	Analysis                 RegistrationAnalysis    `json:"analysis"`
	PilotArtifacts           []RegistrationArtifact  `json:"pilot_artifacts"`
	HeldOutArtifacts         []RegistrationArtifact  `json:"held_out_artifacts"`
	Stopping                 RegistrationStopping    `json:"stopping"`
	MissingData              RegistrationMissingData `json:"missing_data"`
	Limits                   RegistrationLimits      `json:"limits"`
	RegisteredAt             time.Time               `json:"registered_at"`
	Digest                   string                  `json:"digest"`
}

// RegistrationResult separates retained receipt consistency from measurement
// acceptance. MeasurementAcceptance is always unverified in this source-only API.
type RegistrationResult struct {
	Applicability         string `json:"applicability"`
	CellID                string `json:"cell_id"`
	Selected              bool   `json:"selected"`
	Capability            string `json:"capability"`
	Status                string `json:"status"`
	Gate                  string `json:"gate"`
	ReceiptConsistency    bool   `json:"receipt_consistency"`
	MeasurementAcceptance string `json:"measurement_acceptance"`
	Reason                string `json:"reason"`
	Owner                 string `json:"owner"`
	NextAction            string `json:"next_action"`
}

// RegistrationArtifactResult retains the original expected path/digest and a
// bounded acquisition outcome. Consistent bytes are not authentic acquisition.
type RegistrationArtifactResult struct {
	Artifact RegistrationArtifact `json:"artifact"`
	Required bool                 `json:"required"`
	Status   string               `json:"status"`
	Reason   string               `json:"reason"`
}

// RegistrationReport is a deterministic factual report, not a comparative
// claim, health judgment or execution authority. Targets retain unknown facts.
type RegistrationReport struct {
	// ReceiptConsistency covers selected prerequisites and all acquired artifact
	// outcomes. It is false for omitted optional artifacts, never a qualification.
	ReceiptConsistency    bool                         `json:"receipt_consistency"`
	Artifacts             []RegistrationArtifactResult `json:"artifacts"`
	SourceLibraries       []RegistrationLibrary        `json:"source_libraries"`
	RegistrationDigest    string                       `json:"registration_digest"`
	MeasurementAcceptance string                       `json:"measurement_acceptance"`
	Cells                 []RegistrationResult         `json:"cells"`
	Targets               []RegistrationTarget         `json:"targets"`
	FactBinding           string                       `json:"fact_binding"`
}

const registrationLimit = 4 << 20

func registrationSHA(b []byte) string         { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func registrationDigest(s string, n int) bool { return ledgerDigest(s, n) && s == strings.ToLower(s) }
func (r Registration) contentDigest() (string, error) {
	r.Digest = ""
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	return registrationSHA(b), nil
}

// Validate checks registration structure and frozen content, without reading
// artifacts. It does not authenticate declared policies, clocks or measurements.
func (r Registration) Validate() error {
	if err := registrationShapeBounds(reflect.ValueOf(r)); err != nil {
		return err
	}
	if r.Version != 1 || r.ID == "" || r.Task == "" || r.Owner == "" || !ledgerOneOf(r.Mode, "property", "comparison") {
		return errors.New("registration: invalid version or identity")
	}
	if !r.Protocol.Frozen {
		return errors.New("registration: protocol must already be frozen")
	}
	if err := r.Protocol.Validate(); err != nil {
		return fmt.Errorf("registration protocol: %w", err)
	}
	if r.Mode == "comparison" && (r.Protocol.Comparison == nil || r.Protocol.Comparison.Task != r.Task) {
		return errors.New("registration: comparison task missing or inconsistent")
	}
	if r.Mode == "property" && r.Protocol.Comparison != nil {
		return errors.New("registration: property mode has comparison arms")
	}
	if err := r.Source.validate(false); err != nil {
		return fmt.Errorf("registration source: %w", err)
	}
	for _, d := range []string{r.Source.Candidate, r.Source.Tree, r.Source.Base} {
		if d != "" && !registrationDigest(d, 40) {
			return errors.New("registration: noncanonical source digest")
		}
	}
	for _, d := range []string{r.Source.BinarySHA256, r.Source.DependenciesSHA256, r.Source.InputsSHA256, r.Source.OutputsSHA256, r.Source.SignatureMapSHA256} {
		if d != "" && !registrationDigest(d, 64) {
			return errors.New("registration: noncanonical source digest")
		}
	}
	if !registrationDigest(r.PrerequisiteLedgerSHA256, 64) {
		return errors.New("registration: invalid ledger digest")
	}
	names := map[string]bool{}
	for _, lib := range r.SourceLibraries {
		if lib.Name == "" || names[lib.Name] {
			return errors.New("registration: duplicate library fact")
		}
		names[lib.Name] = true
		if err := lib.Fact.validate(); err != nil {
			return err
		}
		if lib.Fact.State == "known" && r.Source.Libraries[lib.Name] != lib.Fact.Value {
			return errors.New("registration: library fact differs from closure")
		}
	}
	for name := range r.Source.Libraries {
		if !names[name] {
			return errors.New("registration: source library lacks fact provenance")
		}
	}
	if (r.ParentDigest == "") != (r.RevisionReason == "") || r.ParentDigest != "" && !registrationDigest(r.ParentDigest, 64) {
		return errors.New("registration: incomplete revision identity")
	}
	if len(r.SelectedCells) == 0 || len(r.SelectedCells) > 4096 || len(r.TargetRows) == 0 || len(r.TargetRows) > 4096 {
		return errors.New("registration: invalid cell or target count")
	}
	ids := map[string]bool{}
	for _, c := range r.SelectedCells {
		if !ledgerOneOf(c.Applicability, "selected", "not-selected", "unsupported") || c.Applicability != "selected" && c.Reason == "" || c.CellID == "" || ids[c.CellID] || c.CapabilityID == "" || !ledgerGate(c.Gate) || !registrationDigest(c.ClosureSHA256, 64) || c.Owner == "" || c.NextAction == "" {
			return errors.New("registration: invalid or duplicate selected cell")
		}
		ids[c.CellID] = true
		if c.ReceiptSHA256 == "" {
			if c.Reason == "" {
				return errors.New("registration: pending receipt has no reason")
			}
		} else if !registrationDigest(c.ReceiptSHA256, 64) {
			return errors.New("registration: invalid receipt digest")
		}
	}
	ids = map[string]bool{}
	for _, t := range r.TargetRows {
		if t.ID == "" || ids[t.ID] || !ledgerOneOf(t.Applicability, "selected", "not-selected", "unsupported") || t.Owner == "" || t.NextAction == "" || t.Applicability != "selected" && t.Reason == "" {
			return errors.New("registration: invalid target row")
		}
		ids[t.ID] = true
		for _, f := range []RegistrationFact{t.Host, t.Device, t.Stack, t.Topology, t.Workload, t.Deployment, t.Observer} {
			if err := f.validate(); err != nil {
				return err
			}
		}
	}
	if err := r.validateContracts(); err != nil {
		return err
	}
	if _, err := r.artifacts(); err != nil {
		return err
	}
	if r.Limits.MaxDurationNanos <= 0 || r.Limits.MaxRecords == 0 || r.Limits.MaxBytes == 0 {
		return errors.New("registration: missing resource limits")
	}
	if r.Digest == "" {
		if !r.RegisteredAt.IsZero() {
			return errors.New("registration: timestamp without frozen digest")
		}
	} else {
		if !registrationDigest(r.Digest, 64) || r.RegisteredAt.IsZero() {
			return errors.New("registration: invalid frozen identity")
		}
		_, offset := r.RegisteredAt.Zone()
		if offset != 0 {
			return errors.New("registration: registration time must be UTC")
		}
		d, err := r.contentDigest()
		if err != nil {
			return err
		}
		if d != r.Digest {
			return errors.New("registration: frozen content changed")
		}
	}
	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("registration encoding: %w", err)
	}
	if len(b)+1 > registrationLimit {
		return errors.New("registration: size limit exceeded")
	}
	if err := registrationTokenBounds(b); err != nil {
		return err
	}
	return registrationKeys(b, reflect.TypeOf(r))
}

func registrationArtifacts(as []RegistrationArtifact) error {
	seen := map[string]bool{}
	for _, a := range as {
		if !fs.ValidPath(a.Path) || a.Path == "." || seen[a.Path] || !registrationDigest(a.SHA256, 64) {
			return errors.New("registration: invalid or duplicate artifact")
		}
		seen[a.Path] = true
	}
	return nil
}

// Freeze returns a detached immutable registration at the caller's UTC time.
// A local timestamp and digest are not premeasurement provenance.
func (r Registration) Freeze(at time.Time) (Registration, error) {
	if r.Digest != "" {
		return Registration{}, errors.New("registration: already frozen")
	}
	if err := r.Validate(); err != nil {
		return Registration{}, err
	}
	_, offset := at.Zone()
	if at.IsZero() || offset != 0 {
		return Registration{}, errors.New("registration: nonzero UTC time required")
	}
	b, err := json.Marshal(r)
	if err != nil {
		return Registration{}, err
	}
	var out Registration
	if err := json.Unmarshal(b, &out); err != nil {
		return Registration{}, err
	}
	out.RegisteredAt = at
	out.Digest, err = out.contentDigest()
	if err != nil {
		return Registration{}, err
	}
	return out, out.Validate()
}

// CheckRegistrationSuccessor requires immutable same-ID content or a new ID
// linked to the frozen predecessor. It never rewrites earlier receipt attempts.
func CheckRegistrationSuccessor(previous, next Registration) error {
	if err := previous.Validate(); err != nil {
		return err
	}
	if err := next.Validate(); err != nil {
		return err
	}
	if previous.Digest == "" || next.Digest == "" {
		return errors.New("registration successor: frozen inputs required")
	}
	if previous.ID == next.ID {
		if previous.Digest != next.Digest || !reflect.DeepEqual(previous, next) {
			return errors.New("registration successor: same ID changed")
		}
		return nil
	}
	if next.ParentDigest != previous.Digest || next.RevisionReason == "" || next.RegisteredAt.Before(previous.RegisteredAt) {
		return errors.New("registration successor: missing parent or revision reason")
	}
	return nil
}

// VerifyRegistration checks the immutable prerequisite snapshot and selected
// cell bindings using the existing ledger verifier. It reports receipt
// consistency only. Even a consistent qualified receipt remains an unverified
// measurement. Registered artifact bytes are checked; field/value bindings remain
// declared. Verification uses a private checked-byte snapshot, removed on return.
// It executes no recorded command or policy method.
func VerifyRegistration(ctx context.Context, r Registration, l QualificationLedger, root string) (RegistrationReport, error) {
	return verifyRegistrationSnapshot(ctx, r, l, func() (string, func() error, []RegistrationArtifactResult, error) {
		return registrationSnapshot(ctx, r, l, root)
	})
}

// The acquisition callback is private and supplies only checked snapshot bytes.
// Standalone verification and paired admission share the unchanged ledger verifier.
func verifyRegistrationSnapshot(ctx context.Context, r Registration, l QualificationLedger, acquire func() (string, func() error, []RegistrationArtifactResult, error)) (RegistrationReport, error) {
	var out RegistrationReport
	if ctx == nil {
		return out, errors.New("verify registration: nil context")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if err := r.Validate(); err != nil {
		return out, err
	}
	if r.Digest == "" {
		return out, errors.New("verify registration: frozen registration required")
	}
	var b bytes.Buffer
	if err := WriteQualificationLedger(&b, l); err != nil {
		return out, err
	}
	if registrationSHA(b.Bytes()) != r.PrerequisiteLedgerSHA256 {
		return out, errors.New("verify registration: prerequisite ledger digest mismatch")
	}
	cells := map[string]QualificationCell{}
	for _, c := range l.Cells {
		cells[c.ID] = c
	}
	selected := map[string]bool{}
	maxDepth := map[string]int{}
	var visit func(string, int) error
	visit = func(id string, depth int) error {
		if depth > 32 {
			return errors.New("verify registration: dependency depth exceeded")
		}
		c, ok := cells[id]
		if !ok {
			return errors.New("verify registration: missing selected dependency")
		}
		if old, ok := maxDepth[id]; ok && old >= depth {
			return nil
		}
		maxDepth[id] = depth
		selected[id] = true
		for _, d := range c.DependsOn {
			if err := visit(d, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	for _, s := range r.SelectedCells {
		c, ok := cells[s.CellID]
		if !ok {
			return out, errors.New("verify registration: missing selected cell")
		}
		cb, err := json.Marshal(c.Closure)
		if err != nil {
			return out, err
		}
		if c.CapabilityID != s.CapabilityID || c.Gate != s.Gate || registrationSHA(cb) != s.ClosureSHA256 {
			return out, errors.New("verify registration: selected scope mismatch")
		}
		if s.ReceiptSHA256 != "" {
			if len(c.Attempts) == 0 || c.Attempts[len(c.Attempts)-1].SHA256 != s.ReceiptSHA256 {
				return out, errors.New("verify registration: selected receipt is not latest attempt")
			}
		}
		if s.Applicability != "selected" {
			continue
		}
		if err := visit(c.ID, 0); err != nil {
			return out, err
		}
	}
	snapshot, cleanup, artifacts, err := acquire()
	if err != nil {
		return out, err
	}
	results, err := VerifyQualificationLedger(ctx, snapshot, l)
	cleanupErr := cleanup()
	if err != nil {
		return out, errors.Join(err, cleanupErr)
	}
	if cleanupErr != nil {
		return out, fmt.Errorf("verify registration cleanup: %w", cleanupErr)
	}
	out.Artifacts = artifacts
	out.ReceiptConsistency = len(selected) > 0 && !r.pendingContracts()
	for _, a := range artifacts {
		if a.Status != "consistent" {
			out.ReceiptConsistency = false
		}
	}
	out.RegistrationDigest = r.Digest
	out.MeasurementAcceptance = "unverified"
	out.FactBinding = "artifact bytes checked; field/value binding declared"
	values := map[string]QualificationResult{}
	blocked := func(v QualificationResult, reason string) QualificationResult {
		v.Accepted = false
		if v.Status == "qualified" {
			v.Status = "pending"
		}
		if v.Reason == "" {
			v.Reason = reason
		}
		return v
	}
	for _, v := range results {
		values[v.CellID] = v
	}
	for _, s := range r.SelectedCells {
		v := values[s.CellID]
		if s.Applicability != "selected" {
			v = blocked(v, s.Reason)
			if s.Applicability == "not-selected" {
				v.Status = "not-selected"
			}

		}
		if s.ReceiptSHA256 == "" {
			v = blocked(v, s.Reason)
		}
		values[s.CellID] = v
	}
	if r.pendingContracts() {
		for id := range selected {
			values[id] = blocked(values[id], "required registered contract or fact is unresolved")
		}
	}
	done := map[string]bool{}
	var resolve func(string)
	resolve = func(id string) {
		if done[id] {
			return
		}
		v := values[id]
		for _, dep := range cells[id].DependsOn {
			resolve(dep)
			if !values[dep].Accepted {
				v = blocked(v, "registered prerequisite is incomplete")
			}
		}
		values[id] = v
		done[id] = true
	}
	for _, v := range results {
		resolve(v.CellID)
	}
	for _, original := range results {
		v := values[original.CellID]
		if selected[v.CellID] && !v.Accepted {
			out.ReceiptConsistency = false
		}
		applicability := "not-selected"
		if selected[v.CellID] {
			applicability = "selected"
		}
		for _, s := range r.SelectedCells {
			if s.CellID == v.CellID {
				applicability = s.Applicability
			}
		}
		out.Cells = append(out.Cells, RegistrationResult{Applicability: applicability, CellID: v.CellID, Selected: selected[v.CellID], Capability: v.Capability, Status: v.Status, Gate: v.Gate, ReceiptConsistency: v.Accepted && selected[v.CellID], MeasurementAcceptance: "unverified", Reason: v.Reason, Owner: v.Owner, NextAction: v.NextAction})
	}
	// Deep detach target facts so modifying a report cannot alter registration.
	tb, err := json.Marshal(r.TargetRows)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(tb, &out.Targets); err != nil {
		return out, err
	}
	lb, err := json.Marshal(r.SourceLibraries)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(lb, &out.SourceLibraries); err != nil {
		return out, err
	}
	sort.Slice(out.SourceLibraries, func(i, j int) bool { return out.SourceLibraries[i].Name < out.SourceLibraries[j].Name })
	sort.Slice(out.Targets, func(i, j int) bool { return out.Targets[i].ID < out.Targets[j].ID })
	return out, nil
}

func (f RegistrationFact) validate() error {
	if !ledgerOneOf(f.State, "known", "unknown", "unavailable", "denied", "conflicting", "not-applicable") || f.Owner == "" || f.NextAction == "" {
		return errors.New("registration: invalid fact state")
	}
	if f.State == "known" {
		if f.Value == "" || f.Field == "" || f.Artifact == nil || !registrationDigest(f.SourceSHA256, 64) || f.Artifact.SHA256 != f.SourceSHA256 {
			return errors.New("registration: known fact missing rooted field provenance")
		}
		return registrationArtifacts([]RegistrationArtifact{*f.Artifact})
	}
	if f.Value != "" || f.SourceSHA256 != "" || f.Artifact != nil || f.Field != "" || f.Reason == "" {
		return errors.New("registration: unresolved fact has guessed provenance")
	}
	return nil
}

func registrationFinite(v float64) bool { return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }
func registrationNames(names []string) bool {
	seen := map[string]bool{}
	for _, name := range names {
		if name == "" || seen[name] {
			return false
		}
		seen[name] = true
	}
	return true
}
func registrationContract(c RegistrationContract, body any) error {
	if !ledgerOneOf(c.State, "declared", "unknown", "unavailable") || c.Owner == "" || c.NextAction == "" {
		return errors.New("registration: invalid contract state")
	}
	nilBody := reflect.ValueOf(body).IsNil()
	if c.State == "declared" {
		if c.ID == "" || nilBody {
			return errors.New("registration: missing typed contract body")
		}
	} else if c.ID != "" || !nilBody || len(c.Artifacts) != 0 || c.Reason == "" {
		return errors.New("registration: unresolved contract has body or no reason")
	}
	return nil
}

func (r Registration) validateContracts() error {
	for _, v := range []struct {
		c RegistrationContract
		b any
	}{
		{r.Workload.Contract, r.Workload.Body}, {r.Oracle.Contract, r.Oracle.Body}, {r.Clock.Contract, r.Clock.Body}, {r.Loss.Contract, r.Loss.Body},
		{r.Observer.Contract, r.Observer.Body}, {r.Analysis.Contract, r.Analysis.Body}, {r.Stopping.Contract, r.Stopping.Body}, {r.MissingData.Contract, r.MissingData.Body},
	} {
		if err := registrationContract(v.c, v.b); err != nil {
			return err
		}
	}
	if w := r.Workload.Body; w != nil {
		if w.Question == "" || w.Mechanism == "" || len(w.Inputs) == 0 || !ledgerOneOf(w.CacheState, "cold", "warm", "mixed", "not-applicable") {
			return errors.New("registration: incomplete workload")
		}
		for _, f := range []RegistrationFact{w.PowerSettings, w.Contention, w.HardwareLease} {
			if err := f.validate(); err != nil {
				return err
			}
		}
	}
	if o := r.Oracle.Body; o != nil {
		if !ledgerOneOf(o.Method, "exact", "absolute-relative", "labelled-ground-truth", "external") || len(o.Expected) == 0 || len(o.Quality) == 0 {
			return errors.New("registration: incomplete oracle")
		}
		names := []string{}
		for _, q := range o.Quality {
			names = append(names, q.Output)
			if q.Unit == "" || !registrationFinite(q.AbsoluteTolerance) || !registrationFinite(q.RelativeTolerance) || !ledgerOneOf(q.Nonfinite, "reject", "explicit-expected") || o.Method == "exact" && (q.AbsoluteTolerance != 0 || q.RelativeTolerance != 0) {
				return errors.New("registration: invalid output quality")
			}
		}
		if !registrationNames(names) {
			return errors.New("registration: duplicate oracle output")
		}
	}
	domains := map[string]RegistrationClockDomain{}
	if c := r.Clock.Body; c != nil {
		if len(c.Domains) == 0 {
			return errors.New("registration: missing clock domains")
		}
		for _, d := range c.Domains {
			if d.ID == "" || domains[d.ID].ID != "" || d.Epoch == "" || d.Unit == "" || !ledgerOneOf(d.Family, "host-wall", "host-monotonic", "device-event", "device-activity", "sensor-window", "counter-duration", "interval-union", "external") || !registrationFinite(d.Uncertainty) || d.ValidityEnd < d.ValidityStart {
				return errors.New("registration: invalid clock epoch or bound")
			}
			domains[d.ID] = d
		}
	}
	if l := r.Loss.Body; l != nil {
		if len(l.Families) == 0 {
			return errors.New("registration: missing loss families")
		}
		names := []string{}
		for _, f := range l.Families {
			names = append(names, f.Name)
			if f.DenominatorUnit == "" || f.ExpectedRecords == 0 || f.MaxMissing > f.ExpectedRecords || f.MaxLost > f.ExpectedRecords-f.MaxMissing || !ledgerOneOf(f.Handling, "withhold", "bounded") || f.Handling == "withhold" && (f.MaxMissing != 0 || f.MaxLost != 0) {
				return errors.New("registration: invalid loss denominator or budget")
			}
		}
		if !registrationNames(names) {
			return errors.New("registration: duplicate loss family")
		}
	}
	if o := r.Observer.Body; o != nil {
		if !ledgerOneOf(o.Mode, "off", "basic", "host", "device-activity", "pc", "pm", "range", "sass", "deep", "serialized", "replay", "external") || len(o.SensorScope) == 0 || !registrationNames(o.SensorScope) || o.FreshnessNanos <= 0 || !registrationFinite(o.DutyCycle) || o.DutyCycle > 1 || !registrationFinite(o.SamplingRateHz) || o.Mode == "off" && (o.DutyCycle != 0 || o.SamplingRateHz != 0) {
			return errors.New("registration: invalid observer scope or duty")
		}
		for _, f := range append([]RegistrationFact{o.Access}, o.OwnerConflicts...) {
			if err := f.validate(); err != nil {
				return err
			}
		}
		tools := map[string]bool{}
		for _, t := range r.Protocol.Tools {
			tools[t.Name] = true
		}
		if !tools[o.Arm] || r.Mode == "comparison" && (o.Arm != r.Protocol.Comparison.Candidate || o.ReferenceArm != r.Protocol.Comparison.Comparator) || r.Mode == "property" && o.ReferenceArm != "" {
			return errors.New("registration: observer arm differs from protocol")
		}
	}
	if a := r.Analysis.Body; a != nil {
		if a.Repetitions == 0 || a.MinSamples == 0 || a.MinSamples > a.Repetitions || !ledgerOneOf(a.Order, "randomized", "interleaved", "fixed") || a.Order == "randomized" && a.Seed == nil || a.Order != "randomized" && a.OrderArtifact == nil {
			return errors.New("registration: invalid sample order or population")
		}
		expected := len(r.Protocol.PropertyLimits)
		if r.Mode == "comparison" {
			expected = len(r.Protocol.Comparison.Properties)
		}
		if len(a.Properties) == 0 || len(a.Properties) != expected {
			return errors.New("registration: selected property scope incomplete")
		}
		names := []string{}
		for _, p := range a.Properties {
			names = append(names, p.Property)
			max, ok := r.Protocol.PropertyLimits[p.Property]
			if !ok || !ledgerOneOf(p.Property, "attribution", "diagnosis", "cost", "reproducibility") || p.Unit == "" || p.Denominator == 0 || p.Direction != "lower" || p.UncertaintyKind != "absolute-bound" || p.Overlap != "inconclusive" || !registrationFinite(p.MaxValue) || p.MaxValue != max {
				return errors.New("registration: invalid property denominator or uncertainty")
			}
			if r.Mode == "comparison" {
				scope, ok := r.Protocol.Comparison.Properties[p.Property]
				if !ok || scope.Unit != p.Unit || scope.Denominator != p.Denominator {
					return errors.New("registration: property differs from comparison scope")
				}
			}
			if p.TimingFamily == "not-applicable" {
				if p.ClockDomain != "" {
					return errors.New("registration: non-temporal metric has clock domain")
				}
			} else {
				if !ledgerOneOf(p.TimingFamily, "host-wall", "host-monotonic", "device-event", "device-activity", "sensor-window", "counter-duration", "interval-union", "external") || p.ClockDomain == "" || !ledgerOneOf(p.Unit, "ns", "us", "ms", "s") {
					return errors.New("registration: invalid temporal metric scope")
				}
				if r.Clock.Body != nil {
					d, ok := domains[p.ClockDomain]
					if !ok || d.Unit != p.Unit || d.Family != p.TimingFamily {
						return errors.New("registration: property clock family/domain/unit mismatch")
					}
				}
			}
		}
		if !registrationNames(names) {
			return errors.New("registration: duplicate analysis property")
		}
	}
	if s := r.Stopping.Body; s != nil {
		if !ledgerOneOf(s.Rule, "fixed-samples", "bounded-sequential") || s.MaxSamples == 0 || s.MaxDurationNanos <= 0 || s.MaxDurationNanos > r.Limits.MaxDurationNanos || s.Rule == "bounded-sequential" && s.Method == nil {
			return errors.New("registration: invalid stopping rule")
		}
		if a := r.Analysis.Body; a != nil {
			if s.MaxSamples < a.MinSamples || s.Rule == "fixed-samples" && s.MaxSamples != a.Repetitions {
				return errors.New("registration: stopping sample scope mismatch")
			}
		}
	}
	if m := r.MissingData.Body; m != nil {
		if !ledgerOneOf(m.Handling, "withhold", "bounded") || m.Denominator != "declared-population" || m.Handling == "withhold" && (m.MaxMissing != 0 || m.MaxWithheld != 0) || m.Handling == "bounded" && m.Method == nil {
			return errors.New("registration: invalid missing data denominator")
		}
	}
	return nil
}

// artifacts traverses only concrete schema fields. No arbitrary method executes.
func (r Registration) artifacts() ([]RegistrationArtifact, error) {
	var out []RegistrationArtifact
	var walk func(reflect.Value)
	walk = func(v reflect.Value) {
		if v.Kind() == reflect.Pointer {
			if !v.IsNil() {
				walk(v.Elem())
			}
			return
		}
		if v.Type() == reflect.TypeFor[RegistrationArtifact]() {
			out = append(out, v.Interface().(RegistrationArtifact))
			return
		}
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if v.Type().Field(i).IsExported() {
					walk(v.Field(i))
				}
			}
		case reflect.Slice:
			for i := 0; i < v.Len(); i++ {
				walk(v.Index(i))
			}
		}
	}
	walk(reflect.ValueOf(r))
	if len(out) > 4096 {
		return nil, errors.New("registration: artifact count limit exceeded")
	}
	paths := map[string]string{}
	for _, a := range out {
		if err := registrationArtifacts([]RegistrationArtifact{a}); err != nil {
			return nil, err
		}
		if d, ok := paths[a.Path]; ok && d != a.SHA256 {
			return nil, errors.New("registration: conflicting rooted artifact identity")
		}
		paths[a.Path] = a.SHA256
	}
	pilotPaths, pilotDigests := map[string]bool{}, map[string]bool{}
	for _, a := range r.PilotArtifacts {
		if pilotPaths[a.Path] {
			return nil, errors.New("registration: duplicate pilot artifact")
		}
		pilotPaths[a.Path] = true
		pilotDigests[a.SHA256] = true
	}
	heldPaths := map[string]bool{}
	for _, a := range r.HeldOutArtifacts {
		if heldPaths[a.Path] || pilotPaths[a.Path] || pilotDigests[a.SHA256] {
			return nil, errors.New("registration: pilot and held-out artifacts overlap")
		}
		heldPaths[a.Path] = true
	}
	return out, nil
}

func (r Registration) pendingContracts() bool {
	for _, c := range []RegistrationContract{r.Workload.Contract, r.Oracle.Contract, r.Clock.Contract, r.Loss.Contract, r.Observer.Contract, r.Analysis.Contract, r.Stopping.Contract, r.MissingData.Contract} {
		if c.State != "declared" {
			return true
		}
	}
	resolved := func(f RegistrationFact) bool { return f.State == "known" || f.State == "not-applicable" }
	if w := r.Workload.Body; w != nil {
		for _, f := range []RegistrationFact{w.PowerSettings, w.Contention, w.HardwareLease} {
			if !resolved(f) {
				return true
			}
		}
	}
	if o := r.Observer.Body; o != nil {
		if !resolved(o.Access) {
			return true
		}
		for _, f := range o.OwnerConflicts {
			if !resolved(f) {
				return true
			}
		}
	}
	for _, l := range r.SourceLibraries {
		if !resolved(l.Fact) {
			return true
		}
	}
	for _, t := range r.TargetRows {
		if t.Applicability == "selected" {
			for _, f := range []RegistrationFact{t.Host, t.Device, t.Stack, t.Topology, t.Workload, t.Deployment, t.Observer} {
				if !resolved(f) {
					return true
				}
			}
		}
	}
	return false
}
