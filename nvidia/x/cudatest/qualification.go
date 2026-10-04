package cudatest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"math/big"
	"regexp"
	"strings"
	"time"
)

// QualificationFile binds a file to its actual SHA-256 bytes. Source paths are
// relative to the current checkout; other paths are relative to retained receipts.
type QualificationFile struct {
	Path   string `json:"path"`
	Digest string `json:"digest"`
}

// QualificationIdentity records observed CUDA compiler, driver, hardware and
// workload identities. Empty values are unknown, not a copy of expected values.
// WorkloadDigest and InputDigest hash the exact retained workload-description
// and input bytes; they are not substitutes for compiler/cache keys.
type QualificationIdentity struct {
	SourceRevision    string `json:"source_revision"`
	WorkloadDigest    string `json:"workload_digest"`
	InputDigest       string `json:"input_digest"`
	Toolchain         string `json:"toolchain"`
	Driver            string `json:"driver"`
	DeviceUUID        string `json:"device_uuid"`
	ComputeCapability string `json:"compute_capability"`
}

// QualificationPolicy is a finite, predeclared packet budget. Variation is the
// permitted relative difference between actual and reference PC sample counts;
// it never permits a zero collector. It is not a GPU-duration policy.
type QualificationPolicy struct {
	Attempts           uint64  `json:"attempts"`
	RunTimeoutMillis   uint64  `json:"run_timeout_millis"`
	TotalTimeoutMillis uint64  `json:"total_timeout_millis"`
	MinimumUserSamples uint64  `json:"minimum_user_samples"`
	SampleVariation    float64 `json:"sample_variation"`
}

// QualificationFacts contains domain-specific observed controls. Counts must
// come from the executed fixture; a pass line or constructed plan supplies none.
// Checked/Expected and GuardsChecked/GuardsExpected describe separate checks.
// User+NonUser+Dropped+Unnamed is the full Offered sampling population.
type QualificationFacts struct {
	PositiveRuns      uint64 `json:"positive_runs"`
	NegativeRuns      uint64 `json:"negative_runs"`
	Launches          uint64 `json:"launches"`
	CompletedLaunches uint64 `json:"completed_launches"`
	Synchronizations  uint64 `json:"synchronizations"`
	// CopyExecutions counts owner CopyWithLayouts invocations that reached
	// device execution, separately from kernel launches and individual driver
	// transfers within a plan. CompletedCopyExecutions counts those invocations
	// whose completion was checked before reading their destination.
	CopyExecutions          uint64 `json:"copy_executions"`
	CompletedCopyExecutions uint64 `json:"completed_copy_executions"`
	Expected                uint64 `json:"expected"`
	Checked                 uint64 `json:"checked"`
	GuardsExpected          uint64 `json:"guards_expected"`
	GuardsChecked           uint64 `json:"guards_checked"`
	ScalarRuns              uint64 `json:"scalar_runs"`
	VectorRuns              uint64 `json:"vector_runs"`
	CollapsedAliasRuns      uint64 `json:"collapsed_alias_runs"`
	AliasRefusals           uint64 `json:"alias_refusals"`
	ExplicitSharedCast      bool   `json:"explicit_shared_cast"`
	ABOutputsChecked        bool   `json:"ab_outputs_checked"`
	MatchedKernelInspection bool   `json:"matched_kernel_inspection"`
	// ConversionCountA and ConversionCountB count supported observed SASS
	// conversion mechanisms in the matched before and after kernels. PTX
	// mnemonic counts cannot stand in for these observations. Qualification
	// requires a positive before control and supported zero after count.
	ConversionCountA           uint64 `json:"conversion_count_a"`
	ConversionCountB           uint64 `json:"conversion_count_b"`
	RegistersA                 uint64 `json:"registers_a"`
	RegistersB                 uint64 `json:"registers_b"`
	RankGeometryMatched        bool   `json:"rank_geometry_matched"`
	PostReadOrdered            bool   `json:"post_read_ordered"`
	CooperativeOracleChecked   bool   `json:"cooperative_oracle_checked"`
	LocalWindowControlDetected bool   `json:"local_window_control_detected"`
	Offered                    uint64 `json:"offered"`
	User                       uint64 `json:"user"`
	NonUser                    uint64 `json:"non_user"`
	Dropped                    uint64 `json:"dropped"`
	Unnamed                    uint64 `json:"unnamed"`
	ReferenceUser              uint64 `json:"reference_user"`
	ExpectedActivity           uint64 `json:"expected_activity"`
	IsolationChecked           bool   `json:"isolation_checked"`
	DropControlOffered         uint64 `json:"drop_control_offered"`
	DropControlDropped         uint64 `json:"drop_control_dropped"`
	WorkloadGeometryMatched    bool   `json:"workload_geometry_matched"`
	PrintedToleranceChecked    bool   `json:"printed_tolerance_checked"`
}

// QualificationRun is a retained producer record. Log contains raw go test
// -json events, not a hand-written PASS summary. Policy binds predeclared JSON
// bytes. Source and artifact files are the actual files used by the fixture.
// Result is pass, fail, skip, unsupported, invalid-config or harness-error.
type QualificationRun struct {
	Version    int                   `json:"version"`
	Kind       string                `json:"kind"`
	Test       string                `json:"test"`
	Package    string                `json:"package"`
	Identity   QualificationIdentity `json:"identity"`
	Workload   QualificationFile     `json:"workload"`
	Inputs     QualificationFile     `json:"inputs"`
	Sources    []QualificationFile   `json:"sources"`
	Artifacts  []QualificationFile   `json:"artifacts"`
	Policy     QualificationFile     `json:"policy"`
	Log        QualificationFile     `json:"log"`
	Command    []string              `json:"command"`
	Result     string                `json:"result"`
	Reason     string                `json:"reason"`
	Native     bool                  `json:"native"`
	ExitCode   int                   `json:"exit_code"`
	WallMillis uint64                `json:"wall_millis"`
	Facts      QualificationFacts    `json:"facts"`
}

// QualificationRow records one required CUDA qualification slice. Kind is copy,
// address-space, dsmem, cupti-range or cupti-pc. Status is qualified, failed,
// unsupported, unmeasured, invalid-config or harness-error. Qualified has no
// reason; every other status requires one. Runs point to producer JSON records.
type QualificationRow struct {
	Kind      string                `json:"kind"`
	Status    string                `json:"status"`
	Reason    string                `json:"reason"`
	Identity  QualificationIdentity `json:"identity"`
	Workload  QualificationFile     `json:"workload"`
	Inputs    QualificationFile     `json:"inputs"`
	Policy    QualificationFile     `json:"policy"`
	Sources   []QualificationFile   `json:"sources"`
	Artifacts []QualificationFile   `json:"artifacts"`
	Runs      []QualificationFile   `json:"runs"`
}

// QualificationMatrix is a partial native-evidence ledger. All five required
// rows are explicit. Complete means each required row has admitted qualification
// evidence or a recorded native unsupported result; it is not package support.
type QualificationMatrix struct {
	Version        int                `json:"version"`
	SourceRevision string             `json:"source_revision"`
	Complete       bool               `json:"complete"`
	Rows           []QualificationRow `json:"rows"`
}

const qualificationJSONLimit = 1 << 20
const qualificationFileLimit = 16 << 20
const qualificationPacketLimit = 128 << 20

// ReadQualification reads one strict, bounded matrix. Validation against live
// file bytes and the independently supplied current revision is a separate step.
func ReadQualification(r io.Reader) (QualificationMatrix, error) {
	var m QualificationMatrix
	if r == nil {
		return m, errors.New("qualification: nil matrix reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, qualificationJSONLimit+1))
	if err != nil {
		return m, err
	}
	if len(b) > qualificationJSONLimit {
		return m, errors.New("qualification: matrix exceeds byte limit")
	}
	if err := qualificationDecode(b, &m); err != nil {
		return QualificationMatrix{}, err
	}
	return m, nil
}

// ValidateQualification checks current source/artifact bindings, actual retained
// producer records, raw test outcomes and each slice's controls. currentRevision
// must be independently obtained from the executing checkout. sources exposes
// that current checkout, not an archived source copy. receipts exposes retained
// artifacts and records. Both are trusted; this does not sandbox an arbitrary fs.FS.
// This is structural admission, not authentication of a dishonest producer or
// driver attestation. A caller must preserve the runner's actual files and logs.
func ValidateQualification(m QualificationMatrix, currentRevision string, sources, receipts fs.FS) error {
	if m.Version != 1 || !qualificationRevision(currentRevision) || m.SourceRevision != currentRevision {
		return errors.New("qualification: stale or invalid source revision")
	}
	if sources == nil || receipts == nil {
		return errors.New("qualification: nil receipt filesystem")
	}
	if len(m.Rows) != 5 {
		return errors.New("qualification: all five rows must be explicit")
	}
	reader := qualificationReader{files: receipts, sources: sources, seen: map[string]string{}}
	kinds := map[string]bool{}
	complete := true
	for _, row := range m.Rows {
		if !qualificationKind(row.Kind) || kinds[row.Kind] {
			return errors.New("qualification: unknown or duplicate row")
		}
		kinds[row.Kind] = true
		if err := reader.row(row, m.SourceRevision); err != nil {
			return fmt.Errorf("qualification %s: %w", row.Kind, err)
		}
		complete = complete && (row.Status == "qualified" || row.Status == "unsupported")
	}
	if m.Complete != complete {
		return errors.New("qualification: complete contradicts required row statuses")
	}
	return nil
}

type qualificationReader struct {
	files   fs.FS
	sources fs.FS
	seen    map[string]string
	bytes   int
}

func (r *qualificationReader) read(f QualificationFile) ([]byte, error) {
	return r.readFrom(f, r.files, "receipt:")
}
func (r *qualificationReader) readFrom(f QualificationFile, files fs.FS, prefix string) ([]byte, error) {
	if !fs.ValidPath(f.Path) || f.Path == "." || !outputDigest(f.Digest) {
		return nil, errors.New("invalid retained file binding")
	}
	if digest, ok := r.seen[prefix+f.Path]; ok && digest != f.Digest {
		return nil, errors.New("contradictory file binding")
	}
	file, err := files.Open(f.Path)
	if err != nil {
		return nil, fmt.Errorf("open retained file: %w", err)
	}
	b, err := io.ReadAll(io.LimitReader(file, qualificationFileLimit+1))
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		return nil, errors.Join(err, closeErr)
	}
	if len(b) > qualificationFileLimit {
		return nil, errors.New("retained file exceeds byte limit")
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != f.Digest {
		return nil, errors.New("stale retained source/artifact/receipt bytes")
	}
	if _, ok := r.seen[prefix+f.Path]; !ok {
		r.bytes += len(b)
		r.seen[prefix+f.Path] = f.Digest
	}
	if r.bytes > qualificationPacketLimit {
		return nil, errors.New("retained packet exceeds byte limit")
	}
	return b, nil
}
func (r *qualificationReader) bindings(files []QualificationFile, source, required bool) error {
	if len(files) > 64 {
		return errors.New("too many file bindings")
	}
	seen := map[string]bool{}
	for _, f := range files {
		if seen[f.Path] {
			return errors.New("duplicate file binding")
		}
		seen[f.Path] = true

		var data []byte
		var err error
		if source {
			data, err = r.readFrom(f, r.sources, "source:")
		} else {
			data, err = r.read(f)
		}
		if err != nil {
			return err
		}
		if required && len(data) == 0 {
			return errors.New("empty required source/artifact content")
		}
	}
	return nil
}

func (r *qualificationReader) row(row QualificationRow, revision string) error {
	switch row.Status {
	case "qualified":
		if row.Reason != "" {
			return errors.New("qualified row has a reason")
		}
	case "failed", "unsupported", "unmeasured", "invalid-config", "harness-error":
		if strings.TrimSpace(row.Reason) == "" {
			return errors.New("nonqualified row lacks reason")
		}
	default:
		return errors.New("unknown status")
	}
	if row.Identity.SourceRevision != "" && row.Identity.SourceRevision != revision {
		return errors.New("stale row identity")
	}
	if err := r.bindings(row.Sources, true, row.Status == "qualified"); err != nil {
		return err
	}
	if err := r.bindings(row.Artifacts, false, row.Status == "qualified"); err != nil {
		return err
	}
	if len(row.Runs) > 32 {
		return errors.New("too many attempts")
	}
	if row.Status == "qualified" || row.Status == "unsupported" || row.Status == "failed" {
		if err := qualificationIdentity(row.Identity, revision); err != nil {
			return err
		}
	} else {

		for _, value := range []string{row.Identity.Toolchain, row.Identity.Driver, row.Identity.DeviceUUID, row.Identity.ComputeCapability} {
			if value != "" && strings.TrimSpace(value) == "" {
				return errors.New("blank observed identity")
			}
		}
		for _, digest := range []string{row.Identity.WorkloadDigest, row.Identity.InputDigest} {
			if digest != "" && !outputDigest(digest) {
				return errors.New("invalid partial identity digest")
			}
		}
	}
	for _, binding := range []struct {
		file   QualificationFile
		digest string
	}{{row.Workload, row.Identity.WorkloadDigest}, {row.Inputs, row.Identity.InputDigest}} {
		if binding.digest == "" && binding.file.Path == "" {
			continue
		}
		if binding.file.Digest != binding.digest {
			return errors.New("workload/input identity differs from retained bytes")
		}
		data, err := r.read(binding.file)
		if err != nil {
			return err
		}
		if row.Status == "qualified" && len(data) == 0 {
			return errors.New("empty required workload/input content")
		}
	}
	var policy QualificationPolicy
	if row.Policy != (QualificationFile{}) {
		policyBytes, err := r.read(row.Policy)
		if err != nil {
			return err
		}
		if err := qualificationDecode(policyBytes, &policy); err != nil {
			return err
		}
		if policy.Attempts == 0 || policy.Attempts > 32 || policy.RunTimeoutMillis == 0 || policy.TotalTimeoutMillis < policy.RunTimeoutMillis || math.IsNaN(policy.SampleVariation) || math.IsInf(policy.SampleVariation, 0) || policy.SampleVariation < 0 || policy.SampleVariation >= 1 {
			return errors.New("invalid finite attempt policy")
		}
	}
	if row.Status == "unmeasured" && len(row.Runs) == 0 {
		return nil
	}
	if len(row.Runs) == 0 {
		return errors.New("status has no executed evidence")
	}
	if row.Policy == (QualificationFile{}) {
		return errors.New("executed evidence lacks attempt policy")
	}
	if uint64(len(row.Runs)) > policy.Attempts {
		return errors.New("attempt count exceeds policy")
	}
	if row.Status == "qualified" && (uint64(len(row.Runs)) != policy.Attempts || len(row.Sources) == 0 || len(row.Artifacts) == 0) {
		return errors.New("qualification missing required attempts/source/artifact")
	}
	seen := map[string]bool{}
	wall := uint64(0)
	observedStatus := false
	for _, ref := range row.Runs {
		if seen[ref.Path] {
			return errors.New("reused attempt receipt")
		}
		seen[ref.Path] = true
		b, err := r.read(ref)
		if err != nil {
			return err
		}
		var run QualificationRun
		if err := qualificationDecode(b, &run); err != nil {
			return err
		}
		if run.Version != 1 || run.Kind != row.Kind || run.Identity != row.Identity || run.Workload != row.Workload || run.Inputs != row.Inputs || run.Policy != row.Policy || !qualificationBindingsEqual(run.Sources, row.Sources) || !qualificationBindingsEqual(run.Artifacts, row.Artifacts) {
			return errors.New("producer source/artifact/identity differs from row")
		}
		if run.WallMillis > policy.RunTimeoutMillis || wall > policy.TotalTimeoutMillis-run.WallMillis {
			return errors.New("attempt exceeds declared packet budget")
		}
		wall += run.WallMillis
		if err := qualificationCommand(run, policy); err != nil {
			return err
		}
		log, err := r.read(run.Log)
		if err != nil {
			return err
		}
		terminal, err := qualificationTestOutcome(log, run.Package, run.Test)
		if err != nil {
			return err
		}
		switch run.Result {
		case "pass":
			if terminal != "pass" || run.ExitCode != 0 || run.Reason != "" {
				return errors.New("pass contradicts raw test outcome")
			}
		case "skip":
			if (terminal != "skip" && terminal != "partial") || strings.TrimSpace(run.Reason) == "" || run.ExitCode != 0 {
				return errors.New("skip contradicts raw outcome")
			}
		case "unsupported":
			if strings.TrimSpace(run.Reason) == "" || !run.Native || run.ExitCode != 0 || (terminal != "skip" && terminal != "pass") {
				return errors.New("unsupported lacks native probe evidence")
			}
		case "fail", "invalid-config", "harness-error":
			if (terminal != "fail" && !(run.Result == "harness-error" && terminal == "not-run" && !run.Native)) || run.ExitCode == 0 || strings.TrimSpace(run.Reason) == "" || run.Result == "fail" && !run.Native {
				return errors.New("failure contradicts raw outcome")
			}
		default:
			return errors.New("unknown attempt result")
		}
		if row.Status == "qualified" {
			if run.Result != "pass" || !run.Native {
				return errors.New("skip, failure or nonnative run cannot qualify")
			}
			if err := qualificationFacts(row.Kind, run.Facts, policy); err != nil {
				return err
			}
		}
		observedStatus = observedStatus || row.Status == "unsupported" && run.Result == "unsupported" || row.Status == "failed" && run.Result == "fail" || row.Status == "invalid-config" && run.Result == "invalid-config" || row.Status == "harness-error" && run.Result == "harness-error" || row.Status == "unmeasured" && run.Result != "pass"
	}
	if row.Status != "qualified" && row.Status != "unmeasured" && !observedStatus {
		return errors.New("row status lacks matching observed attempt")
	}
	return nil
}

func qualificationFacts(kind string, f QualificationFacts, p QualificationPolicy) error {
	if f.PositiveRuns == 0 || f.NegativeRuns == 0 || f.CompletedLaunches != f.Launches || f.Synchronizations == 0 {
		return errors.New("missing executed positive/negative/completion controls")
	}
	if kind != "copy" && f.Launches == 0 {
		return errors.New("missing executed kernel launches")
	}
	switch kind {
	case "copy":
		if f.CopyExecutions == 0 || f.CompletedCopyExecutions != f.CopyExecutions {
			return errors.New("copy lacks actual completed copy invocations")
		}
		if f.Expected == 0 || f.Checked != f.Expected || f.GuardsExpected < 2 || f.GuardsChecked != f.GuardsExpected || f.ScalarRuns == 0 || f.VectorRuns == 0 || f.CollapsedAliasRuns == 0 || f.AliasRefusals == 0 {
			return errors.New("copy lacks full output/guard/path/alias controls")
		}
	case "address-space":
		if f.Expected == 0 || f.Checked != f.Expected || !f.ExplicitSharedCast || !f.ABOutputsChecked || !f.MatchedKernelInspection || f.ConversionCountA == 0 || f.ConversionCountB != 0 || f.RegistersA == 0 || f.RegistersB == 0 {
			return errors.New("address-space lacks actual A/B output and compiled-kernel evidence")
		}
	case "dsmem":
		if f.Expected == 0 || f.Checked != f.Expected || !f.RankGeometryMatched || !f.PostReadOrdered || !f.CooperativeOracleChecked || !f.LocalWindowControlDetected {
			return errors.New("DSMEM lacks checked geometry/oracle/remote-window control")
		}
	case "cupti-range", "cupti-pc":
		total := uint64(0)
		for _, n := range []uint64{f.User, f.NonUser, f.Dropped, f.Unnamed} {
			if n > math.MaxUint64-total {
				return errors.New("sample population overflow")
			}
			total += n
		}
		if total == 0 || total != f.Offered || f.User == 0 || f.ExpectedActivity == 0 || !f.IsolationChecked || f.DropControlOffered == 0 || f.DropControlDropped == 0 || f.DropControlDropped > f.DropControlOffered || !f.WorkloadGeometryMatched || !f.PrintedToleranceChecked {
			return errors.New("CUPTI lacks activity/isolation/drop/geometry evidence or full denominator")
		}
		if kind == "cupti-pc" {
			if p.MinimumUserSamples == 0 || f.User < p.MinimumUserSamples || f.ReferenceUser == 0 || !qualificationSampleVariation(f.User, f.ReferenceUser, p.SampleVariation) {
				return errors.New("PC sampling violates declared nonzero variation policy")
			}
		}
	}
	return nil
}

func qualificationSampleVariation(user, reference uint64, variation float64) bool {
	if reference == 0 {
		return false
	}
	delta := user - reference
	if user < reference {
		delta = reference - user
	}
	ratio := new(big.Rat).SetFrac(new(big.Int).SetUint64(delta), new(big.Int).SetUint64(reference))
	bound := new(big.Rat).SetFloat64(variation)
	return bound != nil && ratio.Cmp(bound) <= 0
}

func qualificationIdentity(id QualificationIdentity, revision string) error {
	if id.SourceRevision != revision || !outputDigest(id.WorkloadDigest) || !outputDigest(id.InputDigest) || !qualificationKnown(id.Toolchain) || !qualificationKnown(id.Driver) || !qualificationKnown(id.DeviceUUID) || !regexp.MustCompile(`^[1-9][0-9]*\.[0-9]+$`).MatchString(id.ComputeCapability) {
		return errors.New("missing observed source/workload/tool/driver/hardware identity")
	}
	return nil
}
func qualificationKnown(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "unknown", "unread", "unmeasured", "n/a", "none":
		return false
	}
	return true
}
func qualificationRevision(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func qualificationKind(s string) bool {
	switch s {
	case "copy", "address-space", "dsmem", "cupti-range", "cupti-pc":
		return true
	}
	return false
}
func qualificationBindingsEqual(a, b []QualificationFile) bool {
	if len(a) != len(b) {
		return false
	}
	set := map[QualificationFile]bool{}
	for _, f := range a {
		if set[f] {
			return false
		}
		set[f] = true
	}
	for _, f := range b {
		if !set[f] {
			return false
		}
	}
	return true
}

func qualificationCommand(run QualificationRun, p QualificationPolicy) error {
	if len(run.Command) < 10 || run.Command[0] != "/home/tmc/ml-explore/bin/gpu-lock" || run.Command[1] != "env" || !regexp.MustCompile(`^Test[A-Za-z0-9_]+$`).MatchString(run.Test) || !strings.HasPrefix(run.Package, "github.com/tmc/lib/nvidia/x/") {
		return errors.New("missing exact native lock command/test")
	}
	i := 2
	env := map[string]bool{}
	for i < len(run.Command) && strings.Contains(run.Command[i], "=") {
		key, _, _ := strings.Cut(run.Command[i], "=")
		if env[key] || key == "GPU_LOCK_HELD" {
			return errors.New("duplicate environment or forged lock holder")
		}
		env[key] = true
		if key == "GOWORK" && run.Command[i] != "GOWORK=off" {
			return errors.New("work environment differs")
		}
		i++
	}
	if !env["GOWORK"] || i+2 >= len(run.Command) || run.Command[i] != "go" || run.Command[i+1] != "test" {
		return errors.New("native command is not direct go test")
	}
	args := run.Command[i+2:]
	flags := map[string]bool{}
	selector := ""
	timeout := time.Duration(0)
	packageArg := ""
	for j := 0; j < len(args); j++ {
		s := args[j]
		switch {
		case s == "-run":
			j++
			if j == len(args) || selector != "" {
				return errors.New("invalid repeated test selector")
			}
			selector = args[j]
		case strings.HasPrefix(s, "-run="):
			if selector != "" {
				return errors.New("repeated test selector")
			}
			selector = strings.TrimPrefix(s, "-run=")
		case strings.HasPrefix(s, "-timeout="):
			if timeout != 0 {
				return errors.New("repeated timeout")
			}
			var err error
			timeout, err = time.ParseDuration(strings.TrimPrefix(s, "-timeout="))
			if err != nil {
				return err
			}
		case s == "-tags=native" || s == "-count=1" || s == "-json" || s == "-v":
			if flags[s] {
				return errors.New("repeated execution flag")
			}
			flags[s] = true
		case strings.HasPrefix(s, "./x/"):
			if packageArg != "" {
				return errors.New("multiple packages in one attempt")
			}
			packageArg = s
		default:
			return errors.New("unsupported qualification command argument")
		}
	}
	if !flags["-tags=native"] || !flags["-count=1"] || !flags["-json"] || selector != "^"+run.Test+"$" || packageArg != "./x/"+strings.TrimPrefix(run.Package, "github.com/tmc/lib/nvidia/x/") || timeout <= 0 || uint64(timeout/time.Millisecond) > p.RunTimeoutMillis {
		return errors.New("native command does not match recorded test/package/budget")
	}
	return nil
}

func qualificationTestOutcome(data []byte, pkg, test string) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	started := false
	terminal := ""
	packageTerminal := ""
	childSkipped, childFailed := false, false
	for {

		var raw json.RawMessage
		if err := dec.Decode(&raw); err == io.EOF {
			break
		} else if err != nil {
			return "", fmt.Errorf("raw test JSON: %w", err)
		}
		if err := qualificationTestKeys(raw); err != nil {
			return "", err
		}
		var event struct {
			Action, Package, Test, Output string
			Elapsed                       float64
			Time, OutputType              string
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return "", fmt.Errorf("raw test JSON: %w", err)
		}
		// Go 1.27 test2json distinguishes framing and error output. These
		// annotations do not replace the run/pass/fail/skip action ledger.
		switch event.OutputType {
		case "":
		case "frame", "error", "error-continue":
			if event.Action != "output" {
				return "", errors.New("output type on non-output test event")
			}
		default:
			return "", errors.New("unknown raw test output type")
		}
		if event.Package != "" && event.Package != pkg {
			return "", errors.New("unexpected package in one recorded attempt")
		}
		if event.Package != pkg {
			continue
		}
		if event.Test == test {
			switch event.Action {
			case "run":
				if started {
					return "", errors.New("test rerun in one attempt")
				}
				started = true
			case "pass", "fail", "skip":
				if terminal != "" || !started {
					return "", errors.New("contradictory test terminal")
				}
				terminal = event.Action
			}
		}
		if event.Test != "" && event.Test != test && !strings.HasPrefix(event.Test, test+"/") {
			return "", errors.New("unexpected test in exact-selector attempt")
		}
		if strings.HasPrefix(event.Test, test+"/") {
			childSkipped = childSkipped || event.Action == "skip"
			childFailed = childFailed || event.Action == "fail"
		}
		if event.Test == "" && (event.Action == "pass" || event.Action == "fail" || event.Action == "skip") {
			if packageTerminal != "" {
				return "", errors.New("contradictory package terminals")
			}
			packageTerminal = event.Action
		}
	}
	if !started && terminal == "" && packageTerminal == "fail" {
		return "not-run", nil
	}
	if !started || terminal == "" || packageTerminal == "" || terminal == "pass" && packageTerminal != "pass" || terminal == "fail" && packageTerminal != "fail" || terminal == "skip" && packageTerminal != "pass" || childFailed && terminal != "fail" {
		return "", errors.New("missing complete raw named test outcome")
	}
	if childSkipped && terminal == "pass" {
		return "partial", nil
	}
	return terminal, nil
}

func qualificationTestKeys(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	start, err := dec.Token()
	if err != nil || start != json.Delim('{') {
		return errors.New("raw test event must be an object")
	}
	seen := map[string]bool{}
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok || seen[key] {
			return errors.New("duplicate raw test event key")
		}
		switch key {
		case "Time", "Action", "Package", "Test", "Elapsed", "Output", "OutputType", "FailedBuild", "ImportPath":
		default:
			return errors.New("unknown or case-aliased raw test event key")
		}
		seen[key] = true
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return err
		}
	}
	_, err = dec.Token()
	return err
}

func qualificationDecode(data []byte, out any) error {
	if len(data) > qualificationJSONLimit {
		return errors.New("qualification JSON exceeds byte limit")
	}
	if trimmed := bytes.TrimSpace(data); len(trimmed) == 0 || trimmed[0] != '{' {
		return errors.New("qualification JSON must be an object")
	}
	scan := json.NewDecoder(bytes.NewReader(data))
	if err := qualificationKeys(scan, 0); err != nil {
		return err
	}
	if _, err := scan.Token(); err != io.EOF {
		return errors.New("trailing qualification JSON")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return fmt.Errorf("qualification JSON: %w", err)
	}
	return nil
}
func qualificationKeys(dec *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("qualification JSON nesting limit")
	}
	token, err := dec.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	if delim == '{' {
		seen := map[string]bool{}
		for dec.More() {
			t, err := dec.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok || key == "" || seen[key] {
				return errors.New("duplicate or invalid qualification key")
			}
			for _, c := range key {
				if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
					return errors.New("noncanonical qualification key")
				}
			}
			seen[key] = true
			if err := qualificationKeys(dec, depth+1); err != nil {
				return err
			}
		}
	} else if delim == '[' {
		for dec.More() {
			if err := qualificationKeys(dec, depth+1); err != nil {
				return err
			}
		}
	} else {
		return errors.New("invalid qualification JSON delimiter")
	}
	_, err = dec.Token()
	return err
}
