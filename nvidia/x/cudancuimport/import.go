package cudancuimport

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

const (
	// SchemaVersion is this project's envelope version.
	SchemaVersion = 1
	// ToolVersion is the supported exporter release, not a collection version.
	ToolVersion = "2026.3.0.0"
	// Format selects the release-pinned NVIDIA details CSV header profile.
	Format = "ncu-details-csv-2026.3.0"
	// MaxBytes bounds encoded envelopes and total candidate image bytes.
	MaxBytes = 32 << 20
	// MaxRows bounds CSV records and explicit launch bindings.
	MaxRows = 65536
)

// LaunchKey is the CSV launch identifier, scoped by process, context and stream.
// Its fields are decimal strings retained exactly from the export.
type LaunchKey struct {
	ID      string `json:"id"`
	Process string `json:"process"`
	Context string `json:"context"`
	Stream  string `json:"stream"`
}

// Binding is an external producer's explicit launch-to-artifact receipt.
// It is not vendor CSV metadata and is never inferred from a capture.
type Binding struct {
	Launch   LaunchKey             `json:"launch"`
	Identity cudaevidence.Identity `json:"identity"`
}

// Envelope versions the project export contract. ReportDigest identifies the
// original opaque report, as supplied by its producer; CSVDigest is checked
// against CSV. SourcePath is provenance only and is never opened.
type Envelope struct {
	SchemaVersion int       `json:"schema_version"`
	ToolVersion   string    `json:"tool_version"`
	Format        string    `json:"format"`
	SourcePath    string    `json:"source_path"`
	ReportDigest  string    `json:"report_digest"`
	CSVDigest     string    `json:"csv_digest"`
	CSV           string    `json:"csv"`
	Bindings      []Binding `json:"bindings"`
}

// Candidate supplies actual artifact and compiler/workload producer receipts
// alongside the capture to which a launch may be joined. Capture and Artifact
// must not be mutated during Import. Missing receipts never acquire defaults.
type Candidate struct {
	Capture     *cudacapture.Capture
	Artifact    *cudakernel.Artifact
	Workload    cudacache.WorkloadIdentity
	CompilerKey string
}

// Metric retains an exported scalar and its classified interpretation.
// Unsigned, when present, is exact even when it cannot be represented by the
// evidence model's float64 number. Such an interpretation remains Unread.
type Metric struct {
	Name        string             `json:"name"`
	Unit        string             `json:"unit"`
	RawValue    string             `json:"raw_value"`
	Unsigned    *uint64            `json:"unsigned,omitempty"`
	Denominator string             `json:"denominator"`
	Value       cudaevidence.Value `json:"value"`
}

// Row retains every CSV field, including unsupported rule diagnostics.
// ModuleID and Generation are set only after a unique verified join.
type Row struct {
	Launch     LaunchKey `json:"launch"`
	Entry      string    `json:"entry"`
	Block      [3]uint64 `json:"block"`
	Grid       [3]uint64 `json:"grid"`
	Target     string    `json:"target"`
	Section    string    `json:"section"`
	Raw        []string  `json:"raw"`
	Metric     *Metric   `json:"metric,omitempty"`
	ModuleID   string    `json:"module_id,omitempty"`
	Generation uint64    `json:"generation,omitempty"`
	JoinReason string    `json:"join_reason,omitempty"`
}

// Report preserves the admitted envelope and all exported rows.
type Report struct {
	Envelope Envelope `json:"envelope"`
	Rows     []Row    `json:"rows"`
}

var detailsHeader = []string{"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Rule Name", "Rule Type", "Rule Description", "Estimated Speedup Type", "Estimated Speedup"}

// Read reads one strict envelope with a positive byte budget no larger than
// MaxBytes. The original encoded input, including whitespace, counts against
// the budget. Duplicate, aliased, unknown, and trailing JSON fields are refused.
func Read(r io.Reader, maxBytes int) (Envelope, error) {
	var e Envelope
	if maxBytes <= 0 || maxBytes > MaxBytes {
		return e, fmt.Errorf("ncu import: invalid byte budget")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(maxBytes)+1))
	if err != nil {
		return e, fmt.Errorf("ncu import: read: %w", err)
	}
	if len(b) > maxBytes {
		return e, fmt.Errorf("ncu import: byte budget exceeded")
	}
	if err := strictJSON(b, &e); err != nil {
		return e, fmt.Errorf("ncu import: envelope: %w", err)
	}
	if err := validateEnvelope(e); err != nil {
		return Envelope{}, err
	}
	return e, nil
}

// Import validates a supported envelope and classifies its rows. Unsupported
// values and unsuccessful joins remain present with explicit Unread reasons.
// Candidate order never resolves an ambiguous join.
func Import(e Envelope, candidates []Candidate) (*Report, error) {
	if err := validateEnvelope(e); err != nil {
		return nil, err
	}
	if len(candidates) > 64 {
		return nil, fmt.Errorf("ncu import: too many join candidates")
	}
	imageBytes := 0
	retainedBytes := 0
	for _, c := range candidates {
		if c.Artifact != nil {
			if len(c.Artifact.Raw) > MaxBytes-imageBytes {
				return nil, fmt.Errorf("ncu import: candidate image byte budget exceeded")
			}
			imageBytes += len(c.Artifact.Raw)
			if len(c.Artifact.Schema.Params) > 4096 || len(c.Artifact.CompilerOpts) > 128 || len(c.Artifact.Dependencies) > 128 {
				return nil, fmt.Errorf("ncu import: candidate descriptor budget exceeded")
			}
		}
		if len(c.Workload.ShapeDims) > 1024 || len(c.Workload.ProgramID) > 4096 || len(c.Workload.NumericalMode) > 256 || len(c.Workload.PrepackID) > 4096 {
			return nil, fmt.Errorf("ncu import: workload receipt budget exceeded")
		}
		if c.Capture != nil {
			if len(c.Capture.Files) > 64 || len(c.Capture.Modules) > 64 {
				return nil, fmt.Errorf("ncu import: capture inventory budget exceeded")
			}
			for _, b := range c.Capture.Files {
				if len(b) > MaxBytes-retainedBytes {
					return nil, fmt.Errorf("ncu import: retained code byte budget exceeded")
				}
				retainedBytes += len(b)
			}
		}
	}
	bindings := make(map[LaunchKey]cudaevidence.Identity, len(e.Bindings))
	bindingBytes := 0
	for _, b := range e.Bindings {
		for _, s := range []string{b.Identity.ArtifactDigest, b.Identity.Entry, b.Identity.Target, b.Identity.CompilerKey, b.Identity.WorkloadKey} {
			if len(s) > 4096 {
				return nil, fmt.Errorf("ncu import: identity field budget exceeded")
			}
			if len(s) > MaxBytes-bindingBytes {
				return nil, fmt.Errorf("ncu import: binding byte budget exceeded")
			}
			bindingBytes += len(s)
		}
		if _, ok := bindings[b.Launch]; ok {
			return nil, fmt.Errorf("ncu import: duplicate launch binding")
		}
		if err := b.Identity.Validate(); err != nil {
			return nil, fmt.Errorf("ncu import: launch identity: %w", err)
		}
		if err := validLaunch(b.Launch); err != nil {
			return nil, err
		}
		bindings[b.Launch] = b.Identity
	}
	// Verify each candidate's receipt once. CSV rows may repeat a launch and
	// must not multiply image hashing or module decoding work.
	compatible := make(map[cudaevidence.Identity][]Candidate)
	for _, c := range candidates {
		actual, err := cudaprof.BindEvidence(c.Artifact, c.Workload, c.CompilerKey)
		if err != nil || c.Capture == nil || c.Artifact.Kind != cudakernel.KindCUBIN {
			continue
		}
		info, err := cudakernel.ParseCUBIN(c.Artifact.Raw)
		if err != nil || info.Arch == "" || info.Arch != actual.Target {
			continue
		}
		if observed := c.Capture.Identity; observed != nil && ((observed.ComputeCap != "" && observed.ComputeCap != actual.Target) || (observed.WorkloadHash != "" && observed.WorkloadHash != actual.WorkloadKey)) {
			continue
		}
		compatible[actual] = append(compatible[actual], c)
	}
	type joined struct {
		module     string
		generation uint64
		reason     string
	}
	joins := make(map[cudaevidence.Identity]joined, len(compatible))
	for id, matching := range compatible {
		m, g, r := join(id, matching)
		joins[id] = joined{m, g, r}
	}
	reader := csv.NewReader(strings.NewReader(e.CSV))
	reader.FieldsPerRecord = -1
	header, err := reader.Read()
	if err != nil || !reflect.DeepEqual(header, detailsHeader) {
		return nil, fmt.Errorf("ncu import: unsupported CSV header")
	}
	e.Bindings = append([]Binding(nil), e.Bindings...)
	report := &Report{Envelope: e}
	seen := make(map[LaunchKey]Row)
	used := make(map[LaunchKey]bool)
	for {
		fields, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("ncu import: CSV: %w", err)
		}
		if len(report.Rows) >= MaxRows {
			return nil, fmt.Errorf("ncu import: row budget exceeded")
		}
		// NVIDIA 2026.3 details metric rows omit the last four empty rule
		// fields. No other truncated row profile is supported.
		if len(fields) != 16 && len(fields) != 20 {
			return nil, fmt.Errorf("ncu import: unsupported row width")
		}
		if len(fields) == 16 && (fields[12] == "" || fields[15] != "") {
			return nil, fmt.Errorf("ncu import: truncated rule row")
		}
		for _, s := range fields {
			if len(s) > 65536 {
				return nil, fmt.Errorf("ncu import: field budget exceeded")
			}
		}
		row := Row{Launch: LaunchKey{fields[0], fields[1], fields[5], fields[6]}, Entry: fields[4], Section: fields[11], Raw: fields}
		if err := validLaunch(row.Launch); err != nil {
			return nil, err
		}
		if row.Entry == "" {
			return nil, fmt.Errorf("ncu import: missing entry")
		}
		row.Block, err = geometry(fields[7])
		if err != nil {
			return nil, err
		}
		row.Grid, err = geometry(fields[8])
		if err != nil {
			return nil, err
		}
		row.Target, err = computeTarget(fields[10])
		if err != nil {
			return nil, err
		}
		if prior, ok := seen[row.Launch]; ok && (prior.Entry != row.Entry || prior.Block != row.Block || prior.Grid != row.Grid || prior.Target != row.Target || !reflect.DeepEqual(prior.Raw[:11], fields[:11])) {
			return nil, fmt.Errorf("ncu import: inconsistent launch metadata")
		}
		seen[row.Launch] = row
		id, hasBinding := bindings[row.Launch]
		used[row.Launch] = true
		row.JoinReason = "external launch identity unavailable"
		if hasBinding {
			if id.Entry != row.Entry || id.Target != row.Target {
				row.JoinReason = "external entry or target differs from CSV"
			} else {
				if j, ok := joins[id]; ok {
					row.ModuleID, row.Generation, row.JoinReason = j.module, j.generation, j.reason
				} else {
					row.JoinReason = "no compatible retained image, entry and workload receipt"
				}
			}
		}
		if fields[12] != "" {
			m, err := metric(fields[12], fields[13], fields[14], row.JoinReason, id, e, len(report.Rows))
			if err != nil {
				return nil, err
			}
			row.Metric = &m
		}
		report.Rows = append(report.Rows, row)
	}
	if len(report.Rows) == 0 {
		return nil, fmt.Errorf("ncu import: empty CSV")
	}
	for key := range bindings {
		if !used[key] {
			return nil, fmt.Errorf("ncu import: binding has no exported launch")
		}
	}
	return report, nil
}

func validateEnvelope(e Envelope) error {
	if e.SchemaVersion != SchemaVersion {
		return fmt.Errorf("ncu import: unknown schema version")
	}
	if e.ToolVersion != ToolVersion || e.Format != Format {
		return fmt.Errorf("ncu import: unsupported exporter or format")
	}
	if strings.TrimSpace(e.SourcePath) == "" || len(e.SourcePath) > 4096 || len(e.CSV) > MaxBytes || len(e.Bindings) > MaxRows {
		return fmt.Errorf("ncu import: missing or oversized provenance")
	}
	if err := digest(e.ReportDigest); err != nil {
		return fmt.Errorf("ncu import: report digest: %w", err)
	}
	if err := digest(e.CSVDigest); err != nil {
		return fmt.Errorf("ncu import: CSV digest: %w", err)
	}
	if fmt.Sprintf("%x", sha256.Sum256([]byte(e.CSV))) != e.CSVDigest {
		return fmt.Errorf("ncu import: CSV digest mismatch")
	}
	return nil
}
func digest(s string) error {
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 32 || s != strings.ToLower(s) {
		return fmt.Errorf("invalid sha256")
	}
	return nil
}
func validLaunch(k LaunchKey) error {
	for _, s := range []string{k.ID, k.Process, k.Context, k.Stream} {
		if len(s) == 0 || len(s) > 20 {
			return fmt.Errorf("ncu import: invalid launch key")
		}
		if _, err := strconv.ParseUint(s, 10, 64); err != nil {
			return fmt.Errorf("ncu import: invalid launch key")
		}
	}
	return nil
}
func geometry(s string) ([3]uint64, error) {
	var out [3]uint64
	if !strings.HasPrefix(s, "(") || !strings.HasSuffix(s, ")") {
		return out, fmt.Errorf("ncu import: invalid geometry")
	}
	parts := strings.Split(s[1:len(s)-1], ",")
	if len(parts) != 3 {
		return out, fmt.Errorf("ncu import: invalid geometry")
	}
	product := uint64(1)
	for i, p := range parts {
		n, err := strconv.ParseUint(strings.TrimSpace(p), 10, 32)
		if err != nil || n == 0 || product > math.MaxUint64/n {
			return out, fmt.Errorf("ncu import: invalid geometry")
		}
		out[i] = n
		product *= n
	}
	return out, nil
}
func computeTarget(s string) (string, error) {
	p := strings.Split(s, ".")
	if len(p) != 2 || len(p[1]) != 1 {
		return "", fmt.Errorf("ncu import: unsupported compute capability")
	}
	a, e := strconv.ParseUint(p[0], 10, 8)
	b, f := strconv.ParseUint(p[1], 10, 8)
	if e != nil || f != nil || a == 0 || b > 9 {
		return "", fmt.Errorf("ncu import: invalid compute capability")
	}
	return fmt.Sprintf("sm_%d", a*10+b), nil
}

func join(id cudaevidence.Identity, candidates []Candidate) (string, uint64, string) {
	var module string
	var generation uint64
	matches := 0
	for _, candidate := range candidates {
		for _, m := range candidate.Capture.Modules {
			if m.ID == "" || m.Generation == 0 || !bytes.Equal(candidate.Capture.Files[m.Code.CUBIN], candidate.Artifact.Raw) {
				continue
			}
			if _, err := cudacapture.VerifyModule(candidate.Capture, m.ID, id.ArtifactDigest, []string{id.Entry}); err != nil {
				continue
			}
			matches++
			if matches > 1 {
				return "", 0, "ambiguous retained module generation"
			}
			module = m.ID
			generation = m.Generation
		}
	}
	if matches == 0 {
		return "", 0, "no compatible retained image, entry and workload receipt"
	}
	if matches != 1 {
		return "", 0, "ambiguous retained module generation"
	}
	return module, generation, ""
}

// This intentionally small table admits direct scalar observations only.
// Rates, percentages, averages and derived metrics need additional denominator
// and aggregation evidence; an attractive numeric spelling does not supply it.
var scalarUnits = map[string]string{
	"gpu__time_duration.sum":               "ns",
	"gpc__cycles_elapsed.max":              "cycle",
	"smsp__inst_executed.sum":              "inst",
	"smsp__inst_issued.sum":                "inst",
	"smsp__inst_executed_op_branch.sum":    "inst",
	"launch__registers_per_thread":         "register/thread",
	"launch__shared_mem_config_size":       "byte",
	"launch__shared_mem_per_block_driver":  "byte/block",
	"launch__shared_mem_per_block_dynamic": "byte/block",
	"launch__shared_mem_per_block_static":  "byte/block",
	"launch__thread_count":                 "thread",
}

func metric(name, unit, raw, joinReason string, id cudaevidence.Identity, e Envelope, index int) (Metric, error) {
	m := Metric{Name: name, Unit: unit, RawValue: raw, Denominator: "none"}
	if _, denominator, ok := strings.Cut(unit, "/"); ok {
		m.Denominator = denominator
	}
	reason := joinReason
	want, supported := scalarUnits[name]
	if !supported {
		reason = "unsupported metric or denominator"
		m.Denominator = "unsupported"
	} else if unit != want {
		reason = "metric unit differs from supported profile"
	}
	n, err := strconv.ParseUint(raw, 10, 64)
	if err == nil {
		m.Unsigned = &n
		if n > 1<<53 {
			reason = "integer exceeds exact evidence number range"
		}
	} else {
		reason = "noninteger, nonfinite or unsupported numeric value"
	}
	r := cudaevidence.Record{Unit: unit, Basis: "external:ncu", Source: cudaevidence.Source{Identity: id, Producer: "external:ncu", Locator: fmt.Sprintf("sha256:%s/csv:%s/row:%d", e.ReportDigest, e.CSVDigest, index+2)}}
	if reason != "" {
		r.Class = cudaevidence.Unread
		r.Reason = reason
	} else {
		number := float64(n)
		r.Class = cudaevidence.Measured
		r.Number = &number
	}
	v, err := cudaevidence.New(r)
	if err != nil {
		return Metric{}, fmt.Errorf("ncu import: evidence: %w", err)
	}
	m.Value = v
	return m, nil
}

func strictJSON(data []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	if err := uniqueKeys(d, 0); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func uniqueKeys(d *json.Decoder, depth int) error {
	if depth > 64 {
		return fmt.Errorf("JSON nesting budget exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			token, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := token.(string)
			if !ok || !asciiJSONKey(key) || seen[key] {
				return fmt.Errorf("duplicate or aliased JSON key")
			}
			seen[key] = true
			if err := uniqueKeys(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := uniqueKeys(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}

func asciiJSONKey(key string) bool {
	if key == "" {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c != '_' && !(c >= 'a' && c <= 'z') && !(c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}
