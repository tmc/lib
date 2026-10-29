package advisecorpus

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// Artifact identifies retained bytes using the existing corpus reference format.
type Artifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Record preserves the fields of an existing, upstream-validated trajectory.
// Decode rejects duplicate JSON fields but does not replace its semantic validator.
type Record map[string]json.RawMessage

// Link is the existing corpus-advise-link/v1 sidecar. Sequence and mechanism
// bindings are read from DecisionReceipt, not inferred from a measured speedup.
type Link struct {
	Schema            string              `json:"schema"`
	RecordID          string              `json:"record_id"`
	SourcePair        map[string]string   `json:"source_pair"`
	Origin            string              `json:"origin"`
	Scope             string              `json:"scope"`
	Outcome           string              `json:"outcome"`
	Mechanism         json.RawMessage     `json:"mechanism"`
	InputArm          string              `json:"input_arm"`
	InputSourceSHA256 string              `json:"input_source_sha256"`
	ToolSourceSHA256  string              `json:"tool_source_sha256"`
	ToolSourceReceipt Artifact            `json:"tool_source_receipt"`
	ToolBinary        Artifact            `json:"tool_binary"`
	InputCapture      Artifact            `json:"input_capture"`
	AdviseReport      Artifact            `json:"advise_report"`
	ItemIndex         *int                `json:"item_index"`
	DecisionReceipt   Artifact            `json:"decision_receipt"`
	CaptureProofs     map[string]Artifact `json:"capture_proofs"`
	HardwareMetrics   []json.RawMessage   `json:"hardware_metrics,omitempty"`
}

// Result retains the existing corpus-advise-consumption/v1 report fields.
// Mechanism causality remains NOT_ESTABLISHED, even when a logged link is valid.
type Result map[string]any

// Decode reads concatenated JSON trajectory objects, including newline-delimited
// native records and explicitly static companions. No model or device work runs.
func Decode(r io.Reader) ([]Record, error) {
	dec := json.NewDecoder(r)
	var records []Record
	ids := make(map[string]bool)
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err == io.EOF {
			return records, nil
		} else if err != nil {
			return nil, fmt.Errorf("decode corpus: %w", err)
		}
		if _, err := canonical(raw); err != nil {
			return nil, fmt.Errorf("decode corpus: %w", err)
		}
		var record Record
		if err := json.Unmarshal(raw, &record); err != nil {
			return nil, fmt.Errorf("decode record: %w", err)
		}
		id, _, err := sources(record)
		if err != nil {
			return nil, err
		}
		if ids[id] {
			return nil, fmt.Errorf("duplicate record id %q", id)
		}
		ids[id] = true
		records = append(records, record)
	}
}

func digest(raw []byte) string { s := sha256.Sum256(raw); return hex.EncodeToString(s[:]) }
func validDigest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && s == strings.ToLower(s)
}
func artifact(root string, ref Artifact) ([]byte, error) {
	if ref.Path == "" || !validDigest(ref.SHA256) || ref.Bytes < 0 {
		return nil, fmt.Errorf("invalid artifact reference")
	}
	name := ref.Path
	if !filepath.IsAbs(name) {
		name = filepath.Join(root, name)
	}
	raw, err := os.ReadFile(name)
	if err != nil {
		return nil, fmt.Errorf("read artifact: %w", err)
	}
	if int64(len(raw)) != ref.Bytes || digest(raw) != ref.SHA256 {
		return nil, fmt.Errorf("artifact identity drift: %s", ref.Path)
	}
	return raw, nil
}
func field[T any](r Record, key string) (T, error) {
	var v T
	raw, ok := r[key]
	if !ok {
		return v, fmt.Errorf("missing record field %s", key)
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		return v, fmt.Errorf("record field %s: %w", key, err)
	}
	return v, nil
}
func sources(r Record) (string, map[string]Artifact, error) {
	id, err := field[string](r, "id")
	if err != nil || strings.TrimSpace(id) == "" {
		return "", nil, fmt.Errorf("missing record id")
	}
	s, err := field[map[string]json.RawMessage](r, "source")
	if err != nil {
		return "", nil, err
	}
	arms := make(map[string]Artifact)
	for _, arm := range []string{"baseline", "candidate"} {
		var ref Artifact
		if err := json.Unmarshal(s[arm], &ref); err != nil || !validDigest(ref.SHA256) || ref.Path == "" || ref.Bytes < 0 {
			return "", nil, fmt.Errorf("invalid %s source", arm)
		}
		arms[arm] = ref
	}
	return id, arms, nil
}
func sameJSON(a, b json.RawMessage) bool {
	x, e := canonical(a)
	y, f := canonical(b)
	return e == nil && f == nil && bytes.Equal(x, y)
}
func load(root string, ref Artifact, v any) error {
	b, err := artifact(root, ref)
	if err != nil {
		return err
	}
	if _, err = canonical(b); err != nil {
		return err
	}
	if err = json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("decode artifact: %w", err)
	}
	return nil
}

// Consume verifies an optional sidecar against exact artifacts, mechanism,
// source pair and decision chronology. Runtime links require already qualified
// same-run timing and complete per-arm capture proofs. Missing links never imply
// a missed optimization; static STOP records cannot receive runtime attribution.
func Consume(root string, r Record, link *Link) (Result, error) {
	raw, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	canonicalRecord, err := canonical(raw)
	if err != nil {
		return nil, err
	}
	id, src, err := sources(r)
	if err != nil {
		return nil, err
	}
	pair := map[string]string{}
	for arm, ref := range src {
		if _, err := artifact(root, ref); err != nil {
			return nil, err
		}
		pair[arm] = ref.SHA256
	}
	mechanism, err := field[json.RawMessage](r, "mechanism")
	if err != nil {
		return nil, err
	}
	var hypothesis string
	if json.Unmarshal(mechanism, &hypothesis) != nil {
		var item struct {
			Hypothesis string `json:"hypothesis"`
		}
		if json.Unmarshal(mechanism, &item) != nil {
			return nil, fmt.Errorf("invalid mechanism")
		}
		hypothesis = item.Hypothesis
	}
	if strings.TrimSpace(hypothesis) == "" {
		return nil, fmt.Errorf("missing mechanism hypothesis")
	}
	comparison, err := field[struct {
		Boundary string `json:"boundary"`
		SameRun  bool   `json:"same_run"`
		Paired   bool   `json:"paired_delta_qualified"`
	}](r, "comparison")
	if err != nil || comparison.Boundary == "" {
		return nil, fmt.Errorf("missing comparison boundary")
	}
	result := Result{"schema": "corpus-advise-consumption/v1", "record_id": id, "record_sha256": digest(canonicalRecord), "classification": "NOT_EVALUATED", "advice_item": nil, "hardware_metrics": []any{}, "mechanism_causality": "NOT_ESTABLISHED", "native_revalidation": "NOT_RUN", "blind_success_claim": false}
	for _, key := range []string{"problem", "trajectory", "origin", "context", "case", "compile", "claim_limit", "source", "mechanism", "observation", "comparison", "stages", "adoption"} {
		if v, ok := r[key]; ok {
			result[key] = v
		} else {
			result[key] = nil
		}
	}
	result["existing_advice"] = r["advise"]
	if link == nil {
		result["reason"] = "No exact-source advice/decision/capture link supplied; retained record is not advice attribution"
		return result, nil
	}
	if link.Schema != "corpus-advise-link/v1" || link.RecordID != id || !reflect.DeepEqual(link.SourcePair, pair) {
		return nil, fmt.Errorf("advice record/source pair mismatch")
	}
	if link.Origin != "NONBLIND_CALIBRATION" {
		return nil, fmt.Errorf("existing calibration cannot be called blind")
	}
	if link.Scope != "STATIC_ONLY" && link.Scope != "RUNTIME_DIAGNOSTIC" {
		return nil, fmt.Errorf("unknown advice evidence scope")
	}
	if link.Outcome != "LEAD" && link.Outcome != "MISS" && link.Outcome != "MISLEAD" {
		return nil, fmt.Errorf("unknown scoped advice outcome")
	}
	if !sameJSON(link.Mechanism, r["mechanism"]) {
		return nil, fmt.Errorf("mechanism changed during join")
	}
	if _, ok := pair[link.InputArm]; !ok || link.InputSourceSHA256 != pair[link.InputArm] {
		return nil, fmt.Errorf("advice input source mismatch")
	}
	if !validDigest(link.ToolSourceSHA256) || link.ToolSourceReceipt.SHA256 != link.ToolSourceSHA256 {
		return nil, fmt.Errorf("tool source identity missing")
	}
	for _, ref := range []Artifact{link.ToolSourceReceipt, link.ToolBinary, link.InputCapture} {
		if _, err := artifact(root, ref); err != nil {
			return nil, err
		}
	}
	var advice struct {
		Items json.RawMessage `json:"items"`
	}
	if err := load(root, link.AdviseReport, &advice); err != nil {
		return nil, err
	}
	var items []json.RawMessage
	if len(advice.Items) == 0 || advice.Items[0] != '[' || json.Unmarshal(advice.Items, &items) != nil {
		return nil, fmt.Errorf("not cudaprof advise report JSON")
	}
	if link.Outcome == "MISS" {
		if link.ItemIndex != nil {
			return nil, fmt.Errorf("miss must not invent a finding")
		}
	} else {
		if link.ItemIndex == nil || *link.ItemIndex < 0 || *link.ItemIndex >= len(items) {
			return nil, fmt.Errorf("advice finding index missing/outside report")
		}
		result["advice_item"] = items[*link.ItemIndex]
	}
	var decision struct {
		RecordID         string            `json:"record_id"`
		SourcePair       map[string]string `json:"source_pair"`
		ReportSHA        string            `json:"advice_report_sha256"`
		Mechanism        json.RawMessage   `json:"mechanism"`
		Outcome          string            `json:"outcome"`
		AdviceSequence   *int64            `json:"advice_sequence"`
		DecisionSequence *int64            `json:"decision_sequence"`
	}
	if err := load(root, link.DecisionReceipt, &decision); err != nil {
		return nil, err
	}
	if decision.RecordID != id || !reflect.DeepEqual(decision.SourcePair, pair) || decision.ReportSHA != link.AdviseReport.SHA256 || !sameJSON(decision.Mechanism, r["mechanism"]) || decision.Outcome != link.Outcome {
		return nil, fmt.Errorf("decision/source/advice/mechanism binding mismatch")
	}
	if decision.AdviceSequence == nil || decision.DecisionSequence == nil || *decision.AdviceSequence >= *decision.DecisionSequence {
		return nil, fmt.Errorf("advice chronology not retained")
	}
	result["link"] = link
	if link.Scope == "STATIC_ONLY" {
		if len(link.HardwareMetrics) > 0 {
			return nil, fmt.Errorf("static artifacts cannot provide hardware metrics")
		}
		result["classification"] = "STATIC_" + link.Outcome + "_LOGGED_NOT_RUNTIME_WIN"
		result["reason"] = "Exact scoped static advice/decision log; no timing share or hardware causal attribution"
		return result, nil
	}
	kind, _ := field[string](r, "kind")
	if kind == "STATIC_COMPILE_CLOSURE" || strings.HasPrefix(comparison.Boundary, "STATIC") || !comparison.SameRun || !comparison.Paired || !runtimeStages(r) {
		return nil, fmt.Errorf("runtime advice requires upstream-qualified same-run pair")
	}
	if len(link.CaptureProofs) != 2 {
		return nil, fmt.Errorf("complete per-arm capture proof missing")
	}
	contextRaw, _ := json.Marshal(map[string]json.RawMessage{"context": null(r["context"]), "case": null(r["case"]), "boundary": mustJSON(comparison.Boundary)})
	contextBytes, err := canonical(contextRaw)
	if err != nil {
		return nil, err
	}
	var metrics []any
	for _, arm := range []string{"baseline", "candidate"} {
		proof, ok := link.CaptureProofs[arm]
		if !ok {
			return nil, fmt.Errorf("missing %s capture proof", arm)
		}
		var p struct {
			Schema          string                       `json:"schema"`
			SourceSHA       string                       `json:"source_sha256"`
			ContextSHA      string                       `json:"context_sha256"`
			Success         bool                         `json:"capture_success"`
			Completeness    string                       `json:"run_completeness"`
			Dropped         *int64                       `json:"dropped_records"`
			Post            string                       `json:"postprofile_numerical_state"`
			Raw             Artifact                     `json:"raw_capture"`
			ValidatorSHA    string                       `json:"validator_source_sha256"`
			ValidatorSource Artifact                     `json:"validator_source"`
			Validator       Artifact                     `json:"validator_receipt"`
			Metrics         []map[string]json.RawMessage `json:"metrics"`
		}
		if err := load(root, proof, &p); err != nil {
			return nil, err
		}
		if p.Schema != "corpus-advise-capture-proof/v1" || p.SourceSHA != pair[arm] || p.ContextSHA != digest(contextBytes) || !p.Success || p.Completeness != "complete" || p.Dropped == nil || *p.Dropped != 0 || p.Post != "PASS" {
			return nil, fmt.Errorf("partial/stale/failed capture cannot attribute advice")
		}
		if !validDigest(p.ValidatorSHA) {
			return nil, fmt.Errorf("upstream validator binding missing")
		}
		if p.ValidatorSource.SHA256 != p.ValidatorSHA {
			return nil, fmt.Errorf("validator source digest mismatch")
		}
		for _, ref := range []Artifact{p.Raw, p.ValidatorSource} {
			if _, err := artifact(root, ref); err != nil {
				return nil, err
			}
		}
		if arm == link.InputArm && link.InputCapture != p.Raw {
			return nil, fmt.Errorf("advice input capture differs from selected arm")
		}
		var validation struct {
			Schema       string   `json:"schema"`
			Status       string   `json:"status"`
			ValidatorSHA string   `json:"validator_source_sha256"`
			SourceSHA    string   `json:"source_sha256"`
			ContextSHA   string   `json:"context_sha256"`
			Raw          Artifact `json:"raw_capture"`
			Completeness string   `json:"run_completeness"`
			Dropped      *int64   `json:"dropped_records"`
			Post         string   `json:"postprofile_numerical_state"`
		}
		if err := load(root, p.Validator, &validation); err != nil {
			return nil, err
		}
		if validation.Schema != "corpus-advise-validator-receipt/v1" || validation.Status != "PASS" || validation.ValidatorSHA != p.ValidatorSHA || validation.SourceSHA != p.SourceSHA || validation.ContextSHA != p.ContextSHA || validation.Raw != p.Raw || validation.Completeness != "complete" || validation.Dropped == nil || *validation.Dropped != 0 || validation.Post != "PASS" {
			return nil, fmt.Errorf("upstream validator receipt failed or mismatched")
		}
		for _, m := range p.Metrics {
			var name, unit, class string
			var value json.Number
			if json.Unmarshal(m["name"], &name) != nil || name == "" || json.Unmarshal(m["unit"], &unit) != nil || unit == "" || json.Unmarshal(m["quantity_class"], &class) != nil || class != "measured" || json.Unmarshal(m["value"], &value) != nil {
				return nil, fmt.Errorf("invalid/unmeasured hardware quantity")
			}
			if n, err := value.Float64(); err != nil || math.IsNaN(n) || math.IsInf(n, 0) || len(m["value"]) == 0 || m["value"][0] == '"' {
				return nil, fmt.Errorf("invalid hardware number")
			}
			row := map[string]any{"arm": arm}
			for k, v := range m {
				row[k] = v
			}
			metrics = append(metrics, row)
		}
	}
	if len(metrics) == 0 {
		return nil, fmt.Errorf("no measured metric evidence")
	}
	result["hardware_metrics"] = metrics
	result["classification"] = "RUNTIME_" + link.Outcome + "_LOGGED_NOT_CAUSAL_PROOF"
	result["reason"] = "Scoped upstream evidence/decision join only; runtime success does not prove proposed mechanism"
	return result, nil
}

// A qualified paired flag does not turn a static STOP or missing native stage
// into a measured runtime trajectory.
func runtimeStages(r Record) bool {
	observation, _ := field[string](r, "observation")
	if strings.Contains(observation, "STATIC_STOP") {
		return false
	}
	stages, err := field[map[string]json.RawMessage](r, "stages")
	if err != nil {
		return false
	}
	for _, name := range []string{"native_numerical", "unprofiled_timing"} {
		var status string
		if json.Unmarshal(stages[name], &status) != nil {
			var item struct {
				Status string `json:"status"`
			}
			if json.Unmarshal(stages[name], &item) != nil {
				return false
			}
			status = item.Status
		}
		if !strings.HasPrefix(status, "PASS") {
			return false
		}
	}
	var context struct {
		GPU string `json:"gpu"`
	}
	var cases map[string]json.RawMessage
	return json.Unmarshal(r["context"], &context) == nil && strings.TrimSpace(context.GPU) != "" && json.Unmarshal(r["case"], &cases) == nil && len(cases) > 0
}

func null(v json.RawMessage) json.RawMessage {
	if len(v) == 0 {
		return json.RawMessage("null")
	}
	return v
}
func mustJSON(s string) json.RawMessage { b, _ := json.Marshal(s); return b }

// Analyze verifies the retained source pair and capture bytes, then invokes the
// real advise.AnalyzeWithEvidence API. Evidence remains caller-supplied provenance,
// not hardware attestation or proof that raw-source and image digests coincide.
// This produces advice only; Consume separately gates any outcome attribution.
func Analyze(root string, r Record, arm string, capture Artifact, evidence cudaevidence.Source) (*advise.Report, error) {
	rawRecord, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if _, err := canonical(rawRecord); err != nil {
		return nil, err
	}
	_, src, err := sources(r)
	if err != nil {
		return nil, err
	}
	if _, ok := src[arm]; !ok {
		return nil, fmt.Errorf("unknown input arm")
	}
	for _, ref := range src {
		if _, err := artifact(root, ref); err != nil {
			return nil, err
		}
	}
	raw, err := artifact(root, capture)
	if err != nil {
		return nil, err
	}
	cap, err := cudacapture.Read(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("read advice capture: %w", err)
	}
	// Only named translation-unit source roles can establish this association.
	// A random attached file or a matching kernel symbol is insufficient. Python
	// wrappers whose embedded CUDA is not mapped require an upstream closure.
	named := []string{cap.Code.Source}
	named = append(named, cap.Code.Sources...)
	for _, m := range cap.Modules {
		named = append(named, m.Code.Source)
		named = append(named, m.Code.Sources...)
	}
	matched := false
	for _, name := range named {
		if name != "" && digest(cap.Files[name]) == src[arm].SHA256 {
			matched = true
		}
	}
	if !matched {
		return nil, fmt.Errorf("capture has no exact selected translation-unit source; wrapper mapping required")
	}
	return advise.AnalyzeWithEvidence(cap, evidence)
}

// Match returns indices whose finding, recommendation and kernel exactly match
// an explicitly enrolled mechanism criterion. It does not use substring guesses,
// headline ratios, or an item's suspected evidence as proof of a mechanism win.
func Match(report *advise.Report, finding, recommendation, kernel string) ([]int, error) {
	if report == nil || finding == "" || recommendation == "" || kernel == "" {
		return nil, fmt.Errorf("missing exact mechanism criterion")
	}
	var indices []int
	for i, item := range report.Items {
		if item.Finding == finding && item.ActionableRecommendation == recommendation && item.Location.Kernel == kernel {
			indices = append(indices, i)
		}
	}
	return indices, nil
}

// DecodeLink reads one existing corpus-advise-link/v1 sidecar, rejecting duplicate
// JSON fields and trailing data. Consume verifies its artifact and outcome gates.
func DecodeLink(r io.Reader) (*Link, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if _, err := canonical(raw); err != nil {
		return nil, fmt.Errorf("decode advice link: %w", err)
	}
	var link Link
	if err := json.Unmarshal(raw, &link); err != nil {
		return nil, fmt.Errorf("decode advice link: %w", err)
	}
	if link.Schema != "corpus-advise-link/v1" || strings.TrimSpace(link.RecordID) == "" {
		return nil, fmt.Errorf("invalid advice link identity")
	}
	return &link, nil
}
