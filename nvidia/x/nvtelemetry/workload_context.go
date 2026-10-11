package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// TrainingUpdate is an application-supplied optimizer outcome. Step is the
// optimizer step, not an experiment logging index. Attempt separates resumed
// runs. Unknown quality cannot qualify useful work.
type TrainingUpdate struct {
	Attempt       string `json:"attempt"`
	Step          uint64 `json:"step"`
	Rank          uint32 `json:"rank"`
	WorldSize     uint32 `json:"world_size"`
	Outcome       string `json:"outcome"` // updated, skipped, failed, unknown
	Quality       string `json:"quality"` // accepted, rejected, unknown
	QualityPolicy string `json:"quality_policy,omitempty"`
	Evidence      string `json:"evidence"`
}

// TrainingContext is a consumer interchange, not a vendor export or native
// measurement. Evidence identifies an independent application artifact; its
// contents and quality policy are not verified by this reader.
type TrainingContext struct {
	Version int              `json:"version"`
	RunID   string           `json:"run_id"`
	Updates []TrainingUpdate `json:"updates"`
}

// TrainingContextSummary counts complete single-rank or multi-rank updates.
// Unknown and unsuccessful work remain visible. No durations or GPU shares
// are inferred, and an accepted quality assertion is not a convergence proof.
type TrainingContextSummary struct {
	Accepted, Rejected, Skipped, Failed, Unknown, Incomplete uint64
}

// ReadTrainingContext reads version 1 application-supplied optimizer outcomes.
// It returns original bytes and their digest in the existing workload export.
// Duplicate JSON keys, unknown fields, repeated rank/step/attempt identities,
// inconsistent world sizes, and unsupported statuses are refused.
func ReadTrainingContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, TrainingContext, error) {
	if s.Provider != "application" {
		return WorkloadExport{}, TrainingContext{}, errors.New("training context requires application source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, TrainingContext{}, err
	}
	var c TrainingContext
	if err := decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("decode training context: %w", err)
	}
	if err := requireContextIntegers(e.Raw, "updates", "step", "rank", "world_size"); err != nil {
		return e, c, err
	}
	if err := validateTrainingContext(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

// SummarizeTrainingContext reports optimizer outcomes by attempt and global
// step. Every expected rank must be present before a step is classified.
// A rejected rank rejects the step even if other ranks report accepted quality.
func SummarizeTrainingContext(c TrainingContext, l WorkloadLimits) (TrainingContextSummary, error) {
	l, err := workloadLimits(l)
	if err != nil {
		return TrainingContextSummary{}, err
	}
	if err := validateTrainingContext(c, c.RunID, l); err != nil {
		return TrainingContextSummary{}, err
	}
	type key struct {
		attempt string
		step    uint64
	}
	groups := make(map[key][]TrainingUpdate)
	for _, u := range c.Updates {
		k := key{u.Attempt, u.Step}
		groups[k] = append(groups[k], u)
	}
	var out TrainingContextSummary
	for _, group := range groups {
		if len(group) != int(group[0].WorldSize) {
			out.Incomplete++
			continue
		}
		failed, skipped, unknown, rejected := false, false, false, false
		for _, u := range group {
			failed = failed || u.Outcome == "failed"
			skipped = skipped || u.Outcome == "skipped"
			unknown = unknown || u.Outcome == "unknown" || u.Quality == "unknown"
			rejected = rejected || u.Quality == "rejected"
		}
		switch {
		case failed:
			out.Failed++
		case skipped:
			out.Skipped++
		case rejected:
			out.Rejected++
		case unknown:
			out.Unknown++
		default:
			out.Accepted++
		}
	}
	return out, nil
}

func validateTrainingContext(c TrainingContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || c.RunID == "" || c.RunID != run || c.Updates == nil || len(c.Updates) > l.MaxMetrics || len(c.RunID) > l.MaxStringBytes {
		return errors.New("invalid training context version, run or update count")
	}
	type stepKey struct {
		attempt string
		step    uint64
	}
	type rankKey struct {
		stepKey
		rank uint32
	}
	seen := make(map[rankKey]bool)
	worlds := make(map[stepKey]uint32)
	policies := make(map[stepKey]string)
	for _, u := range c.Updates {
		for _, s := range []string{u.Attempt, u.Evidence, u.QualityPolicy} {
			if len(s) > l.MaxStringBytes {
				return errors.New("training context string limit exceeded")
			}
		}
		if u.Attempt == "" || u.Evidence == "" || u.WorldSize == 0 || u.WorldSize > uint32(l.MaxMetrics) || u.Rank >= u.WorldSize {
			return errors.New("invalid training update identity")
		}
		switch u.Outcome {
		case "updated", "skipped", "failed", "unknown":
		default:
			return errors.New("unsupported optimizer outcome")
		}
		switch u.Quality {
		case "accepted", "rejected":
			if u.QualityPolicy == "" {
				return errors.New("missing training quality policy")
			}
		case "unknown":
			if u.QualityPolicy != "" {
				return errors.New("unknown quality has policy assertion")
			}
		default:
			return errors.New("unsupported training quality")
		}
		k := stepKey{u.Attempt, u.Step}
		rk := rankKey{k, u.Rank}
		if seen[rk] || worlds[k] != 0 && worlds[k] != u.WorldSize {
			return errors.New("duplicate training rank or contradictory world size")
		}
		if u.QualityPolicy != "" {
			if prior := policies[k]; prior != "" && prior != u.QualityPolicy {
				return errors.New("contradictory training quality policies")
			}
			policies[k] = u.QualityPolicy
		}
		seen[rk] = true
		worlds[k] = u.WorldSize
	}
	return nil
}

// WriteTrainingContext writes the versioned application interchange after
// validating identities and bounds. It does not create application evidence.
func WriteTrainingContext(w io.Writer, c TrainingContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing training context writer")
	}
	if err := validateTrainingContext(c, c.RunID, l); err != nil {
		return err
	}
	raw, err := marshalContext(c.RunID, "updates", c.Updates, l)
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("training context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

// ServingAttempt retains a caller-supplied request outcome. RequestID alone is
// not unique: attempts, worker and restart epoch are required. Token counts are
// present only when the application reports them. No GPU attribution is made.
type ServingAttempt struct {
	RequestID    string  `json:"request_id"`
	Attempt      string  `json:"attempt"`
	Worker       string  `json:"worker"`
	WorkerEpoch  string  `json:"worker_epoch"`
	Role         string  `json:"role"`    // prefill, decode, combined, unknown
	Outcome      string  `json:"outcome"` // accepted, failed, cancelled, partial, unknown
	OutputTokens *uint64 `json:"output_tokens,omitempty"`
	OutputPolicy string  `json:"output_policy,omitempty"`
	Evidence     string  `json:"evidence"`
}

// ServingContext is a versioned application interchange. It is not an SGLang
// or Dynamo exporter mapping, and cannot expand aggregate scrapes into requests.
type ServingContext struct {
	Version  int              `json:"version"`
	RunID    string           `json:"run_id"`
	Attempts []ServingAttempt `json:"attempts"`
}

// ServingContextSummary keeps accepted attempts and unsuccessful work separate.
// Accepted attempts are not necessarily distinct successful requests; retries
// remain separate. Unknown token counts never become measured zero.
type ServingContextSummary struct {
	Accepted, Failed, Cancelled, Partial, Unknown    uint64
	KnownAcceptedTokenAttempts, AcceptedOutputTokens uint64
}

// ReadServingContext reads version 1 caller-supplied outcomes with bounded
// original bytes and digest. Supplied identity is not a verified kernel join.
func ReadServingContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, ServingContext, error) {
	if s.Provider != "application" {
		return WorkloadExport{}, ServingContext{}, errors.New("serving context requires application source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, ServingContext{}, err
	}
	var c ServingContext
	if err := decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("decode serving context: %w", err)
	}
	if err := validateServingContext(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

// WriteServingContext validates and writes the application interchange without
// generating any worker, request or transfer identifiers.
func WriteServingContext(w io.Writer, c ServingContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing serving context writer")
	}
	if err := validateServingContext(c, c.RunID, l); err != nil {
		return err
	}
	raw, err := marshalContext(c.RunID, "attempts", c.Attempts, l)
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("serving context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

// SummarizeServingContext counts supplied outcomes without assigning device
// time, energy, cache memory, or shared batch work to individual requests.
func SummarizeServingContext(c ServingContext, l WorkloadLimits) (ServingContextSummary, error) {
	l, err := workloadLimits(l)
	if err != nil {
		return ServingContextSummary{}, err
	}
	if err := validateServingContext(c, c.RunID, l); err != nil {
		return ServingContextSummary{}, err
	}
	var out ServingContextSummary
	for _, a := range c.Attempts {
		switch a.Outcome {
		case "accepted":
			out.Accepted++
			if a.OutputTokens != nil {
				if ^uint64(0)-out.AcceptedOutputTokens < *a.OutputTokens {
					return ServingContextSummary{}, errors.New("accepted token count overflow")
				}
				out.AcceptedOutputTokens += *a.OutputTokens
				out.KnownAcceptedTokenAttempts++
			}
		case "failed":
			out.Failed++
		case "cancelled":
			out.Cancelled++
		case "partial":
			out.Partial++
		case "unknown":
			out.Unknown++
		}
	}
	return out, nil
}

func validateServingContext(c ServingContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || c.RunID == "" || c.RunID != run || len(c.RunID) > l.MaxStringBytes || c.Attempts == nil || len(c.Attempts) > l.MaxMetrics {
		return errors.New("invalid serving context version, run or attempt count")
	}
	type key struct{ request, attempt, worker, epoch string }
	seen := make(map[key]bool)
	for _, a := range c.Attempts {
		for _, s := range []string{a.RequestID, a.Attempt, a.Worker, a.WorkerEpoch, a.OutputPolicy, a.Evidence} {
			if len(s) > l.MaxStringBytes {
				return errors.New("serving context string limit exceeded")
			}
		}
		if a.RequestID == "" || a.Attempt == "" || a.Worker == "" || a.WorkerEpoch == "" || a.Evidence == "" {
			return errors.New("missing serving attempt identity or evidence")
		}
		switch a.Role {
		case "prefill", "decode", "combined", "unknown":
		default:
			return errors.New("unsupported serving worker role")
		}
		switch a.Outcome {
		case "accepted":
			if a.OutputPolicy == "" {
				return errors.New("missing accepted output policy")
			}
		case "failed", "cancelled", "partial", "unknown":
		default:
			return errors.New("unsupported serving outcome")
		}
		k := key{a.RequestID, a.Attempt, a.Worker, a.WorkerEpoch}
		if seen[k] {
			return errors.New("duplicate serving attempt")
		}
		seen[k] = true
	}
	return nil
}

// marshalContext checks the encoded budget one bounded record at a time. A
// rejected export never builds an unbounded whole-document JSON allocation.
func marshalContext[T any](run, field string, records []T, l WorkloadLimits) ([]byte, error) {
	name, _ := json.Marshal(run)
	raw := append([]byte(`{"version":1,"run_id":`), name...)
	raw = append(raw, []byte(`,"`+field+`":[`)...)
	if int64(len(raw))+2 > l.MaxBytes {
		return nil, errors.New("context byte limit exceeded")
	}
	for i, record := range records {
		encoded, err := json.Marshal(record)
		if err != nil {
			return nil, err
		}
		extra := len(encoded)
		if i > 0 {
			extra++
		}
		if int64(len(raw))+int64(extra)+2 > l.MaxBytes {
			return nil, errors.New("context byte limit exceeded")
		}
		if i > 0 {
			raw = append(raw, ',')
		}
		raw = append(raw, encoded...)
	}
	return append(raw, ']', '}'), nil
}

// Zero is a valid optimizer step or rank. An absent or null field must not
// silently turn into that explicitly supplied zero during JSON decoding.
func requireContextIntegers(raw []byte, field string, names ...string) error {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return err
	}
	var records []map[string]json.RawMessage
	if err := json.Unmarshal(body[field], &records); err != nil {
		return err
	}
	for _, record := range records {
		for _, name := range names {
			value, ok := record[name]
			if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return errors.New("missing training integer field")
			}
		}
	}
	return nil
}
