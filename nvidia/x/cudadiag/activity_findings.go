package cudadiag

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// ActivityAnalysis is a versioned annotation-to-activity finding result. Joins
// retain original record order and IDs. Findings are sorted independently of
// record order. An exact identity link does not establish a clock mapping or cause.
type ActivityAnalysis struct {
	Version  int                         `json:"version"`
	Joins    []cudaworkload.ActivityJoin `json:"joins"`
	Findings []tracecontract.Finding     `json:"findings"`
}

// AnalyzeActivities joins a bounded frozen producer batch and admits each
// operation's activity duration independently. The receipt must identify the
// actual raw export and producer clock. Missing identity or timestamps withhold
// that row; loss of another record family cannot upgrade or invalidate it.
// This entry point does not perform a native collection or infer a scheduler join.
func AnalyzeActivities(batch cudaworkload.ActivityBatch, annotations []cudaworkload.HostSubmission, receipt tracecontract.Receipt) (ActivityAnalysis, error) {
	var out ActivityAnalysis
	if _, err := receipt.JSON(); err != nil {
		return out, fmt.Errorf("activity findings receipt: %w", err)
	}
	joins, err := cudaworkload.JoinActivities(batch, annotations)
	if err != nil {
		return out, err
	}
	out.Version = 1
	out.Joins = joins
	for _, join := range joins {
		raw, err := json.Marshal(join.Record)
		if err != nil {
			return ActivityAnalysis{}, err
		}
		hash := sha256.Sum256(raw)
		id := "activity-" + hex.EncodeToString(hash[:])
		r := join.Record
		timed := r.Complete && r.Start != 0 && r.End != 0 && r.Start <= r.End && r.End <= math.MaxInt64
		correlation := &tracecontract.CorrelationOutcome{Status: string(join.Join.Outcome)}
		if join.Join.Outcome == cudaworkload.OutcomeExact {
			correlation.Matched = 1
		} else {
			correlation.Missing = 1
		}
		ctx := tracecontract.EvidenceContext{Receipt: receipt, Artifacts: []string{receipt.InputSHA256}, Correlation: correlation}
		var value *float64
		if timed {
			v := float64(r.End - r.Start)
			value = &v
			ctx.Intervals = []tracecontract.Interval{{Start: int64(r.Start), End: int64(r.End), Clock: join.Clock}}
		}
		reqs := []tracecontract.Requirement{{Name: "operation_identity", RequireCorrelation: true}, {Name: "producer_clock", ClockDomain: join.Clock}, {Name: "raw_export", ArtifactDigest: receipt.InputSHA256}}
		finding := tracecontract.Admit(id, "operation_activity_duration", tracecontract.FindingObserved, "duration of an explicitly correlated captured activity; not causal or complete workload timing", value, "ns", reqs, ctx)
		if !timed {
			finding.Class = tracecontract.FindingUnavailable
			finding.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: "activity timestamps incomplete or invalid"}
		}
		out.Findings = append(out.Findings, finding)
	}
	sort.Slice(out.Findings, func(i, j int) bool { return out.Findings[i].ID < out.Findings[j].ID })
	return out, nil
}

// AnalyzeActivityJSONL reads the existing cuptiactivity.Record JSON-lines export.
// Metadata identifies the producer session, process, run and timestamp epoch.
// It refuses truncated lines, unknown fields, more than 32 MiB, and a raw hash
// differing from the receipt. It never derives producer identity from timestamps.
func AnalyzeActivityJSONL(reader io.Reader, metadata cudaworkload.ActivityBatch, annotations []cudaworkload.HostSubmission, receipt tracecontract.Receipt) (ActivityAnalysis, error) {
	if reader == nil || len(metadata.Records) != 0 {
		return ActivityAnalysis{}, fmt.Errorf("activity findings: invalid reader or nonempty metadata records")
	}
	const maxBytes = 32 << 20
	data, err := io.ReadAll(io.LimitReader(reader, maxBytes+1))
	if err != nil {
		return ActivityAnalysis{}, err
	}
	if len(data) == 0 || len(data) > maxBytes || data[len(data)-1] != '\n' {
		return ActivityAnalysis{}, fmt.Errorf("activity findings: empty, oversized or truncated export")
	}
	hash := sha256.Sum256(data)
	if hex.EncodeToString(hash[:]) != receipt.InputSHA256 {
		return ActivityAnalysis{}, fmt.Errorf("activity findings: raw export hash mismatch")
	}
	limit := metadata.MaxRecords
	if limit == 0 {
		limit = 4096
	}
	if limit < 1 || limit > 65536 {
		return ActivityAnalysis{}, fmt.Errorf("activity findings: invalid record limit")
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for scanner.Scan() {
		if len(metadata.Records) >= limit {
			return ActivityAnalysis{}, fmt.Errorf("activity findings: record limit reached")
		}
		if err := validateActivityObject(scanner.Bytes()); err != nil {
			return ActivityAnalysis{}, err
		}
		var record cuptiactivity.Record
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&record); err != nil {
			return ActivityAnalysis{}, fmt.Errorf("activity findings: record: %w", err)
		}
		var extra any
		if err := decoder.Decode(&extra); err != io.EOF {
			return ActivityAnalysis{}, fmt.Errorf("activity findings: trailing record payload")
		}
		metadata.Records = append(metadata.Records, record)
	}
	if err := scanner.Err(); err != nil {
		return ActivityAnalysis{}, err
	}
	return AnalyzeActivities(metadata, annotations, receipt)
}

var activityFields = func() map[string]bool {
	fields := make(map[string]bool)
	typeOf := reflect.TypeOf(cuptiactivity.Record{})
	for i := 0; i < typeOf.NumField(); i++ {
		field := typeOf.Field(i)
		if !field.IsExported() {
			continue
		}
		name := strings.Split(field.Tag.Get("json"), ",")[0]
		if name == "-" {
			continue
		}
		if name == "" {
			name = field.Name
		}
		fields[name] = true
	}
	return fields
}()

// Native exports use one canonical field spelling. Encoding/json otherwise
// accepts duplicate keys and case aliases, which can hide conflicting raw IDs.
func validateActivityObject(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return fmt.Errorf("activity findings: record object required")
	}
	seen := make(map[string]bool)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok || !activityFields[key] || seen[key] {
			return fmt.Errorf("activity findings: unknown, aliased or duplicate field %q", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
	}
	if _, err := decoder.Token(); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return fmt.Errorf("activity findings: trailing record payload")
	}
	return nil
}
