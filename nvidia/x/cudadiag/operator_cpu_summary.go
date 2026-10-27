package cudadiag

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sort"
	"strconv"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// OperatorCPUAnnotation locates an annotation in its raw-hash-scoped reader
// profile. Its ordinal is not a native operation or contributing member ID.
type OperatorCPUAnnotation struct {
	Annotation nvtelemetry.WorkloadAnnotation `json:"annotation"`
	Source     nvtelemetry.OperatorReference  `json:"source"`
}

// OperatorCPUSummary projects imported CPU annotations without native joins.
// The observed annotation count is not a complete producer denominator.
// CodeFacts are retained references, never contributor edges or clock links.
type OperatorCPUSummary struct {
	Profile                string                          `json:"profile"`
	Version                int                             `json:"version"`
	ContextSHA256          string                          `json:"context_sha256"`
	Source                 nvtelemetry.WorkloadSource      `json:"source"`
	TraceSource            nvtelemetry.WorkloadSource      `json:"trace_source"`
	Trace                  nvtelemetry.OperatorReference   `json:"trace"`
	Annotations            []OperatorCPUAnnotation         `json:"annotations"`
	ObservedCPUAnnotations uint64                          `json:"observed_cpu_annotation_count"`
	NativeAnalysisStatus   string                          `json:"native_analysis_status"`
	NativePopulation       *OperatorPopulation             `json:"native_population"`
	MembershipStatus       string                          `json:"membership_status"`
	CodeFacts              []nvtelemetry.OperatorReference `json:"code_facts,omitempty"`
	Evidence               []nvtelemetry.OperatorReference `json:"evidence,omitempty"`
	RawArtifactSHA256      []string                        `json:"raw_artifact_sha256"`
	PopulationLimits       []string                        `json:"population_limits"`
}

// AnalyzeOperatorCPUJSON reads application operator-cpu-context-v1 metadata
// and projects the existing PyTorch annotation reader's ordered CPU records.
// It does not consume activity JSONL, match kernels, or infer membership. A
// caller-owned blocking Reader cannot be interrupted by cancellation.
func AnalyzeOperatorCPUJSON(ctx context.Context, metadata io.Reader, source nvtelemetry.WorkloadSource, limits nvtelemetry.WorkloadLimits) (OperatorCPUSummary, error) {
	var out OperatorCPUSummary
	if ctx == nil || metadata == nil {
		return out, fmt.Errorf("operator CPU summary: missing context or reader")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	export, c, err := nvtelemetry.ReadOperatorCPUContext(telemetryReader{ctx, metadata}, source, limits)
	if err != nil {
		return out, fmt.Errorf("operator CPU summary context: %w", err)
	}
	trace, annotations, err := nvtelemetry.ReadPyTorchAnnotations(telemetryReader{ctx, bytes.NewReader(c.Artifacts[c.Trace.SHA256])}, c.TraceSource, limits)
	if err != nil {
		return out, fmt.Errorf("operator CPU summary trace: %w", err)
	}
	if trace.RawSHA256 != c.Trace.SHA256 {
		return out, fmt.Errorf("operator CPU summary trace hash mismatch")
	}
	out = OperatorCPUSummary{Profile: "operator-cpu-summary-v1", Version: 1, ContextSHA256: export.RawSHA256, Source: export.Source, TraceSource: trace.Source, Trace: c.Trace,
		ObservedCPUAnnotations: uint64(len(annotations)), NativeAnalysisStatus: "not_collected", NativePopulation: nil, MembershipStatus: "unavailable", CodeFacts: c.CodeFacts, Evidence: c.Evidence,
		Annotations:      make([]OperatorCPUAnnotation, 0, len(annotations)),
		PopulationLimits: []string{"imported CPU annotations only; native activity was not collected", "observed annotation count is not producer completeness", "process start, loss coverage, cross-clock mapping and Kineto source pairing unverified", "code facts are references without event-to-code or contributor membership", "annotation durations are not GPU durations or performance qualification"}}
	for i, annotation := range annotations {
		if err := ctx.Err(); err != nil {
			return OperatorCPUSummary{}, err
		}
		out.Annotations = append(out.Annotations, OperatorCPUAnnotation{Annotation: annotation, Source: nvtelemetry.OperatorReference{SHA256: trace.RawSHA256, Locator: "user_annotation/" + strconv.Itoa(i)}})
	}
	for hash := range c.Artifacts {
		out.RawArtifactSHA256 = append(out.RawArtifactSHA256, hash)
	}
	sort.Strings(out.RawArtifactSHA256)
	if err := ctx.Err(); err != nil {
		return OperatorCPUSummary{}, err
	}
	return out, nil
}
