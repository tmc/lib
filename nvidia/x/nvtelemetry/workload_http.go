package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
)

// WorkloadHTTPResult contains an unchanged response export and, for PyTorch,
// explicit CPU annotations. An HTTP observation does not establish sensor time.
type WorkloadHTTPResult struct {
	Export      WorkloadExport
	Annotations []WorkloadAnnotation
}

// FetchWorkload reads one caller-selected API response or artifact download.
// The caller owns endpoint selection, request encoding, credentials and paging.
// The request must identify a pinned format supported by the corresponding
// reader; this function does not invent framework endpoints or fetch pages.
//
// Redirects are refused. The client is copied and its timeout capped at 30
// seconds. Response bytes use WorkloadLimits; request bodies must be bounded by
// the caller. Errors omit URLs, credentials, response bodies and transport text.
// A custom transport must honor request cancellation, as net/http transports do.
func FetchWorkload(ctx context.Context, client *http.Client, request *http.Request, source WorkloadSource, limits WorkloadLimits) (WorkloadHTTPResult, error) {
	var result WorkloadHTTPResult
	if ctx == nil || client == nil || request == nil || request.URL == nil {
		return result, errors.New("missing workload HTTP context, client or request")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if (request.URL.Scheme != "http" && request.URL.Scheme != "https") || request.URL.Host == "" || request.URL.User != nil {
		return result, errors.New("invalid workload HTTP endpoint")
	}
	// Validate the format, source and limits before transmitting credentials.
	if !supportedWorkloadHTTP(source) {
		return result, errors.New("unsupported workload HTTP provider or version")
	}
	if _, _, err := readWorkload(bytes.NewReader(nil), source, limits); err != nil {
		return result, err
	}
	limits, err := workloadLimits(limits)
	if err != nil {
		return result, err
	}
	raw, err := fetchHTTPBytes(ctx, client, request, limits.MaxBytes)
	if err != nil {
		return result, err
	}
	result, err = parseWorkloadResponse(ctx, bytes.NewReader(raw), source, limits)
	if err != nil {
		return WorkloadHTTPResult{}, errors.New("invalid workload HTTP response format")
	}
	return result, nil
}

func supportedWorkloadHTTP(s WorkloadSource) bool {
	switch s.Provider {
	case "triton":
		return s.Version == "25.06"
	case "vllm":
		return s.Version == "0.8.5"
	case "ray":
		return s.Version == "2.44.0"
	case "sglang":
		return s.Version == "0.5.21"
	case "dynamo":
		return s.Version == "1.5.1"
	case "mlflow":
		return s.Version == "2.22.0"
	case "wandb":
		return s.Version == "0.19.11"
	case "clearml":
		return s.Version == "1.18.0/api-2.9"
	case "pytorch":
		return s.Version == "2.7.0"
	}
	return false
}

func parseWorkloadResponse(ctx context.Context, r io.Reader, s WorkloadSource, l WorkloadLimits) (result WorkloadHTTPResult, err error) {
	switch s.Provider {
	case "triton", "vllm", "ray", "sglang", "dynamo":
		result.Export, err = ReadServingExposition(ctx, r, s, l)
	case "mlflow":
		result.Export, err = ReadMLflowHistory(r, s, l)
	case "wandb":
		result.Export, err = ReadWandBHistory(r, s, l)
	case "clearml":
		result.Export, err = ReadClearMLScalars(r, s, l)
	case "pytorch":
		result.Export, result.Annotations, err = ReadPyTorchAnnotations(r, s, l)
	default:
		err = errors.New("unsupported workload HTTP provider or version")
	}
	return
}
