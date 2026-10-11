package nvtelemetry

import (
	"context"
	"errors"
	"net/http"
	"strings"
)

// FetchMLflowHistoryPages follows MLflow 2.22.0 get-history page tokens for
// one explicitly selected run and metric. Each original page and hash is
// retained. No run-wide completeness or stable server snapshot is inferred.
// maxPages must be 1..32. limits bound the aggregate raw bytes and metrics,
// not merely each page. On any refusal no partial history is returned.
// The GET request must contain matching run_id and a nonempty metric_key;
// only page_token is changed, on a cloned request at the same endpoint.
func FetchMLflowHistoryPages(ctx context.Context, client *http.Client, request *http.Request, source WorkloadSource, limits WorkloadLimits, maxPages int) ([]WorkloadHTTPResult, error) {
	if ctx == nil || request == nil || request.URL == nil || request.Method != http.MethodGet || request.Body != nil && request.Body != http.NoBody || source.Provider != "mlflow" || source.Version != "2.22.0" || maxPages < 1 || maxPages > 32 {
		return nil, errors.New("invalid MLflow pagination configuration")
	}
	q := request.URL.Query()
	if len(q["run_id"]) != 1 || q.Get("run_id") != source.RunID || len(q["metric_key"]) != 1 || q.Get("metric_key") == "" || len(q["page_token"]) > 1 || !strings.HasSuffix(request.URL.Path, "/api/2.0/mlflow/metrics/get-history") {
		return nil, errors.New("MLflow pagination requires an explicit matching run and metric history endpoint")
	}
	limits, err := workloadLimits(limits)
	if err != nil {
		return nil, err
	}
	key := q.Get("metric_key")
	if len(key) > limits.MaxStringBytes || len(q.Get("page_token")) > limits.MaxStringBytes {
		return nil, errors.New("MLflow query exceeds string limit")
	}
	next := request.Clone(ctx)
	u := *request.URL
	next.URL = &u
	seen := map[string]bool{q.Get("page_token"): true}
	var pages []WorkloadHTTPResult
	var usedBytes int64
	usedMetrics := 0
	for {
		if len(pages) >= maxPages {
			return nil, errors.New("MLflow pagination page limit exceeded")
		}
		pageLimits := limits
		pageLimits.MaxBytes -= usedBytes
		pageLimits.MaxMetrics -= usedMetrics
		if pageLimits.MaxBytes <= 0 || pageLimits.MaxMetrics <= 0 {
			return nil, errors.New("MLflow pagination aggregate limit exceeded")
		}
		page, err := FetchWorkload(ctx, client, next, source, pageLimits)
		if err != nil {
			return nil, err
		}
		for _, metric := range page.Export.Metrics {
			if metric.Name != key {
				return nil, errors.New("MLflow history metric identity mismatch")
			}
		}
		usedBytes += int64(len(page.Export.Raw))
		usedMetrics += len(page.Export.Metrics)
		pages = append(pages, page)
		token := page.Export.NextPageToken
		if token == "" {
			return pages, nil
		}
		if seen[token] {
			return nil, errors.New("MLflow pagination token cycle")
		}
		seen[token] = true
		q.Set("page_token", token)
		next.URL.RawQuery = q.Encode()
	}
}
