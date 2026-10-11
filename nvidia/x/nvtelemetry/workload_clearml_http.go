package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
)

// FetchClearMLScalarPages follows ClearML 1.18.0 API v2.9 events.get_task_events
// scroll pagination for one explicitly selected task.
// Pinned references: clearml/backend_api/services/v2_9/events.py lines 1809..1921
// (request event_type string, batch_size, scroll_id), lines 1924..2008
// (response events, total number/null, returned int, scroll_id string), and
// clearml/backend_api/session/callresult.py (envelope data, meta.result_code == 200).
//
// Each original page and hash is retained. No task-wide completeness or stable
// server snapshot is inferred.
// maxPages must be 1..32. limits bound the aggregate raw bytes and metrics,
// not merely each page. All raw events (including filtered non-scalar events)
// count against the aggregate metric/event bound.
// On any refusal no partial history is returned.
// GetBody is used when available; otherwise the caller-owned Body is consumed.
// Custom body readers must honor cancellation; blocked reads cannot be interrupted.
// The POST request must target events.get_task_events with matching task in
// the JSON request body and an initial empty scroll_id. Only scroll_id is updated
// for subsequent pages, preserving all other request fields byte-for-byte.
func FetchClearMLScalarPages(ctx context.Context, client *http.Client, request *http.Request, source WorkloadSource, limits WorkloadLimits, maxPages int) ([]WorkloadHTTPResult, error) {
	if ctx == nil || client == nil || request == nil || request.URL == nil {
		return nil, errors.New("missing ClearML pagination context, client or request")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if request.Method != http.MethodPost {
		return nil, errors.New("clearml events.get_task_events requires POST method")
	}
	if source.Provider != "clearml" || source.Version != "1.18.0/api-2.9" {
		return nil, errors.New("unsupported ClearML provider or version")
	}
	if source.RunID == "" {
		return nil, errors.New("missing ClearML task ID in source.RunID")
	}
	if maxPages < 1 || maxPages > 32 {
		return nil, errors.New("invalid ClearML maxPages limit (1..32)")
	}
	if !strings.HasSuffix(request.URL.Path, "events.get_task_events") {
		return nil, errors.New("clearml pagination requires an explicit events.get_task_events endpoint")
	}
	limits, err := workloadLimits(limits)
	if err != nil {
		return nil, err
	}

	// Read initial request body with context awareness and error redaction.
	// Prefer request.GetBody if available to avoid consuming reusable requests.
	var bodyReader io.Reader
	if request.GetBody != nil {
		rc, err := request.GetBody()
		if err != nil {
			return nil, clearMLBodyError(ctx, err, "failed to acquire ClearML request body")
		}
		if rc == nil {
			return nil, errors.New("missing ClearML request body reader")
		}
		defer rc.Close()
		bodyReader = rc
	} else if request.Body != nil && request.Body != http.NoBody {
		bodyReader = request.Body
	} else {
		return nil, errors.New("missing ClearML request body")
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rawBody, err := io.ReadAll(io.LimitReader(workloadContextReader{ctx, bodyReader}, 64<<10+1))
	if err != nil {
		return nil, clearMLBodyError(ctx, err, "failed to read ClearML request body")
	}
	if len(rawBody) == 0 {
		return nil, errors.New("empty ClearML request body")
	}
	if len(rawBody) > 64<<10 {
		return nil, errors.New("clearml request body exceeds 64KB limit")
	}

	var reqObj struct {
		Task      string          `json:"task"`
		EventType json.RawMessage `json:"event_type"`
		BatchSize *int            `json:"batch_size"`
		ScrollID  string          `json:"scroll_id"`
	}
	if err := decodeWorkloadJSON(rawBody, &reqObj); err != nil {
		return nil, errors.New("invalid ClearML request body JSON")
	}
	if reqObj.Task != source.RunID {
		return nil, errors.New("clearml request body task contradicts source.RunID")
	}
	if reqObj.BatchSize != nil && (*reqObj.BatchSize <= 0 || *reqObj.BatchSize > 10000) {
		return nil, errors.New("clearml request batch_size out of bounds (1..10000)")
	}
	// Initial traversal must start fresh; nonempty initial scroll_id is rejected.
	if reqObj.ScrollID != "" {
		return nil, errors.New("clearml pagination requires an initial request without scroll_id")
	}
	if len(reqObj.EventType) > 0 && string(bytes.TrimSpace(reqObj.EventType)) != "null" {
		var singleType string
		if err := json.Unmarshal(reqObj.EventType, &singleType); err != nil {
			return nil, errors.New("clearml scalar pagination requires string event_type")
		}
		if singleType != "training_stats_scalar" {
			return nil, errors.New("clearml scalar pagination requires training_stats_scalar event_type")
		}
	}

	// Preserve all caller-supplied request fields without floating-point conversion.
	var baseMap map[string]json.RawMessage
	if err := json.Unmarshal(rawBody, &baseMap); err != nil {
		return nil, errors.New("invalid ClearML request body JSON")
	}

	for key := range baseMap {
		switch strings.ToLower(key) {
		case "task", "order", "event_type", "batch_size", "scroll_id":
			if key != strings.ToLower(key) {
				return nil, errors.New("clearml request field case alias refused")
			}
		}
	}

	var pages []WorkloadHTTPResult
	var usedBytes int64
	var usedEvents int
	var hasAdvertisedTotal bool
	var advertisedTotal int
	seenPageHashes := make(map[string]bool)
	currentBody := rawBody

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(pages) >= maxPages {
			return nil, errors.New("clearml pagination page limit exceeded")
		}
		pageLimits := limits
		pageLimits.MaxBytes -= usedBytes
		pageLimits.MaxMetrics -= usedEvents
		if pageLimits.MaxBytes <= 0 || pageLimits.MaxMetrics <= 0 {
			return nil, errors.New("clearml pagination aggregate limit exceeded")
		}

		pageBody := currentBody
		nextReq := request.Clone(ctx)
		nextReq.Body = io.NopCloser(bytes.NewReader(pageBody))
		nextReq.ContentLength = int64(len(pageBody))
		nextReq.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(pageBody)), nil
		}
		if nextReq.Header == nil {
			nextReq.Header = make(http.Header)
		}
		nextReq.Header.Set("Content-Type", "application/json")

		page, err := FetchWorkload(ctx, client, nextReq, source, pageLimits)
		if err != nil {
			return nil, err
		}

		// Response cycle detection on identical page hashes.
		if seenPageHashes[page.Export.RawSHA256] {
			return nil, errors.New("clearml pagination response cycle detected")
		}
		seenPageHashes[page.Export.RawSHA256] = true

		eventsCount, pageTotal, hasPageTotal, scrollID, err := inspectClearMLPage(page.Export.Raw, source.RunID, limits)
		if err != nil {
			return nil, err
		}

		if hasPageTotal {
			if !hasAdvertisedTotal {
				hasAdvertisedTotal = true
				advertisedTotal = pageTotal
			} else if pageTotal != advertisedTotal {
				return nil, errors.New("clearml pagination advertised total contradiction across pages")
			}
		}

		usedBytes += int64(len(page.Export.Raw))
		usedEvents += eventsCount
		if usedBytes > limits.MaxBytes || usedEvents > limits.MaxMetrics {
			return nil, errors.New("clearml pagination aggregate limit exceeded")
		}
		if hasAdvertisedTotal && usedEvents > advertisedTotal {
			return nil, errors.New("clearml pagination event count exceeds advertised total")
		}

		pages = append(pages, page)

		// Termination rules:
		// 1. Advertised total reached.
		if hasAdvertisedTotal && usedEvents == advertisedTotal {
			return pages, nil
		}
		// 2. Terminal page: zero events returned or empty scroll token.
		if eventsCount == 0 || scrollID == "" {
			if hasAdvertisedTotal && usedEvents < advertisedTotal {
				return nil, errors.New("clearml pagination terminated with unmet advertised total")
			}
			return pages, nil
		}

		scrollBytes, err := json.Marshal(scrollID)
		if err != nil {
			return nil, err
		}
		baseMap["scroll_id"] = scrollBytes
		currentBody, err = json.Marshal(baseMap)
		if err != nil {
			return nil, err
		}
	}
}

func inspectClearMLPage(raw []byte, expectedTask string, l WorkloadLimits) (eventsCount int, total int, hasTotal bool, scrollID string, err error) {
	var envelope struct {
		Data json.RawMessage `json:"data"`
		Meta struct {
			ResultCode int `json:"result_code"`
		} `json:"meta"`
	}
	if err := decodeWorkloadJSON(raw, &envelope); err != nil {
		return 0, 0, false, "", err
	}
	payload := raw
	if envelope.Data != nil {
		if envelope.Meta.ResultCode != 200 {
			return 0, 0, false, "", errors.New("clearml response is not successful")
		}
		payload = envelope.Data
	}
	var body struct {
		Events []struct {
			Type string `json:"type"`
			Task string `json:"task"`
		} `json:"events"`
		ScrollID string          `json:"scroll_id"`
		Total    json.RawMessage `json:"total"`
		Returned json.RawMessage `json:"returned"`
	}
	if err := decodeWorkloadJSON(payload, &body); err != nil {
		return 0, 0, false, "", err
	}
	if len(body.ScrollID) > l.MaxStringBytes {
		return 0, 0, false, "", errors.New("clearml scroll token limit exceeded")
	}
	for _, ev := range body.Events {
		if ev.Task != "" && ev.Task != expectedTask {
			return 0, 0, false, "", errors.New("clearml response contains mixed task IDs")
		}
	}

	ret, hasRet, err := parseClearMLCount(body.Returned, "returned count")
	if err != nil {
		return 0, 0, false, "", err
	}
	if hasRet && ret != len(body.Events) {
		return 0, 0, false, "", errors.New("clearml response returned count contradicts event list length")
	}

	tot, hasTot, err := parseClearMLCount(body.Total, "event total")
	if err != nil {
		return 0, 0, false, "", err
	}

	return len(body.Events), tot, hasTot, body.ScrollID, nil
}

func parseClearMLCount(raw json.RawMessage, fieldName string) (int, bool, error) {
	trimmed := string(bytes.TrimSpace(raw))
	if len(trimmed) == 0 || trimmed == "null" {
		return 0, false, nil
	}

	count, ok := boundedDecimal(trimmed)
	if !ok {
		return 0, false, errors.New("invalid ClearML " + fieldName)
	}
	if count.Sign() < 0 {
		return 0, false, errors.New("negative ClearML " + fieldName)
	}
	if !count.IsInt() {
		return 0, false, errors.New("fractional ClearML " + fieldName)
	}
	if !count.Num().IsInt64() || count.Num().Int64() > math.MaxInt32 {
		return 0, false, errors.New("overflow ClearML " + fieldName)
	}
	return int(count.Num().Int64()), true, nil
}

func clearMLBodyError(ctx context.Context, err error, message string) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	return errors.New(message)
}
