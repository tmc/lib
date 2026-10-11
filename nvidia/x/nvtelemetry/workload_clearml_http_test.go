package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClearMLScalarPagination(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer secret-token" {
			t.Error("authorization header lost")
		}
		var req struct {
			Task     string `json:"task"`
			ScrollID string `json:"scroll_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		if req.Task != "task-123" {
			t.Errorf("expected task-123, got %s", req.Task)
		}

		if req.ScrollID == "" {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [
						{"type":"training_stats_scalar","task":"task-123","metric":"loss","variant":"train","value":0.8,"iter":0,"timestamp":1000},
						{"type":"training_stats_scalar","task":"task-123","metric":"loss","variant":"train","value":0.6,"iter":1,"timestamp":1010}
					],
					"scroll_id": "tok-page2",
					"returned": 2,
					"total": 3
				},
				"meta": {"result_code": 200}
			}`)
		} else if req.ScrollID == "tok-page2" {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [
						{"type":"training_stats_scalar","task":"task-123","metric":"loss","variant":"train","value":0.4,"iter":2,"timestamp":1020}
					],
					"scroll_id": "",
					"returned": 1,
					"total": 3
				},
				"meta": {"result_code": 200}
			}`)
		} else {
			t.Errorf("unexpected scroll_id: %s", req.ScrollID)
		}
	}))
	defer server.Close()

	reqBody := []byte(`{"task":"task-123","event_type":"training_stats_scalar","batch_size":2}`)
	request, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", bytes.NewReader(reqBody))
	request.Header.Set("Authorization", "Bearer secret-token")
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-123"}

	pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), request, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || len(pages) != 2 {
		t.Fatalf("expected 2 pages and 2 calls, got pages=%d calls=%d", len(pages), calls)
	}
	if pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 {
		t.Fatal("identical page hashes")
	}
	if len(pages[0].Export.Metrics) != 2 || len(pages[1].Export.Metrics) != 1 {
		t.Fatalf("unexpected metrics count across pages: %+v", pages)
	}
}

func TestClearMLPaginationRepeatedScrollID(t *testing.T) {
	// Demonstrates that a legitimate repeated scroll ID from the backend
	// does not cause a false cycle refusal when distinct events are returned.
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req struct {
			ScrollID string `json:"scroll_id"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		switch calls {
		case 1:
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-456","metric":"acc","variant":"val","value":0.8,"iter":0,"timestamp":100}],
					"scroll_id": "constant-scroll-ctx",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		case 2:
			// Server returns same scroll_id "constant-scroll-ctx" but new event!
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-456","metric":"acc","variant":"val","value":0.9,"iter":1,"timestamp":200}],
					"scroll_id": "constant-scroll-ctx",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		case 3:
			// Terminal page: 0 events returned.
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [],
					"scroll_id": "constant-scroll-ctx",
					"returned": 0
				},
				"meta": {"result_code": 200}
			}`)
		default:
			t.Errorf("unexpected call %d", calls)
		}
	}))
	defer server.Close()

	reqBody := []byte(`{"task":"task-456","event_type":"training_stats_scalar"}`)
	request, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", bytes.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-456"}

	pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), request, source, WorkloadLimits{}, 5)
	if err != nil {
		t.Fatalf("unexpected error for legitimate repeated scroll ID: %v", err)
	}
	if calls != 3 || len(pages) != 3 {
		t.Fatalf("expected 3 pages and 3 calls, got pages=%d calls=%d", len(pages), calls)
	}
}

func TestClearMLPaginationZeroMetricsNotEOF(t *testing.T) {
	// Demonstrates that a page containing non-scalar events (yielding 0 normalized metrics)
	// does not prematurely terminate pagination.
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		switch calls {
		case 1:
			// Page has 1 non-scalar event (filtered out by ReadClearMLScalars).
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"log","task":"task-789","timestamp":100}],
					"scroll_id": "tok-1",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		case 2:
			// Page has 1 scalar metric.
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-789","metric":"loss","variant":"train","value":0.1,"iter":5,"timestamp":200}],
					"scroll_id": "",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		default:
			t.Errorf("unexpected call %d", calls)
		}
	}))
	defer server.Close()

	reqBody := []byte(`{"task":"task-789"}`)
	request, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", bytes.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-789"}

	pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), request, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pages) != 2 {
		t.Fatalf("expected 2 pages, got %d", len(pages))
	}
	if len(pages[0].Export.Metrics) != 0 {
		t.Fatalf("expected 0 metrics on page 0, got %d", len(pages[0].Export.Metrics))
	}
	if len(pages[1].Export.Metrics) != 1 {
		t.Fatalf("expected 1 metric on page 1, got %d", len(pages[1].Export.Metrics))
	}
}

func TestClearMLPaginationAggregateRawEventsBound(t *testing.T) {
	// Verifies that non-scalar raw events count against aggregate limit,
	// preventing filtered events from evading the events bound.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [
					{"type":"log","task":"task-bound","timestamp":100},
					{"type":"log","task":"task-bound","timestamp":101},
					{"type":"log","task":"task-bound","timestamp":102},
					{"type":"log","task":"task-bound","timestamp":103},
					{"type":"log","task":"task-bound","timestamp":104}
				],
				"scroll_id": "tok-next",
				"returned": 5
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-bound"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-bound"}

	// MaxMetrics=4, but page returns 5 non-scalar events. Must be refused.
	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{MaxMetrics: 4}, 5)
	if err == nil {
		t.Fatal("expected aggregate limit exceeded on non-scalar events")
	}
}

func TestClearMLPaginationReturnedCountMismatch(t *testing.T) {
	// Verifies rejection when returned != len(events).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [
					{"type":"training_stats_scalar","task":"task-ret","metric":"loss","variant":"train","value":0.5,"iter":0,"timestamp":100},
					{"type":"training_stats_scalar","task":"task-ret","metric":"loss","variant":"train","value":0.4,"iter":1,"timestamp":101}
				],
				"scroll_id": "",
				"returned": 1
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-ret"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-ret"}

	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "returned count contradicts") {
		t.Fatalf("expected returned count mismatch error, got: %v", err)
	}
}

func TestClearMLPaginationContradictoryTotal(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-tot","metric":"m","variant":"v","value":1,"timestamp":1}],
					"scroll_id": "tok-2",
					"returned": 1,
					"total": 5
				},
				"meta": {"result_code": 200}
			}`)
		} else {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-tot","metric":"m","variant":"v","value":2,"timestamp":2}],
					"scroll_id": "",
					"returned": 1,
					"total": 6
				},
				"meta": {"result_code": 200}
			}`)
		}
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-tot"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-tot"}

	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "advertised total contradiction") {
		t.Fatalf("expected total contradiction error, got: %v", err)
	}
}

func TestClearMLPaginationExceedsTotal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [
					{"type":"training_stats_scalar","task":"task-exc","metric":"m","variant":"v","value":1,"timestamp":1},
					{"type":"training_stats_scalar","task":"task-exc","metric":"m","variant":"v","value":2,"timestamp":2}
				],
				"scroll_id": "",
				"returned": 2,
				"total": 1
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-exc"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-exc"}

	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "exceeds advertised total") {
		t.Fatalf("expected exceeds advertised total error, got: %v", err)
	}
}

func TestClearMLPaginationUnmetTotal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [
					{"type":"training_stats_scalar","task":"task-unmet","metric":"m","variant":"v","value":1,"timestamp":1}
				],
				"scroll_id": "",
				"returned": 1,
				"total": 5
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-unmet"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-unmet"}

	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "unmet advertised total") {
		t.Fatalf("expected unmet advertised total error, got: %v", err)
	}
}

func TestClearMLPaginationPreservesLargeIntegers(t *testing.T) {
	// Asserts that integers > 2^53 (e.g. 9007199254740993 = 2^53 + 1)
	// are preserved across page requests without float64 precision loss.
	calls := 0
	const largeIntStr = "9007199254740993"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(bodyBytes), `"large_seed":`+largeIntStr) {
			t.Errorf("call %d lost large integer precision: %s", calls, string(bodyBytes))
		}
		if calls == 1 {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-big","metric":"m","variant":"v","value":1,"timestamp":1}],
					"scroll_id": "tok-2",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		} else {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [{"type":"training_stats_scalar","task":"task-big","metric":"m","variant":"v","value":2,"timestamp":2}],
					"scroll_id": "",
					"returned": 1
				},
				"meta": {"result_code": 200}
			}`)
		}
	}))
	defer server.Close()

	reqBody := `{"task":"task-big","large_seed":` + largeIntStr + `}`
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-big"}

	pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pages) != 2 || calls != 2 {
		t.Fatalf("expected 2 pages and 2 calls, got pages=%d calls=%d", len(pages), calls)
	}
}

func TestClearMLPaginationRejectsMultiTypeArray(t *testing.T) {
	reqBody := `{"task":"task-arr","event_type":["training_stats_scalar"]}`
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, "http://localhost/events.get_task_events", strings.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-arr"}

	_, err := FetchClearMLScalarPages(context.Background(), http.DefaultClient, req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "string event_type") {
		t.Fatalf("expected string event_type error, got: %v", err)
	}
}

func TestClearMLPaginationRejectsFractionalTotal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [{"type":"training_stats_scalar","task":"task-frac","metric":"m","variant":"v","value":1,"timestamp":1}],
				"scroll_id": "",
				"returned": 1,
				"total": 2.5
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-frac"}`))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-frac"}

	_, err := FetchClearMLScalarPages(context.Background(), server.Client(), req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "fractional ClearML") {
		t.Fatalf("expected fractional total error, got: %v", err)
	}
}

func TestClearMLPaginationRejectsInitialScrollID(t *testing.T) {
	reqBody := `{"task":"task-scroll","scroll_id":"existing-token"}`
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, "http://localhost/events.get_task_events", strings.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-scroll"}

	_, err := FetchClearMLScalarPages(context.Background(), http.DefaultClient, req, source, WorkloadLimits{}, 4)
	if err == nil || !strings.Contains(err.Error(), "without scroll_id") {
		t.Fatalf("expected without scroll_id error, got: %v", err)
	}
}

type errSecretReader struct{}

func (errSecretReader) Read([]byte) (int, error) {
	return 0, errors.New("reader failed with secret-token-xyz")
}

func TestClearMLPaginationReaderErrorRedacted(t *testing.T) {
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, "http://localhost/events.get_task_events", errSecretReader{})
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-redact"}

	_, err := FetchClearMLScalarPages(context.Background(), http.DefaultClient, req, source, WorkloadLimits{}, 4)
	if err == nil {
		t.Fatal("expected reader error")
	}
	if strings.Contains(err.Error(), "secret-token-xyz") {
		t.Fatalf("error leaked secret reader text: %v", err)
	}
}

func TestClearMLPaginationLimitsAndErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [{"type":"training_stats_scalar","task":"task-err","metric":"m","variant":"v","value":1,"timestamp":10}],
				"scroll_id": "tok-next",
				"returned": 1
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task-err"}

	// Page limit exceeded: maxPages=1 but server returned scroll_id "tok-next".
	req1, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-err"}`))
	if pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), req1, source, WorkloadLimits{}, 1); err == nil || pages != nil {
		t.Fatal("expected page budget error and nil pages on exceeded limit")
	}

	// Task mismatch in request body vs source.RunID.
	req2, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"other-task"}`))
	if pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), req2, source, WorkloadLimits{}, 4); err == nil || pages != nil {
		t.Fatal("expected task contradiction error")
	}

	// Wrong HTTP method (GET instead of POST).
	req3, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL+"/events.get_task_events", strings.NewReader(`{"task":"task-err"}`))
	if pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), req3, source, WorkloadLimits{}, 4); err == nil || pages != nil {
		t.Fatal("expected POST method error")
	}
}

func ExampleFetchClearMLScalarPages() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"data": {
				"events": [
					{"type":"training_stats_scalar","task":"demo-task","metric":"loss","variant":"train","value":0.42,"iter":1,"timestamp":1700000000}
				],
				"scroll_id": "",
				"returned": 1,
				"total": 1
			},
			"meta": {"result_code": 200}
		}`)
	}))
	defer server.Close()

	reqBody := []byte(`{"task":"demo-task","event_type":"training_stats_scalar","batch_size":100}`)
	request, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/events.get_task_events", bytes.NewReader(reqBody))
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "demo-task"}

	pages, err := FetchClearMLScalarPages(context.Background(), server.Client(), request, source, WorkloadLimits{}, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(pages), len(pages[0].Export.Metrics))
	// Output: 1 1
}

func TestClearMLExactCountAndBodyCancellationControls(t *testing.T) {
	for _, raw := range []string{"2.00000000000000001", "1.99999999999999999", "-0.000000000000000001"} {
		if _, _, err := parseClearMLCount(json.RawMessage(raw), "event total"); err == nil {
			t.Fatal("rounded fractional count accepted", raw)
		}
	}
	for _, err := range []error{context.Canceled, context.DeadlineExceeded} {
		request, _ := http.NewRequest("POST", "https://example.invalid/events.get_task_events", strings.NewReader(`{"task":"task"}`))
		request.GetBody = func() (io.ReadCloser, error) { return nil, fmt.Errorf("private body: %w", err) }
		pages, got := FetchClearMLScalarPages(t.Context(), http.DefaultClient, request, WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task"}, WorkloadLimits{}, 4)
		if !errors.Is(got, err) || pages != nil || strings.Contains(got.Error(), "private") {
			t.Fatal("cancellation became success or leaked", got)
		}
	}
	request, _ := http.NewRequest("POST", "https://example.invalid/events.get_task_events", strings.NewReader(`{"task":"task"}`))
	request.GetBody = func() (io.ReadCloser, error) { return nil, nil }
	if _, err := FetchClearMLScalarPages(t.Context(), http.DefaultClient, request, WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task"}, WorkloadLimits{}, 4); err == nil {
		t.Fatal("nil body reader accepted")
	}
}
func TestClearMLRequestCaseAliasRefusedBeforeAuthentication(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; t.Error("case alias transmitted") }))
	defer server.Close()
	request, _ := http.NewRequest("POST", server.URL+"/events.get_task_events", strings.NewReader(`{"task":"wrong","Task":"task"}`))
	if _, err := FetchClearMLScalarPages(t.Context(), server.Client(), request, WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task"}, WorkloadLimits{}, 4); err == nil || calls != 0 {
		t.Fatal("case alias bypassed task admission", err)
	}
}
