package nvtelemetry

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMLflowHistoryPagination(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer private" {
			t.Error("auth lost")
		}
		if r.URL.Query().Get("page_token") == "" {
			fmt.Fprint(w, `{"metrics":[{"key":"loss","value":0.5,"timestamp":1,"step":0}],"next_page_token":"page 2/+"}`)
		} else if r.URL.Query().Get("page_token") == "page 2/+" {
			fmt.Fprint(w, `{"metrics":[{"key":"loss","value":0.25,"timestamp":2,"step":1}]}`)
		} else {
			t.Error("bad token")
		}
	}))
	defer server.Close()
	request, _ := http.NewRequest("GET", server.URL+"/api/2.0/mlflow/metrics/get-history?run_id=run&metric_key=loss", nil)
	request.Header.Set("Authorization", "Bearer private")
	source := WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}
	pages, err := FetchMLflowHistoryPages(t.Context(), server.Client(), request, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || len(pages) != 2 || pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 || request.URL.Query().Get("page_token") != "" {
		t.Fatal(pages, calls)
	}
	if pages, err := FetchMLflowHistoryPages(t.Context(), server.Client(), request, source, WorkloadLimits{}, 1); err == nil || pages != nil {
		t.Fatal("page budget returned partial history")
	}
	if pages, err := FetchMLflowHistoryPages(t.Context(), server.Client(), request, source, WorkloadLimits{MaxBytes: 150}, 4); err == nil || pages != nil {
		t.Fatal("aggregate bytes exceeded")
	}
}
func TestMLflowHistoryTokenCycleAndWrongMetric(t *testing.T) {
	for _, raw := range []string{`{"metrics":[],"next_page_token":"repeat"}`, `{"metrics":[{"key":"other","value":1}]}`} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, raw) }))
		request, _ := http.NewRequest("GET", server.URL+"/api/2.0/mlflow/metrics/get-history?run_id=run&metric_key=loss", nil)
		pages, err := FetchMLflowHistoryPages(t.Context(), server.Client(), request, WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}, WorkloadLimits{}, 4)
		server.Close()
		if err == nil || pages != nil {
			t.Fatal("contradictory history accepted")
		}
	}
}
func ExampleFetchMLflowHistoryPages() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"metrics":[{"key":"loss","value":0.5,"timestamp":1,"step":0}]}`)
	}))
	defer server.Close()
	request, _ := http.NewRequest("GET", server.URL+"/api/2.0/mlflow/metrics/get-history?run_id=run&metric_key=loss", nil)
	pages, err := FetchMLflowHistoryPages(context.Background(), server.Client(), request, WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}, WorkloadLimits{}, 4)
	fmt.Println(len(pages), err)
	// Output: 1 <nil>
}
