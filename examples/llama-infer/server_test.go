package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestTelemetryServer(t *testing.T) {
	rec := metalprof.NewFlightRecorder(128)
	rec.Record("Test_GEMV_Kernel", 120*time.Microsecond, [3]int{2048, 1, 1}, [3]int{128, 1, 1}, map[string]string{
		"layer": "0",
		"op":    "gemv",
	})
	rec.Record("Test_Attention_Kernel", 250*time.Microsecond, [3]int{32, 1, 1}, [3]int{128, 1, 1}, map[string]string{
		"layer": "0",
		"op":    "flash_attn",
	})

	eng := &ModelEngine{
		recorder:   rec,
		egraphSats: 4,
	}

	cfg := Config{
		Name:       "Test-Model-7B",
		Dim:        2048,
		NumLayers:  32,
		FuseEGraph: true,
	}

	srv := NewTelemetryServer(eng, cfg)

	// 1. Test Dashboard HTML Endpoint
	reqDash := httptest.NewRequest("GET", "/", nil)
	wDash := httptest.NewRecorder()
	srv.ServeHTTP(wDash, reqDash)

	if wDash.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for dashboard, got %d", wDash.Code)
	}
	body := wDash.Body.String()
	if !stringsContains(body, "tmc/lib Adaptive E-Graph Telemetry") {
		t.Errorf("expected dashboard title in HTML, got: %s", body)
	}
	if !stringsContains(body, "IROH P2P MODEL SYNC") || !stringsContains(body, "Node Detail") {
		t.Error("dashboard is missing the P2P sync controls")
	}

	// 2. Test JSON API Telemetry Endpoint
	reqAPI := httptest.NewRequest("GET", "/api/telemetry", nil)
	wAPI := httptest.NewRecorder()
	srv.ServeHTTP(wAPI, reqAPI)

	if wAPI.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for api, got %d", wAPI.Code)
	}

	var resp TelemetryResponse
	if err := json.Unmarshal(wAPI.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed decoding JSON telemetry: %v", err)
	}

	if resp.ModelName != "Test-Model-7B" || resp.TotalDispatches != 2 {
		t.Errorf("unexpected telemetry response: %+v", resp)
	}
	if resp.EGraphSats != 4 {
		t.Errorf("expected 4 egraph saturations, got %d", resp.EGraphSats)
	}

	// 3. Test GPU FlightRecorder pprof endpoint
	reqPprof := httptest.NewRequest("GET", "/debug/pprof/gpu", nil)
	wPprof := httptest.NewRecorder()
	srv.ServeHTTP(wPprof, reqPprof)

	if wPprof.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for gpu pprof, got %d", wPprof.Code)
	}
	if wPprof.Body.Len() == 0 {
		t.Fatal("expected non-empty pprof profile bytes")
	}

	// 4. Test standard Go runtime pprof index endpoint
	reqRuntimePprof := httptest.NewRequest("GET", "/debug/pprof/", nil)
	wRuntimePprof := httptest.NewRecorder()
	srv.ServeHTTP(wRuntimePprof, reqRuntimePprof)

	if wRuntimePprof.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for runtime pprof index, got %d", wRuntimePprof.Code)
	}
	if !stringsContains(wRuntimePprof.Body.String(), "Types of profiles available") {
		t.Errorf("expected pprof index content in response, got: %s", wRuntimePprof.Body.String())
	}

	// 5. Mesh API rejects a sync without a running cluster node and exposes a
	// stable, empty status list before any sync has started.
	reqModels := httptest.NewRequest(http.MethodGet, "/api/mesh/models", nil)
	wModels := httptest.NewRecorder()
	srv.ServeHTTP(wModels, reqModels)
	if wModels.Code != http.StatusOK {
		t.Errorf("mesh models status = %d, want 200", wModels.Code)
	}

	reqSync := httptest.NewRequest(http.MethodPost, "/api/mesh/sync", strings.NewReader(`{"model_id":"example/model","source_node_id":"peer"}`))
	wSync := httptest.NewRecorder()
	srv.ServeHTTP(wSync, reqSync)
	if wSync.Code != http.StatusServiceUnavailable {
		t.Errorf("mesh sync status = %d, want 503", wSync.Code)
	}

	reqStatus := httptest.NewRequest(http.MethodGet, "/api/mesh/sync/status", nil)
	wStatus := httptest.NewRecorder()
	srv.ServeHTTP(wStatus, reqStatus)
	if wStatus.Code != http.StatusOK || wStatus.Body.String() != "[]\n" {
		t.Errorf("mesh sync status response = %d %q, want 200 []", wStatus.Code, wStatus.Body.String())
	}
}

func stringsContains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (len(s) > 0 && len(substr) > 0 && searchSubstr(s, substr)))
}

func searchSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
