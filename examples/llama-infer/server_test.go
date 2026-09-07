//go:build !tpu
package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestLoadDashboardJS(t *testing.T) {
	want := []byte("window.dashboard = 'development';\n")
	got := loadDashboardJS(func(name string) ([]byte, error) {
		if name == "dashboard.js" {
			return want, nil
		}
		return nil, errors.New("not found")
	}, "dashboard.js")
	if string(got) != string(want) {
		t.Fatalf("loadDashboardJS() = %q, want development copy %q", got, want)
	}

	got = loadDashboardJS(func(string) ([]byte, error) {
		return nil, errors.New("not found")
	}, "dashboard.js")
	if string(got) != string(embeddedDashboardJS) {
		t.Fatal("loadDashboardJS() did not fall back to the embedded script")
	}
}

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
		recorder:    rec,
		egraphSats:  4,
		egraphKnown: true,
	}

	cfg := Config{
		Name:       "Test-Model-7B",
		Dim:        2048,
		NumLayers:  32,
		NumKVHeads: 8,
		HeadDim:    128,
		MaxSeqLen:  4096,
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
	if !stringsContains(body, "IROH P2P MODEL SYNC") || !stringsContains(body, "Node Detail") || !stringsContains(body, "Mesh Model Selection") {
		t.Error("dashboard is missing the P2P sync controls")
	}
	if !stringsContains(body, `src="/dashboard.js"`) || !stringsContains(body, `id="roofline-canvas"`) || !stringsContains(body, `id="mesh-topology"`) || !stringsContains(body, `id="mesh-sync-hud"`) || !stringsContains(body, `id="pgo-trials"`) || !stringsContains(body, `id="memory-kv"`) || !stringsContains(body, `id="memory-prefix"`) || !stringsContains(body, `id="memory-precision"`) || !stringsContains(body, `id="memory-speculation"`) || !stringsContains(body, `id="memory-batching"`) {
		t.Error("dashboard does not load the external script")
	}

	reqJS := httptest.NewRequest("GET", "/dashboard.js", nil)
	wJS := httptest.NewRecorder()
	srv.ServeHTTP(wJS, reqJS)
	if wJS.Code != http.StatusOK || !stringsContains(wJS.Body.String(), "refreshModelSelector") || !stringsContains(wJS.Body.String(), "renderRoofline") || !stringsContains(wJS.Body.String(), "renderMeshTopology") || !stringsContains(wJS.Body.String(), "renderMeshSyncHUD") || !stringsContains(wJS.Body.String(), "renderPGO") || !stringsContains(wJS.Body.String(), "renderMemoryTelemetry") || !stringsContains(wJS.Body.String(), "refreshMemoryTelemetry") {
		t.Error("dashboard script endpoint did not serve the embedded script")
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
	if resp.EGraphSats == nil || *resp.EGraphSats != 4 {
		t.Errorf("expected 4 egraph rewrite merges, got %v", resp.EGraphSats)
	}
	if len(resp.RecentTimeline) != 2 {
		t.Errorf("expected 2 timeline events, got %d", len(resp.RecentTimeline))
	}
	if len(resp.Roofline.Ceilings) != 1 || resp.Roofline.Ceilings[0].Source != "heuristic capability estimate" {
		t.Errorf("roofline ceilings = %+v, want one explicitly heuristic ceiling", resp.Roofline.Ceilings)
	}
	if len(resp.Roofline.Points) != 2 {
		t.Fatalf("roofline points = %d, want 2", len(resp.Roofline.Points))
	}
	for _, point := range resp.Roofline.Points {
		if point.State != "unmeasured" || point.ArithmeticIntensity != nil || point.AttainedTFLOPs != nil {
			t.Errorf("roofline point = %+v, want explicitly unmeasured values", point)
		}
	}
	var rawPayload map[string]json.RawMessage
	if err := json.Unmarshal(wAPI.Body.Bytes(), &rawPayload); err != nil {
		t.Fatalf("decode raw telemetry: %v", err)
	}
	var rawRoofline struct {
		Points []map[string]json.RawMessage `json:"points"`
	}
	if err := json.Unmarshal(rawPayload["roofline"], &rawRoofline); err != nil {
		t.Fatalf("decode raw roofline: %v", err)
	}
	for _, point := range rawRoofline.Points {
		if _, ok := point["arithmetic_intensity"]; ok {
			t.Error("unmeasured roofline point included arithmetic_intensity")
		}
		if _, ok := point["attained_tflops"]; ok {
			t.Error("unmeasured roofline point included attained_tflops")
		}
	}

	reqPlan := httptest.NewRequest("GET", "/api/graph/plan", nil)
	wPlan := httptest.NewRecorder()
	srv.ServeHTTP(wPlan, reqPlan)
	if wPlan.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for graph plan, got %d", wPlan.Code)
	}
	var plan GraphPlan
	if err := json.Unmarshal(wPlan.Body.Bytes(), &plan); err != nil {
		t.Fatalf("decode graph plan: %v", err)
	}
	if plan.Before == nil || plan.After == nil || len(plan.Rewrites) != 2 {
		t.Fatalf("unexpected graph plan: %+v", plan)
	}

	reqPGO := httptest.NewRequest(http.MethodGet, "/api/optimizer/pgo", nil)
	wPGO := httptest.NewRecorder()
	srv.ServeHTTP(wPGO, reqPGO)
	if wPGO.Code != http.StatusOK {
		t.Fatalf("PGO status = %d, want 200", wPGO.Code)
	}
	var inactivePGO PGOResponse
	if err := json.Unmarshal(wPGO.Body.Bytes(), &inactivePGO); err != nil {
		t.Fatalf("decode inactive PGO: %v", err)
	}
	if inactivePGO.State != "inactive" || inactivePGO.Data != nil || inactivePGO.FlightRecorderDispatches != 2 {
		t.Errorf("inactive PGO = %+v, want no synthetic PGO data", inactivePGO)
	}
	var rawPGO map[string]json.RawMessage
	if err := json.Unmarshal(wPGO.Body.Bytes(), &rawPGO); err != nil {
		t.Fatalf("decode raw PGO: %v", err)
	}
	if _, ok := rawPGO["data"]; ok {
		t.Error("inactive PGO response included synthetic data")
	}

	tracker := egraph.NewPGOTracker("test kernel")
	tracker.RecordTrial("test kernel", [3]int{32, 8, 1}, 12.5, 3)
	srv.SetPGOTracker(tracker)
	wPGO = httptest.NewRecorder()
	srv.ServeHTTP(wPGO, reqPGO)
	var observedPGO PGOResponse
	if err := json.Unmarshal(wPGO.Body.Bytes(), &observedPGO); err != nil {
		t.Fatalf("decode observed PGO: %v", err)
	}
	if observedPGO.State != "observed" || observedPGO.Data == nil || len(observedPGO.Data.AutoTuningTrials) != 1 {
		t.Errorf("observed PGO = %+v, want tracker trial data", observedPGO)
	}

	reqPGOPost := httptest.NewRequest(http.MethodPost, "/api/optimizer/pgo", nil)
	wPGOPost := httptest.NewRecorder()
	srv.ServeHTTP(wPGOPost, reqPGOPost)
	if wPGOPost.Code != http.StatusMethodNotAllowed {
		t.Errorf("PGO POST status = %d, want 405", wPGOPost.Code)
	}

	reqMemory := httptest.NewRequest(http.MethodGet, "/api/memory/kvcache", nil)
	wMemory := httptest.NewRecorder()
	srv.ServeHTTP(wMemory, reqMemory)
	if wMemory.Code != http.StatusOK {
		t.Fatalf("memory telemetry status = %d, want 200", wMemory.Code)
	}
	var memory MemoryTelemetryResponse
	if err := json.Unmarshal(wMemory.Body.Bytes(), &memory); err != nil {
		t.Fatalf("decode memory telemetry: %v", err)
	}
	if memory.KVCache.State != "configured" || memory.KVCache.MaxSequenceTokens != 4096 || memory.KVCache.Layers != 32 || memory.KVCache.KVHeads != 8 || memory.KVCache.HeadDim != 128 {
		t.Errorf("KV cache telemetry = %+v, want configured dimensions", memory.KVCache)
	}
	if memory.Precision.State != "unavailable" || len(memory.Precision.DTypes) != 0 {
		t.Errorf("precision telemetry = %+v, want no synthetic dtype aggregate", memory.Precision)
	}
	for name, capability := range map[string]CapabilityTelemetry{
		"prefix cache": memory.PrefixCache,
		"speculation":  memory.Speculation,
		"batching":     memory.Batching,
	} {
		if capability.State != "unavailable" {
			t.Errorf("%s telemetry = %+v, want unavailable without runtime counters", name, capability)
		}
	}
	var rawMemory map[string]json.RawMessage
	if err := json.Unmarshal(wMemory.Body.Bytes(), &rawMemory); err != nil {
		t.Fatalf("decode raw memory telemetry: %v", err)
	}
	for _, unsupported := range []string{"allocation", "occupancy", "fragmentation", "blocks", "prefix_hits", "draft_tree", "acceptance", "batch_slots"} {
		if _, ok := rawMemory[unsupported]; ok {
			t.Errorf("memory telemetry contains unsupported field %q", unsupported)
		}
	}
	var rawPrecision map[string]json.RawMessage
	if err := json.Unmarshal(rawMemory["precision"], &rawPrecision); err != nil {
		t.Fatalf("decode raw precision telemetry: %v", err)
	}
	if _, ok := rawPrecision["dtypes"]; ok {
		t.Error("unavailable precision telemetry included synthetic dtypes")
	}

	reqMemoryPost := httptest.NewRequest(http.MethodPost, "/api/memory/kvcache", nil)
	wMemoryPost := httptest.NewRecorder()
	srv.ServeHTTP(wMemoryPost, reqMemoryPost)
	if wMemoryPost.Code != http.StatusMethodNotAllowed {
		t.Errorf("memory telemetry POST status = %d, want 405", wMemoryPost.Code)
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

func TestMeshHUDTelemetry(t *testing.T) {
	cfg := Config{Name: "Mesh Test", Dim: 64, NumLayers: 2}
	srv := NewTelemetryServer(&ModelEngine{}, cfg)
	srv.SetClusterNode(&IrohClusterNode{
		self: NodeTelemetry{
			NodeID:        "local-node",
			EndpointAddrs: []string{"127.0.0.1:7777"},
			OS:            "darwin",
			Arch:          "arm64",
			ComputeTFLOPs: 40,
			BandwidthGBS:  800,
		},
		peers: map[string]NodeTelemetry{
			"peer-node": {
				NodeID:        "peer-node",
				EndpointAddrs: []string{"192.0.2.1:7777"},
				OS:            "linux",
				Arch:          "arm64",
				ComputeTFLOPs: 120,
				BandwidthGBS:  300,
			},
		},
		cfg: cfg,
	})
	srv.syncJobs["sync-1"] = &SyncJobStatus{
		ID:              "sync-1",
		ModelID:         "example/model",
		SourceNodeID:    "peer-node",
		CurrentFile:     "model.safetensors",
		DownloadedBytes: 50,
		TotalBytes:      100,
		Percent:         50,
		ThroughputMBs:   12.5,
		State:           "running",
		StartTime:       time.Unix(1, 0),
	}

	stats := srv.getStats()
	if stats.ClusterMesh.TotalNodes != 2 || len(stats.ClusterMesh.Peers) != 1 {
		t.Fatalf("mesh = %+v, want one local node and one peer", stats.ClusterMesh)
	}
	if len(stats.ActiveSyncs) != 1 || stats.ActiveSyncs[0].Percent != 50 {
		t.Fatalf("sync jobs = %+v, want one reported sync", stats.ActiveSyncs)
	}

	encoded, err := json.Marshal(stats)
	if err != nil {
		t.Fatalf("marshal mesh telemetry: %v", err)
	}
	for _, unsupported := range []string{"\"link_mode\"", "\"eta\"", "\"sha"} {
		if strings.Contains(string(encoded), unsupported) {
			t.Errorf("mesh telemetry contains unsupported field %q: %s", unsupported, encoded)
		}
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

func TestEGraphSaturationProvenance(t *testing.T) {
	for _, tc := range []struct {
		name   string
		engine *ModelEngine
		want   *int
	}{
		{"absent engine", nil, nil},
		{"unrecorded zero", &ModelEngine{}, nil},
		{"unrecorded nonzero", &ModelEngine{egraphSats: 7}, nil},
		{"recorded zero", &ModelEngine{egraphKnown: true}, new(int)},
		{"recorded merges", &ModelEngine{egraphSats: 4, egraphKnown: true}, newIntForTelemetryTest(4)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			count, source := egraphSaturationTelemetry(tc.engine)
			if (count == nil) != (tc.want == nil) || count != nil && *count != *tc.want {
				t.Fatalf("count=%v want %v", count, tc.want)
			}
			wantSource := "unmeasured"
			if tc.want != nil {
				wantSource = "compile-time rewrite merges"
			}
			if source != wantSource {
				t.Fatalf("source=%q want %q", source, wantSource)
			}
			for _, value := range []any{TelemetryResponse{EGraphSats: count}, GraphPlan{Saturations: count}} {
				data, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(data, &fields); err != nil {
					t.Fatal(err)
				}
				for _, name := range []string{"egraph_saturations", "saturations"} {
					if raw, ok := fields[name]; ok && tc.want == nil && string(raw) != "null" {
						t.Fatalf("unmeasured counter emitted a value: %s", data)
					}
				}
			}
			if tc.want == nil && formatEGraphSaturations(count) != "Rewrite merges unmeasured" {
				t.Fatal("dashboard presents an unmeasured count as observed")
			}
		})
	}
}

func newIntForTelemetryTest(n int) *int { return &n }
