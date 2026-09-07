//go:build !tpu
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRuntimeSpeculationEndpoint(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	localEngine := &ModelEngine{metrics: NewRuntimeMetrics(4)}
	remoteEngine := &ModelEngine{metrics: NewRuntimeMetrics(4)}
	local, err := NewIrohClusterNode(ctx, localEngine, Config{}, 0)
	if err != nil {
		t.Fatalf("create local node: %v", err)
	}
	defer local.Close()
	remote, err := NewIrohClusterNode(ctx, remoteEngine, Config{}, 0)
	if err != nil {
		t.Fatalf("create remote node: %v", err)
	}
	defer remote.Close()
	if err := connectRuntimeSpeculationPeer(ctx, local, remote); err != nil {
		t.Fatalf("connect remote target: %v", err)
	}

	srv := NewTelemetryServer(localEngine, Config{})
	srv.SetClusterNode(local)

	wMissingTarget := httptest.NewRecorder()
	srv.ServeHTTP(wMissingTarget, httptest.NewRequest(http.MethodPost, "/api/runtime/speculation", bytes.NewBufferString(`{"prompt":"paired","max_tokens":2}`)))
	if wMissingTarget.Code != http.StatusBadRequest {
		t.Fatalf("missing target status = %d, want 400", wMissingTarget.Code)
	}

	body, err := json.Marshal(RuntimeSpeculationRequest{
		TargetNodeID: remote.NodeID(),
		Prompt:       "paired execution",
		MaxTokens:    2,
	})
	if err != nil {
		t.Fatalf("marshal speculation request: %v", err)
	}
	wPost := httptest.NewRecorder()
	srv.ServeHTTP(wPost, httptest.NewRequest(http.MethodPost, "/api/runtime/speculation", bytes.NewReader(body)))
	if wPost.Code != http.StatusOK {
		t.Fatalf("speculation status = %d, want 200: %s", wPost.Code, wPost.Body.String())
	}
	var trace SpeculationTrace
	if err := json.Unmarshal(wPost.Body.Bytes(), &trace); err != nil {
		t.Fatalf("decode speculation trace: %v", err)
	}
	if trace.TargetNodeID != remote.NodeID() || len(trace.Nodes) == 0 {
		t.Fatalf("trace = %+v, want paired target nodes", trace)
	}
	for _, node := range trace.Nodes {
		if node.ID == "" || (node.Engine != "both" && node.Engine != "draft" && node.Engine != "target") || (node.State != "accepted" && node.State != "rejected" && node.State != "target") {
			t.Errorf("trace node = %+v, want actual node fields", node)
		}
	}

	wGet := httptest.NewRecorder()
	srv.ServeHTTP(wGet, httptest.NewRequest(http.MethodGet, "/api/runtime/scheduling", nil))
	if wGet.Code != http.StatusOK {
		t.Fatalf("scheduling status = %d, want 200", wGet.Code)
	}
	var telemetry RuntimeSchedulingTelemetry
	if err := json.Unmarshal(wGet.Body.Bytes(), &telemetry); err != nil {
		t.Fatalf("decode scheduling telemetry: %v", err)
	}
	if telemetry.Speculation.State != "observed" || telemetry.Speculation.Trace == nil || telemetry.Speculation.Trace.TargetNodeID != remote.NodeID() {
		t.Errorf("scheduling speculation = %+v, want latest observed trace", telemetry.Speculation)
	}

	wGetMethod := httptest.NewRecorder()
	srv.ServeHTTP(wGetMethod, httptest.NewRequest(http.MethodGet, "/api/runtime/speculation", nil))
	if wGetMethod.Code != http.StatusMethodNotAllowed {
		t.Errorf("speculation GET status = %d, want 405", wGetMethod.Code)
	}
}

func connectRuntimeSpeculationPeer(ctx context.Context, local, remote *IrohClusterNode) error {
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		if err = local.ConnectIroh(ctx, testEndpointAddr(remote)); err == nil {
			return nil
		}
		if attempt < 2 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(100 * time.Millisecond):
			}
		}
	}
	return err
}
