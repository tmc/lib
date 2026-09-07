//go:build !tpu
package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRuntimeDashboardElements(t *testing.T) {
	srv := NewTelemetryServer(&ModelEngine{}, Config{Name: "dashboard test"})

	wDashboard := httptest.NewRecorder()
	srv.ServeHTTP(wDashboard, httptest.NewRequest(http.MethodGet, "/", nil))
	if wDashboard.Code != http.StatusOK {
		t.Fatalf("dashboard status = %d, want 200", wDashboard.Code)
	}
	for _, id := range []string{
		`id="request-slot-queue"`,
		`id="speculative-tree-canvas"`,
		`id="speculative-tree-detail"`,
		`role="status"`,
		`aria-modal="true"`,
		`id="node-modal-close"`,
	} {
		if !stringsContains(wDashboard.Body.String(), id) {
			t.Errorf("dashboard is missing %s", id)
		}
	}

	wJS := httptest.NewRecorder()
	srv.ServeHTTP(wJS, httptest.NewRequest(http.MethodGet, "/dashboard.js", nil))
	if wJS.Code != http.StatusOK {
		t.Fatalf("dashboard script status = %d, want 200", wJS.Code)
	}
	for _, name := range []string{
		"renderRequestSlotQueue",
		"renderSpeculativeTree",
		"refreshRuntimeScheduling",
		"point.flops",
		"point.bytes_moved",
		"trace.nodes",
		"OpenAI generation is not accelerated",
		"nodeModalFocusables",
		"event.key === 'Escape'",
	} {
		if !stringsContains(wJS.Body.String(), name) {
			t.Errorf("dashboard script is missing %s", name)
		}
	}

	wPost := httptest.NewRecorder()
	srv.ServeHTTP(wPost, httptest.NewRequest(http.MethodPost, "/api/runtime/scheduling", nil))
	if wPost.Code != http.StatusMethodNotAllowed {
		t.Errorf("runtime scheduling POST status = %d, want 405", wPost.Code)
	}
}

func TestRooflineKeepsPointsUnmeasuredWithTimingOnly(t *testing.T) {
	metrics := NewRuntimeMetrics(1)
	metrics.RecordForward(ForwardMeasurement{GPUTime: time.Microsecond, WallTime: 2 * time.Microsecond})
	srv := NewTelemetryServer(&ModelEngine{metrics: metrics}, Config{})

	points := srv.getStats().Roofline.Points
	if len(points) == 0 {
		t.Fatal("roofline points are empty")
	}
	for _, point := range points {
		if point.State != "unmeasured" || point.FLOPs != nil || point.BytesMoved != nil || point.ArithmeticIntensity != nil || point.AttainedTFLOPs != nil {
			t.Errorf("roofline point = %+v, want no config-derived operational data", point)
		}
		if !stringsContains(point.Reason, "GPU timing is measured") {
			t.Errorf("roofline reason = %q, want measured timing boundary", point.Reason)
		}
	}
}
