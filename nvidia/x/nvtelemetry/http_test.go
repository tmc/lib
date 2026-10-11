package nvtelemetry

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func httpReading() Reading {
	return Reading{Name: MetricPowerUsageMW, Unit: UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 9007199254740993}
}
func TestMetricsHTTPFormats(t *testing.T) {
	h, err := NewMetricsHandler(ExportOptions{}, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	readings := []Reading{httpReading()}
	if err := h.Update(readings); err != nil {
		t.Fatal(err)
	}
	readings[0].IntValue = 0
	for _, tt := range []struct {
		name, accept, method string
		code                 int
		open                 bool
	}{
		{"default", "", "GET", 200, false}, {"open", "application/openmetrics-text;version=1.0.0", "GET", 200, true},
		{"weighted", "application/openmetrics-text;q=0.2,text/plain;q=0.8", "GET", 200, false},
		{"exclude open", "application/openmetrics-text;q=0,*/*;q=1", "GET", 200, false},
		{"exclude text", "text/plain;q=0,*/*;q=1", "GET", 200, true},
		{"unknown", "application/json", "GET", 406, false}, {"version", "application/openmetrics-text;version=2.0", "GET", 406, false},
		{"quality", "text/plain;q=NaN", "GET", 406, false},
		{"exponent", "text/plain;q=1e0", "GET", 406, false}, {"excluded", "*/*;q=0", "GET", 406, false},
		{"head", "", "HEAD", 200, false}, {"method", "", "POST", 405, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/metrics", nil)
			req.Header.Set("Accept", tt.accept)
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != tt.code {
				t.Fatalf("status %d want %d", rec.Code, tt.code)
			}
			if tt.code != 200 {
				return
			}
			if strings.Contains(rec.Header().Get("Content-Type"), "openmetrics") != tt.open {
				t.Fatal(rec.Header())
			}
			if tt.method == "HEAD" {
				if rec.Body.Len() != 0 {
					t.Fatal("HEAD body")
				}
				return
			}
			if !strings.Contains(rec.Body.String(), "9007199254740993") {
				t.Fatal("integer or caller ownership lost")
			}
			if strings.HasSuffix(rec.Body.String(), "# EOF\n") != tt.open {
				t.Fatal("wrong terminator")
			}
		})
	}
}
func TestMetricsHTTPReadiness(t *testing.T) {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Second)
	check := func(want int) {
		t.Helper()
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
		if rec.Code != want {
			t.Fatal(rec.Code, want)
		}
	}
	if h.Ready() {
		t.Fatal("unpublished ready")
	}
	check(503)
	if err := h.Update([]Reading{httpReading()}); err != nil {
		t.Fatal(err)
	}
	if !h.Ready() {
		t.Fatal("published not ready")
	}
	check(200)
	h.mu.Lock()
	h.published = time.Now().Add(-2 * time.Second)
	h.mu.Unlock()
	check(503)
	if h.Ready() {
		t.Fatal("stale ready")
	}
	_ = h.Update([]Reading{httpReading()})
	bad := httpReading()
	bad.Unit = "wrong"
	if h.Update([]Reading{bad}) == nil || h.Ready() {
		t.Fatal("invalid update retained snapshot")
	}
	h.Invalidate()
	check(503)
	var zero MetricsHandler
	if zero.Update(nil) == nil || zero.Ready() {
		t.Fatal("zero usable")
	}
	if _, err := NewMetricsHandler(ExportOptions{}, 0); err == nil {
		t.Fatal("unbounded freshness")
	}
}
func TestMetricsHTTPConcurrent(t *testing.T) {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	server := httptest.NewServer(h)
	defer server.Close()
	var wg sync.WaitGroup
	wg.Go(func() {
		for i := 0; i < 100; i++ {
			if err := h.Update([]Reading{httpReading()}); err != nil {
				t.Error(err)
			}
		}
	})
	for i := 0; i < 100; i++ {
		resp, err := server.Client().Get(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 && resp.StatusCode != 503 {
			t.Fatal(resp.StatusCode)
		}
	}
	wg.Wait()
}
func ExampleNewMetricsHandler() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	fmt.Println(h.Ready())
	// Output: false
}
func ExampleMetricsHandler_Update() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	fmt.Println(h.Update([]Reading{{Name: MetricPowerUsageMW, Unit: UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, Value: 12500}}), h.Ready())
	// Output: <nil> true
}
func ExampleMetricsHandler_Ready() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	fmt.Println(h.Ready())
	// Output: false
}
func ExampleMetricsHandler_Invalidate() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	_ = h.Update(nil)
	h.Invalidate()
	fmt.Println(h.Ready())
	// Output: false
}
func ExampleMetricsHandler_ServeHTTP() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	_ = h.Update(nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	fmt.Println(rec.Code)
	// Output: 200
}

func ExampleMetricsHandler() {
	h, _ := NewMetricsHandler(ExportOptions{}, time.Minute)
	_ = h.Update(nil)
	fmt.Println(h.Ready())
	// Output: true
}
