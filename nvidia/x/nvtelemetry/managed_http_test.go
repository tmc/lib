package nvtelemetry

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func managedHTTPConfig() ManagedImport {
	return ManagedImport{SourceID: "exporter-a", Producer: "dcgm-exporter", ProducerVersion: "4.6.1", NativeOrigin: "dcgm", Fields: DCGMExporterFields()}
}
func TestFetchManagedEndpoint(t *testing.T) {
	calls := 0
	raw := "# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter\nDCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{UUID=\"GPU-a\"} 9007199254740993\n"
	clock := NewFakeClock(0, 1000)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer private" {
			t.Error("missing credentials")
		}
		clock.Advance(100)
		fmt.Fprint(w, raw)
	}))
	defer server.Close()
	req, _ := http.NewRequest("GET", server.URL, nil)
	req.Header.Set("Authorization", "Bearer private")
	sample, err := FetchManaged(t.Context(), server.Client(), req, managedHTTPConfig(), clock, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(sample.Readings) != 1 || sample.Readings[0].IntValue != 9007199254740993 {
		t.Fatal(sample)
	}
	r := sample.Readings[0]
	if !r.HasQueryBounds || r.QueryStartNanos != 0 || r.QueryEndNanos != 100 || r.HasSourceTimestamp || r.HasWallClockUncertainty {
		t.Fatal("invented timing", r)
	}
	bad := managedHTTPConfig()
	bad.SourceID = ""
	if _, err := FetchManaged(t.Context(), server.Client(), req, bad, clock, 4096); err == nil || calls != 1 {
		t.Fatal("bad provenance transmitted credentials")
	}
}
func TestFetchManagedRefusals(t *testing.T) {
	for _, status := range []int{302, 401, 200} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", "http://private.invalid")
			w.WriteHeader(status)
			fmt.Fprint(w, strings.Repeat("private", 20))
		}))
		req, _ := http.NewRequest("GET", server.URL, nil)
		_, err := FetchManaged(context.Background(), server.Client(), req, managedHTTPConfig(), NewFakeClock(0, 1000), 16)
		server.Close()
		if err == nil || strings.Contains(err.Error(), "private") {
			t.Fatal(status, err)
		}
	}
}
func ExampleFetchManaged() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.5\n")
	}))
	defer server.Close()
	request, _ := http.NewRequest("GET", server.URL, nil)
	sample, err := FetchManaged(context.Background(), server.Client(), request, ManagedImport{SourceID: "demo", Producer: "dcgm-exporter", ProducerVersion: "4.6.1", NativeOrigin: "dcgm", Fields: DCGMExporterFields()}, RealClock{}, 4096)
	fmt.Println(len(sample.Readings), err)
	// Output: 1 <nil>
}
