package hostmetrics

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func ExampleNewRedfishHTTPClient() {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/power","Id":"power","ReadingType":"Power","ReadingUnits":"W","Implementation":"PhysicalSensor","Status":{"State":"Enabled","Health":"OK"},"Reading":250.0}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		Headers:              http.Header{"X-Auth-Token": []string{"caller-token"}},
		MaxResponseBytes:     4096,
		RequestTimeout:       5 * time.Second,
		MaxCollectionMembers: 64,
		MaxPages:             4,
	})
	if err != nil {
		panic(err)
	}

	hostCtx := Context{"node-1", "bmc", "1.0", "mono", "boot", 100, 200}
	sensor, err := client.GetSensor(context.Background(), hostCtx, "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.1f %s %s\n", *sensor.Reading, sensor.Unit, sensor.Health)
	// Output: 250.0 W OK
}

func TestRedfishHTTPClient_GetSensorSuccess(t *testing.T) {
	var authSeen atomic.Value
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authSeen.Store(r.Header.Get("Authorization"))
		if r.URL.Path != "/redfish/v1/Chassis/1/Sensors/temp" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Accept") != "application/json" {
			t.Errorf("missing accept header")
		}
		_, _ = io.WriteString(w, `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/temp","Id":"temp","ReadingType":"Temperature","ReadingUnits":"Cel","Implementation":"PhysicalSensor","Status":{"State":"Enabled","Health":"OK"},"Reading":42.5}`)
	}))
	defer s.Close()

	headers := http.Header{"Authorization": []string{"Bearer secret-key-12345"}}
	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		Headers:              headers,
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Test with relative URI.
	sensor, err := client.GetSensor(context.Background(), fixture(), "/redfish/v1/Chassis/1/Sensors/temp", "temp")
	if err != nil {
		t.Fatal(err)
	}
	if sensor.Reading == nil || *sensor.Reading != 42.5 || sensor.Unit != "Cel" || sensor.Health != "OK" {
		t.Fatalf("unexpected sensor data: %+v", sensor)
	}

	// Test with absolute same-origin URI.
	sensorAbs, err := client.GetSensor(context.Background(), fixture(), s.URL+"/redfish/v1/Chassis/1/Sensors/temp", "temp")
	if err != nil {
		t.Fatal(err)
	}
	if sensorAbs.Reading == nil || *sensorAbs.Reading != 42.5 {
		t.Fatalf("unexpected absolute URI sensor data: %+v", sensorAbs)
	}

	if auth, ok := authSeen.Load().(string); !ok || auth != "Bearer secret-key-12345" {
		t.Fatalf("expected auth header sent, got %v", auth)
	}
}

func TestRedfishHTTPClient_ResolveURINegativeControls(t *testing.T) {
	// Dummy server: if a request reaches here, the negative control failed to reject upfront!
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("request sent to server for invalid URI: %s", r.RequestURI)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	invalidURIs := []string{
		"//outside.invalid/redfish/v1/Sensors/a",
		"/redfish/v1/%252e%252e/outside",
		"/redfish/v1/%255coutside",
		"/redfish/v1/../../outside",
		"/redfish/v1/./outside",
		"/redfish/v1/sub/../..",
		"/redfish/v1/%2e%2e/outside",
		"/redfish/v1/%2E%2E/outside",
		"/redfish/v1/%2f/escape",
		"/redfish/v1/%2F/escape",
		"/redfish/v1/path\\backslash",
		"/redfish/v1/path%5cbackslash",
		"/redfish/v1/path%5Cbackslash",
		"/redfish/v1/Chassis#fragment",
		"/redfish/v1/Chassis//double",
		"http://other-bmc.example.com/redfish/v1/Chassis/1",
		"https://other-bmc.example.com/redfish/v1/Chassis/1",
		"/outside/v1/Chassis",
		"",
	}

	for _, invalidURI := range invalidURIs {
		t.Run(invalidURI, func(t *testing.T) {
			_, err := client.GetSensor(context.Background(), fixture(), invalidURI, "sensor")
			if err == nil {
				t.Fatalf("expected error for invalid URI %q, got nil", invalidURI)
			}
			_, err = client.GetCollection(context.Background(), invalidURI)
			if err == nil {
				t.Fatalf("expected error for invalid collection URI %q, got nil", invalidURI)
			}
		})
	}
}

func TestRedfishHTTPClient_RedirectRefused(t *testing.T) {
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("request was forwarded to redirect destination!")
	}))
	defer sink.Close()

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, sink.URL, http.StatusMovedPermanently)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		Headers:              http.Header{"Authorization": []string{"Bearer secret-token"}},
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetSensor(context.Background(), fixture(), "/redfish/v1/Chassis/1/Sensors/temp", "temp")
	if err == nil {
		t.Fatal("expected error on redirect response")
	}
	if !strings.Contains(err.Error(), "unexpected status 301") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishHTTPClient_ErrorRedaction(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, "invalid token sensitive_debug_dump_with_passwords")
	}))
	defer s.Close()

	token := "sensitive-token-secret-xyz"
	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		Headers:              http.Header{"X-Auth-Token": []string{token}},
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetSensor(context.Background(), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err == nil {
		t.Fatal("expected auth error")
	}
	errStr := err.Error()
	if strings.Contains(errStr, token) || strings.Contains(errStr, "sensitive") || strings.Contains(errStr, "password") {
		t.Fatalf("error message leaks sensitive information: %s", errStr)
	}
	if strings.Contains(errStr, s.URL) {
		t.Fatalf("error message leaks full URL: %s", errStr)
	}
}

func TestRedfishHTTPClient_OversizeResponseRefused(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat("x", 2048))
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     512,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetSensor(context.Background(), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err == nil {
		t.Fatal("expected error on oversized response")
	}
	if !strings.Contains(err.Error(), "byte limit") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishHTTPClient_TimeoutAndCancellation(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     1024,
		RequestTimeout:       20 * time.Millisecond,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetSensor(context.Background(), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err == nil {
		t.Fatal("expected timeout error")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = client.GetSensor(ctx, fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestRedfishHTTPClient_CancellationAfterBodyRead(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Cancel the context while server responds.
		cancel()
		_, _ = io.WriteString(w, `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/temp","Id":"temp","ReadingType":"Temperature","ReadingUnits":"Cel","Implementation":"PhysicalSensor","Status":{"State":"Enabled","Health":"OK"},"Reading":42.5}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       5 * time.Second,
		MaxCollectionMembers: 16,
		MaxPages:             2,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetSensor(ctx, fixture(), "/redfish/v1/Chassis/1/Sensors/temp", "temp")
	if err == nil {
		t.Fatal("expected cancellation error, got success")
	}
}

func TestRedfishHTTPClient_CollectionPaginationSuccess(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/redfish/v1/Chassis/1/Sensors":
			_, _ = io.WriteString(w, `{
				"@odata.id": "/redfish/v1/Chassis/1/Sensors",
				"Members@odata.count": 3,
				"Members": [
					{"@odata.id": "/redfish/v1/Chassis/1/Sensors/power_0"},
					{"@odata.id": "/redfish/v1/Chassis/1/Sensors/power_1"}
				],
				"Members@odata.nextLink": "/redfish/v1/Chassis/1/Sensors?$skip=2"
			}`)
		case "/redfish/v1/Chassis/1/Sensors?$skip=2":
			_, _ = io.WriteString(w, `{
				"@odata.id": "/redfish/v1/Chassis/1/Sensors",
				"Members@odata.count": 3,
				"Members": [
					{"@odata.id": "/redfish/v1/Chassis/1/Sensors/power_2"}
				]
			}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             5,
	})
	if err != nil {
		t.Fatal(err)
	}

	col, err := client.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors")
	if err != nil {
		t.Fatal(err)
	}
	if col.Count != 3 || len(col.Members) != 3 || !col.HasAdvertisedCount || col.AdvertisedCount != 3 {
		t.Fatalf("expected 3 members with AdvertisedCount=3, got %+v", col)
	}
	expected := []string{
		"/redfish/v1/Chassis/1/Sensors/power_0",
		"/redfish/v1/Chassis/1/Sensors/power_1",
		"/redfish/v1/Chassis/1/Sensors/power_2",
	}
	for i, m := range col.Members {
		if m != expected[i] {
			t.Fatalf("member %d: expected %s, got %s", i, expected[i], m)
		}
	}
}

func TestRedfishHTTPClient_CollectionAdvertisedCountContradiction(t *testing.T) {
	// Server advertises 5 members, but delivers only 2 without nextLink.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"@odata.id": "/redfish/v1/Chassis/1/Sensors",
			"Members@odata.count": 5,
			"Members": [
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/0"},
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/1"}
			]
		}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             5,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors")
	if err == nil {
		t.Fatal("expected error on advertised count contradiction")
	}
	if !strings.Contains(err.Error(), "contradicts advertised count") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishHTTPClient_CollectionInconsistentAdvertisedCount(t *testing.T) {
	// Page 1 advertises count 5; page 2 advertises count 6.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/redfish/v1/Chassis/1/Sensors":
			_, _ = io.WriteString(w, `{
				"@odata.id": "/redfish/v1/Chassis/1/Sensors",
				"Members@odata.count": 5,
				"Members": [{"@odata.id": "/redfish/v1/Chassis/1/Sensors/0"}],
				"Members@odata.nextLink": "/redfish/v1/Chassis/1/Sensors?$skip=1"
			}`)
		case "/redfish/v1/Chassis/1/Sensors?$skip=1":
			_, _ = io.WriteString(w, `{
				"@odata.id": "/redfish/v1/Chassis/1/Sensors",
				"Members@odata.count": 6,
				"Members": [{"@odata.id": "/redfish/v1/Chassis/1/Sensors/1"}]
			}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             5,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors")
	if err == nil {
		t.Fatal("expected error on inconsistent advertised count across pages")
	}
	if !strings.Contains(err.Error(), "inconsistent advertised collection count") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishHTTPClient_CollectionCycleDetected(t *testing.T) {
	// Page points back to itself.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"@odata.id": "/redfish/v1/Chassis/1/Sensors",
			"Members": [{"@odata.id": "/redfish/v1/Chassis/1/Sensors/0"}],
			"Members@odata.nextLink": "/redfish/v1/Chassis/1/Sensors"
		}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             10,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors")
	if err == nil {
		t.Fatal("expected cycle detection error")
	}
	if !strings.Contains(err.Error(), "pagination cycle detected") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishHTTPClient_ConfigLimits(t *testing.T) {
	// RequestTimeout upper bound: > 60s must fail.
	_, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              "http://127.0.0.1",
		Client:               http.DefaultClient,
		MaxResponseBytes:     1024,
		RequestTimeout:       70 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             2,
	})
	if err == nil || !strings.Contains(err.Error(), "timeout must be between") {
		t.Fatalf("expected timeout bound error, got %v", err)
	}

	// Header validation: invalid character in key.
	_, err = NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              "http://127.0.0.1",
		Client:               http.DefaultClient,
		Headers:              http.Header{"bad:header": []string{"val"}},
		MaxResponseBytes:     1024,
		RequestTimeout:       5 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             2,
	})
	if err == nil || !strings.Contains(err.Error(), "invalid characters in header key") {
		t.Fatalf("expected header key validation error, got %v", err)
	}

	// Header validation: control character in value.
	_, err = NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              "http://127.0.0.1",
		Client:               http.DefaultClient,
		Headers:              http.Header{"X-Auth": []string{"val\r\ninjection"}},
		MaxResponseBytes:     1024,
		RequestTimeout:       5 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             2,
	})
	if err == nil || !strings.Contains(err.Error(), "control characters in header value") {
		t.Fatalf("expected header value validation error, got %v", err)
	}
}

func TestRedfishHTTPClient_CollectionLimits(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"@odata.id": "/redfish/v1/Chassis/1/Sensors",
			"Members": [
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/0"},
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/1"},
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/2"}
			],
			"Members@odata.nextLink": "/redfish/v1/Chassis/1/Sensors?page=2"
		}`)
	}))
	defer s.Close()

	// Member limit
	clientMembers, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 2,
		MaxPages:             5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := clientMembers.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors"); err == nil {
		t.Fatal("expected collection member limit error")
	}

	// Page limit
	clientPages, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 100,
		MaxPages:             1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := clientPages.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors"); err == nil {
		t.Fatal("expected collection page limit error")
	}
}

func TestRedfishHTTPClient_CollectionCrossOriginRefused(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"@odata.id": "/redfish/v1/Chassis/1/Sensors",
			"Members": [
				{"@odata.id": "/redfish/v1/Chassis/1/Sensors/0"}
			],
			"Members@odata.nextLink": "https://evil.external.net/redfish/v1/Chassis/1/Sensors?skip=1"
		}`)
	}))
	defer s.Close()

	client, err := NewRedfishHTTPClient(RedfishHTTPConfig{
		BaseURL:              s.URL,
		Client:               s.Client(),
		MaxResponseBytes:     4096,
		RequestTimeout:       2 * time.Second,
		MaxCollectionMembers: 10,
		MaxPages:             5,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.GetCollection(context.Background(), "/redfish/v1/Chassis/1/Sensors")
	if err == nil {
		t.Fatal("expected error on cross-origin nextLink")
	}
	if !strings.Contains(err.Error(), "cross-origin") && !strings.Contains(err.Error(), "invalid nextLink") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestRedfishArtifactMembersRefuseNamespaceEscapes(t *testing.T) {
	for _, member := range []string{"/redfish/v1/../outside", "/redfish/v1/%252e%252e/outside", "/redfish/v1/path%2fescape", "/redfish/v1/Chassis#fragment", "/redfish/v1/Chassis?token=private", "/redfish/v1/Chassis//duplicate"} {
		raw, err := json.Marshal(map[string]any{"@odata.id": "/redfish/v1/Chassis", "Members": []any{map[string]string{"@odata.id": member}}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ReadRedfishCollection(bytes.NewReader(raw), "/redfish/v1/Chassis", 4096); err == nil {
			t.Fatal("unsafe artifact member accepted", member)
		}
	}
}
