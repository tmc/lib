package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const deliveryFixture = `{"resourceMetrics":[{"resource":{"attributes":[{"key":"host.id","value":{"stringValue":"node-a"}}]},"scopeMetrics":[{"scope":{"name":"nvtelemetry"},"metrics":[{"name":"nvtelemetry.power","unit":"W","gauge":{"dataPoints":[{"timeUnixNano":"10","asDouble":46}]}}]}]}]}`

func deliveryTestConfig(server *httptest.Server) DeliveryConfig {
	// Response semantics tests do not measure loopback server scheduling speed.
	// Timeout behavior has separate controls with explicitly shorter bounds.
	return DeliveryConfig{Endpoint: server.URL + "/v1/metrics", Client: server.Client(), MaxQueuedBatches: 4, MaxQueuedBytes: 8192, MaxBatchBytes: 4096, MaxBatchDataPoints: 4, MaxResponseBytes: 512, MaxAttempts: 3, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond, BatchTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second, ShutdownTimeout: 10 * time.Second}
}

func TestDeliveryRetryAndRoundTrip(t *testing.T) {
	var calls atomic.Int32
	var gotMu sync.Mutex
	var got [][]byte
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/metrics" || r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Authorization") != "Bearer caller-secret" {
			t.Error("request configuration lost")
		}
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		gotMu.Lock()
		got = append(got, b)
		gotMu.Unlock()
		if calls.Add(1) == 1 {
			w.WriteHeader(503)
			_, _ = io.WriteString(w, "unavailable")
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()
	c := deliveryTestConfig(s)
	c.Headers = http.Header{"Authorization": []string{"Bearer caller-secret"}}
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	c.Headers.Set("Authorization", "mutated")
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	stats := d.Stats()
	if stats.SucceededBatches != 1 || stats.RetryAttempts != 1 || stats.OutageResponses != 1 || stats.PendingBytes != 0 {
		t.Fatalf("stats=%+v", stats)
	}
	gotMu.Lock()
	defer gotMu.Unlock()
	if len(got) != 2 || string(got[0]) != deliveryFixture || string(got[1]) != deliveryFixture {
		t.Fatal("protocol bytes changed")
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); !errors.Is(err, ErrDeliveryClosed) {
		t.Fatal(err)
	}
}

func TestDeliveryCanonicalEncoderReceiver(t *testing.T) {
	type point struct {
		Value      string `json:"asInt"`
		Timestamp  string `json:"timeUnixNano"`
		Start      string `json:"startTimeUnixNano"`
		Attributes []struct {
			Key   string `json:"key"`
			Value struct {
				String string `json:"stringValue"`
			} `json:"value"`
		} `json:"attributes"`
	}
	type instrument struct {
		Points      []point `json:"dataPoints"`
		Temporality int     `json:"aggregationTemporality"`
		Monotonic   bool    `json:"isMonotonic"`
	}
	var received struct {
		Resources []struct {
			Scopes []struct {
				Scope struct {
					Name string `json:"name"`
				} `json:"scope"`
				Metrics []struct {
					Name  string      `json:"name"`
					Unit  string      `json:"unit"`
					Gauge *instrument `json:"gauge"`
					Sum   *instrument `json:"sum"`
				} `json:"metrics"`
			} `json:"scopeMetrics"`
		} `json:"resourceMetrics"`
	}
	accepted := make(chan struct{})
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		close(accepted)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()
	readings := []Reading{{Name: MetricPowerUsageMW, Unit: UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 12500, TimestampUnix: 100}, {Name: MetricTotalEnergyMJ, Unit: UnitMillijoules, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 9007199254740993, TimestampUnix: 100}}
	var encoded bytes.Buffer
	if err := WriteOTLPMetrics(&encoded, readings, ExportOptions{MaxReadings: 4, MaxBytes: 4096, HasCounterStart: true, CounterStartUnixNanos: 50}); err != nil {
		t.Fatal(err)
	}
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), encoded.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	<-accepted
	if len(received.Resources) != 1 || len(received.Resources[0].Scopes) != 1 {
		t.Fatal("resource scope lost")
	}
	scope := received.Resources[0].Scopes[0]
	if scope.Scope.Name != "github.com/tmc/lib/nvidia/x/nvtelemetry" || len(scope.Metrics) != 2 {
		t.Fatal("scope lost")
	}
	gauge, counter := scope.Metrics[0], scope.Metrics[1]
	if gauge.Name != "nvtelemetry.power_milliwatts" || gauge.Unit != "mW" || gauge.Gauge == nil || len(gauge.Gauge.Points) != 1 || gauge.Gauge.Points[0].Value != "12500" {
		t.Fatal("gauge semantics lost")
	}
	if counter.Name != "nvtelemetry.total_energy_millijoules" || counter.Unit != "mJ" || counter.Sum == nil || counter.Sum.Temporality != 2 || !counter.Sum.Monotonic || len(counter.Sum.Points) != 1 || counter.Sum.Points[0].Value != "9007199254740993" || counter.Sum.Points[0].Start != "50" {
		t.Fatal("counter semantics lost")
	}
	for _, p := range []point{gauge.Gauge.Points[0], counter.Sum.Points[0]} {
		if p.Timestamp != "100" || len(p.Attributes) < 1 || p.Attributes[0].Key != "nvtelemetry.device.uuid" || p.Attributes[0].Value.String != "GPU-a" {
			t.Fatal("identity or time lost")
		}
		attrs := map[string]string{}
		for _, a := range p.Attributes {
			attrs[a.Key] = a.Value.String
		}
		if attrs["nvtelemetry.evidence"] != "measured" || attrs["nvtelemetry.source_status"] != "unknown" || attrs["nvtelemetry.sensor_timestamp_known"] != "false" || attrs["nvtelemetry.wall_uncertainty_known"] != "false" || attrs["nvtelemetry.calibration_known"] != "false" {
			t.Fatal("evidence or unknown clock status lost", attrs)
		}
	}
}

func TestDeliveryTerminalResponses(t *testing.T) {
	for _, tt := range []struct {
		name                             string
		status                           int
		body                             string
		partial, rejected, auth, invalid uint64
	}{
		{"partial", 200, `{"partialSuccess":{"rejectedDataPoints":"1","errorMessage":"credential must not leak"}}`, 1, 1, 0, 0},
		{"numeric partial", 200, `{"partialSuccess":{"rejectedDataPoints":1},"futureField":true}`, 1, 1, 0, 0},
		{"zero warning", 200, `{"partialSuccess":{"rejectedDataPoints":"0","errorMessage":"warning"}}`, 1, 0, 0, 0},
		{"invalid partial", 200, `{"partialSuccess":{"rejectedDataPoints":"-1"}}`, 0, 0, 0, 1},
		{"bad json", 200, `not json secret`, 0, 0, 0, 1},
		{"null envelope", 200, `null`, 0, 0, 0, 1},
		{"empty body", 200, ``, 0, 0, 0, 1},
		{"oversize", 200, strings.Repeat(" ", 513), 0, 0, 0, 1},
		{"auth", 401, `secret authorization detail`, 0, 0, 1, 0},
		{"rejected payload", 400, `invalid secret payload`, 0, 0, 0, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			defer s.Close()
			d, err := NewDelivery(deliveryTestConfig(s))
			if err != nil {
				t.Fatal(err)
			}
			if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
				t.Fatal(err)
			}
			err = d.Close(context.Background())
			if !errors.Is(err, ErrDeliveryIncomplete) || strings.Contains(err.Error(), "secret") {
				t.Fatalf("unsafe or missing error: %v", err)
			}
			if calls.Load() != 1 {
				t.Fatal("terminal response retried")
			}
			st := d.Stats()
			if st.FailedBatches != 1 || st.PartialBatches != tt.partial || st.RejectedDataPoints != tt.rejected || st.AuthFailures != tt.auth || st.InvalidResponses != tt.invalid {
				t.Fatalf("stats=%+v", st)
			}
			if !errors.Is(d.Close(context.Background()), ErrDeliveryIncomplete) {
				t.Fatal("outcome not retained")
			}
		})
	}
}

func TestDeliveryQueueIncludesInflight(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var body []byte
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		close(entered)
		<-release
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()
	c := deliveryTestConfig(s)
	c.MaxQueuedBatches = 1
	c.MaxQueuedBytes = len(deliveryFixture)
	c.MaxBatchBytes = len(deliveryFixture)
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}
	payload[0] = 'X'
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not send")
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); !errors.Is(err, ErrDeliveryFull) {
		t.Fatal("inflight excluded from budget")
	}
	if st := d.Stats(); st.PendingBatches != 1 || st.PendingBytes != len(deliveryFixture) || st.DroppedBatches != 1 {
		t.Fatalf("stats=%+v", st)
	}
	close(release)
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if string(body) != deliveryFixture {
		t.Fatal("caller payload alias")
	}
}

func TestDeliveryBoundedShutdownAndCancellation(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		entered <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer s.Close()
	defer close(release)
	c := deliveryTestConfig(s)
	c.ShutdownTimeout = 20 * time.Millisecond
	c.RequestTimeout = 5 * time.Second
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := d.Enqueue(cancelled, []byte(deliveryFixture)); !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled admission")
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker not entered")
	}
	start := time.Now()
	if err := d.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("shutdown unbounded")
	}
	select {
	case <-d.done:
	case <-time.After(5 * time.Second):
		t.Fatal("context-aware worker did not stop")
	}
	if st := d.Stats(); st.FailedBatches != 2 || st.PendingBatches != 0 || st.PendingBytes != 0 {
		t.Fatalf("stats=%+v", st)
	}
}

func TestDeliveryRetryBudget(t *testing.T) {
	var calls atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); w.WriteHeader(429) }))
	defer s.Close()
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	if calls.Load() != 3 || d.Stats().RetryAttempts != 2 {
		t.Fatalf("unbounded retry %+v %d", d.Stats(), calls.Load())
	}
}

func TestDeliveryRetryAfterHonorsElapsedBudget(t *testing.T) {
	s := httptest.NewServer(http.NotFoundHandler())
	defer s.Close()
	c := deliveryTestConfig(s)
	transport := new(deliveryRetryAfterTransport)
	c.Client = &http.Client{Transport: transport}
	c.BatchTimeout = 2 * time.Second
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	if transport.calls.Load() != 1 || d.Stats().RetryAttempts != 0 {
		t.Fatal("server retry delay waived")
	}
	now := time.Now()
	if got := deliveryRetryAfter(now.Add(time.Minute).UTC().Format(http.TimeFormat), now); got < 59*time.Second || got > time.Minute {
		t.Fatal("http-date not honored")
	}
}

type deliveryRetryAfterTransport struct{ calls atomic.Int32 }

func (t *deliveryRetryAfterTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.calls.Add(1)
	return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": []string{"60"}}, Body: io.NopCloser(strings.NewReader(""))}, nil
}

type deliveryErrorTransport struct{ calls atomic.Int32 }

func (t *deliveryErrorTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.calls.Add(1)
	return nil, errors.New("private-token-in-transport-error")
}

type deliveryBlockingTransport struct {
	entered chan struct{}
	release chan struct{}
}

func (t deliveryBlockingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	close(t.entered)
	<-t.release
	return nil, errors.New("late transport failure")
}

func TestDeliveryShutdownReturnsWithUncooperativeTransport(t *testing.T) {
	s := httptest.NewServer(http.NotFoundHandler())
	defer s.Close()
	c := deliveryTestConfig(s)
	transport := deliveryBlockingTransport{make(chan struct{}), make(chan struct{})}
	c.Client = &http.Client{Transport: transport}
	c.ShutdownTimeout = 20 * time.Millisecond
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	<-transport.entered
	finished := make(chan error, 1)
	go func() { finished <- d.Close(context.Background()) }()
	select {
	case err := <-finished:
		if !errors.Is(err, ErrDeliveryIncomplete) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		close(transport.release)
		t.Fatal("shutdown waited on uncooperative transport")
	}
	if d.Stats().PendingBatches != 1 {
		t.Fatal("inflight ownership fabricated as released")
	}
	close(transport.release)
	select {
	case <-d.done:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not stop after transport returned")
	}
}

func TestDeliveryTransportErrorsAreBoundedAndPrivate(t *testing.T) {
	s := httptest.NewServer(http.NotFoundHandler())
	defer s.Close()
	c := deliveryTestConfig(s)
	transport := new(deliveryErrorTransport)
	c.Client = &http.Client{Transport: transport}
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	err = d.Close(context.Background())
	if !errors.Is(err, ErrDeliveryIncomplete) || strings.Contains(err.Error(), "private-token") {
		t.Fatal(err)
	}
	if transport.calls.Load() != 3 || d.Stats().TransportFailures != 3 {
		t.Fatal("transport retry budget")
	}
}

func TestDeliveryConfiguration(t *testing.T) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"additiveResponseField":true}`)
	}))
	defer s.Close()
	c := deliveryTestConfig(s)
	for _, endpoint := range []string{"", "://invalid", "file:///file", "https://user:secret@example.com/v1/metrics", "https://example.com/v1/metrics?token=secret"} {
		invalid := c
		invalid.Endpoint = endpoint
		if _, err := NewDelivery(invalid); err == nil {
			t.Fatal("invalid endpoint accepted")
		}
	}
	d, err := NewDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDeliveryBatchControls(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("invalid batch reached network") }))
	defer s.Close()
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close(context.Background())
	for _, payload := range []string{`null`, `[]`, `{}`, `{"resourceMetrics":[]}`, strings.Replace(deliveryFixture, `{"timeUnixNano":"10","asDouble":46}`, `null`, 1), strings.Replace(deliveryFixture, `{"timeUnixNano":"10","asDouble":46}`, `42`, 1), strings.Replace(deliveryFixture, `"asDouble":46`, `"asDouble":46,"asDouble":0`, 1), strings.Replace(deliveryFixture, `"gauge":`, `"sum":{"dataPoints":[{}]},"gauge":`, 1), strings.Repeat(" ", 4097), deliveryFixture + `{}`} {
		if err := d.Enqueue(context.Background(), []byte(payload)); err == nil {
			t.Fatalf("accepted invalid batch %s", payload)
		}
	}
	tooMany := strings.Replace(deliveryFixture, `[{"timeUnixNano":"10","asDouble":46}]`, `[{},{},{},{},{}]`, 1)
	if err := d.Enqueue(context.Background(), []byte(tooMany)); err == nil {
		t.Fatal("data point bound bypassed")
	}
	if d.Stats().EnqueuedBatches != 0 {
		t.Fatal("invalid batch admitted")
	}
}

func TestDeliveryNilContextPreservesAdmission(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, `{}`) }))
	defer s.Close()
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(nil, []byte(deliveryFixture)); err == nil {
		t.Fatal("nil enqueue context accepted")
	}
	if err := d.Close(nil); err == nil {
		t.Fatal("nil close context accepted")
	}
	if stats := d.Stats(); stats.EnqueuedBatches != 0 || stats.PendingBytes != 0 || stats.PendingBatches != 0 {
		t.Fatalf("nil context mutated budgets: %+v", stats)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatalf("nil close stopped admission: %v", err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if d.Stats().SucceededBatches != 1 {
		t.Fatal("valid delivery did not survive nil context controls")
	}
}

func TestDeliveryConcurrentClose(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, `{}`) }))
	defer s.Close()
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = d.Enqueue(context.Background(), []byte(deliveryFixture))
			_ = d.Close(context.Background())
			d.Stats()
		}()
	}
	wg.Wait()
}

func TestDeliveryDoesNotFollowRedirect(t *testing.T) {
	var unexpected atomic.Int32
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { unexpected.Add(1); _, _ = io.WriteString(w, `{}`) }))
	defer sink.Close()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, sink.URL, http.StatusTemporaryRedirect)
	}))
	defer s.Close()
	d, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	if unexpected.Load() != 0 || d.Stats().RejectedBatches != 1 {
		t.Fatal("redirect delivered outside explicit endpoint")
	}
}

func FuzzDeliveryResponse(f *testing.F) {
	f.Add(`{}`, false)
	f.Add(`{"partialSuccess":{"rejectedDataPoints":"1"}}`, false)
	f.Add(`{"partialSuccess":{"rejectedDataPoints":-1}}`, false)
	f.Add(`{"partialSuccess":{"rejectedLogRecords":"1"}}`, true)
	f.Add(`{"partialSuccess":{"rejectedLogRecords":-1}}`, true)
	f.Fuzz(func(t *testing.T, body string, isLogs bool) {
		if len(body) > 4096 {
			t.Skip()
		}
		partial, rejected, valid := deliveryResponse([]byte(body), isLogs)
		if valid && (!partial && rejected != 0 || rejected > uint64(1<<63-1)) {
			t.Fatal("invalid response classified accepted")
		}
	})
}

const deliveryLogsFixture = `{"resourceLogs":[{"resource":{"attributes":[{"key":"host.id","value":{"stringValue":"node-a"}}]},"scopeLogs":[{"scope":{"name":"nvtelemetry"},"logRecords":[{"timeUnixNano":"100","body":{"stringValue":"device healthy"}}]}]}]}`

func deliveryLogsTestConfig(server *httptest.Server) DeliveryConfig {
	return DeliveryConfig{
		Endpoint:           server.URL + "/v1/logs",
		Client:             server.Client(),
		MaxQueuedBatches:   4,
		MaxQueuedBytes:     8192,
		MaxBatchBytes:      4096,
		MaxBatchDataPoints: 4,
		MaxResponseBytes:   512,
		MaxAttempts:        3,
		InitialBackoff:     time.Millisecond,
		MaxBackoff:         2 * time.Millisecond,
		BatchTimeout:       10 * time.Second,
		RequestTimeout:     5 * time.Second,
		ShutdownTimeout:    10 * time.Second,
	}
}

func TestLogsDeliverySuccessAndPartialSuccess(t *testing.T) {
	var calls atomic.Int32
	var gotMu sync.Mutex
	var got [][]byte
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/logs" || r.Header.Get("Content-Type") != "application/json" {
			t.Error("request configuration lost")
		}
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		gotMu.Lock()
		got = append(got, b)
		gotMu.Unlock()
		call := calls.Add(1)
		if call == 1 {
			_, _ = io.WriteString(w, `{}`)
			return
		}
		_, _ = io.WriteString(w, `{"partialSuccess":{"rejectedLogRecords":"2","errorMessage":"sample rejected"}}`)
	}))
	defer s.Close()

	d, err := NewLogsDelivery(deliveryLogsTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	// First batch: full success.
	if err := d.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err != nil {
		t.Fatal(err)
	}
	// Second batch: partial success (server rejects 2 log records).
	if err := d.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err != nil {
		t.Fatal(err)
	}
	// Close will report incomplete delivery because the second batch had a partial rejection.
	if err := d.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete, got %v", err)
	}
	stats := d.Stats()
	if stats.SucceededBatches != 1 || stats.PartialBatches != 1 || stats.RejectedLogRecords != 2 || stats.RejectedDataPoints != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	gotMu.Lock()
	defer gotMu.Unlock()
	if len(got) != 2 || string(got[0]) != deliveryLogsFixture || string(got[1]) != deliveryLogsFixture {
		t.Fatal("protocol bytes changed")
	}
}

func TestLogsDeliverySignalSeparation(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	dLogs, err := NewLogsDelivery(deliveryLogsTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	defer dLogs.Close(context.Background())

	dMetrics, err := NewDelivery(deliveryTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	defer dMetrics.Close(context.Background())

	// Enqueuing metrics payload into logs delivery must fail.
	if err := dLogs.Enqueue(context.Background(), []byte(deliveryFixture)); err == nil {
		t.Fatal("expected error enqueuing metrics into logs delivery")
	} else if !strings.Contains(err.Error(), "resourceMetrics") && !strings.Contains(err.Error(), "logs request") {
		t.Fatalf("unexpected error message: %v", err)
	}

	// Enqueuing logs payload into metrics delivery must fail.
	if err := dMetrics.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err == nil {
		t.Fatal("expected error enqueuing logs into metrics delivery")
	} else if !strings.Contains(err.Error(), "resourceLogs") && !strings.Contains(err.Error(), "metrics request") {
		t.Fatalf("unexpected error message: %v", err)
	}

	// Payload containing both resourceLogs and resourceMetrics must fail for both.
	mixedPayload := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"timeUnixNano":"100","body":{"stringValue":"log"}}]}]}],"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"m","gauge":{"dataPoints":[{"asDouble":1}]}}]}]}]}`)
	if err := dLogs.Enqueue(context.Background(), mixedPayload); err == nil {
		t.Fatal("expected error enqueuing mixed payload into logs delivery")
	}
	if err := dMetrics.Enqueue(context.Background(), mixedPayload); err == nil {
		t.Fatal("expected error enqueuing mixed payload into metrics delivery")
	}
}

func TestLogsDeliverySignalResponseSeparation(t *testing.T) {
	// If server sends rejectedDataPoints to logs delivery, it's invalid response.
	sLogs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"partialSuccess":{"rejectedDataPoints":1}}`)
	}))
	defer sLogs.Close()

	dLogs, err := NewLogsDelivery(deliveryLogsTestConfig(sLogs))
	if err != nil {
		t.Fatal(err)
	}
	if err := dLogs.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err != nil {
		t.Fatal(err)
	}
	if err := dLogs.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete, got %v", err)
	}
	if dLogs.Stats().InvalidResponses != 1 || dLogs.Stats().RejectedLogRecords != 0 {
		t.Fatalf("expected 1 invalid response for metrics partialSuccess to logs delivery: %+v", dLogs.Stats())
	}

	// If server sends rejectedLogRecords to metrics delivery, it's invalid response.
	sMetrics := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"partialSuccess":{"rejectedLogRecords":1}}`)
	}))
	defer sMetrics.Close()

	dMetrics, err := NewDelivery(deliveryTestConfig(sMetrics))
	if err != nil {
		t.Fatal(err)
	}
	if err := dMetrics.Enqueue(context.Background(), []byte(deliveryFixture)); err != nil {
		t.Fatal(err)
	}
	if err := dMetrics.Close(context.Background()); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete, got %v", err)
	}
	if dMetrics.Stats().InvalidResponses != 1 || dMetrics.Stats().RejectedDataPoints != 0 {
		t.Fatalf("expected 1 invalid response for logs partialSuccess to metrics delivery: %+v", dMetrics.Stats())
	}
}

func TestLogsDeliveryOversizeAndBounds(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	c := deliveryLogsTestConfig(s)
	c.MaxBatchDataPoints = 2
	d, err := NewLogsDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close(context.Background())

	// 3 records exceeds MaxBatchDataPoints = 2.
	oversize := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"body":{"stringValue":"a"}},{"body":{"stringValue":"b"}},{"body":{"stringValue":"c"}}]}]}]}`)
	if err := d.Enqueue(context.Background(), oversize); err == nil {
		t.Fatal("expected error for exceeding record limit")
	}

	// Empty logs request.
	empty := []byte(`{"resourceLogs":[]}`)
	if err := d.Enqueue(context.Background(), empty); err == nil {
		t.Fatal("expected error for empty logs request")
	}

	// Invalid record object (not a json object).
	invalidRecord := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":["not-an-object"]}]}]}`)
	if err := d.Enqueue(context.Background(), invalidRecord); err == nil {
		t.Fatal("expected error for invalid record object")
	}
}

func TestLogsDeliveryRetryAndOutage(t *testing.T) {
	var calls atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(503)
			_, _ = io.WriteString(w, "temporarily unavailable")
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	d, err := NewLogsDelivery(deliveryLogsTestConfig(s))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	stats := d.Stats()
	if stats.SucceededBatches != 1 || stats.RetryAttempts != 1 || stats.OutageResponses != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestLogsDeliveryAuthFailureAndRedaction(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
		_, _ = io.WriteString(w, "unauthorized secret-sensitive-token")
	}))
	defer s.Close()

	c := deliveryLogsTestConfig(s)
	c.Headers = http.Header{"Authorization": []string{"Bearer super-secret-key"}}
	d, err := NewLogsDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Enqueue(context.Background(), []byte(deliveryLogsFixture)); err != nil {
		t.Fatal(err)
	}
	closeErr := d.Close(context.Background())
	if !errors.Is(closeErr, ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete, got %v", closeErr)
	}
	// Verify that error string does not contain token or body.
	if strings.Contains(closeErr.Error(), "secret") || strings.Contains(closeErr.Error(), "sensitive") {
		t.Fatal("error message leaks sensitive information")
	}
	stats := d.Stats()
	if stats.AuthFailures != 1 || stats.FailedBatches != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestDeliveryEmptyPartialSuccessIsFullAcceptance(t *testing.T) {
	for _, logs := range []bool{false, true} {
		signal := "rejectedDataPoints"
		if logs {
			signal = "rejectedLogRecords"
		}
		for _, body := range []string{`{}`, `{"partialSuccess":{}}`, `{"partialSuccess":{"` + signal + `":"0","errorMessage":""}}`} {
			partial, rejected, valid := deliveryResponse([]byte(body), logs)
			if !valid || partial || rejected != 0 {
				t.Fatal(logs, body, partial, rejected, valid)
			}
		}
		partial, rejected, valid := deliveryResponse([]byte(`{"partialSuccess":{"`+signal+`":"0","errorMessage":"producer warning"}}`), logs)
		if !valid || !partial || rejected != 0 {
			t.Fatal("warning should remain explicit incomplete", logs)
		}
	}
}
