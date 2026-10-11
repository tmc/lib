package nvtelemetry_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleNewDelivery() {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, `{}`) }))
	defer s.Close()
	d, err := nvtelemetry.NewDelivery(nvtelemetry.DeliveryConfig{Endpoint: s.URL + "/v1/metrics", Client: s.Client(), MaxQueuedBatches: 2, MaxQueuedBytes: 4096, MaxBatchBytes: 2048, MaxBatchDataPoints: 10, MaxResponseBytes: 1024, MaxAttempts: 1, BatchTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second, ShutdownTimeout: 10 * time.Second})
	if err != nil {
		panic(err)
	}
	// Each item is a complete encoded OTLP ExportMetricsServiceRequest.
	payload := []byte(`{"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"nvtelemetry.power","unit":"W","gauge":{"dataPoints":[{"asDouble":46}]}}]}]}]}`)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		panic(err)
	}
	if err := d.Close(context.Background()); err != nil {
		panic(err)
	}
	fmt.Println(d.Stats().SucceededBatches)
	// Output: 1
}

func ExampleNewLogsDelivery() {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, `{}`) }))
	defer s.Close()
	d, err := nvtelemetry.NewLogsDelivery(nvtelemetry.DeliveryConfig{
		Endpoint:           s.URL + "/v1/logs",
		Client:             s.Client(),
		MaxQueuedBatches:   2,
		MaxQueuedBytes:     4096,
		MaxBatchBytes:      2048,
		MaxBatchDataPoints: 10,
		MaxResponseBytes:   1024,
		MaxAttempts:        1,
		BatchTimeout:       10 * time.Second,
		RequestTimeout:     5 * time.Second,
		ShutdownTimeout:    10 * time.Second,
	})
	if err != nil {
		panic(err)
	}
	// Each item is a complete encoded OTLP ExportLogsServiceRequest.
	payload := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"timeUnixNano":"100","body":{"stringValue":"gpu throttle resolved"}}]}]}]}`)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		panic(err)
	}
	if err := d.Close(context.Background()); err != nil {
		panic(err)
	}
	fmt.Println(d.Stats().SucceededBatches)
	// Output: 1
}
