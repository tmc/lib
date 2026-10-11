package nvtelemetry_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleNewDurableDelivery() {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir, err := os.MkdirTemp("", "spool-example-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(spoolDir)

	d, err := nvtelemetry.NewDurableDelivery(nvtelemetry.DurableDeliveryConfig{
		Delivery: nvtelemetry.DeliveryConfig{
			Endpoint:           s.URL + "/v1/metrics",
			Client:             s.Client(),
			MaxQueuedBatches:   2,
			MaxQueuedBytes:     4096,
			MaxBatchBytes:      2048,
			MaxBatchDataPoints: 10,
			MaxResponseBytes:   1024,
			MaxAttempts:        1,
			BatchTimeout:       5 * time.Second,
			RequestTimeout:     2 * time.Second,
			ShutdownTimeout:    5 * time.Second,
		},
		SpoolDir:        filepath.Join(spoolDir, "task-spool"),
		Owner:           "example-task",
		MaxSpoolBatches: 10,
		MaxSpoolBytes:   65536,
	})
	if err != nil {
		panic(err)
	}

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

func ExampleNewLogsDurableDelivery() {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir, err := os.MkdirTemp("", "spool-logs-example-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(spoolDir)

	d, err := nvtelemetry.NewLogsDurableDelivery(nvtelemetry.DurableDeliveryConfig{
		Delivery: nvtelemetry.DeliveryConfig{
			Endpoint:           s.URL + "/v1/logs",
			Client:             s.Client(),
			MaxQueuedBatches:   2,
			MaxQueuedBytes:     4096,
			MaxBatchBytes:      2048,
			MaxBatchDataPoints: 10,
			MaxResponseBytes:   1024,
			MaxAttempts:        1,
			BatchTimeout:       5 * time.Second,
			RequestTimeout:     2 * time.Second,
			ShutdownTimeout:    5 * time.Second,
		},
		SpoolDir:        filepath.Join(spoolDir, "task-spool"),
		Owner:           "example-logs-task",
		MaxSpoolBatches: 10,
		MaxSpoolBytes:   65536,
	})
	if err != nil {
		panic(err)
	}

	payload := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"timeUnixNano":"100","body":{"stringValue":"gpu throttle cleared"}}]}]}]}`)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		panic(err)
	}
	if err := d.Close(context.Background()); err != nil {
		panic(err)
	}

	fmt.Println(d.Stats().SucceededBatches)
	// Output: 1
}
