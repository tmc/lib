package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func newEndpointDelivery(path string, logs bool) (*nvtelemetry.Delivery, func(), error) {
	c, client, request, err := readEndpointConfig(path)
	if err != nil {
		return nil, nil, err
	}
	if c.Workload != nil || c.Managed != nil || len(c.Devices) != 0 || c.Method != "" && c.Method != "GET" || c.BodyFile != "" || c.MaxPages != 0 || c.MaxBytes > 4<<20 {
		client.CloseIdleConnections()
		return nil, nil, errors.New("OTLP endpoint requires transport-only configuration and at most 4 MiB batches")
	}
	cfg := nvtelemetry.DeliveryConfig{Endpoint: c.URL, Client: client, Headers: request.Header, MaxQueuedBatches: 16, MaxQueuedBytes: 8 << 20, MaxBatchBytes: int(c.MaxBytes), MaxBatchDataPoints: 4096, MaxResponseBytes: 64 << 10, MaxAttempts: 3, InitialBackoff: 100 * time.Millisecond, MaxBackoff: time.Second, BatchTimeout: 30 * time.Second, RequestTimeout: c.Timeout, ShutdownTimeout: 5 * time.Second}
	var delivery *nvtelemetry.Delivery
	if logs {
		delivery, err = nvtelemetry.NewLogsDelivery(cfg)
	} else {
		delivery, err = nvtelemetry.NewDelivery(cfg)
	}
	if err != nil {
		client.CloseIdleConnections()
		return nil, nil, err
	}
	return delivery, client.CloseIdleConnections, nil
}

func runDeliver(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("deliver", flag.ContinueOnError)
	fs.SetOutput(stderr)
	config := fs.String("config", "", "transport-only OTLP endpoint configuration JSON")
	input := fs.String("input", "", "encoded OTLP HTTP JSON request file")
	signal := fs.String("signal", "metrics", "metrics or logs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *config == "" || *input == "" || (*signal != "metrics" && *signal != "logs") {
		return errors.New("deliver requires config, input and metrics or logs signal")
	}
	payload, err := readEndpointFile(*input, 4<<20)
	if err != nil {
		return err
	}
	delivery, cleanup, err := newEndpointDelivery(*config, *signal == "logs")
	if err != nil {
		return err
	}
	defer cleanup()
	enqueueErr := delivery.Enqueue(context.Background(), payload)
	closeErr := delivery.Close(context.Background())
	outputErr := json.NewEncoder(stdout).Encode(delivery.Stats())
	return errors.Join(enqueueErr, closeErr, outputErr)
}
