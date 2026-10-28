package nvtelemetry_test

import (
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleDelivery_Done() {
	d, err := nvtelemetry.NewDelivery(nvtelemetry.DeliveryConfig{Endpoint: "http://offline.invalid/v1/metrics", Client: &http.Client{}, MaxQueuedBatches: 1, MaxQueuedBytes: 1024, MaxBatchBytes: 1024, MaxBatchDataPoints: 1, MaxResponseBytes: 1024, MaxAttempts: 1, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second})
	if err != nil {
		fmt.Println(err)
		return
	}
	err = d.Close(context.Background())
	<-d.Done()
	fmt.Println(err)
	// Output: <nil>
}

func TestDeliveryDoneWithoutBatches(t *testing.T) {
	var zero nvtelemetry.Delivery
	if zero.Done() != nil {
		t.Fatal("zero delivery has a completion channel")
	}
	var nilDelivery *nvtelemetry.Delivery
	if nilDelivery.Done() != nil {
		t.Fatal("nil delivery has a completion channel")
	}
	d, err := nvtelemetry.NewDelivery(nvtelemetry.DeliveryConfig{Endpoint: "http://offline.invalid/v1/metrics", Client: &http.Client{}, MaxQueuedBatches: 1, MaxQueuedBytes: 1024, MaxBatchBytes: 1024, MaxBatchDataPoints: 1, MaxResponseBytes: 1024, MaxAttempts: 1, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-d.Done():
	default:
		t.Fatal("successful close did not finish worker")
	}
}
