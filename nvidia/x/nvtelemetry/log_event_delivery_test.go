package nvtelemetry

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestSuppliedLogEventDelivery(t *testing.T) {
	var payload bytes.Buffer
	if err := WriteOTLPLogs(&payload, []Reading{logEventFixture("ecc_error_event"), logEventFixture("device_reset_event")}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	received := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		received <- b
		fmt.Fprint(w, `{}`)
	}))
	defer server.Close()
	d, err := NewLogsDelivery(DeliveryConfig{Endpoint: server.URL, Client: server.Client(), MaxQueuedBatches: 1, MaxQueuedBytes: 1 << 20, MaxBatchBytes: 1 << 20, MaxBatchDataPoints: 2, MaxResponseBytes: 4096, MaxAttempts: 1, InitialBackoff: time.Millisecond, MaxBackoff: time.Millisecond, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	if err := d.Enqueue(ctx, payload.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-received:
		if !bytes.Equal(got, payload.Bytes()) {
			t.Fatal("delivery changed event encoding")
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if s := d.Stats(); s.SucceededBatches != 1 || s.RejectedLogRecords != 0 || s.PendingBatches != 0 {
		t.Fatal(s)
	}
}
