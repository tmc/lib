package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestDeliverSuppliedLogEvents(t *testing.T) {
	one := uint64(1)
	r := nvtelemetry.Reading{Name: "job_start_event", Unit: "code", DeviceUUID: "GPU-fixture", Evidence: nvtelemetry.EvidenceMeasured, Number: &nvtelemetry.Number{Uint: &one}, Observation: &nvtelemetry.Observation{Kind: "event", SourceClass: "synthetic"}}
	var payload bytes.Buffer
	if err := nvtelemetry.WriteOTLPLogs(&payload, []nvtelemetry.Reading{r}, nvtelemetry.ExportOptions{}); err != nil {
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
	input := filepath.Join(t.TempDir(), "events.json")
	if err := os.WriteFile(input, payload.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	config := endpointConfigFile(t, endpointConfig{URL: server.URL, Timeout: time.Second, MaxBytes: 1 << 20})
	var out bytes.Buffer
	if err := runDeliver([]string{"-config", config, "-input", input, "-signal", "logs"}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	var stats nvtelemetry.DeliveryStats
	if err := json.Unmarshal(out.Bytes(), &stats); err != nil {
		t.Fatal(err)
	}
	if stats.SucceededBatches != 1 || stats.PendingBatches != 0 {
		t.Fatal(stats)
	}
	select {
	case got := <-received:
		if !bytes.Equal(got, payload.Bytes()) {
			t.Fatal("CLI changed supplied-event bytes")
		}
	default:
		t.Fatal("no local consumer")
	}
}
