package advise

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvise_WarnsOnDroppedRecords(t *testing.T) {
	for _, statusBytes := range [][]byte{
		[]byte(`{"status":{"capture_success":true,"run_completeness":"partial","activity_records":36504,"dropped_records":48161}}`),
		[]byte(`{"capture_success":true,"run_completeness":"partial","activity_records":36504,"dropped_records":48161}`),
	} {
		cap := &cudacapture.Capture{
			Files: map[string][]byte{
				"capture-status.json": statusBytes,
			},
		}
		report := &Report{}
		auditCaptureStatus(cap, report)

		var found bool
		for _, item := range report.Items {
			if item.Severity == SeverityCost && strings.Contains(item.Finding, "Lossy capture: 48161 records dropped") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected warning for dropped records, got items: %+v", report.Items)
		}
	}
}
