package capturefiles

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// Status returns the capture-status.json object carried by an
// artifact, unwrapping the optional "status" nesting.
func Status(cap *cudacapture.Capture) map[string]any {
	if cap == nil {
		return nil
	}
	statusData, ok := cap.Files["capture-status.json"]
	if !ok {
		return nil
	}
	var raw map[string]any
	if err := json.Unmarshal(statusData, &raw); err != nil {
		return nil
	}
	if nested, ok := raw["status"].(map[string]any); ok {
		return nested
	}
	return raw
}

// Failure reports whether an artifact records a capture that did not
// succeed, and describes it in terms a reader can act on: the outcome names
// which transport gave up, and capture_error names the stage it gave up at.
// "transport-failed" alone is a token, not a diagnosis.
func Failure(cap *cudacapture.Capture) (string, bool) {
	status := Status(cap)
	if status == nil {
		return "", false
	}
	success, ok := status["capture_success"].(bool)
	if !ok || success {
		return "", false
	}
	reason := "capture marked as failed"
	if failure, ok := status["failure"].(string); ok && failure != "" {
		reason = failure
	} else if captureErr, ok := status["capture_error"].(string); ok && captureErr != "" {
		reason = captureErr
	}
	if outcome, ok := status["outcome"].(string); ok && outcome != "" {
		reason = fmt.Sprintf("%s (outcome %s)", reason, outcome)
	}
	return reason, true
}
