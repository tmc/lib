package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func nvSentinelFixture(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/health-capacity/nvsentinel-v1.26-request.json")
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
func nvSentinelFixtureRequest(t *testing.T) nvSentinelRequest {
	t.Helper()
	var request nvSentinelRequest
	if err := json.Unmarshal(nvSentinelFixture(t), &request); err != nil {
		t.Fatal(err)
	}
	return request
}

func TestHealthCapacityNVSentinelPreservesProducer(t *testing.T) {
	raw := nvSentinelFixture(t)
	hash := sha256.Sum256(raw)
	got, err := readNVSentinel(raw, hex.EncodeToString(hash[:]))
	if err != nil {
		t.Fatal(err)
	}
	if got.Producer != "NVSentinel" || got.Release != "v1.26.0" || got.UpstreamCommit != "a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5" || got.SHA256 != hex.EncodeToString(hash[:]) || !got.ExpectedDigestChecked {
		t.Fatal("source/schema/raw digest lost")
	}
	if got.StatusDisposition != "producer_reported" || got.ProducerPhase != "Succeeded" || got.TestSelection != "explicit_request" {
		t.Fatal("producer-reported state not retained")
	}
	request := got.Request
	if string(request.Metadata["name"]) != `"constructed-request"` || string(request.Metadata["uid"]) != `"constructed-uid"` || string(request.Metadata["resourceVersion"]) != `"42"` || len(request.Spec.Nodes) != 2 || request.Spec.Nodes[0].Name != "node-a" || request.Spec.Nodes[1].Name != "node-b" || !slices.Equal(request.Spec.Tests, []string{"dcgm-diag-test", "bandwidth"}) {
		t.Fatal("manually checked request membership lost")
	}
	status := request.Status
	if status == nil || status.StartTime == nil || status.CompletionTime == nil || len(status.TestGroups) != 1 || len(status.Conditions) != 1 || status.StartTime.Format(time.RFC3339) != "2026-01-01T00:00:30Z" || status.CompletionTime.Format(time.RFC3339) != "2026-01-01T00:05:00Z" {
		t.Fatal("complete status missing")
	}
	group := status.TestGroups[0]
	if group.Phase != "Succeeded" || group.Name != "diagnostics" || group.Provider != "k8s-job-provider" || !slices.Equal(group.Nodes, []string{"node-a"}) || !slices.Equal(group.Tests, []string{"dcgm-diag-test"}) || len(group.Attempts) != 2 {
		t.Fatal("group/provider/membership/retries lost")
	}
	first, last := group.Attempts[0], group.Attempts[1]
	if first.StartTime == nil || first.EndTime == nil || last.StartTime == nil || last.EndTime == nil {
		t.Fatal("attempt timestamps lost")
	}
	if first.ObjectName != "constructed-job-0" || first.Phase != "Failed" || first.FailureReason != "TestTimeout" || !slices.Equal(first.FailedNodes, []string{"node-a"}) || first.StartTime.Format(time.RFC3339) != "2026-01-01T00:01:00Z" || first.EndTime.Format(time.RFC3339) != "2026-01-01T00:02:00Z" || last.ObjectName != "constructed-job-1" || last.Phase != "Succeeded" || last.StartTime.Format(time.RFC3339) != "2026-01-01T00:03:00Z" || last.EndTime.Format(time.RFC3339) != "2026-01-01T00:04:00Z" {
		t.Fatal("failed attempt erased by later producer success")
	}
	if status.Skipped == nil || !slices.Equal(status.Skipped.Nodes, []string{"node-b"}) || !slices.Equal(status.Skipped.Tests, []string{"bandwidth"}) || status.Conditions[0].Reason != "ValidationSucceeded" || status.Conditions[0].ObservedGeneration == nil || *status.Conditions[0].ObservedGeneration != 1 {
		t.Fatal("skips/conditions lost")
	}
	condition := status.Conditions[0]
	if condition.Type != "Complete" || condition.Status != "True" || condition.Message == nil || *condition.Message != "Constructed producer judgment; one node and test skipped" || condition.LastTransitionTime.Format(time.RFC3339) != "2026-01-01T00:05:00Z" {
		t.Fatal("condition fields lost")
	}
	if got.Coverage != "unverified" || got.Authorization != "unavailable" || got.OriginProof != "unavailable" || got.DeviceAssociation != "unavailable" || got.ActiveConfiguration != "unavailable" || got.ProviderObjects != "unavailable" || got.OutputEvidence != "unavailable" || !slices.Contains(got.WithheldReasons, "producer_reported_skips") {
		t.Fatal("producer success promoted into physical validation")
	}
}

func TestHealthCapacityNVSentinelUnknownStates(t *testing.T) {
	for _, state := range []string{"no status", "empty status", "Pending", "Running", "Succeeded", "Failed", "default tests", "missing attempts"} {
		t.Run(state, func(t *testing.T) {
			request := nvSentinelFixtureRequest(t)
			switch state {
			case "no status":
				request.Status = nil
			case "empty status":
				request.Status = &nvSentinelStatus{}
			case "default tests":
				request.Spec.Tests = nil
			case "missing attempts":
				request.Status.TestGroups[0].Attempts = nil
			default:
				request.Status = &nvSentinelStatus{Phase: state}
			}
			raw, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			got, err := readNVSentinel(raw, "")
			if err != nil {
				t.Fatal(err)
			}
			if got.Coverage != "unverified" || got.Authorization != "unavailable" {
				t.Fatal("unknown state implied authority/coverage")
			}
			switch state {
			case "no status":
				if got.Request.Status != nil || got.StatusDisposition != "unobserved" || got.ProducerPhase != "unobserved" {
					t.Fatal("absent status conflated with pending/failed")
				}
			case "empty status":
				if got.Request.Status == nil || got.StatusDisposition != "producer_reported" || got.ProducerPhase != "unreported" {
					t.Fatal("empty status conflated with absent")
				}
			case "default tests":
				if got.TestSelection != "defaults_unresolved" || !slices.Contains(got.WithheldReasons, "default_tests_unresolved") || len(got.Request.Spec.Tests) != 0 {
					t.Fatal("chart defaults invented")
				}
			case "missing attempts":
				if !slices.Contains(got.WithheldReasons, "attempts_unavailable") || got.ProducerPhase != "Succeeded" {
					t.Fatal("missing attempts promoted or producer state erased")
				}
			default:
				if got.ProducerPhase != state || got.StatusDisposition != "producer_reported" {
					t.Fatal("phase lost")
				}
			}
		})
	}
}

func TestHealthCapacityNVSentinelCommandRefusals(t *testing.T) {
	args, _, _, _, _ := commandHealthFiles(t)
	raw := nvSentinelFixture(t)
	hash := sha256.Sum256(raw)
	baseHash := hex.EncodeToString(hash[:])
	cases := []struct {
		name     string
		raw      []byte
		expected string
	}{
		{"wrong digest", raw, strings.Repeat("a", 64)},
		{"invalid digest", raw, "not-a-digest"},
		{"byte changed old digest", append(append([]byte{}, raw...), byte(' ')), baseHash},
		{"wrong api", bytes.Replace(raw, []byte("nvsentinel.nvidia.com/v1alpha1"), []byte("nvsentinel.nvidia.com/v9"), 1), ""},
		{"wrong kind", bytes.Replace(raw, []byte(`"ValidationRequest"`), []byte(`"ValidationResult"`), 1), ""},
		{"malformed", []byte("{"), ""},
		{"invalid UTF8", bytes.Replace(raw, []byte("node-a"), []byte{0xff}, 1), ""},
		{"duplicate", bytes.Replace(raw, []byte(`"kind":`), []byte(`"kind":"ValidationRequest","kind":`), 1), ""},
		{"alias duplicate", bytes.Replace(raw, []byte(`"kind":`), []byte(`"Kind":"ValidationRequest","kind":`), 1), ""},
		{"unicode alias duplicate", bytes.Replace(raw, []byte(`"status":`), []byte(`"ſtatus":{},"status":`), 1), ""},
		{"trailing", append(append([]byte{}, raw...), []byte(" {}")...), ""},
		{"oversize", append(append([]byte{}, raw...), bytes.Repeat([]byte{' '}, 8<<20)...), ""},
		{"deep", bytes.Replace(raw, []byte(`"metadata": {`), []byte(`"metadata": {"deep":`+strings.Repeat("[", 33)+"0"+strings.Repeat("]", 33)+`,`), 1), ""},
		{"record overflow", bytes.Replace(raw, []byte(`"metadata": {`), []byte(`"metadata": {"records":[`+strings.Repeat("0,", 4096)+`0],`), 1), ""},
		{"string overflow", bytes.Replace(raw, []byte(`"metadata": {`), []byte(`"metadata": {"value":"`+strings.Repeat("x", 32769)+`",`), 1), ""},
		{"ADR-only field", bytes.Replace(raw, []byte(`{"name": "node-a"}`), []byte(`{"name":"node-a","GPUUUID":"GPU-fake","exclusiveAccess":true}`), 1), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "request.json")
			if err := os.WriteFile(path, tc.raw, 0600); err != nil {
				t.Fatal(err)
			}
			commandArgs := append(append([]string{}, args...), "-nvsentinel", path)
			if tc.expected != "" {
				commandArgs = append(commandArgs, "-nvsentinel-sha256", tc.expected)
			}
			var output bytes.Buffer
			err := runHealthCapacity(context.Background(), commandArgs, &output)
			t.Logf("command refusal: %v; stdout: %q", err, output.String())
			if err == nil || output.Len() != 0 {
				t.Fatal("invalid import composed/emitted command output")
			}
		})
	}
}

func TestHealthCapacityNVSentinelTimeAndStateRefusals(t *testing.T) {
	args, _, _, _, _ := commandHealthFiles(t)
	cases := []struct {
		name   string
		change func(*nvSentinelRequest)
	}{
		{"reversed overall", func(r *nvSentinelRequest) { v := r.Status.StartTime.Add(-time.Second); r.Status.CompletionTime = &v }},
		{"retry overlap", func(r *nvSentinelRequest) {
			v := r.Status.TestGroups[0].Attempts[0].EndTime.Add(-time.Second)
			r.Status.TestGroups[0].Attempts[1].StartTime = &v
		}},
		{"reversed attempt", func(r *nvSentinelRequest) {
			a := &r.Status.TestGroups[0].Attempts[0]
			v := a.StartTime.Add(-time.Second)
			a.EndTime = &v
		}},
		{"attempt before request", func(r *nvSentinelRequest) {
			v := r.Status.StartTime.Add(-time.Second)
			r.Status.TestGroups[0].Attempts[0].StartTime = &v
		}},
		{"attempt after request", func(r *nvSentinelRequest) {
			v := r.Status.CompletionTime.Add(time.Second)
			r.Status.TestGroups[0].Attempts[1].EndTime = &v
		}},
		{"nonterminal completion", func(r *nvSentinelRequest) { r.Status.Phase = "Running" }},
		{"nonterminal attempt end", func(r *nvSentinelRequest) { r.Status.TestGroups[0].Attempts[1].Phase = "Running" }},
		{"unknown phase", func(r *nvSentinelRequest) { r.Status.Phase = "Healthy"; r.Status.CompletionTime = nil }},
		{"unknown attempt phase", func(r *nvSentinelRequest) { r.Status.TestGroups[0].Attempts[0].Phase = "Validated" }},
		{"unknown failure reason", func(r *nvSentinelRequest) { r.Status.TestGroups[0].Attempts[0].FailureReason = "NotAReason" }},
		{"success failure contradiction", func(r *nvSentinelRequest) { r.Status.TestGroups[0].Attempts[0].Phase = "Succeeded" }},
		{"duplicate requested node", func(r *nvSentinelRequest) { r.Spec.Nodes = append(r.Spec.Nodes, r.Spec.Nodes[0]) }},
		{"duplicate condition", func(r *nvSentinelRequest) { r.Status.Conditions = append(r.Status.Conditions, r.Status.Conditions[0]) }},
		{"condition enum", func(r *nvSentinelRequest) { r.Status.Conditions[0].Status = "Success" }},
		{"condition negative generation", func(r *nvSentinelRequest) { v := int64(-1); r.Status.Conditions[0].ObservedGeneration = &v }},
		{"missing condition message", func(r *nvSentinelRequest) { r.Status.Conditions[0].Message = nil }},
		{"missing nodes", func(r *nvSentinelRequest) { r.Spec.Nodes = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := nvSentinelFixtureRequest(t)
			tc.change(&request)
			raw, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "request.json")
			if err := os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			var output bytes.Buffer
			err = runHealthCapacity(context.Background(), append(append([]string{}, args...), "-nvsentinel", path), &output)
			t.Logf("command refusal: %v; stdout: %q", err, output.String())
			if err == nil || output.Len() != 0 {
				t.Fatal("inconsistent producer state composed")
			}
		})
	}
}
