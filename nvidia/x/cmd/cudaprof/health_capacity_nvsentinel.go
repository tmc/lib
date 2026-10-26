package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"
)

const nvSentinelCommit = "a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5"

// These private wire types follow the shipped v1.26 types and CRD, not ADR049.
// Metadata remains a complete bounded JSON object; Kubernetes metadata is not
// an authorization record, artifact hash, device identity or global join key.
type nvSentinelRequest struct {
	APIVersion string                     `json:"apiVersion"`
	Kind       string                     `json:"kind"`
	Metadata   map[string]json.RawMessage `json:"metadata,omitempty"`
	Spec       nvSentinelSpec             `json:"spec"`
	Status     *nvSentinelStatus          `json:"status,omitempty"`
}
type nvSentinelSpec struct {
	Nodes []nvSentinelNode `json:"nodes"`
	Tests []string         `json:"tests,omitempty"`
}
type nvSentinelNode struct {
	Name string `json:"name"`
}
type nvSentinelStatus struct {
	Phase          string                `json:"phase,omitempty"`
	TestGroups     []nvSentinelGroup     `json:"testGroups,omitempty"`
	Skipped        *nvSentinelSkipped    `json:"skipped,omitempty"`
	StartTime      *time.Time            `json:"startTime,omitempty"`
	CompletionTime *time.Time            `json:"completionTime,omitempty"`
	Conditions     []nvSentinelCondition `json:"conditions,omitempty"`
}
type nvSentinelGroup struct {
	Name     string              `json:"name"`
	Provider string              `json:"provider"`
	Tests    []string            `json:"tests,omitempty"`
	Nodes    []string            `json:"nodes,omitempty"`
	Phase    string              `json:"phase,omitempty"`
	Attempts []nvSentinelAttempt `json:"attempts,omitempty"`
}
type nvSentinelAttempt struct {
	ObjectName    string     `json:"objectName,omitempty"`
	Phase         string     `json:"phase,omitempty"`
	FailureReason string     `json:"failureReason,omitempty"`
	FailedNodes   []string   `json:"failedNodes,omitempty"`
	StartTime     *time.Time `json:"startTime,omitempty"`
	EndTime       *time.Time `json:"endTime,omitempty"`
}
type nvSentinelSkipped struct {
	Nodes []string `json:"nodes,omitempty"`
	Tests []string `json:"tests,omitempty"`
}
type nvSentinelCondition struct {
	Type               string    `json:"type"`
	Status             string    `json:"status"`
	ObservedGeneration *int64    `json:"observedGeneration,omitempty"`
	LastTransitionTime time.Time `json:"lastTransitionTime"`
	Reason             string    `json:"reason"`
	Message            *string   `json:"message"`
}

type nvSentinelImport struct {
	Producer              string            `json:"producer"`
	Release               string            `json:"release"`
	UpstreamCommit        string            `json:"upstream_commit"`
	SHA256                string            `json:"sha256"`
	ExpectedDigestChecked bool              `json:"expected_digest_checked"`
	Request               nvSentinelRequest `json:"request"`
	StatusDisposition     string            `json:"status_disposition"`
	ProducerPhase         string            `json:"producer_phase"`
	TestSelection         string            `json:"test_selection"`
	OriginProof           string            `json:"origin_proof"`
	Authorization         string            `json:"authorization"`
	DeviceAssociation     string            `json:"device_association"`
	ActiveConfiguration   string            `json:"active_configuration"`
	ProviderObjects       string            `json:"provider_objects"`
	OutputEvidence        string            `json:"output_evidence"`
	Coverage              string            `json:"coverage"`
	WithheldReasons       []string          `json:"withheld_reasons"`
}

func readNVSentinel(raw []byte, expected string) (nvSentinelImport, error) {
	out := nvSentinelImport{Producer: "NVSentinel", Release: "v1.26.0", UpstreamCommit: nvSentinelCommit, SHA256: healthCapacityDigest(raw), StatusDisposition: "unobserved", ProducerPhase: "unobserved", TestSelection: "defaults_unresolved", OriginProof: "unavailable", Authorization: "unavailable", DeviceAssociation: "unavailable", ActiveConfiguration: "unavailable", ProviderObjects: "unavailable", OutputEvidence: "unavailable", Coverage: "unverified",
		WithheldReasons: []string{"producer_origin_unverified", "independent_authority_unavailable", "node_device_allocation_binding_unavailable", "active_configuration_unavailable", "raw_provider_objects_unavailable", "raw_test_output_unavailable"}}
	if len(raw) == 0 || len(raw) > 8<<20 {
		return out, errors.New("NVSentinel request byte limit exceeded")
	}
	if expected != "" {
		digest, err := hex.DecodeString(expected)
		if err != nil || len(digest) != 32 || expected != out.SHA256 {
			return out, errors.New("NVSentinel raw SHA256 mismatch or invalid digest")
		}
		out.ExpectedDigestChecked = true
	}
	if err := decodeHealthCapacityJSON(raw, &out.Request); err != nil {
		return out, fmt.Errorf("NVSentinel request: %w", err)
	}
	request := &out.Request
	if request.APIVersion != "nvsentinel.nvidia.com/v1alpha1" || request.Kind != "ValidationRequest" {
		return out, errors.New("unsupported NVSentinel apiVersion or kind")
	}
	if request.Spec.Nodes == nil || len(request.Spec.Nodes) > 1024 || len(request.Spec.Tests) > 1024 {
		return out, errors.New("NVSentinel request node/test bounds")
	}
	nodes := make(map[string]bool)
	for _, node := range request.Spec.Nodes {
		if node.Name == "" || nodes[node.Name] {
			return out, errors.New("NVSentinel missing or duplicate requested node")
		}
		nodes[node.Name] = true
	}
	if err := nvSentinelNames(request.Spec.Tests); err != nil {
		return out, err
	}
	if len(request.Spec.Tests) > 0 {
		out.TestSelection = "explicit_request"
	} else {
		out.WithheldReasons = append(out.WithheldReasons, "default_tests_unresolved")
	}
	status := request.Status
	if status == nil {
		out.WithheldReasons = append(out.WithheldReasons, "status_unobserved")
		return out, nil
	}
	out.StatusDisposition = "producer_reported"
	out.ProducerPhase = status.Phase
	if out.ProducerPhase == "" {
		out.ProducerPhase = "unreported"
		out.WithheldReasons = append(out.WithheldReasons, "phase_unreported")
	}
	if !nvSentinelPhase(status.Phase) {
		return out, errors.New("NVSentinel invalid overall phase")
	}
	if len(status.TestGroups) > 256 || len(status.Conditions) > 128 {
		return out, errors.New("NVSentinel status record limit exceeded")
	}
	if err := nvSentinelTimes(status.StartTime, status.CompletionTime); err != nil {
		return out, err
	}
	if status.CompletionTime != nil && !nvSentinelTerminal(status.Phase) {
		return out, errors.New("NVSentinel nonterminal request has completion time")
	}
	if status.StartTime == nil || nvSentinelTerminal(status.Phase) && status.CompletionTime == nil {
		out.WithheldReasons = append(out.WithheldReasons, "request_times_incomplete")
	}
	if len(status.TestGroups) == 0 {
		out.WithheldReasons = append(out.WithheldReasons, "test_groups_unavailable")
	}
	seenGroups := make(map[string]bool)
	attempts := 0
	for _, group := range status.TestGroups {
		if group.Name == "" || group.Provider == "" || seenGroups[group.Name] || !nvSentinelPhase(group.Phase) {
			return out, errors.New("NVSentinel invalid or duplicate test group")
		}
		seenGroups[group.Name] = true
		if len(group.Attempts) > 64 {
			return out, errors.New("NVSentinel attempt limit exceeded")
		}
		attempts += len(group.Attempts)
		if attempts > 4096 {
			return out, errors.New("NVSentinel total attempt limit exceeded")
		}
		if err := nvSentinelNames(group.Nodes); err != nil {
			return out, err
		}
		if err := nvSentinelNames(group.Tests); err != nil {
			return out, err
		}
		for _, node := range group.Nodes {
			if !nodes[node] {
				out.WithheldReasons = append(out.WithheldReasons, "group_node_not_requested")
			}
		}
		if len(request.Spec.Tests) > 0 {
			for _, test := range group.Tests {
				if !slices.Contains(request.Spec.Tests, test) {
					out.WithheldReasons = append(out.WithheldReasons, "group_test_not_requested")
				}
			}
		}
		if len(group.Nodes) == 0 || len(group.Tests) == 0 {
			out.WithheldReasons = append(out.WithheldReasons, "group_membership_incomplete")
		}
		if len(group.Attempts) == 0 {
			out.WithheldReasons = append(out.WithheldReasons, "attempts_unavailable")
		}

		if status.Phase == "Succeeded" && group.Phase != "Succeeded" {
			out.WithheldReasons = append(out.WithheldReasons, "request_group_phase_mismatch")
		}
		var previousEnd *time.Time
		for index, attempt := range group.Attempts {
			if index > 0 {
				if previousEnd != nil && attempt.StartTime != nil && attempt.StartTime.Before(*previousEnd) {
					return out, errors.New("NVSentinel retry attempts overlap or move backwards")
				}
				if previousEnd == nil || attempt.StartTime == nil {
					out.WithheldReasons = append(out.WithheldReasons, "attempt_order_unavailable")
				}
			}
			previousEnd = attempt.EndTime

			if !nvSentinelPhase(attempt.Phase) || !nvSentinelFailureReason(attempt.FailureReason) {
				return out, errors.New("NVSentinel invalid attempt phase or failure reason")
			}
			if attempt.Phase != "Failed" && (attempt.FailureReason != "" || len(attempt.FailedNodes) > 0) {
				return out, errors.New("NVSentinel failure fields contradict attempt phase")
			}
			if err := nvSentinelNames(attempt.FailedNodes); err != nil {
				return out, err
			}
			for _, node := range attempt.FailedNodes {
				if !slices.Contains(group.Nodes, node) {
					out.WithheldReasons = append(out.WithheldReasons, "failed_node_not_in_group")
				}
			}
			if err := nvSentinelTimes(attempt.StartTime, attempt.EndTime); err != nil {
				return out, err
			}
			if attempt.EndTime != nil && !nvSentinelTerminal(attempt.Phase) {
				return out, errors.New("NVSentinel nonterminal attempt has end time")
			}
			if status.StartTime != nil && attempt.StartTime != nil && attempt.StartTime.Before(*status.StartTime) || status.CompletionTime != nil && attempt.EndTime != nil && attempt.EndTime.After(*status.CompletionTime) {
				return out, errors.New("NVSentinel attempt outside request interval")
			}
			if attempt.ObjectName == "" || attempt.StartTime == nil || nvSentinelTerminal(attempt.Phase) && attempt.EndTime == nil {
				out.WithheldReasons = append(out.WithheldReasons, "attempt_evidence_incomplete")
			}
			// objectName remains only a provider-resource reference. Its GVK, namespace,
			// UID, controller ownership and actual output are not supplied by this file.
		}
		if len(group.Attempts) > 0 && group.Phase != group.Attempts[len(group.Attempts)-1].Phase {
			out.WithheldReasons = append(out.WithheldReasons, "group_attempt_phase_mismatch")
		}
	}
	if status.Skipped != nil {
		if err := nvSentinelNames(status.Skipped.Nodes); err != nil {
			return out, err
		}
		if err := nvSentinelNames(status.Skipped.Tests); err != nil {
			return out, err
		}
		if len(status.Skipped.Nodes)+len(status.Skipped.Tests) > 0 {
			out.WithheldReasons = append(out.WithheldReasons, "producer_reported_skips")
		}
	}
	conditions := make(map[string]bool)
	for _, condition := range status.Conditions {
		if condition.Type == "" || condition.Reason == "" || condition.Message == nil || condition.LastTransitionTime.IsZero() || conditions[condition.Type] || condition.ObservedGeneration != nil && *condition.ObservedGeneration < 0 {
			return out, errors.New("NVSentinel invalid or duplicate condition")
		}
		conditions[condition.Type] = true
		if condition.Status != "True" && condition.Status != "False" && condition.Status != "Unknown" {
			return out, errors.New("NVSentinel invalid condition status")
		}
	}
	slices.Sort(out.WithheldReasons)
	out.WithheldReasons = slices.Compact(out.WithheldReasons)
	return out, nil
}

func nvSentinelPhase(phase string) bool {
	return phase == "" || phase == "Pending" || phase == "Running" || nvSentinelTerminal(phase)
}
func nvSentinelTerminal(phase string) bool { return phase == "Succeeded" || phase == "Failed" }
func nvSentinelFailureReason(reason string) bool {
	return reason == "" || reason == "TestFailed" || reason == "TestTimeout" || reason == "NodeReadinessViolation" || reason == "NodeDeleted" || reason == "BatchMinimumNotMet"
}
func nvSentinelNames(names []string) error {
	if len(names) > 1024 {
		return errors.New("NVSentinel membership limit exceeded")
	}
	seen := make(map[string]bool)
	for _, name := range names {
		if name == "" || seen[name] {
			return errors.New("NVSentinel empty or duplicate membership")
		}
		seen[name] = true
	}
	return nil
}
func nvSentinelTimes(start, end *time.Time) error {
	if start != nil && start.IsZero() || end != nil && end.IsZero() || start != nil && end != nil && end.Before(*start) {
		return errors.New("NVSentinel inconsistent time interval")
	}
	return nil
}
