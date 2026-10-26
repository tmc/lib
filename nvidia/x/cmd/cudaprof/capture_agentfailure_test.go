package main

import (
	"strings"
	"testing"
)

func TestAgentFailureMessage(t *testing.T) {
	tests := []struct {
		failure, detail string
		want            []string
		not             []string
	}{
		{"cupti_unavailable", "", []string{"cupti_unavailable"}, []string{":", "cudadoctor"}},
		{"cupti_unavailable", "CUPTI runtime/header version mismatch",
			[]string{"cupti_unavailable: CUPTI runtime/header version mismatch"}, []string{"cudadoctor", "RestrictProfiling"}},
		{"cupti_unavailable", "cuptiActivityRegisterCallbacks: CUPTI_ERROR_UNKNOWN",
			[]string{"cupti_unavailable: cuptiActivityRegisterCallbacks: CUPTI_ERROR_UNKNOWN (", "could not reach the driver or device", "cudadoctor -health"}, nil},
		{"counters_unavailable", "cuptiProfilerInitialize: CUPTI_ERROR_INSUFFICIENT_PRIVILEGES",
			[]string{"NVreg_RestrictProfilingToAdminUsers=0", "ERR_NVGPUCTRPERM"}, []string{"cudadoctor"}},
	}
	for _, tt := range tests {
		got := agentFailureMessage(tt.failure, tt.detail)
		for _, w := range tt.want {
			if !strings.Contains(got, w) {
				t.Errorf("agentFailureMessage(%q, %q) = %q, want %q", tt.failure, tt.detail, got, w)
			}
		}
		for _, w := range tt.not {
			if strings.Contains(got, w) {
				t.Errorf("agentFailureMessage(%q, %q) = %q, must not contain %q", tt.failure, tt.detail, got, w)
			}
		}
	}
}
