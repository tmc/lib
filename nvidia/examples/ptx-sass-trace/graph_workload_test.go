package main

import (
	"context"
	"testing"
)

func TestGraphWorkloadValidation(t *testing.T) {
	steps := []PipelineStep{
		{Name: "step1"},
	}

	_, _, err := runWorkloadInGraph(context.Background(), nil, nil, "invalid_mode", 1, steps)
	if err == nil {
		t.Error("expected error for invalid graph mode, got nil")
	}

	_, _, err = runWorkloadInGraph(context.Background(), nil, nil, "capture", 1, steps)
	if err == nil {
		t.Error("expected error for nil stream, got nil")
	}

	_, _, err = runWorkloadInGraph(context.Background(), nil, nil, "capture", 1, nil)
	if err == nil {
		t.Error("expected error for empty steps, got nil")
	}
}
