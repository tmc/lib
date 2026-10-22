package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestStreamingCUDARefusal(t *testing.T) {
	for _, o := range []streamingOptions{
		{enabled: true},
		{enabled: true, autoCache: true},
	} {
		if err := o.validate("cuda"); err == nil || !strings.Contains(err.Error(), "explicit budget") {
			t.Fatalf("CUDA streaming silently accepted: %v", err)
		}
	}
	if err := (streamingOptions{enabled: true, budget: "6", cold: true}).validate("cuda"); err != nil {
		t.Fatal(err)
	}
}

func TestStreamingCPUParseRefusal(t *testing.T) {
	_, err := parseConfig([]string{"-model", "not-opened", "-backend", "cpu", "-ssd-streaming", "-prompt", "x"}, strings.NewReader(""), new(bytes.Buffer))
	if err == nil || !strings.Contains(err.Error(), "ssd-streaming backend") {
		t.Fatalf("unsupported streaming reached execution: %v", err)
	}
}
