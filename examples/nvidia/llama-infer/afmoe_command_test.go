package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAfmoeCommandRejectsIncompleteOptions(t *testing.T) {
	for _, opt := range []afmoeRunOptions{{}, {Model: "unused", TokenIDs: "1", Steps: 128, Context: 128}, {Model: "unused", TokenIDs: "1", Steps: 128, Context: 131072, Scatter: "fp32-then-bf16"}} {
		if err := runAfmoeMini(opt); err == nil {
			t.Fatal("unprepared command accepted")
		}
	}
}

func TestAfmoeDefaultGateBeforeCUDA(t *testing.T) {
	old := *modelPath
	defer func() { *modelPath = old }()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), afmoeMiniFixture(t), 0600); err != nil {
		t.Fatal(err)
	}
	*modelPath = dir
	t.Setenv("LLAMA_AFMOE_NATIVE_ENABLE", "")
	routed, err := routeAfmoeFromFlags()
	if !routed || err == nil || !strings.Contains(err.Error(), "gated") {
		t.Fatalf("default AFMoE guard routed=%v error=%v", routed, err)
	}
}
