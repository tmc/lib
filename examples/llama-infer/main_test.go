package main

import (
	"testing"
)

func TestCrossPlatformLlamaInfer(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	if err := runPlatform(cfg); err != nil {
		t.Fatalf("runPlatform failed: %v", err)
	}
}
