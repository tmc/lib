//go:build darwin && arm64

package main

import (
	"strings"
	"testing"
)

func TestMetalServerRejectsCUDAOptions(t *testing.T) {
	for _, test := range []struct {
		name    string
		ballast float64
		warm    bool
	}{{"ballast", 1, false}, {"warm", 0, true}} {
		t.Run(test.name, func(t *testing.T) {
			backend, err := openServerBackend("nonexistent.gguf", 2048, test.ballast, test.warm)
			if backend != nil || err == nil || !strings.Contains(err.Error(), "requires CUDA") {
				t.Fatalf("openServerBackend = %v, %v; want CUDA-only option rejection before model load", backend, err)
			}
		})
	}
}
