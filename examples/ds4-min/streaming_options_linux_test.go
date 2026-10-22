//go:build linux

package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestStreamingCUDAParseAndDispatchRefusal(t *testing.T) {
	for _, backend := range []string{"auto", "cuda"} {
		_, err := parseConfig([]string{"-model", "not-opened", "-backend", backend, "-ssd-streaming", "-prompt", "x"}, strings.NewReader(""), new(bytes.Buffer))
		if err == nil || !strings.Contains(err.Error(), "CUDA execution not implemented") {
			t.Fatalf("backend %s ignored streaming at parse: %v", backend, err)
		}
	}
	session, closer, err := openSessionBackend(config{backend: "cuda", model: "not-opened", streaming: streamingOptions{enabled: true}})
	if session != nil || closer != nil || err == nil || !strings.Contains(err.Error(), "CUDA execution not implemented") {
		t.Fatalf("CUDA dispatch reached constructor: %v", err)
	}
	backend, err := openConfiguredServerBackend("not-opened", 2, 0, false, streamingOptions{cold: true})
	if backend != nil || err == nil || !strings.Contains(err.Error(), "require ssd-streaming") {
		t.Fatalf("server ignored orphan streaming options: %v", err)
	}
}
