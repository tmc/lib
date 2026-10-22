package main

import (
	"os"
	"strings"
	"testing"
)

func TestCUDAStreamingWiringSource(t *testing.T) {
	data, err := os.ReadFile("cuda_session_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	start := strings.Index(s, "func (r *cudaRuntime) evalLayerFFNPost(")
	if start < 0 {
		t.Fatal("FFN body missing")
	}
	end := strings.Index(s[start:], "func (r *cudaRuntime) evalLayerFFNResidual(")
	if end < 0 {
		t.Fatal("FFN body missing")
	}
	body := s[start : start+end]
	streaming := strings.Index(body, "r.evalStreamingExperts(s, index)")
	resident := strings.Index(body, "r.evalQ4Experts(s, index)")
	aligned := strings.Index(body, "r.encodeMatvecIQ2SelectedQ81Aligned(")
	if streaming < 0 || streaming > resident || streaming > aligned {
		t.Fatal("streaming branch follows resident expert read")
	}
	if !strings.Contains(s, "if foundation.streamingExperts == nil {\n\t\tmoeLayers := cudaMoEArtifactLayers()") {
		t.Fatal("streaming constructor reaches routed repack")
	}
	data, err = os.ReadFile("cuda_streaming_experts_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	body = string(data)
	for _, required := range []string{"o.selections.Select(index, ids, false)", "syncStream(r.stream())", "s.expertIDs.CopyToHost(ids32[:])", "s.expertW, s.expertMid, 6, 1", "consumer.launch.down(o.down, s.expertMid, s.expertDown)"} {
		if !strings.Contains(body, required) {
			t.Fatalf("inference wiring missing %q", required)
		}
	}
}
