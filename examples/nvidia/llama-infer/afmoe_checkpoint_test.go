package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestAFMoECheckpointHeaders reads only original checkpoint metadata. It
// needs no CUDA driver and never allocates device memory or reads tensors.
func TestAFMoECheckpointHeaders(t *testing.T) {
	path := os.Getenv("LLAMA_AFMOE_TEST_MODEL")
	if path == "" {
		t.Skip("set LLAMA_AFMOE_TEST_MODEL to the admitted Mini snapshot for CPU-only header validation")
	}
	if !filepath.IsAbs(path) {
		t.Fatal("LLAMA_AFMOE_TEST_MODEL must be an absolute snapshot path")
	}
	canonical, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatal(err)
	}
	ck, cfg, err := openAfmoeCheckpoint(canonical)
	if ck != nil {
		t.Cleanup(func() {
			if err := ck.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, spec := range cfg.tensors() {
		info, ok := ck.Info(spec.Name)
		if !ok {
			t.Fatalf("missing %s after header preflight", spec.Name)
		}
		var n int64 = 1
		for _, dim := range spec.Shape {
			n *= int64(dim)
		}
		if info.Size != n*2 {
			t.Fatalf("%s size=%d want%d", spec.Name, info.Size, n*2)
		}
		total += info.Size
	}
	if total != 52247948800 || len(cfg.tensors()) != 12031 {
		t.Fatalf("contract tensors=%d sourcebytes=%d", len(cfg.tensors()), total)
	}
	t.Logf("AFMOE_HEADER_CONTRACT model=%q tensors=12031 source_bytes=%d payload_reads=0 device_allocations=0", canonical, total)
}
