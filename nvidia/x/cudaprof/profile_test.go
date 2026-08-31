package cudaprof_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"runtime/pprof"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestProfileSession(t *testing.T) {
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)

	ctx := pprof.WithLabels(context.Background(), pprof.Labels(
		"model_layer", "transformer_block_0",
		"op", "gemm_forward",
	))

	session.RecordSample(ctx, "GemmKernel", "gemm.cu", 42, 0x7fff00001040, 100)
	session.RecordSample(ctx, "SoftmaxKernel", "softmax.cu", 18, 0x7fff00002080, 50)

	if err := session.Close(); err != nil {
		t.Fatalf("session.Close failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty profile buffer")
	}

	// Decompress and verify
	gr, err := gzip.NewReader(&buf)
	if err != nil {
		t.Fatalf("gzip reader failed: %v", err)
	}
	defer gr.Close()

	decompressed, err := io.ReadAll(gr)
	if err != nil {
		t.Fatalf("decompression failed: %v", err)
	}

	if len(decompressed) == 0 {
		t.Fatal("expected non-empty decompressed profile payload")
	}
}
