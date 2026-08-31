package cudagraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudatest"
)



func TestGraphCapture(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	g, err := cudagraph.Capture(stream, func() {
		// Captured operations on stream
	})
	if err != nil {
		t.Fatalf("Capture failed: %v", err)
	}
	defer g.Destroy()

	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	defer exec.Destroy()

	if err := exec.Launch(stream); err != nil {
		t.Fatalf("Launch failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
}

func TestBucketManager(t *testing.T) {
	bm := cudagraph.NewBucketManager([]int{128, 256, 512, 1024}, func(s *cuda.Stream, dim int) (*cudagraph.Graph, error) {
		return cudagraph.New()
	})
	defer bm.Close()

	if bm == nil {
		t.Fatal("expected non-nil bucket manager")
	}
}

