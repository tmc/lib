package main

import (
	"context"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// syncStream waits for the default stream inside a span. Kernel launches
// are asynchronous, so this wait is where the encoded work is actually
// paid for, and everything between two of these spans is host cost —
// the same reading the Metal lane's batch.wait span gives.
func syncStream(s *cuda.Stream) error {
	defer beginOp("stream.sync").end()
	return s.Synchronize(context.Background())
}
