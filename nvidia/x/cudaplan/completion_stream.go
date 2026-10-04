package cudaplan

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// NewStreamCompletion adopts releases for work already queued on stream.
// Wait synchronizes the whole stream; it does not record an event or submit
// work. Cancellation or a drain error retains every release for a later Wait.
// After a successful drain, releases run exactly once. ExecutedSteps is empty
// because this constructor has no plan dispatch receipts.
//
// The caller must keep the device, stream and queued resources live until Wait
// succeeds, and exclude concurrent Close or further use of borrowed resources.
// A release must not call methods on this completion. The callback list is
// copied; the callbacks themselves remain the caller's responsibility.
// On error no callback is adopted or called, so the caller still owns the
// resources and their drain responsibility.
func NewStreamCompletion(stream *cuda.Stream, releases ...func()) (*Completion, error) {
	if stream == nil || stream.Device() == nil {
		return nil, fmt.Errorf("cudaplan: stream has no device")
	}
	if err := stream.WithHandle(func(driver.Stream) error { return nil }); err != nil {
		return nil, fmt.Errorf("cudaplan: adopt stream: %w", err)
	}
	owned := append([]func(){}, releases...)
	return newHostCompletion(stream.Synchronize, nil, owned...), nil
}
