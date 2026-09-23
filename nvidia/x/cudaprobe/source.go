package cudaprobe

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// Track consumes source into tracker until target exit (io.EOF). A decode,
// ordering, cancellation or source error marks the interval incomplete and is
// returned with context. Track does not own or close the source's attachments.
// It retains only the tracker's bounded set of unmatched allocations.
func Track(ctx context.Context, source Source, tracker *Tracker) error {
	if source == nil || tracker == nil || tracker.active == nil {
		return fmt.Errorf("cudaprobe: missing source or tracker")
	}
	for {
		if err := ctx.Err(); err != nil {
			tracker.Gap()
			return err
		}
		capture, err := source.Next(ctx)
		if errors.Is(err, io.EOF) {
			tracker.Exit()
			return nil
		}
		if err != nil {
			tracker.Gap()
			return fmt.Errorf("cudaprobe: read source: %w", err)
		}
		event, err := Decode(capture)
		if err != nil {
			tracker.Gap()
			return err
		}
		if err := tracker.Observe(event); err != nil {
			tracker.Gap()
			return err
		}
	}
}
