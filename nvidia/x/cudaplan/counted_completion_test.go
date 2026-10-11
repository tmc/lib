package cudaplan

import (
	"context"
	"errors"
	"testing"
)

func TestCountedOutcomeWaitDrainsBeforeReleasing(t *testing.T) {
	outcomeErr := errors.New("worker rejected completion counts")
	drainErr := context.Canceled
	var order []string
	c := newHostCompletion(func(context.Context) error { order = append(order, "drain"); return drainErr }, nil, func() { order = append(order, "release") })
	c.state.outcome = func() error { order = append(order, "outcome"); return outcomeErr }
	if err := c.Wait(context.Background()); !errors.Is(err, context.Canceled) || c.Done() || len(order) != 1 {
		t.Fatalf("failed drain released state: %v %v", err, order)
	}
	drainErr = nil
	if err := c.Wait(nil); !errors.Is(err, outcomeErr) || !c.Done() {
		t.Fatalf("worker error lost after successful drain: %v", err)
	}
	if len(order) != 4 || order[0] != "drain" || order[1] != "drain" || order[2] != "outcome" || order[3] != "release" {
		t.Fatalf("unsafe drain/outcome/release order %v", order)
	}
	if err := c.Wait(nil); !errors.Is(err, outcomeErr) || len(order) != 4 {
		t.Fatalf("repeated wait changed outcome or released twice: %v %v", err, order)
	}
}
