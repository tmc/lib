//go:build !linux

package fswatch

import (
	"context"
	"time"
)

type notifier struct{}

func openNotify([]string, bool) (*notifier, error) { return &notifier{}, nil }
func (*notifier) wait(ctx context.Context, interval time.Duration) error {
	return waitPoll(ctx, interval)
}
func (*notifier) close() {}
