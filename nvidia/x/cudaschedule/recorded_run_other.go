//go:build !linux

package cudaschedule

import (
	"context"
	"fmt"
	"time"
)

func runRecordedAttempt(context.Context, uint64, string, RecordedRunOptions, time.Duration, bool) (RunRecord, error) {
	return RunRecord{}, fmt.Errorf("recorded run: Linux custody required")
}
func runRecordedWorkerReady(context.Context) error {
	return fmt.Errorf("recorded helper: Linux custody required")
}
func runRecordedHelperMode(context.Context, string, string, string) error {
	return fmt.Errorf("recorded helper: Linux custody required")
}
