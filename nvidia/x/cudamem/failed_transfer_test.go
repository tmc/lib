package cudamem

import "testing"

func TestFailedTransferCleanup(t *testing.T) {
	for _, attempted := range []bool{false, true} {
		released := 0
		var cleanup func()
		failedTransferCleanup(attempted, func(fn func()) { cleanup = fn }, func() { released++ })
		if attempted {
			if released != 0 || cleanup == nil {
				t.Fatal("attempted enqueue released before stream completion")
			}
			cleanup()
		} else if cleanup != nil {
			t.Fatal("pre-enqueue failure retained a stream cleanup")
		}
		if released != 1 {
			t.Fatalf("attempted=%v releases=%d", attempted, released)
		}
	}
}
