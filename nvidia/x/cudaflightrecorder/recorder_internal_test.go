package cudaflightrecorder

import (
	"errors"
	"testing"
	"time"
)

func TestUncertaintyAfter(t *testing.T) {
	for _, tt := range []struct {
		name    string
		elapsed time.Duration
		want    int64
	}{
		{"now", 0, 5}, {"ten_seconds", 10 * time.Second, 110005},
		{"one_hour", time.Hour, 39600005}, {"negative", -time.Second, 5},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := uncertaintyAfter(5, tt.elapsed); got != tt.want {
				t.Fatalf("uncertainty = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestCalibrateWithDiscardsColdStart pins the warm-up read. CUPTI initializes
// lazily, so bracketing its first call charges the library's cold start to the
// calibration uncertainty; a cold start is milliseconds and a warm read is
// microseconds, so the reported bracket was off by three orders of magnitude.
func TestCalibrateWithDiscardsColdStart(t *testing.T) {
	const coldStart = 40 * time.Millisecond
	calls := 0
	read := func() (uint64, error) {
		calls++
		if calls == 1 {
			time.Sleep(coldStart)
		}
		return uint64(1000 * calls), nil
	}
	_, deviceNs, uncertaintyNs, err := calibrateWith(read)
	if err != nil {
		t.Fatalf("calibrateWith: %v", err)
	}
	if calls != 2 {
		t.Errorf("read calls = %d, want 2 (one discarded warm-up, one bracketed)", calls)
	}
	if deviceNs != 2000 {
		t.Errorf("deviceNs = %d, want 2000 (the second read, not the cold one)", deviceNs)
	}
	if uncertaintyNs >= int64(coldStart/2) {
		t.Errorf("uncertaintyNs = %d, want well under %d: the cold start was charged to the bracket", uncertaintyNs, int64(coldStart/2))
	}
}

func TestCalibrateWithPropagatesWarmUpError(t *testing.T) {
	want := errors.New("boom")
	_, _, _, err := calibrateWith(func() (uint64, error) { return 0, want })
	if !errors.Is(err, want) {
		t.Errorf("err = %v, want %v", err, want)
	}
}
