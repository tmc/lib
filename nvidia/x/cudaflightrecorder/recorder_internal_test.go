package cudaflightrecorder

import (
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
