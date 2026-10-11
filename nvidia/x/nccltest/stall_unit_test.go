package nccltest

import (
	"testing"
	"time"
)

func TestDelayOptions(t *testing.T) {
	opts := Options{
		DelayRank: 1,
		PeerDelay: 10 * time.Millisecond,
	}

	if opts.DelayRank != 1 {
		t.Errorf("DelayRank = %d, want 1", opts.DelayRank)
	}
	if opts.PeerDelay != 10*time.Millisecond {
		t.Errorf("PeerDelay = %v, want 10ms", opts.PeerDelay)
	}

	res := Result{
		DelayRank: 1,
		PeerDelay: 10 * time.Millisecond,
		RankTimes: []time.Duration{time.Millisecond, 11 * time.Millisecond},
	}
	if len(res.RankTimes) != 2 || res.RankTimes[1] <= res.RankTimes[0] {
		t.Errorf("invalid RankTimes: %v", res.RankTimes)
	}
}
