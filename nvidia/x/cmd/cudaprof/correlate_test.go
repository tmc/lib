package main

import "testing"

func TestCorrelateNegativeLimits(t *testing.T) {
	for _, flag := range []string{"-max-records=-1", "-journal-bytes=-1"} {
		if err := runCorrelateCommand([]string{flag, "missing.pprof"}); err == nil || err.Error() != "correlate: journal limits must be nonnegative" {
			t.Fatalf("%s: %v", flag, err)
		}
	}
}
