package main

import "testing"

func TestDSparkHistoryPartialAllocationCleanup(t *testing.T) {
	runDSparkPartialAllocationCleanup(t, "history")
}
