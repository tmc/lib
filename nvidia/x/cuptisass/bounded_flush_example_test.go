package cuptisass

import "fmt"

func ExampleSession_FlushBounded() {
	limits := FlushLimits{MaxRecords: 1024, MaxInstances: 128, MaxBytes: 8 << 20, MaxFunctionNameBytes: 256}
	// Use a live session returned by NewSession.
	collect := func(session *Session) ([]Record, error) {
		return session.FlushBounded(limits)
	}
	_ = collect
	fmt.Println(limits.MaxRecords, limits.MaxFunctionNameBytes)
	// Output: 1024 256
}
