//go:build darwin && arm64

package main

import (
	"fmt"
	"testing"
)

func oracleComparisonShape(candidate, oracle int) error {
	if candidate <= 0 || oracle <= 0 {
		return fmt.Errorf("empty comparison: candidate %d, oracle %d", candidate, oracle)
	}
	if candidate != oracle {
		return fmt.Errorf("comparison length: candidate %d, oracle %d", candidate, oracle)
	}
	return nil
}

func TestOracleComparisonShape(t *testing.T) {
	for _, tt := range []struct {
		name              string
		candidate, oracle int
		wantError         bool
	}{
		{"complete", 4, 4, false},
		{"truncated candidate", 3, 4, true},
		{"truncated oracle", 4, 3, true},
		{"empty both", 0, 0, true},
		{"empty candidate", 0, 4, true},
		{"empty oracle", 4, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := oracleComparisonShape(tt.candidate, tt.oracle); (err != nil) != tt.wantError {
				t.Fatalf("comparison error = %v, want error %v", err, tt.wantError)
			}
		})
	}
}
