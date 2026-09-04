package cublasltx

import (
	"strings"
	"testing"
)

func TestValidateMatMulBuffers(t *testing.T) {
	tests := []struct {
		name    string
		aLen    int
		m, k    int
		bLen, n int
		dLen    int
		want    string
	}{
		{name: "valid", aLen: 6, m: 2, k: 3, bLen: 12, n: 4, dLen: 8},
		{name: "zero dimension", m: 0, k: 3, n: 4, want: "dimensions must be positive"},
		{name: "short A", aLen: 5, m: 2, k: 3, bLen: 12, n: 4, dLen: 8, want: "matrix A length 5 is smaller than 6"},
		{name: "short B", aLen: 6, m: 2, k: 3, bLen: 11, n: 4, dLen: 8, want: "matrix B length 11 is smaller than 12"},
		{name: "short D", aLen: 6, m: 2, k: 3, bLen: 12, n: 4, dLen: 7, want: "matrix D length 7 is smaller than 8"},
		{name: "overflow", aLen: 1, m: int(^uint(0) >> 1), k: 2, bLen: 1, n: 2, dLen: 1, want: "dimensions overflow"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMatMulLengths(tt.aLen, tt.m, tt.k, tt.bLen, tt.n, tt.dLen)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateMatMulBuffers: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateMatMulBuffers error = %v, want substring %q", err, tt.want)
			}
		})
	}
}
