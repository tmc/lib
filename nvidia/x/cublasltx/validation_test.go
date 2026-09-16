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

func TestValidateLinearBuffers(t *testing.T) {
	tests := []struct {
		name                         string
		xLen, tokens, inFeatures     int
		wLen, outFeatures            int
		yLen                         int
		want                         string
	}{
		{name: "valid", xLen: 6, tokens: 2, inFeatures: 3, wLen: 12, outFeatures: 4, yLen: 8},
		{name: "zero dimension", tokens: 0, inFeatures: 3, outFeatures: 4, want: "dimensions must be positive"},
		{name: "short X", xLen: 5, tokens: 2, inFeatures: 3, wLen: 12, outFeatures: 4, yLen: 8, want: "matrix X length 5 is smaller than 6"},
		{name: "short W", xLen: 6, tokens: 2, inFeatures: 3, wLen: 11, outFeatures: 4, yLen: 8, want: "matrix W length 11 is smaller than 12"},
		{name: "short Y", xLen: 6, tokens: 2, inFeatures: 3, wLen: 12, outFeatures: 4, yLen: 7, want: "matrix Y length 7 is smaller than 8"},
		{name: "overflow", xLen: 1, tokens: int(^uint(0) >> 1), inFeatures: 2, wLen: 1, outFeatures: 2, yLen: 1, want: "dimensions overflow"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateLinearLengths(tt.xLen, tt.tokens, tt.inFeatures, tt.wLen, tt.outFeatures, tt.yLen)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateLinearLengths: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateLinearLengths error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestFloat32ToF16Bits(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{0.0, 0x0000},
		{1.0, 0x3c00},
		{-1.0, 0xbc00},
		{2.0, 0x4000},
	}
	for _, c := range cases {
		got := float32ToF16Bits(c.f)
		if got != c.want {
			t.Errorf("float32ToF16Bits(%v) = 0x%04x, want 0x%04x", c.f, got, c.want)
		}
	}
}

