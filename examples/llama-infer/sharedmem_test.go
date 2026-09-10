package main

import (
	"strings"
	"testing"
)

func TestDynamicSharedLimit(t *testing.T) {
	tests := []struct {
		name                    string
		optin, perBlock, static int
		want                    int
	}{
		{"default only", 0, 49152, 0, 49152},
		{"optin raises", 232448, 49152, 0, 232448},
		{"static subtracts", 232448, 49152, 1024, 231424},
		{"optin below default is ignored", 16384, 49152, 0, 49152},
		{"static exceeds ceiling", 0, 49152, 65536, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dynamicSharedLimit(tt.optin, tt.perBlock, tt.static); got != tt.want {
				t.Fatalf("dynamicSharedLimit(%d, %d, %d) = %d, want %d", tt.optin, tt.perBlock, tt.static, got, tt.want)
			}
		})
	}
}

func TestSharedMemFits(t *testing.T) {
	tests := []struct {
		name         string
		bytes, limit int
		wantErr      string
	}{
		{"zero", 0, 49152, ""},
		{"at limit", 49152, 49152, ""},
		{"one over", 49153, 49152, "49153 bytes exceeds the 49152"},
		{"dense decode at 12289 tokens", 4 * 12289, 49152, "49156 bytes exceeds"},
		{"negative", -4, 49152, "is negative"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := sharedMemFits(tt.bytes, tt.limit)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("sharedMemFits(%d, %d) = %v, want nil", tt.bytes, tt.limit, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("sharedMemFits(%d, %d) = %v, want error containing %q", tt.bytes, tt.limit, err, tt.wantErr)
			}
		})
	}
}
