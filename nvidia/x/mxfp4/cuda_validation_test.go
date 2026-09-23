package mxfp4

import (
	"strings"
	"testing"
)

func TestValidateLinear(t *testing.T) {
	tests := []struct {
		name                             string
		x, packed, scales, out, in, rows int
		group                            int
		want                             string
	}{
		{"valid", 32, 32, 2, 2, 32, 2, 32, ""},
		{"odd group", 32, 32, 2, 2, 32, 2, 3, "invalid linear dimensions"},
		{"partial group", 30, 30, 2, 2, 30, 2, 16, "invalid linear dimensions"},
		{"short packed", 32, 31, 2, 2, 32, 2, 32, "slice dimensions"},
		{"short scales", 32, 32, 1, 2, 32, 2, 32, "slice dimensions"},
		{"wrong output", 32, 32, 2, 1, 32, 2, 32, "slice dimensions"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateLinear(test.x, test.packed, test.scales, test.out, test.in, test.rows, test.group)
			if test.want == "" && err != nil {
				t.Fatal(err)
			}
			if test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestValidateDequantize(t *testing.T) {
	tests := []struct {
		name                          string
		packed, scales, out, n, group int
		want                          string
	}{
		{"valid", 16, 1, 32, 32, 32, ""},
		{"odd elements", 17, 2, 33, 33, 3, "invalid dequantize dimensions"},
		{"partial group", 24, 2, 48, 48, 32, "invalid dequantize dimensions"},
		{"short packed", 15, 1, 32, 32, 32, "slice dimensions"},
		{"short scales", 16, 0, 32, 32, 32, "slice dimensions"},
		{"wrong output", 16, 1, 31, 32, 32, "slice dimensions"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDequantize(test.packed, test.scales, test.out, test.n, test.group)
			if test.want == "" && err != nil {
				t.Fatal(err)
			}
			if test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestNilEngine(t *testing.T) {
	var engine *Engine
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := NewEngine(nil); err == nil {
		t.Fatal("NewEngine(nil) succeeded")
	}
}
