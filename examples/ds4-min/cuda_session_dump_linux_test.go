//go:build linux

package main

import (
	"os"
	"testing"
)

func TestDumpPositionParsing(t *testing.T) {
	tests := []struct {
		name     string
		env      *string
		want     int
		explicit bool
		wantErr  bool
	}{
		{name: "absent"},
		{name: "zero", env: stringPtr("0"), want: 0, explicit: true},
		{name: "large", env: stringPtr("2049"), want: 2049, explicit: true},
		{name: "negative", env: stringPtr("-1"), wantErr: true},
		{name: "malformed", env: stringPtr("step-2"), wantErr: true},
		{name: "empty", env: stringPtr(""), wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.env == nil {
				if err := os.Unsetenv("DS4_DUMP_POSITION"); err != nil {
					t.Fatal(err)
				}
			} else {
				t.Setenv("DS4_DUMP_POSITION", *test.env)
			}
			got, explicit, err := dumpPosition()
			if test.wantErr {
				if err == nil {
					t.Fatal("dumpPosition returned nil error")
				}
				return
			}
			if err != nil || got != test.want || explicit != test.explicit {
				t.Fatalf("dumpPosition() = (%d, %t, %v), want (%d, %t, nil)", got, explicit, err, test.want, test.explicit)
			}
		})
	}
}

func stringPtr(s string) *string { return &s }
