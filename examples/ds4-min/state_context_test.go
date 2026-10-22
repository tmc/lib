package main

import "testing"

func TestValidateStateContext(t *testing.T) {
	tests := []struct {
		name    string
		context int
		max     int
		wantErr bool
	}{
		{name: "smallest", context: 2, max: 16},
		{name: "smaller request", context: 8, max: 16},
		{name: "owner size", context: 16, max: 16},
		{name: "zero", context: 0, max: 16, wantErr: true},
		{name: "one", context: 1, max: 16, wantErr: true},
		{name: "too large", context: 17, max: 16, wantErr: true},
		{name: "invalid owner", context: 2, max: 1, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateStateContext(test.context, test.max)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateStateContext(%d, %d) error = %v, wantErr %v", test.context, test.max, err, test.wantErr)
			}
		})
	}
}
