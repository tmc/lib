package cudasass_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestStallProfileValidation(t *testing.T) {
	for _, tt := range []struct {
		name    string
		profile cudasass.StallProfile
		wantErr bool
	}{
		{"measured NaN", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: math.NaN()}, true},
		{"measured positive infinity", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: math.Inf(1)}, true},
		{"measured negative infinity", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: math.Inf(-1)}, true},
		{"measured below zero", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: -0.01}, true},
		{"measured above hundred", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: 100.01}, true},
		{"measured zero", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: 0}, false},
		{"measured hundred", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: 100}, false},
		{"measured interior", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: 31.5}, false},
		{"measured without source", cudasass.StallProfile{Measured: true, Percent: 31.5}, true},
		{"measured sentinel", cudasass.StallProfile{Measured: true, Source: "ncu", Percent: cudasass.StallSampleUnavailable}, true},
		{"absent sentinel", cudasass.StallProfile{Percent: cudasass.StallSampleUnavailable}, false},
		{"absent zero", cudasass.StallProfile{}, true},
		{"absent with source", cudasass.StallProfile{Source: "ncu", Percent: cudasass.StallSampleUnavailable}, true},
		{"absent NaN", cudasass.StallProfile{Percent: math.NaN()}, true},
		{"absent positive infinity", cudasass.StallProfile{Percent: math.Inf(1)}, true},
		{"absent negative infinity", cudasass.StallProfile{Percent: math.Inf(-1)}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.profile.Valid()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Valid() = %v, want error %t", err, tt.wantErr)
			}
		})
	}
}
