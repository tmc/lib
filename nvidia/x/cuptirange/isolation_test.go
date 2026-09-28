package cuptirange_test

import (
	"errors"
	"math"
	"testing"
)

func checkIsolation(h1, l1, h2, l2 float64, static uint64, trips1, trips2 uint32) error {
	if static == 0 || trips1 == 0 || trips2 <= trips1 {
		return errors.New("invalid isolation bound")
	}
	for _, value := range []float64{h1, l1, h2, l2} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return errors.New("invalid range count")
		}
	}
	if l1 == 0 {
		return errors.New("light range count is zero")
	}
	if h1-l1 < float64(static)*float64(trips1) {
		return errors.New("heavy range does not meet the static work bound")
	}
	if h2-h1 < float64(static)*float64(trips2-trips1) {
		return errors.New("heavy range did not scale by the static work bound")
	}
	if l1 != l2 {
		return errors.New("light range changed when heavy work changed")
	}
	return nil
}

func TestCheckIsolation(t *testing.T) {
	const static, trips1, trips2 = uint64(3), uint32(10), uint32(20)
	tests := []struct {
		name    string
		h1, l1  float64
		h2, l2  float64
		wantErr bool
	}{
		{name: "isolated", h1: 130, l1: 100, h2: 160, l2: 100},
		{name: "single-range-absorbs-heavy", h1: 300, l1: 300, h2: 600, l2: 600, wantErr: true},
		{name: "first-bound-only", h1: 110, l1: 100, h2: 150, l2: 100, wantErr: true},
		{name: "scaling-bound-only", h1: 130, l1: 100, h2: 159, l2: 100, wantErr: true},
		{name: "light-zero", h1: 100, l1: 0, h2: 130, l2: 0, wantErr: true},
		{name: "light-range-leaks", h1: 130, l1: 100, h2: 160, l2: 101, wantErr: true},
		{name: "nan", h1: math.NaN(), l1: 100, h2: 160, l2: 100, wantErr: true},
		{name: "positive-infinity", h1: math.Inf(1), l1: 100, h2: 160, l2: 100, wantErr: true},
		{name: "negative-count", h1: 130, l1: -1, h2: 160, l2: 100, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := checkIsolation(test.h1, test.l1, test.h2, test.l2, static, trips1, trips2)
			if (err != nil) != test.wantErr {
				t.Fatalf("checkIsolation() error = %v, wantErr %t", err, test.wantErr)
			}
		})
	}
}

func TestCheckIsolationRejectsInvalidBounds(t *testing.T) {
	tests := []struct {
		name           string
		static         uint64
		trips1, trips2 uint32
	}{
		{"zero-static", 0, 10, 20},
		{"zero-trips", 3, 0, 20},
		{"non-increasing-trips", 3, 20, 20},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkIsolation(130, 100, 160, 100, tt.static, tt.trips1, tt.trips2); err == nil {
				t.Fatal("checkIsolation accepted invalid bounds")
			}
		})
	}
}

func TestCheckIsolationRejectsEachInputGuardAlone(t *testing.T) {
	tests := []struct {
		name           string
		h1, l1, h2, l2 float64
		trips1, trips2 uint32
	}{
		{"negative", 130, -1, 160, -1, 10, 20},
		{"infinite", math.Inf(1), 100, math.Inf(1), 100, 10, 20},
		{"zero-trips1", 130, 100, 190, 100, 0, 20},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkIsolation(tt.h1, tt.l1, tt.h2, tt.l2, 3, tt.trips1, tt.trips2); err == nil {
				t.Fatal("checkIsolation accepted isolated invalid input")
			}
		})
	}
}
