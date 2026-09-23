package cudalayer

import (
	"errors"
	"math"
	"testing"
)

func TestProjectionNormConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  ProjectionNormConfig
		want int
	}{
		{"valid", ProjectionNormConfig{Dim: 128, Threads: 64, Epsilon: 1e-5}, 256},
		{"zero dimension", ProjectionNormConfig{Threads: 64, Epsilon: 1e-5}, 0},
		{"non-power-of-two threads", ProjectionNormConfig{Dim: 128, Threads: 48, Epsilon: 1e-5}, 0},
		{"too many threads", ProjectionNormConfig{Dim: 128, Threads: 2048, Epsilon: 1e-5}, 0},
		{"zero epsilon", ProjectionNormConfig{Dim: 128, Threads: 64}, 0},
		{"nan epsilon", ProjectionNormConfig{Dim: 128, Threads: 64, Epsilon: float32(math.NaN())}, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.cfg.SharedMemoryBytes()
			if test.want == 0 {
				if !errors.Is(err, ErrInvalidConfig) {
					t.Fatalf("SharedMemoryBytes error = %v, want %v", err, ErrInvalidConfig)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("SharedMemoryBytes = %d, %v, want %d, nil", got, err, test.want)
			}
		})
	}
}
