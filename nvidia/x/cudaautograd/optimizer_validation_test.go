package cudaautograd_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
)

func TestSGDOptimizerValidation(t *testing.T) {
	tests := []struct {
		name     string
		lr       float32
		momentum float32
		wantErr  string
	}{
		{"nan learning rate", float32(math.NaN()), 0, "invalid learning rate"},
		{"positive infinite learning rate", float32(math.Inf(1)), 0, "invalid learning rate"},
		{"negative infinite learning rate", float32(math.Inf(-1)), 0, "invalid learning rate"},
		{"zero learning rate", 0, 0, "invalid learning rate"},
		{"negative learning rate", -1, 0, "invalid learning rate"},
		{"nan momentum", 0.1, float32(math.NaN()), "invalid momentum"},
		{"positive infinite momentum", 0.1, float32(math.Inf(1)), "invalid momentum"},
		{"negative infinite momentum", 0.1, float32(math.Inf(-1)), "invalid momentum"},
		{"negative momentum", 0.1, -0.1, "invalid momentum"},
		{"unit momentum", 0.1, 1, "invalid momentum"},
		{"zero momentum", 0.1, 0, ""},
		{"half momentum", 0.1, 0.5, ""},
		{"smallest learning rate", math.SmallestNonzeroFloat32, 0, ""},
		{"largest learning rate", math.MaxFloat32, 0, ""},
		{"largest momentum below one", 0.1, math.Nextafter32(1, 0), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opt, err := cudaautograd.NewSGDOptimizer(tt.lr, tt.momentum)
			if tt.wantErr != "" {
				if opt != nil || err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("NewSGDOptimizer(%g, %g) = %v, %v; want nil optimizer and %q error", tt.lr, tt.momentum, opt, err, tt.wantErr)
				}
				return
			}
			if err != nil || opt == nil {
				t.Fatalf("NewSGDOptimizer(%g, %g) = %v, %v", tt.lr, tt.momentum, opt, err)
			}
			if opt.LearningRate() != tt.lr || opt.Momentum() != tt.momentum {
				t.Fatalf("optimizer changed admitted settings: learning rate %g, momentum %g", opt.LearningRate(), opt.Momentum())
			}
		})
	}
}

func TestSGDOptimizerFiniteStep(t *testing.T) {
	tests := []struct {
		name     string
		momentum float32
	}{
		{"without momentum", 0},
		{"with momentum", 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opt, err := cudaautograd.NewSGDOptimizer(0.25, tt.momentum)
			if err != nil {
				t.Fatal(err)
			}
			params := map[string][]float32{"w": {4}}
			grads := map[string][]float32{"w": {2}}
			if err := opt.Step(params, grads); err != nil {
				t.Fatal(err)
			}
			if params["w"][0] != 3.5 {
				t.Fatalf("first step = %g, want 3.5", params["w"][0])
			}
			if err := opt.Step(params, grads); err != nil {
				t.Fatal(err)
			}
			want := float32(3) - tt.momentum*0.5
			if params["w"][0] != want {
				t.Fatalf("second step = %g, want %g", params["w"][0], want)
			}
		})
	}
}
