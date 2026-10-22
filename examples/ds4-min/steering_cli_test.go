package main

import (
	"io"
	"strings"
	"testing"
)

func TestSteeringCLIConfig(t *testing.T) {
	for _, tt := range []struct {
		name      string
		args      []string
		attn, ffn float64
		bad       bool
	}{
		{"default", nil, 0, 0, false},
		{"file", []string{"-dir-steering-file", "vectors"}, 0, 1, false},
		{"zero", []string{"-dir-steering-file", "vectors", "-dir-steering-attn", "0"}, 0, 0, false},
		{"negative", []string{"-dir-steering-file", "vectors", "-dir-steering-ffn", "-0.5"}, 0, -0.5, false},
		{"missing", []string{"-dir-steering-ffn", "1"}, 0, 0, true},
		{"nan", []string{"-dir-steering-file", "vectors", "-dir-steering-ffn", "NaN"}, 0, 0, true},
		{"infinity", []string{"-dir-steering-file", "vectors", "-dir-steering-attn", "+Inf"}, 0, 0, true},
		{"range", []string{"-dir-steering-file", "vectors", "-dir-steering-attn", "100.000001"}, 0, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			args := append([]string{"-model", "fixture", "-interactive"}, tt.args...)
			cfg, err := parseConfig(args, strings.NewReader(""), io.Discard)
			if tt.bad {
				if err == nil {
					t.Fatal("accepted invalid steering flags")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if cfg.dirSteeringAttn != tt.attn || cfg.dirSteeringFFN != tt.ffn {
				t.Fatalf("scales %v %v", cfg.dirSteeringAttn, cfg.dirSteeringFFN)
			}
		})
	}
}

func TestSteeringZeroSkipsFileLoad(t *testing.T) {
	if err := configureSteering(nil, config{dirSteeringFile: "/does/not/exist"}); err != nil {
		t.Fatal(err)
	}
}
