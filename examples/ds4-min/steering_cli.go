package main

import (
	"errors"
	"flag"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type steeringLoader interface {
	loadSteering(*steeringVectors, float32, float32) error
}

func steeringFlags(fs *flag.FlagSet, cfg *config) {
	fs.StringVar(&cfg.dirSteeringFile, "dir-steering-file", "", "raw little-endian per-layer steering directions")
	fs.Float64Var(&cfg.dirSteeringAttn, "dir-steering-attn", 0, "attention steering scale in [-100,100]")
	fs.Float64Var(&cfg.dirSteeringFFN, "dir-steering-ffn", 0, "FFN steering scale in [-100,100]")
}

func validateSteeringConfig(cfg *config, fs *flag.FlagSet) error {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "dir-steering-attn" || f.Name == "dir-steering-ffn" {
			set = true
		}
	})
	if cfg.dirSteeringFile != "" && !set {
		cfg.dirSteeringFFN = 1
	}
	for _, scale := range []float64{cfg.dirSteeringAttn, cfg.dirSteeringFFN} {
		if math.IsNaN(scale) || math.IsInf(scale, 0) || scale < -100 || scale > 100 {
			return errors.New("steering scale must be between -100 and 100")
		}
	}
	if cfg.dirSteeringFile == "" && (cfg.dirSteeringAttn != 0 || cfg.dirSteeringFFN != 0) {
		return errors.New("steering scale requires dir-steering-file")
	}
	return nil
}

func requestState(b sessionBackend) dsv4.State {
	switch b := b.(type) {
	case *stateSessionBackend:
		return b.state
	case *batchStateSessionBackend:
		return b.state
	}
	return nil
}

func configureSteering(b sessionBackend, cfg config) error {
	// Upstream skips file loading when both startup scales are zero. In
	// particular, this does not promise that /steer can enable unloaded data.
	if cfg.dirSteeringAttn == 0 && cfg.dirSteeringFFN == 0 {
		return nil
	}
	loader, ok := requestState(b).(steeringLoader)
	if !ok {
		return fmt.Errorf("directional steering: %w", dsv4.ErrUnimplemented)
	}
	v, err := loadSteeringVectors(cfg.dirSteeringFile)
	if err != nil {
		return err
	}
	return loader.loadSteering(v, float32(cfg.dirSteeringAttn), float32(cfg.dirSteeringFFN))
}
