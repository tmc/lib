//go:build !linux || !arm64

package sampler

import "math"

func exp(x float32) float32 { return float32(math.Exp(float64(x))) }
