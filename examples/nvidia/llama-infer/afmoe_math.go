package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

func afmoeBF16(v float32) float32 {
	return safetensors.BFloat16ToFloat32(safetensors.Float32ToBFloat16(v))
}

// afmoeRMSNorm is a scalar CPU reference for one BF16 row. The normalized
// value rounds before the learned weight multiplication, as AfmoeRMSNorm
// does. It is not a claim about a CUDA reduction's bitwise result.
func afmoeRMSNorm(values, weights []float32, eps float32) ([]float32, error) {
	if len(values) == 0 || len(values) != len(weights) || !afmoeFinite(eps) || eps <= 0 {
		return nil, fmt.Errorf("afmoe rms norm: invalid row, weights or epsilon")
	}
	var sum float32
	for i, v := range values {
		if !afmoeFinite(v) || !afmoeFinite(weights[i]) {
			return nil, fmt.Errorf("afmoe rms norm: nonfinite input at %d", i)
		}
		v = afmoeBF16(v)
		sum = float32(sum + float32(v*v))
	}
	variance := float32(sum / float32(len(values)))
	scale := float32(1 / math.Sqrt(float64(float32(variance+eps))))
	out := make([]float32, len(values))
	for i, v := range values {
		normal := afmoeBF16(float32(afmoeBF16(v) * scale))
		out[i] = afmoeBF16(float32(normal * afmoeBF16(weights[i])))
	}
	return out, nil
}

func afmoeScaleEmbedding(values []float32, hidden int) ([]float32, error) {
	if hidden < 1 || len(values)%hidden != 0 {
		return nil, fmt.Errorf("afmoe embedding: invalid hidden size or row length")
	}
	scale := float32(math.Sqrt(float64(hidden)))
	out := make([]float32, len(values))
	for i, v := range values {
		if !afmoeFinite(v) {
			return nil, fmt.Errorf("afmoe embedding: nonfinite input at %d", i)
		}
		out[i] = afmoeBF16(float32(afmoeBF16(v) * scale))
	}
	return out, nil
}
