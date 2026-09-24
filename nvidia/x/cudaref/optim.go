package cudaref

import "math"

// AdamW holds the hyperparameters of one AdamW parameter group, following
// torch.optim.AdamW: bias-corrected moments and weight decay decoupled from
// the gradient.
type AdamW struct {
	LR, Beta1, Beta2, Eps, WeightDecay float64
}

// Step applies one AdamW update in float64 to the parameters p and the moment
// estimates m and v, in place. g is the gradient, which is multiplied by
// clip before use (pass 1 for no clipping), and step is the 1-based number of
// this update. p, m, v and g must have the same length.
func (c AdamW) Step(p, m, v, g []float64, step int, clip float64) {
	bc1 := 1 - math.Pow(c.Beta1, float64(step))
	bc2 := 1 - math.Pow(c.Beta2, float64(step))
	for i := range p {
		gi := g[i] * clip
		p[i] *= 1 - c.LR*c.WeightDecay
		m[i] = c.Beta1*m[i] + (1-c.Beta1)*gi
		v[i] = c.Beta2*v[i] + (1-c.Beta2)*gi*gi
		denom := math.Sqrt(v[i])/math.Sqrt(bc2) + c.Eps
		p[i] -= c.LR / bc1 * m[i] / denom
	}
}

// GlobalNorm returns the L2 norm of all the gradients taken together, as
// torch.nn.utils.clip_grad_norm_ computes it.
func GlobalNorm(grads ...[]float64) float64 {
	var sum float64
	for _, g := range grads {
		for _, x := range g {
			sum += x * x
		}
	}
	return math.Sqrt(sum)
}

// ClipScale returns the factor torch.nn.utils.clip_grad_norm_ multiplies
// gradients by: min(maxNorm/(norm+1e-6), 1). A maxNorm of zero or less
// disables clipping and returns 1.
func ClipScale(norm, maxNorm float64) float64 {
	if maxNorm <= 0 {
		return 1
	}
	return math.Min(maxNorm/(norm+1e-6), 1)
}
