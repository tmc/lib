//go:build !linux || !arm64

package sampler

// Preserve direct division outside the independently inspected Linux owner.
// In particular, no Darwin release reciprocal schedule has been established.
type temperatureScale float32

func newTemperatureScale(temperature float32) temperatureScale {
	return temperatureScale(temperature)
}

func (scale temperatureScale) apply(delta float32) float32 {
	return float32(delta / float32(scale))
}
