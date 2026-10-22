//go:build linux && arm64

package sampler

// temperatureScale follows the pinned Linux ARM64 GCC 13.3 release owner:
// one rounded float32 reciprocal, then a rounded float32 multiply. The owning
// -ffast-math sampler rewrites source division this way before calling expf.
// This is that release's schedule, not an other-libc/compiler parity claim.
type temperatureScale float32

func newTemperatureScale(temperature float32) temperatureScale {
	return temperatureScale(float32(1 / temperature))
}

func (scale temperatureScale) apply(delta float32) float32 {
	return float32(delta * float32(scale))
}
