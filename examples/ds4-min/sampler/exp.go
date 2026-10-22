package sampler

// Exp returns the float32 exponential used by sampling. On Linux ARM64 it
// follows the pinned glibc 2.39 scalar expf arithmetic in round-to-nearest mode;
// other platforms retain the Go float64 exponential rounded to float32.
// Exhaustive owning-libm equality is qualified only on the sampling domain
// [-104, 0], not on all positive inputs or NaN payloads. Other platforms require
// their own owning-libm qualification.
func Exp(x float32) float32 { return exp(x) }
