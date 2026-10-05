package cudaoutputcoverage

import "github.com/tmc/lib/nvidia/x/cudatest"

// RecordedOutput retains the completed readbacks and policy used to check a
// campaign run. PTXDigest identifies the runner's loaded PTX; SourceDigest in
// TwoFill identifies the CUDA source. Neither digest substitutes for the other.
// These are producer claims, not independent device or process attestation.
// A campaign snapshots its slices when the runner returns. The runner must
// complete device work before returning and must not mutate evidence concurrently.
type RecordedOutput struct {
	PTXDigest string
	TwoFill   TwoFill
	Ordinary  []uint32
	Policy    cudatest.CUDAOutputPolicy
}
