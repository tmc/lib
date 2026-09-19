//go:build !unix

package cudaprofagent

// The agent is a shared object injected by cudaprof, which ships on unix only.
// Elsewhere there is nothing to find and nothing to finalize.
func lookupFinalize() func() { return nil }
