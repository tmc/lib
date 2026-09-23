//go:build darwin

package main

import apple "github.com/tmc/apple/metal"

// countedComputeEncoder counts commands at the physical Metal encoding
// boundary. A count is published only after its command buffer completes.
type countedComputeEncoder struct {
	apple.MTLComputeCommandEncoder
	dispatches int
}

func countDispatches(enc apple.MTLComputeCommandEncoder) *countedComputeEncoder {
	return &countedComputeEncoder{MTLComputeCommandEncoder: enc}
}

func (e *countedComputeEncoder) DispatchThreadsThreadsPerThreadgroup(grid, group apple.MTLSize) {
	e.MTLComputeCommandEncoder.DispatchThreadsThreadsPerThreadgroup(grid, group)
	e.dispatches++
}

func (e *countedComputeEncoder) DispatchThreadgroupsThreadsPerThreadgroup(grid, group apple.MTLSize) {
	e.MTLComputeCommandEncoder.DispatchThreadgroupsThreadsPerThreadgroup(grid, group)
	e.dispatches++
}
