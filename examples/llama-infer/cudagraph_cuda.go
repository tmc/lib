//go:build !darwin && !tpu

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// cudaGraphEnabled reports whether the single-stream decode step should be
// replayed as a CUDA graph. Off unless LLAMA_INFER_CUDA_GRAPH=1.
func cudaGraphEnabled() bool { return os.Getenv("LLAMA_INFER_CUDA_GRAPH") == "1" }

// decodeGraph replays the single-stream decode step as one CUDA graph launch
// instead of the several dozen kernel launches the step otherwise submits.
//
// UNQUALIFIED. Nothing here has run on a device: no host that has built it
// has one. The mechanism is the documented one, stream capture into a graph,
// cuGraphInstantiate once, cuGraphExecUpdate on every later step; what has
// not been observed is a token coming back from a replay, or any difference
// in step time. Whether replay beats loose launches is a measurement, and
// this package's own doc says it does not promise one.
//
// Every step is re-captured. The position, and with it the attention
// kernel's shared-memory request, change on every token, so the graph's
// kernel parameters change on every token; capture submits nothing to the
// device, so its cost is host time only, and Update patches the executable
// in place when the topology is unchanged. If Update refuses, the executable
// is rebuilt, and rebuilds are counted so a run can tell that it was paying
// for instantiation every step.
//
// The token copy stays outside the capture: it is a pageable host copy, which
// capture rejects, and it is ordered on the same stream ahead of the launch
// anyway. The launch counter still counts the captured launches, so the
// figures a run reports under this mode are launches recorded, not submitted.
type decodeGraph struct {
	exec     *cudagraph.Exec
	replays  uint64
	rebuilds uint64
}

// step runs one decode step through the graph and returns the sampled token.
func (g *decodeGraph) step(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn *boundedKernel,
	sampleFn *cuda.Kernel,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
	counter *cudalaunch.Counter,
) (int32, error) {
	if g == nil {
		return 0, errors.New("cuda graph: nil")
	}
	if stream == nil {
		stream = sp.arena.Device().DefaultStream()
	}
	if err := gpuStageToken(sp, tokenID, stream); err != nil {
		return 0, err
	}

	var bodyErr error
	graph, err := cudagraph.Capture(stream, func() {
		bodyErr = gpuForwardPassBody(gpuW, cfg, kv, sp, gm, attnFn, sampleFn, pos, stream, counter)
	})
	if err != nil {
		if bodyErr != nil {
			return 0, fmt.Errorf("cuda graph: capture decode step at position %d: %w", pos, errors.Join(bodyErr, err))
		}
		return 0, fmt.Errorf("cuda graph: capture decode step at position %d: %w", pos, err)
	}
	defer graph.Destroy()
	if bodyErr != nil {
		return 0, bodyErr
	}

	if g.exec == nil {
		if g.exec, err = graph.Instantiate(); err != nil {
			return 0, fmt.Errorf("cuda graph: instantiate: %w", err)
		}
		g.rebuilds++
	} else if err := g.exec.Update(graph); err != nil {
		// A changed topology, not a changed parameter: the fused paths pick
		// different kernels for different layers only by config, so this
		// should not happen between two steps of one engine. Rebuild rather
		// than fail, and count it.
		_ = g.exec.Destroy()
		if g.exec, err = graph.Instantiate(); err != nil {
			return 0, fmt.Errorf("cuda graph: reinstantiate after update failure: %w", err)
		}
		g.rebuilds++
	}

	if err := g.exec.Launch(stream); err != nil {
		return 0, fmt.Errorf("cuda graph: launch at position %d: %w", pos, err)
	}
	g.replays++
	return gpuReadToken(sp, stream)
}

// close releases the executable graph. The caller must have drained the
// stream it last launched on.
func (g *decodeGraph) close() error {
	if g == nil || g.exec == nil {
		return nil
	}
	err := g.exec.Destroy()
	g.exec = nil
	return err
}
