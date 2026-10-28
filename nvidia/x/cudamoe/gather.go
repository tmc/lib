package cudamoe

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// GatherGeometry returns the one-dimensional grid and block for GatherSource.
// It accounts for inactive columns at the end of each tile-aligned row and
// rejects dimensions outside the kernel's signed-int ABI or CUDA grid limit.
// The layouts of the actual input and output remain tightly packed; padding
// applies only to the logical thread schedule, not to tensor storage.
func GatherGeometry(tokens, width, tile int) (grid, block [3]uint32, err error) {
	if tokens <= 0 || width <= 0 || tokens > math.MaxInt32 || width > math.MaxInt32 || (tile != 16 && tile != 32) {
		return grid, block, fmt.Errorf("cudamoe: invalid gather dimensions or tile")
	}
	return gatherGeometry(tokens, width, tile)
}

func gatherGeometry(tokens, width, tile int) (grid, block [3]uint32, err error) {
	rowThreads := (uint64(width) + uint64(tile) - 1) / uint64(tile) * uint64(tile)
	threads := uint64(tokens) * rowThreads
	blocks := (threads + 127) / 128
	if blocks > math.MaxInt32 {
		return grid, block, fmt.Errorf("cudamoe: gather exceeds one-dimensional grid limit")
	}
	return [3]uint32{uint32(blocks), 1, 1}, [3]uint32{128, 1, 1}, nil
}

// GatherCandidate describes one gather schedule. Layouts are expert data,
// output and routes in ABI pointer order. Data and output are row-major float32;
// routes are int32. Tile is zero for the scalar baseline and 16 or 32 for a
// vectorized tile. Each candidate needs separate, nonoverlapping buffers.
// Compiled resource requirements must be obtained from its artifact.
type GatherCandidate struct {
	Name        string
	Tile        int
	Grid, Block [3]uint32
	Layouts     [3]cudalayout.Tile
}

// GatherCandidates returns scalar, tile16 and tile32 alternatives with checked
// launch geometry and buffer footprints. Routes must contain tokens indices in
// [0,experts); validation of their actual contents belongs to the caller. The
// kernels leave invalid routes untouched. Candidate order is not a performance
// ranking. Schedules exceeding the grid limit are omitted; valid scalar or
// smaller-tile alternatives remain available. This function allocates no tensor
// storage and performs no CUDA calls.
func GatherCandidates(tokens, width, experts int) ([]GatherCandidate, error) {
	if tokens <= 0 || width <= 0 || experts <= 0 || tokens > math.MaxInt32 || width > math.MaxInt32 || experts > math.MaxInt32 {
		return nil, fmt.Errorf("cudamoe: invalid gather dimensions")
	}
	var layouts [3]cudalayout.Tile
	for i, dims := range [][2]int{{experts, width}, {tokens, width}, {1, tokens}} {
		layout, err := cudalayout.NewTile(dims[0], dims[1], 4, cudalayout.WithAlignment(4))
		if err != nil {
			return nil, fmt.Errorf("cudamoe: gather buffer %d: %w", i, err)
		}
		if _, err := layout.FootprintBytes(); err != nil {
			return nil, err
		}
		layouts[i] = layout
	}
	var out []GatherCandidate
	for _, choice := range []struct {
		name string
		tile int
	}{{"gather_scalar", 0}, {"gather16", 16}, {"gather32", 32}} {
		tile := choice.tile
		if tile == 0 {
			tile = 1
		}
		grid, block, err := gatherGeometry(tokens, width, tile)
		if err != nil {
			continue
		}
		out = append(out, GatherCandidate{Name: choice.name, Tile: choice.tile, Grid: grid, Block: block, Layouts: layouts})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("cudamoe: no gather schedule fits the grid limit")
	}
	return out, nil
}

// GatherScalarSource emits a scalar reference alternative to GatherSource.
// Use the scalar entry from GatherCandidates for its unpadded launch geometry.
// It has the same ABI, routing bounds and tightly packed layouts as the tiled
// kernels and copies each output with a separate float32 load and store.
func GatherScalarSource() string {
	return `#include <cuda_runtime.h>
extern "C" __global__ void moe_gather_scalar(const float* experts, float* out,
    const int* route, int tokens, int width, int expert_count) {
  if (tokens <= 0 || width <= 0 || expert_count <= 0) return;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t t = i / (size_t)width;
  if (t >= (size_t)tokens) return;
  int e = route[t];
  if (e < 0 || e >= expert_count) return;
  out[i] = experts[(size_t)e * width + i % (size_t)width];
}`
}

// GatherScalarArtifact compiles GatherScalarSource with the same ABI as
// GatherArtifact. It requires NVRTC and CUDA headers but does not launch work.
func GatherScalarArtifact(target string) (*cudakernel.Artifact, error) {
	return compileGather(GatherScalarSource(), "moe_gather_scalar", target, func(source string, opts ...nvrtcx.Option) (gatherModule, error) {
		return nvrtcx.Compile(source, opts...)
	})
}

// GatherArtifact compiles the tile16 or tile32 gather for a physical CUDA
// target, preserving compiled resources and options. It requires NVRTC and
// CUDA headers but does not load or launch GPU work. Its ABI is expert, output
// and int32 route pointers, followed by int32 token count, width and expert
// count. Use GatherGeometry with the same tile for launch dimensions.
func GatherArtifact(tile int, target string) (*cudakernel.Artifact, error) {
	return gatherArtifact(tile, target, func(source string, opts ...nvrtcx.Option) (gatherModule, error) {
		return nvrtcx.Compile(source, opts...)
	})
}

type gatherModule interface {
	ToArtifact(string, cudakernel.Schema, string) (*cudakernel.Artifact, error)
}

func gatherArtifact(tile int, target string, compile func(string, ...nvrtcx.Option) (gatherModule, error)) (*cudakernel.Artifact, error) {
	source, err := GatherSource(tile)
	if err != nil {
		return nil, err
	}
	return compileGather(source, "moe_gather", target, compile)
}

func compileGather(source, entry, target string, compile func(string, ...nvrtcx.Option) (gatherModule, error)) (*cudakernel.Artifact, error) {
	major, minor, suffix, err := cudakernel.ParseArch(target)
	if err != nil || !strings.HasPrefix(target, "sm_") || target != fmt.Sprintf("sm_%d%d%s", major, minor, suffix) || (suffix != "" && suffix != "a" && suffix != "f") {
		return nil, fmt.Errorf("cudamoe: invalid gather physical target %q", target)
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "experts", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "route", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "tokens", Type: "s32"}, cudakernel.Param{Name: "width", Type: "s32"}, cudakernel.Param{Name: "expert_count", Type: "s32"})
	if err != nil {
		return nil, err
	}
	opts := []nvrtcx.Option{nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithStandard("c++17"), nvrtcx.WithToolkitIncludes(), nvrtcx.WithOutput(nvrtcx.OutputCUBIN)}
	flags, err := nvrtcx.CompileFlags(opts...)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: gather compiler options: %w", err)
	}
	module, err := compile(source, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: compile gather: %w", err)
	}
	if module == nil {
		return nil, fmt.Errorf("cudamoe: compile gather returned no module")
	}
	artifact, err := module.ToArtifact(entry, schema, target)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: gather artifact: %w", err)
	}
	if artifact == nil {
		return nil, fmt.Errorf("cudamoe: compile gather returned no artifact")
	}
	spec := artifact.Spec()
	spec.CompilerOpts = flags
	return cudakernel.NewArtifactFromSpec(spec)
}
