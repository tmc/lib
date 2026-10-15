package egraph

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// OnlineAttentionSchedule owns one checked candidate's packed input, existing
// lane09 schedule, synchronous CUDA C++ source and fixed pointer ABI. Its zero
// value is invalid. Accessors return copies. Source generation and CPU evidence
// do not qualify vendor compilation, device output, resources or timing.
type OnlineAttentionSchedule struct {
	candidate string
	identity  string
	name      string
	source    string
	schema    cudakernel.Schema
	schedule  *cudapipeline.PipelineSchedule
	packed    []float32
}

// NewOnlineAttentionSchedule publishes a checked row to lane09's finite
// producer/consumer schedule and actual synchronous attention source consumer.
// It packs exact copied score/value words without altering the original graph
// or bindings. Padding and final-tile storage are never additional logical keys.
// At least two stages are required; lane09's eight-stage/48 KiB source budget
// can refuse larger stage choices. Two stages support every admitted row size.
// No compiler, device, asynchronous transfer or timing operation is invoked.
func NewOnlineAttentionSchedule(candidate *OnlineAttentionCandidate, stages int, name string) (*OnlineAttentionSchedule, error) {
	if candidate == nil || candidate.snapshot == nil || candidate.identity == "" {
		return nil, errors.New("egraph: missing checked online attention candidate")
	}
	scores, values, _, _, err := checkedAttentionRegion(candidate.snapshot)
	if err != nil {
		return nil, err
	}
	n, d := scores.Shape[0], values.Shape[1]
	trace := candidate.Derivation()
	layout := cudapipeline.AttentionLayout{Keys: n, HeadDim: d, TileSize: trace.TileSize, FullyMaskedPolicy: trace.FullyMaskedPolicy}
	if trace.Mask != nil {
		layout.Excluded = trace.Mask.Dense()
	}
	payload, err := layout.PayloadBytes()
	if err != nil {
		return nil, fmt.Errorf("egraph: attention schedule layout: %w", err)
	}
	tiles := (n + trace.TileSize - 1) / trace.TileSize
	if len(trace.Tiles) != tiles {
		return nil, errors.New("egraph: attention derivation tile count mismatch")
	}
	for i, tile := range trace.Tiles {
		if tile.Start != i*trace.TileSize || tile.End != min((i+1)*trace.TileSize, n) {
			return nil, errors.New("egraph: attention derivation tile boundaries mismatch")
		}
	}
	schedule, err := cudapipeline.BuildWarpSpecializedSchedule(stages, payload, tiles)
	if err != nil {
		return nil, fmt.Errorf("egraph: attention schedule: %w", err)
	}
	source, schema, err := schedule.AttentionSource(name, layout)
	if err != nil {
		return nil, fmt.Errorf("egraph: attention schedule source: %w", err)
	}
	words := int(payload / 4)
	packed := make([]float32, tiles*words)
	for i, tile := range trace.Tiles {
		row := packed[i*words : (i+1)*words]
		copy(row, candidate.inputs[scores.Name][tile.Start:tile.End])
		copy(row[trace.TileSize:], candidate.inputs[values.Name][tile.Start*d:tile.End*d])
	}
	bound := &OnlineAttentionSchedule{candidate: candidate.Identity(), name: name, source: source,
		schema: schema, schedule: schedule, packed: packed}
	encoded, err := json.Marshal(struct {
		Candidate string
		Source    string
		Schema    cudakernel.Schema
		Schedule  *cudapipeline.PipelineSchedule
	}{bound.candidate, source, schema, schedule})
	if err != nil {
		return nil, fmt.Errorf("egraph: attention schedule identity: %w", err)
	}
	digest := sha256.Sum256(encoded)
	bound.identity = hex.EncodeToString(digest[:])
	return bound, nil
}

// Identity returns the candidate/source/schema/schedule binding digest.
func (s *OnlineAttentionSchedule) Identity() string { return s.identity }

// CandidateIdentity returns the original checked graph/input/policy digest.
func (s *OnlineAttentionSchedule) CandidateIdentity() string { return s.candidate }

// Source returns the synchronous attention kernel. It is not compiled PTX.
func (s *OnlineAttentionSchedule) Source() string { return s.source }

// PackedInput returns owned FP32 tiles, including unread zero padding.
func (s *OnlineAttentionSchedule) PackedInput() []float32 { return slices.Clone(s.packed) }

// Schema returns the two-pointer kernel parameter ABI. Parameter-bank alignment
// is eight bytes; it is not a claim about device allocation alignment.
func (s *OnlineAttentionSchedule) Schema() cudakernel.Schema {
	out := s.schema
	out.Params = slices.Clone(out.Params)
	return out
}

// Schedule returns the detached existing lane09 schedule, including tile stage
// and epoch mapping: iteration = epoch*Stages + stage.
func (s *OnlineAttentionSchedule) Schedule() *cudapipeline.PipelineSchedule {
	out := *s.schedule
	out.Roles = slices.Clone(out.Roles)
	out.Steps = slices.Clone(out.Steps)
	out.Expectations = slices.Clone(out.Expectations)
	return &out
}

// Launch returns the sole admitted grid and block dimensions.
func (s *OnlineAttentionSchedule) Launch() (grid, block [3]uint32) {
	return [3]uint32{1, 1, 1}, [3]uint32{64, 1, 1}
}

// SharedBytes returns declared stage storage, not observed compiled allocation
// or occupancy. Compiler registers, spills and device timing remain unmeasured.
func (s *OnlineAttentionSchedule) SharedBytes() int64 {
	return int64(s.schedule.Stages) * s.schedule.PayloadBytes
}

// Compile invokes the actual NVRTC compiler separately from source generation
// and returns authentic PTX through the existing ToArtifact adapter. It never
// loads or launches a kernel. The caller must admit the compiler environment;
// compilation does not promote the CPU candidate to device-qualified output.
// Failed compiler diagnostics remain in the error; successful logs are returned.
func (s *OnlineAttentionSchedule) Compile(ctx context.Context, major, minor int) (*cudakernel.Artifact, string, error) {
	if s == nil || s.source == "" || ctx == nil || major < 8 || major > 99 || minor < 0 || minor > 9 {
		return nil, "", errors.New("egraph: invalid attention compilation request")
	}
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	options := []nvrtcx.Option{nvrtcx.WithName(s.name + ".cu"), nvrtcx.WithArch(major, minor),
		nvrtcx.WithStandard("c++20"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithToolkitIncludes(), nvrtcx.WithOutput(nvrtcx.OutputPTX)}
	flags, err := nvrtcx.CompileFlags(options...)
	if err != nil {
		return nil, "", fmt.Errorf("egraph: attention compiler options: %w", err)
	}
	compiled, err := nvrtcx.CompileContext(ctx, s.source, options...)
	if err != nil {
		return nil, "", fmt.Errorf("egraph: attention compilation: %w", err)
	}
	artifact, err := compiled.ToArtifact(s.name, s.Schema(), fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		return nil, compiled.Log(), fmt.Errorf("egraph: attention artifact: %w", err)
	}
	compilerMajor, compilerMinor, err := nvrtc.Version()
	if err != nil {
		return nil, compiled.Log(), fmt.Errorf("egraph: attention compiler version: %w", err)
	}
	artifact.Toolchain = fmt.Sprintf("nvrtc %d.%d", compilerMajor, compilerMinor)
	artifact.CompilerOpts = slices.Clone(flags)
	return artifact, compiled.Log(), nil
}
