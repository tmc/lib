package cudapipeline

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// AttentionLayout binds a finite row to packed score/value tiles. Each tile
// contains TileSize score words, then TileSize*HeadDim value words, then zero
// padding to a 16-byte extent. The final tile's unused keys are not evaluated.
// Excluded uses true for exclusion; nil includes every key. Callers must prove
// included scores and all values finite, and excluded scores already -Inf.
// This layout alone does not prove equivalence to a source graph.
type AttentionLayout struct {
	Keys, HeadDim, TileSize int
	Excluded                []bool
	// FullyMaskedPolicy must be "canonical-nan-exact-fp32" for an
	// all-excluded row. Source emission does not qualify device NaN behavior.
	FullyMaskedPolicy string
}

// PayloadBytes returns the padded extent of one validated tile. Keys, HeadDim
// and TileSize are bounded to [1,64], with TileSize no greater than Keys.
func (l AttentionLayout) PayloadBytes() (int64, error) {
	if l.Keys < 1 || l.Keys > 64 || l.HeadDim < 1 || l.HeadDim > 64 || l.TileSize < 1 || l.TileSize > l.Keys {
		return 0, fmt.Errorf("cudapipeline: invalid attention layout dimensions")
	}
	if len(l.Excluded) != 0 && len(l.Excluded) != l.Keys {
		return 0, fmt.Errorf("cudapipeline: invalid attention exclusion extent")
	}
	if l.FullyMaskedPolicy != "" && l.FullyMaskedPolicy != "canonical-nan-exact-fp32" {
		return 0, fmt.Errorf("cudapipeline: unsupported fully masked attention policy")
	}
	all := len(l.Excluded) != 0
	for _, excluded := range l.Excluded {
		all = all && excluded
	}
	if all && l.FullyMaskedPolicy == "" {
		return 0, fmt.Errorf("cudapipeline: fully masked attention needs explicit nan policy")
	}
	bytes := int64(4 * l.TileSize * (1 + l.HeadDim))
	return (bytes + 15) &^ 15, nil
}

// AttentionSource emits a synchronous CUDA C++ attention kernel and its
// pointer ABI from this validated schedule. Launch exactly one 64-thread CTA;
// the first warp copies packed tiles and thread 32 consumes them in row order.
// packed must contain ceil(Keys/TileSize)*PayloadBytes bytes; out holds HeadDim
// floats. Their storage must be disjoint and float-aligned.
//
// Shared storage is capped at 48 KiB and eight stages. Abstract StepIssueTMA
// steps become ordinary scalar copies; waits/releases become CTA barriers.
// No TMA, cp.async, overlap, occupancy, compilation or device qualification is
// asserted. The FP64 recurrence is reassociated relative to materialized FP32
// attention; actual compiler/device output needs an independent policy check.
func (s *PipelineSchedule) AttentionSource(name string, layout AttentionLayout) (string, cudakernel.Schema, error) {
	payload, err := layout.PayloadBytes()
	if err != nil {
		return "", cudakernel.Schema{}, err
	}
	if err := checkedAttentionSchedule(s, layout, payload); err != nil {
		return "", cudakernel.Schema{}, err
	}
	if name == "" || len(name) > 128 {
		return "", cudakernel.Schema{}, fmt.Errorf("cudapipeline: invalid attention kernel name")
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return "", cudakernel.Schema{}, fmt.Errorf("cudapipeline: invalid attention kernel identifier")
		}
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "packed", Type: "f32", ByteWidth: 8, Alignment: 8, IsPointer: true},
		cudakernel.Param{Name: "out", Type: "f32", ByteWidth: 8, Alignment: 8, IsPointer: true})
	if err != nil {
		return "", cudakernel.Schema{}, fmt.Errorf("cudapipeline: attention schema: %w", err)
	}
	var excluded uint64
	for key, masked := range layout.Excluded {
		if masked {
			excluded |= uint64(1) << key
		}
	}
	words := payload / 4
	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n#include <math.h>\n")
	fmt.Fprintf(&b, "extern \"C\" __global__ void %s(const float* packed, float* out) {\n", name)
	b.WriteString("  if (blockDim.x != 64 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1) return;\n")
	fmt.Fprintf(&b, "  __shared__ __align__(16) float stage[%d];\n", words*int64(s.Stages))
	fmt.Fprintf(&b, "  const unsigned long long excluded = 0x%016xULL;\n", excluded)
	fmt.Fprintf(&b, "  double m = 0.0, l = 0.0, u[%d] = {};\n", layout.HeadDim)
	b.WriteString("  bool has_prior = false;\n  unsigned int tid = threadIdx.x;\n")
	for _, step := range s.Steps {
		iteration := step.Epoch*s.Stages + step.Stage
		offset := int64(step.Stage) * words
		switch step.Kind {
		case StepIssueTMA:
			fmt.Fprintf(&b, "  if (tid < 32) for (unsigned int j = tid; j < %d; j += 32) stage[%d+j] = packed[%d+j];\n", words, offset, int64(iteration)*words)
		case StepWaitParity, StepReleaseStage:
			b.WriteString("  __syncthreads();\n")
		case StepConsumeStage:
			start := iteration * layout.TileSize
			count := min(layout.TileSize, layout.Keys-start)
			fmt.Fprintf(&b, "  if (tid == 32) { // tile %d, stage %d, epoch %d\n", iteration, step.Stage, step.Epoch)
			fmt.Fprintf(&b, "    const int start = %d, count = %d;\n", start, count)
			b.WriteString("    bool valid = false;\n    double tile_max = 0.0;\n    for (int k = 0; k < count; k++) if (((excluded >> (start+k)) & 1ULL) == 0) {\n")
			fmt.Fprintf(&b, "      double score = (double)stage[%d+k];\n", offset)
			b.WriteString("      if (!valid || score > tile_max) tile_max = score;\n      valid = true;\n    }\n    if (valid) {\n      double m_new = has_prior ? fmax(m, tile_max) : tile_max;\n      double alpha = has_prior ? exp(m - m_new) : 1.0;\n")
			fmt.Fprintf(&b, "      double p[%d] = {}, sum = 0.0;\n", layout.TileSize)
			b.WriteString("      for (int k = 0; k < count; k++) if (((excluded >> (start+k)) & 1ULL) == 0) {\n")
			fmt.Fprintf(&b, "        p[k] = exp((double)stage[%d+k] - m_new);\n", offset)
			b.WriteString("        sum += p[k];\n      }\n")
			fmt.Fprintf(&b, "      for (int d = 0; d < %d; d++) {\n        double pv = 0.0;\n", layout.HeadDim)
			fmt.Fprintf(&b, "        for (int k = 0; k < count; k++) if (p[k] != 0.0) pv += p[k] * (double)stage[%d+k*%d+d];\n", offset+int64(layout.TileSize), layout.HeadDim)
			b.WriteString("        u[d] = u[d]*alpha + pv;\n      }\n      l = l*alpha + sum;\n      m = m_new;\n      has_prior = true;\n    }\n  }\n")
		}
	}
	fmt.Fprintf(&b, "  if (tid == 32) for (int d = 0; d < %d; d++) out[d] = has_prior ? (float)(u[d]/l) : __uint_as_float(0x7fc00000u);\n", layout.HeadDim)
	b.WriteString("}\n")
	return b.String(), schema, nil
}

func checkedAttentionSchedule(s *PipelineSchedule, l AttentionLayout, payload int64) error {
	if failures := s.Validate(); len(failures) != 0 {
		return fmt.Errorf("cudapipeline: invalid attention schedule: %v", failures)
	}
	if s.PayloadBytes != payload || s.Stages > 8 || payload*int64(s.Stages) > 48<<10 {
		return fmt.Errorf("cudapipeline: attention payload, stage or shared storage budget exceeded")
	}
	tiles := (l.Keys + l.TileSize - 1) / l.TileSize
	issued := make([]bool, tiles)
	next := 0
	for index, step := range s.Steps {
		if step.StepIndex != index {
			return fmt.Errorf("cudapipeline: attention schedule step identity mismatch")
		}
		if step.Kind != StepIssueTMA && step.Kind != StepConsumeStage {
			continue
		}
		iteration := step.Epoch*s.Stages + step.Stage
		if iteration < 0 || iteration >= tiles || step.Bytes != payload {
			return fmt.Errorf("cudapipeline: attention tile or payload mismatch")
		}
		if step.Kind == StepIssueTMA {
			if issued[iteration] {
				return fmt.Errorf("cudapipeline: duplicate attention tile transfer")
			}
			issued[iteration] = true
		} else {
			if iteration != next || !issued[iteration] {
				return fmt.Errorf("cudapipeline: attention tiles must be consumed once in row order")
			}
			next++
		}
	}
	if next != tiles {
		return fmt.Errorf("cudapipeline: attention schedule does not cover the row")
	}
	return nil
}
