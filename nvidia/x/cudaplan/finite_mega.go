package cudaplan

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// WithFiniteMega requests one finite, single-CTA FP32 worker for the complete
// plan. It admits contiguous add, mul, neg, relu and sum operations only.
// Unsupported graphs are refused, not replaced by the owner's uint32 scalar
// task fixture. The worker has no spin loop or cross-CTA residency requirement.
// Graph replay and finite Mega are separate execution modes.
func WithFiniteMega(enable bool) PlanOption {
	return func(c *planConfig) error { c.finiteMega = enable; return nil }
}

func finiteMegaArtifact(snap nodeSnapshot, target string) (*cudakernel.Artifact, []string, error) {
	if snap == nil {
		return nil, nil, ErrUnsupportedBackend
	}
	order := snap.Order()
	if len(order) == 0 || len(order) > 256 {
		return nil, nil, fmt.Errorf("%w: finite worker requires 1..256 nodes", ErrUnsupportedBackend)
	}
	indices := make(map[string]int, len(order))
	var names []string
	var params []cudakernel.Param
	var source strings.Builder
	fmt.Fprintf(&source, ".version 9.0\n.target %s\n.address_size 64\n.visible .entry finite_plan(\n", target)
	for i, n := range order {
		if n.Type != cudair.Float32 || n.NumElements() <= 0 || uint64(n.NumElements()) > math.MaxUint32-255 || !cudashape.IsContiguousRowMajor(n.Shape, n.Strides) {
			return nil, nil, fmt.Errorf("%w: finite worker requires nonempty contiguous FP32 node %s", ErrUnsupportedBackend, n.Name)
		}
		indices[n.Name] = i
		names = append(names, n.Name)
		params = append(params, cudakernel.Param{Name: fmt.Sprintf("p%d", i), Type: "ptr", IsPointer: true})
		if i != 0 {
			source.WriteString(",\n")
		}
		fmt.Fprintf(&source, ".param .u64 p%d", i)
	}
	source.WriteString("\n) {\n.reg .pred pred;\n.reg .u32 i, k, base, outer, inner;\n.reg .u64 offset, a, b, out;\n.reg .f32 x, y, z;\n.reg .f64 total, widened;\n")
	for j, n := range order {
		if n.Kind == cudair.KindInput {
			continue
		}
		valid := false
		switch n.Kind {
		case cudair.KindUnary:
			valid = len(n.Inputs) == 1 && (n.Op == "neg" || n.Op == "relu")
		case cudair.KindBinary:
			valid = len(n.Inputs) == 2 && (n.Op == "add" || n.Op == "mul" || n.Op == "multiply") && n.Inputs[0].NumElements() == n.NumElements() && n.Inputs[1].NumElements() == n.NumElements()
		case cudair.KindReduce:
			valid = len(n.Inputs) == 1 && n.Op == "sum" && len(n.Axes) == 1 && n.Axes[0] >= 0 && n.Axes[0] < len(n.Inputs[0].Shape)
		}
		if !valid {
			return nil, nil, fmt.Errorf("%w: finite worker operation %s", ErrUnsupportedBackend, n.Op)
		}
		fmt.Fprintf(&source, "mov.u32 i, %%tid.x;\nstep%d:\nsetp.ge.u32 pred, i, %d;\n@pred bra done%d;\nld.param.u64 out, [p%d];\nmul.wide.u32 offset, i, 4;\nadd.u64 out, out, offset;\n", j, n.NumElements(), j, j)
		input := n.Inputs[0]
		fmt.Fprintf(&source, "ld.param.u64 a, [p%d];\n", indices[input.Name])
		if n.Kind == cudair.KindReduce {
			axis := n.Axes[0]
			stride := 1
			for _, extent := range input.Shape[axis+1:] {
				stride *= extent
			}
			extent := input.Shape[axis]
			// cudaref's authoritative sum uses a sequential float64 accumulator
			// and rounds once to float32. Keep that boundary in the device code.
			fmt.Fprintf(&source, "div.u32 outer, i, %d;\nrem.u32 inner, i, %d;\nmul.lo.u32 base, outer, %d;\nadd.u32 base, base, inner;\nmul.wide.u32 offset, base, 4;\nadd.u64 a, a, offset;\nmov.f64 total, 0d0000000000000000;\nmov.u32 k, 0;\nreduce%d:\nld.global.f32 x, [a];\ncvt.f64.f32 widened, x;\nadd.rn.f64 total, total, widened;\nadd.u64 a, a, %d;\nadd.u32 k, k, 1;\nsetp.lt.u32 pred, k, %d;\n@pred bra reduce%d;\ncvt.rn.f32.f64 z, total;\n", stride, stride, stride*extent, j, stride*4, extent, j)
		} else {
			source.WriteString("add.u64 a, a, offset;\nld.global.f32 x, [a];\n")
			switch n.Op {
			case "neg":
				source.WriteString("neg.f32 z, x;\n")
			case "relu":
				source.WriteString("max.NaN.f32 z, x, 0f00000000;\n")
			default:
				fmt.Fprintf(&source, "ld.param.u64 b, [p%d];\nadd.u64 b, b, offset;\nld.global.f32 y, [b];\n", indices[n.Inputs[1].Name])
				op := "mul"
				if n.Op == "add" {
					op = "add"
				}
				fmt.Fprintf(&source, "%s.rn.f32 z, x, y;\n", op)
			}
		}
		fmt.Fprintf(&source, "st.global.f32 [out], z;\nadd.u32 i, i, 256;\nbra step%d;\ndone%d:\nbar.sync 0;\n", j, j)
	}
	source.WriteString("membar.sys;\nret;\n}\n")
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, nil, err
	}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(source.String()), "finite_plan", schema, target)
	return artifact, names, err
}
