package cudaautograd_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// backwardFixtureKernel is a fixture-only translation of one actual AD node.
// The ABI is the planner's ordered input pointers followed by its output
// pointer. Extents and scalar indexing are compiled in, without an int n ABI.
type backwardFixtureKernel struct {
	nodeID       int
	name, output string
	inputs       []string
	elements     int
	source       string
	schema       cudakernel.Schema
}

func backwardFixtureElements(n *cudair.Node) (int, error) {
	if n == nil || n.ID < 0 || n.Type != cudair.Float32 || n.Layout != cudashape.LayoutRowMajor || !cudashape.IsContiguousRowMajor(n.Shape, n.Strides) {
		return 0, errors.New("backward fixture: requires contiguous FP32 nodes")
	}
	shape, err := cudashape.New(n.Shape...)
	if err != nil {
		return 0, err
	}
	count, err := shape.NumElements()
	if err != nil || count < 1 || count > 4096 {
		return 0, errors.New("backward fixture: extent must be in [1, 4096]")
	}
	return int(count), nil
}

func emitBackwardFixtureKernel(n *cudair.Node) (backwardFixtureKernel, error) {
	var k backwardFixtureKernel
	count, err := backwardFixtureElements(n)
	if err != nil {
		return k, err
	}
	if n.Kind == cudair.KindReshape {
		if n.Op != "reshape" || len(n.Inputs) != 1 || n.Inputs[0] == nil || n.Effect != cudair.EffectInPlaceAlias || n.StorageAlias == nil || n.StorageAlias.ID != n.Inputs[0].ID {
			return k, errors.New("backward fixture: unsupported reshape alias")
		}
	} else if n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil {
		return k, errors.New("backward fixture: unsupported storage effect")
	}
	k.nodeID, k.output, k.elements = n.ID, n.Name, count
	k.name = fmt.Sprintf("backward_node_%d", n.ID)
	var source strings.Builder
	fmt.Fprintf(&source, "// cudair node %d shape %v\nextern \"C\" __global__ void %s(\n", n.ID, n.Shape, k.name)
	var params []cudakernel.Param
	inputCounts := make([]int, len(n.Inputs))
	for i, input := range n.Inputs {
		inputCounts[i], err = backwardFixtureElements(input)
		if err != nil {
			return k, err
		}
		k.inputs = append(k.inputs, input.Name)
		name := fmt.Sprintf("input%d", i)
		params = append(params, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true})
		fmt.Fprintf(&source, "    const float* %s,\n", name)
	}
	params = append(params, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	fmt.Fprintln(&source, "    float* output) {")
	fmt.Fprintln(&source, "    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;")
	fmt.Fprintln(&source, "    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;")
	fmt.Fprintf(&source, "    if (idx >= %d) return;\n", count)
	switch n.Kind {
	case cudair.KindConstant:
		if n.Op != "constant" || len(n.Inputs) != 0 || len(n.ConstValues) != count {
			return k, errors.New("backward fixture: invalid constant extent")
		}
		fmt.Fprintf(&source, "    const float values[%d] = {", count)
		for i, value := range n.ConstValues {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return k, errors.New("backward fixture: nonfinite constant unsupported")
			}
			if i != 0 {
				source.WriteString(", ")
			}
			source.WriteString(strconv.FormatFloat(float64(value), 'x', -1, 32) + "f")
		}
		fmt.Fprintln(&source, "};\n    output[idx] = values[idx];")
	case cudair.KindReshape:
		if inputCounts[0] != count {
			return k, errors.New("backward fixture: reshape changes element count")
		}
		// Materialize the alias's values into the planner-owned result. The
		// fixture never writes the source or observes alias pointer identity.
		fmt.Fprintln(&source, "    output[idx] = input0[idx];")
	case cudair.KindBinary:
		if len(n.Inputs) != 2 || (n.Op != "mul" && n.Op != "add") {
			return k, errors.New("backward fixture: unsupported binary operation")
		}
		indices := []string{"idx", "idx"}
		for i, inputCount := range inputCounts {
			if inputCount == 1 {
				indices[i] = "0"
			} else if inputCount != count || !slices.Equal(n.Inputs[i].Shape, n.Shape) {
				return k, errors.New("backward fixture: unsupported broadcast")
			}
		}
		op := "+"
		if n.Op == "mul" {
			op = "*"
		}
		fmt.Fprintf(&source, "    output[idx] = input0[%s] %s input1[%s];\n", indices[0], op, indices[1])
	default:
		return k, fmt.Errorf("backward fixture: unsupported node %d kind %s op %q", n.ID, n.Kind, n.Op)
	}
	fmt.Fprintln(&source, "}")
	k.source = source.String()
	k.schema, err = cudakernel.NewSchema(params...)
	return k, err
}

func backwardFixtureSources(snapshot *cudair.Snapshot) ([]backwardFixtureKernel, error) {
	if snapshot == nil {
		return nil, errors.New("backward fixture: nil snapshot")
	}
	if len(snapshot.Order()) > 32 {
		return nil, errors.New("backward fixture: graph exceeds 32 nodes")
	}
	var kernels []backwardFixtureKernel
	for _, n := range snapshot.Order() {
		if _, err := backwardFixtureElements(n); err != nil {
			return nil, err
		}
		if n.Kind == cudair.KindInput {
			continue
		}
		if n.Kind == cudair.KindMatMul {
			if n.Op != "matmul" || n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil || len(n.Inputs) != 2 || len(n.Shape) != 2 || len(n.Inputs[0].Shape) != 2 || len(n.Inputs[1].Shape) != 2 || n.Inputs[0].Shape[1] != n.Inputs[1].Shape[0] || n.Shape[0] != n.Inputs[0].Shape[0] || n.Shape[1] != n.Inputs[1].Shape[1] {
				return nil, errors.New("backward fixture: unsupported vendor matmul")
			}
			continue // The real planner selects cuBLASLt for this node.
		}
		kernel, err := emitBackwardFixtureKernel(n)
		if err != nil {
			return nil, err
		}
		kernels = append(kernels, kernel)
	}
	return kernels, nil
}

// backwardFixtureCompileOptions selects one physical target and CUBIN output.
func backwardFixtureCompileOptions(target string) ([]nvrtcx.Option, nvrtcx.OutputKind) {
	// nvrtcx requires CUBIN bytes for physical-architecture compilation.
	// Use the same output selection for compilation and its result check.
	output := nvrtcx.OutputCUBIN
	return []nvrtcx.Option{
		nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithOutput(output),
		nvrtcx.WithStandard("c++17"), nvrtcx.WithNoToolkitIncludes(), nvrtcx.WithNoCache(),
		nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"),
	}, output
}

// compileBackwardFixture is called only inside the enabled native gate. It
// compiles actual node computations; it never installs reference answers or
// fabricates an artifact when compilation fails.
func compileBackwardFixture(ctx context.Context, snapshot *cudair.Snapshot, target string) (*cudaplan.KernelRegistry, error) {
	kernels, err := backwardFixtureSources(snapshot)
	if err != nil {
		return nil, err
	}
	registry := cudaplan.NewKernelRegistry()
	for _, kernel := range kernels {
		options, output := backwardFixtureCompileOptions(target)
		compiled, err := nvrtcx.CompileContext(ctx, kernel.source, options...)
		if err != nil {
			return nil, fmt.Errorf("compile backward node %d: %w", kernel.nodeID, err)
		}
		if compiled.Output() != output || len(compiled.CUBIN()) == 0 {
			return nil, fmt.Errorf("compile backward node %d: required CUBIN output missing", kernel.nodeID)
		}
		artifact, err := compiled.ToArtifact(kernel.name, kernel.schema, target)
		if err != nil {
			return nil, err
		}
		if artifact.Kind != cudakernel.KindCUBIN || artifact.Target != target {
			return nil, fmt.Errorf("compile backward node %d: artifact kind or target mismatch", kernel.nodeID)
		}
		artifact.Toolchain = "nvrtc compiler-key=" + compiled.Key()
		artifact.Dependencies = append(artifact.Dependencies, cudakernel.Dependency{
			Path: fmt.Sprintf("backward-node-%d.cu", kernel.nodeID), Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(kernel.source))), Tracked: true,
		})
		if err := registry.Register(kernel.output, artifact); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

func TestBackwardFixtureCompilerRecipe(t *testing.T) {
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("recipe control requires unloaded NVIDIA libraries")
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("recipe control loaded an NVIDIA library")
		}
	})
	for _, target := range []string{"sm_100a", "sm_121a"} {
		t.Run(target, func(t *testing.T) {
			options, output := backwardFixtureCompileOptions(target)
			if output != nvrtcx.OutputCUBIN {
				t.Fatalf("physical compiler output = %v, want OutputCUBIN", output)
			}
			flags, err := nvrtcx.CompileFlags(options...)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"--gpu-architecture=" + target, "-std=c++17", "--fmad=false", "--ftz=false"} {
				if !slices.Contains(flags, want) {
					t.Fatalf("compiler flags %v omit %q", flags, want)
				}
			}
			if cudakernel.KindCUBIN.RequiresLink() {
				t.Fatal("CUBIN must be directly loadable by the planner")
			}
		})
	}
}

func retainBackwardCompilerError(t *testing.T, err error) {
	t.Helper()
	var owner io.Closer
	if errors.As(err, &owner) {
		t.Cleanup(func() {
			if closeErr := owner.Close(); closeErr != nil {
				t.Errorf("close failed compiler owner: %v", closeErr)
			}
		})
	}
}

func TestBackwardFixtureArtifactSource(t *testing.T) {
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("source control requires unloaded NVIDIA libraries")
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("source control loaded an NVIDIA library")
		}
	})
	f := newNativeBackwardFixture(t)
	kernels, err := backwardFixtureSources(f.snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var expected int
	for _, n := range f.snapshot.Order() {
		if n.Kind != cudair.KindInput && n.Kind != cudair.KindMatMul {
			expected++
		}
	}
	if len(kernels) != expected || expected != 8 {
		t.Fatalf("emitted %d nodes, want all %d non-vendor nodes", len(kernels), expected)
	}
	for _, kernel := range kernels {
		node := f.snapshot.NodeByID(kernel.nodeID)
		if node == nil || node.Name != kernel.output || node.NumElements() != kernel.elements || len(kernel.schema.Params) != len(node.Inputs)+1 || len(kernel.inputs) != len(node.Inputs) {
			t.Fatalf("source/schema/node identity mismatch: %+v", kernel)
		}
		for i, input := range node.Inputs {
			if kernel.inputs[i] != input.Name {
				t.Fatal("input order differs from actual node")
			}
		}
		for _, param := range kernel.schema.Params {
			if !param.IsPointer || param.Type != "ptr" || param.ByteWidth != 8 {
				t.Fatal("planner ABI contains a non-pointer parameter")
			}
		}
		if err := kernel.schema.Validate(); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(kernel.source, fmt.Sprintf("if (idx >= %d) return;", node.NumElements())) || !strings.Contains(kernel.source, kernel.name+"(") {
			t.Fatal("source lost static extent or entry identity")
		}
		mutant := *node
		mutant.Op = "unsupported"
		if _, err := emitBackwardFixtureKernel(&mutant); err == nil {
			t.Fatal("unsupported operation admitted")
		}
		mutant = *node
		mutant.ID++
		changed, err := emitBackwardFixtureKernel(&mutant)
		if err != nil || changed.source == kernel.source || changed.name == kernel.name {
			t.Fatal("node ID did not change generated source identity")
		}
		if node.Kind == cudair.KindConstant {
			mutant = *node
			mutant.ConstValues = slices.Clone(node.ConstValues)
			mutant.ConstValues[0] = 2
			changed, err := emitBackwardFixtureKernel(&mutant)
			if err != nil || changed.source == kernel.source {
				t.Fatal("constant payload did not change generated computation")
			}
		}
		if node.Kind == cudair.KindBinary {
			mutant = *node
			mutant.Op = "add"
			if node.Op == "add" {
				mutant.Op = "mul"
			}
			changed, err := emitBackwardFixtureKernel(&mutant)
			if err != nil || changed.source == kernel.source {
				t.Fatal("binary operation did not change generated computation")
			}
		}
		mutant = *node
		mutant.Type = cudair.Float16
		if _, err := emitBackwardFixtureKernel(&mutant); err == nil {
			t.Fatal("unsupported numerical type admitted")
		}
	}
	unsupported := *f.snapshot.Outputs()[0]
	unsupported.Kind = cudair.KindUnary
	unsupported.Op = "neg"
	if _, err := emitBackwardFixtureKernel(&unsupported); err == nil {
		t.Fatal("unsupported node kind admitted")
	}
}
