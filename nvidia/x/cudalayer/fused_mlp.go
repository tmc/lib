package cudalayer

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// ErrInvalidMLPConfig reports an invalid MLP architecture configuration.
var ErrInvalidMLPConfig = errors.New("cudalayer: invalid fused MLP configuration")

// MLPLayer represents a single dense linear layer with activation in an MLP.
type MLPLayer struct {
	name        string
	inputWidth  int
	outputWidth int
	paddedWidth int
	activation  string
	paramOffset int
}

// NewMLPLayer creates an MLP layer with input and output widths and an activation function.
func NewMLPLayer(name string, inWidth, outWidth int, activation string) (*MLPLayer, error) {
	if inWidth <= 0 || outWidth <= 0 {
		return nil, fmt.Errorf("%w: widths must be positive (in=%d, out=%d)", ErrInvalidMLPConfig, inWidth, outWidth)
	}
	const maxExtent = int(^uint32(0) >> 1)
	if inWidth > maxExtent/(4*32) || outWidth > maxExtent/(4*32)-inWidth || inWidth > maxExtent/outWidth || outWidth > maxExtent-15 {
		return nil, fmt.Errorf("%w: dimensions overflow CUDA extents", ErrInvalidMLPConfig)
	}
	switch strings.ToLower(activation) {
	case "", "none", "relu", "silu":
	default:
		return nil, fmt.Errorf("%w: unsupported activation %q", ErrInvalidMLPConfig, activation)
	}
	// Pad to 16 for tensor alignment if necessary, matching tiny-cuda-nn
	padded := outWidth
	if padded%16 != 0 {
		padded += 16 - (padded % 16)
	}
	return &MLPLayer{
		name:        name,
		inputWidth:  inWidth,
		outputWidth: outWidth,
		paddedWidth: padded,
		activation:  activation,
	}, nil
}

// Name returns the identifier of this layer.
func (l *MLPLayer) Name() string {
	if l.name != "" {
		return l.name
	}
	return "mlp_layer"
}

// InputWidth returns input feature dimension.
func (l *MLPLayer) InputWidth() int { return l.inputWidth }

// OutputWidth returns unpadded output feature dimension.
func (l *MLPLayer) OutputWidth() int { return l.outputWidth }

// PaddedOutputWidth returns padded output feature dimension.
func (l *MLPLayer) PaddedOutputWidth() int { return l.paddedWidth }

// FwdCtxBytes returns the forward context bytes required by this layer.
func (l *MLPLayer) FwdCtxBytes() (int, error) { return 4 * (l.inputWidth + l.outputWidth), nil }

// FwdCtxAlignedPerElement reports whether context is aligned per element.
func (l *MLPLayer) FwdCtxAlignedPerElement() bool { return true }

// EmitForward renders the layer as a self-contained CUDA __device__ function.
func (l *MLPLayer) EmitForward(name string) (string, error) {
	actCode := ""
	switch strings.ToLower(l.activation) {
	case "relu":
		actCode = "val = val > 0.0f ? val : 0.0f;"
	case "silu":
		actCode = "val = val / (1.0f + __expf(-val));"
	case "none", "":
		actCode = "// identity"
	default:
		return "", fmt.Errorf("cudalayer: unsupported activation %q", l.activation)
	}

	body := fmt.Sprintf(`	for (int in_idx = 0; in_idx < %d; ++in_idx) {
		bool saved_input = __cudair_ctx.StoreFloat(in_idx * 4, input[in_idx]);
		(void)saved_input;
	}
	tvec<float, %d> out;
	for (int out_idx = 0; out_idx < %d; ++out_idx) {
		float val = 0.0f;
		for (int in_idx = 0; in_idx < %d; ++in_idx) {
			val += input[in_idx] * params[in_idx * %d + out_idx];
		}
		bool saved_value = __cudair_ctx.StoreFloat((%d + out_idx) * 4, val);
		(void)saved_value;
		%s
		out[out_idx] = val;
	}
	return out;`, l.inputWidth, l.outputWidth, l.outputWidth, l.inputWidth, l.outputWidth, l.inputWidth, actCode)

	return cudair.EmitFromBody(l, name, body)
}

// EmitBackward emits input gradients without parameter writes.
func (l *MLPLayer) EmitBackward(name string, nThreads int) (string, error) {
	return l.EmitBackwardMode(name, nThreads, cudair.GradientModeIgnore)
}

// EmitBackwardMode emits derivatives using the saved input and preactivation.
// Parameter outputs are per sample in input-major order: in*OutputWidth+out.
// Overwrite and accumulate operate on a disjoint sample buffer, not a shared
// batch buffer. Input gradients are always overwritten when requested.
func (l *MLPLayer) EmitBackwardMode(name string, nThreads int, mode cudair.GradientMode) (string, error) {
	if nThreads <= 0 || mode < cudair.GradientModeIgnore || mode > cudair.GradientModeAccumulate {
		return "", fmt.Errorf("cudalayer: invalid backward threads or gradient mode")
	}
	derivative := "1.0f"
	switch strings.ToLower(l.activation) {
	case "relu":
		derivative = "(saved > 0.0f ? 1.0f : 0.0f)"
	case "silu":
		derivative = "sigmoid * (1.0f + saved * (1.0f - sigmoid))"
	}
	var body strings.Builder
	fmt.Fprintf(&body, `	tvec<float, %d> delta;
	for (int out_idx = 0; out_idx < %d; ++out_idx) {
		float saved = 0.0f;
		bool loaded = __cudair_ctx.LoadFloat((%d + out_idx) * 4, &saved);
		if (!loaded) return;
		float sigmoid = 1.0f / (1.0f + __expf(-saved));
		delta[out_idx] = dL_dy[out_idx] * %s;
	}
	if (dL_dx) {
		for (int in_idx = 0; in_idx < %d; ++in_idx) {
			float grad = 0.0f;
			for (int out_idx = 0; out_idx < %d; ++out_idx) {
				grad += delta[out_idx] * params[in_idx * %d + out_idx];
			}
			(*dL_dx)[in_idx] = grad;
		}
	}
`, l.outputWidth, l.outputWidth, l.inputWidth, derivative, l.inputWidth, l.outputWidth, l.outputWidth)
	if mode != cudair.GradientModeIgnore {
		op := "="
		if mode == cudair.GradientModeAccumulate {
			op = "+="
		}
		fmt.Fprintf(&body, `	if (dL_dparams) {
		for (int in_idx = 0; in_idx < %d; ++in_idx) {
			float saved_input = 0.0f;
			bool loaded_input = __cudair_ctx.LoadFloat(in_idx * 4, &saved_input);
			if (!loaded_input) return;
			for (int out_idx = 0; out_idx < %d; ++out_idx) {
				dL_dparams[in_idx * %d + out_idx] %s saved_input * delta[out_idx];
			}
		}
	}
`, l.inputWidth, l.outputWidth, l.outputWidth, op)
	}

	return cudair.EmitBackwardFromBody(l, name, body.String())
}

// BackwardShmemBytes reports shared memory demand for backward pass.
func (l *MLPLayer) BackwardShmemBytes(nThreads int, mode cudair.GradientMode) int {
	return 0
}

// FusedMLP expresses dense layers as a composable cudair tree. Forward child
// results use local vectors; each child saves its input and preactivation in
// bounded context for backward. Compiler register residency is not guaranteed.
type FusedMLP struct {
	name        string
	inputWidth  int
	hiddenWidth int
	outputWidth int
	layers      []*MLPLayer
}

// NewFusedMLP constructs an MLP tree consisting of numLayers dense layers.
// For a 3-layer width-64 MLP, pass (64, 64, 64, 3).
func NewFusedMLP(inputWidth, hiddenWidth, outputWidth, numLayers int) (*FusedMLP, error) {
	return NewFusedMLPWithActivation(inputWidth, hiddenWidth, outputWidth, numLayers, "relu")
}

// NewFusedMLPWithActivation selects none, relu or silu for hidden layers.
// The output layer remains linear. ReLU's derivative at zero is zero. SiLU
// uses NVRTC __expf in both forward and backward. Architectures admit at most
// 1024 layers and checked signed-32-bit parameter and warp-context extents.
func NewFusedMLPWithActivation(inputWidth, hiddenWidth, outputWidth, numLayers int, activation string) (*FusedMLP, error) {
	activation = strings.ToLower(activation)
	if activation == "" {
		activation = "none"
	}
	if activation != "none" && activation != "relu" && activation != "silu" {
		return nil, fmt.Errorf("%w: unsupported hidden activation %q", ErrInvalidMLPConfig, activation)
	}
	if inputWidth <= 0 || hiddenWidth <= 0 || outputWidth <= 0 || numLayers < 1 || numLayers > 1024 {
		return nil, fmt.Errorf("%w: invalid dimensions or layer count", ErrInvalidMLPConfig)
	}

	layers := make([]*MLPLayer, numLayers)
	parameterCount := 0
	contextBytes := 0
	for i := 0; i < numLayers; i++ {
		layerIn := hiddenWidth
		if i == 0 {
			layerIn = inputWidth
		}
		layerOut := hiddenWidth
		act := activation
		if i == numLayers-1 {
			layerOut = outputWidth
			act = "none" // output activation is identity by default
		}

		layerName := fmt.Sprintf("layer_%d", i)
		layer, err := NewMLPLayer(layerName, layerIn, layerOut, act)
		if err != nil {
			return nil, err
		}
		count := layerIn * layerOut
		if count > int(^uint32(0)>>1)-31-parameterCount {
			return nil, fmt.Errorf("%w: parameter extent overflow", ErrInvalidMLPConfig)
		}
		parameterCount += count
		bytes, _ := layer.FwdCtxBytes()
		if bytes > int(^uint32(0)>>1)/32-contextBytes {
			return nil, fmt.Errorf("%w: warp context extent overflow", ErrInvalidMLPConfig)
		}
		contextBytes += bytes
		layers[i] = layer
	}

	return &FusedMLP{
		name:        "fused_mlp",
		inputWidth:  inputWidth,
		hiddenWidth: hiddenWidth,
		outputWidth: outputWidth,
		layers:      layers,
	}, nil
}

// Name returns the identifier of this MLP.
func (m *FusedMLP) Name() string { return m.name }

// InputWidth returns model input width.
func (m *FusedMLP) InputWidth() int { return m.inputWidth }

// OutputWidth returns model output width.
func (m *FusedMLP) OutputWidth() int { return m.outputWidth }

// PaddedOutputWidth returns model padded output width.
func (m *FusedMLP) PaddedOutputWidth() int {
	return m.layers[len(m.layers)-1].PaddedOutputWidth()
}

// FwdCtxBytes returns the parent's own context size, zero. cudair.EmitTree
// includes the separately owned child regions in Unit.FwdCtxBytes.
func (m *FusedMLP) FwdCtxBytes() (int, error) { return 0, nil }

// FwdCtxAlignedPerElement reports whether context is aligned per element.
func (m *FusedMLP) FwdCtxAlignedPerElement() bool { return true }

// Children returns the ordered list of child layers for cudair.EmitTree.
func (m *FusedMLP) Children() []cudair.Emitter {
	res := make([]cudair.Emitter, len(m.layers))
	for i, l := range m.layers {
		res[i] = l
	}
	return res
}

// EmitForward requires the tree emitter to assign child symbols and context.
func (m *FusedMLP) EmitForward(name string) (string, error) {
	return "", fmt.Errorf("cudalayer: fused MLP must be emitted with cudair.EmitTree")
}

// EmitForwardWithChildren composes the layers using symbols and context offsets
// assigned by cudair.EmitTree.
func (m *FusedMLP) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	if len(m.layers) == 0 {
		return "", fmt.Errorf("cudalayer: empty MLP layers")
	}
	if len(children) != len(m.layers) {
		return "", fmt.Errorf("cudalayer: got %d child symbols for %d layers", len(children), len(m.layers))
	}

	var calls []string
	paramOffset := 0

	for i, layer := range m.layers {
		child := children[i]
		inVar := "input"
		if i > 0 {
			inVar = fmt.Sprintf("h%d", i-1)
		}

		numWeights := layer.InputWidth() * layer.OutputWidth()
		ctx := child.FwdCtxArg()
		var call string
		if i == len(m.layers)-1 {
			call = fmt.Sprintf("\treturn %s(%s, params + %d, %s);", child.Name, inVar, paramOffset, ctx)
		} else {
			call = fmt.Sprintf("\tauto h%d = %s(%s, params + %d, %s);", i, child.Name, inVar, paramOffset, ctx)
		}
		calls = append(calls, call)
		paramOffset += numWeights
	}

	body := strings.Join(calls, "\n")
	return cudair.EmitFromBody(m, name, body)
}

// EmitBackward composes input gradients. Parameter writes are selected by
// cudair.EmitBackwardTree's gradient mode.
func (m *FusedMLP) EmitBackward(name string, nThreads int) (string, error) {
	return "", fmt.Errorf("%w: fused MLP requires cudair.EmitBackwardTree for child context offsets", cudair.ErrBackwardEmissionUnsupported)
}

// BackwardShmemBytes reports the maximum demand of the child backward passes.
func (m *FusedMLP) BackwardShmemBytes(nThreads int, mode cudair.GradientMode) int {
	maximum := 0
	for _, layer := range m.layers {
		if size := layer.BackwardShmemBytes(nThreads, mode); size > maximum {
			maximum = size
		}
	}
	return maximum
}

// EmitBackwardWithChildren composes backward functions in reverse layer order
// using the child symbols selected by cudair.EmitBackwardTree.
func (m *FusedMLP) EmitBackwardWithChildren(name string, children []cudair.ChildSymbol, nThreads int, mode cudair.GradientMode) (string, error) {
	if len(m.layers) == 0 || len(children) != len(m.layers) {
		return "", fmt.Errorf("cudalayer: backward child count does not match layers")
	}
	offsets := make([]int, len(m.layers))
	for i := 1; i < len(m.layers); i++ {
		offsets[i] = offsets[i-1] + m.layers[i-1].InputWidth()*m.layers[i-1].OutputWidth()
	}
	var body strings.Builder
	for i := len(m.layers) - 1; i >= 0; i-- {
		dy, dx := "dL_dy", "dL_dx"
		if i < len(m.layers)-1 {
			dy = fmt.Sprintf("grad%d", i+1)
		}
		if i > 0 {
			fmt.Fprintf(&body, "\ttvec<float, %d> grad%d;\n", m.layers[i].InputWidth(), i)
			dx = fmt.Sprintf("&grad%d", i)
		}
		fmt.Fprintf(&body, "\t%s(%s, params + %d, %s, (dL_dparams ? dL_dparams + %d : nullptr), %s);\n", children[i].Name, dy, offsets[i], children[i].FwdCtxArg(), offsets[i], dx)
	}
	return cudair.EmitBackwardFromBody(m, name, body.String())
}

// OuterKernelTemplate is the outer kernel template that splices a fused MLP
// device function. Template authors should write C++ nested-brace initializers
// such as {{x}} as { {x} }, because {{IDENT}} is reserved placeholder syntax.
const OuterKernelTemplate = `
using uint8_t = unsigned char;

template <typename T, int N>
struct tvec {
    T data[N];
    __device__ const T& operator[](int i) const { return data[i]; }
    __device__ T& operator[](int i) { return data[i]; }
};

static constexpr int WARP_SIZE = 32;
#define lane_id() (threadIdx.x & (WARP_SIZE - 1))

{{FORWARD_DEVICE_FUNCTION}}
{{LOSS_DEVICE_FUNCTION}}

extern "C" __global__ void {{KERNEL_NAME}}(
    const int n_elements,
    const float* __restrict__ in_data,
    const float* __restrict__ params,
    float* __restrict__ out_data
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    tvec<float, {{IN_WIDTH}}> in_val;
    #pragma unroll
    for (int i = 0; i < {{IN_WIDTH}}; ++i) {
        in_val[i] = in_data[idx * {{IN_WIDTH}} + i];
    }

    tvec<float, {{OUT_WIDTH}}> out_val = {{FUNC_NAME}}(in_val, params, nullptr);

    #pragma unroll
    for (int i = 0; i < {{OUT_WIDTH}}; ++i) {
        out_data[idx * {{OUT_WIDTH}} + i] = out_val[i];
    }
}
`
