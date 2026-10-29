package egraph

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

type rmsNormConfig struct {
	hasWeight bool
	wName     string
	wVal      float32
	hasWVal   bool
	epsName   string
	epsVal    float32
	hasEps    bool
}

func numericValue(data any) (float32, bool) {
	switch v := data.(type) {
	case float32:
		return v, true
	case float64:
		return float32(v), true
	case int:
		return float32(v), true
	case int32:
		return float32(v), true
	case int64:
		return float32(v), true
	case string:
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f), true
		}
	}
	return 0, false
}

func extractRMSNormConfig(expr *Expr) rmsNormConfig {
	cfg := rmsNormConfig{
		hasWeight: true,
		wName:     "W_norm",
		epsName:   "eps",
		epsVal:    1e-5,
	}
	if expr == nil {
		return cfg
	}

	var wExpr, epsExpr *Expr
	switch expr.Op {
	case "rmsnorm", "tile_rmsnorm", "fused_rmsnorm":
		if len(expr.Children) == 2 {
			cfg.hasWeight = false
			epsExpr = expr.Children[1]
		} else if len(expr.Children) >= 3 {
			cfg.hasWeight = true
			wExpr = expr.Children[1]
			epsExpr = expr.Children[2]
		}
	case "tile_rmsnorm_qkv", "fused_rmsnorm_qkv", "tile_rmsnorm_proj", "fused_rmsnorm_proj", "fused_rmsnorm_gemv":
		if len(expr.Children) == 3 {
			cfg.hasWeight = false
			epsExpr = expr.Children[1]
		} else if len(expr.Children) >= 4 {
			cfg.hasWeight = true
			wExpr = expr.Children[1]
			epsExpr = expr.Children[2]
		}
	case "tile_rmsnorm_qkv_split", "fused_rmsnorm_qkv_split":
		if len(expr.Children) == 5 {
			cfg.hasWeight = false
			epsExpr = expr.Children[1]
		} else if len(expr.Children) >= 6 {
			cfg.hasWeight = true
			wExpr = expr.Children[1]
			epsExpr = expr.Children[2]
		}
	case "tile_rmsnorm_gemv_swiglu", "tile_rmsnorm_swiglu", "fused_rmsnorm_gemv_swiglu", "fused_rmsnorm_swiglu":
		if len(expr.Children) == 6 {
			cfg.hasWeight = false
			epsExpr = expr.Children[1]
		} else if len(expr.Children) >= 7 {
			cfg.hasWeight = true
			wExpr = expr.Children[1]
			epsExpr = expr.Children[2]
		}
	default:
		for _, ch := range expr.Children {
			if ch != nil && (ch.Op == "rmsnorm" || strings.Contains(ch.Op, "rmsnorm")) {
				return extractRMSNormConfig(ch)
			}
		}
	}

	if wExpr != nil {
		if wExpr.Op != "" {
			cfg.wName = wExpr.Op
		}
		if len(wExpr.Children) == 0 {
			if v, ok := numericValue(wExpr.Data); ok {
				cfg.wVal = v
				cfg.hasWVal = true
			} else if v, err := strconv.ParseFloat(wExpr.Op, 32); err == nil {
				cfg.wVal = float32(v)
				cfg.hasWVal = true
			}
		}
	}

	if epsExpr != nil {
		if epsExpr.Op != "" {
			cfg.epsName = epsExpr.Op
		}
		if len(epsExpr.Children) == 0 {
			if v, ok := numericValue(epsExpr.Data); ok {
				cfg.epsVal = v
				cfg.hasEps = true
			} else if v, err := strconv.ParseFloat(epsExpr.Op, 32); err == nil {
				cfg.epsVal = float32(v)
				cfg.hasEps = true
			}
		}
	}

	return cfg
}

func sanitizeTileIdent(s string) string {
	var sb strings.Builder
	for i, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || (i > 0 && r >= '0' && r <= '9') {
			sb.WriteRune(r)
		} else if i == 0 && r >= '0' && r <= '9' {
			sb.WriteByte('_')
			sb.WriteRune(r)
		} else {
			sb.WriteByte('_')
		}
	}
	out := sb.String()
	if out == "" {
		return "sym"
	}
	return out
}

func formatTileFloat(v float32) string {
	s := strconv.FormatFloat(float64(v), 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func declarativeRMSNormPrelude(elemType string, cfg rmsNormConfig) (epsSetup, weightView, normTileExpr string) {
	epsSetup = "    const float norm_eps = eps;"
	if cfg.hasEps {
		epsSetup = fmt.Sprintf("    const float norm_eps = %sf;\n    (void)eps;", strconv.FormatFloat(float64(cfg.epsVal), 'g', -1, 32))
	} else if cfg.epsName != "" && cfg.epsName != "eps" {
		epsSetup = fmt.Sprintf("    const float norm_eps = eps; // eps: %s", sanitizeTileIdent(cfg.epsName))
	}

	if !cfg.hasWeight {
		weightView = "    (void)W_norm;"
		normTileExpr = fmt.Sprintf("tile<%s, shape<TM, TK>>{x.load(m, k) * rms_scale}", elemType)
		return epsSetup, weightView, normTileExpr
	}

	if cfg.hasWVal {
		weightView = fmt.Sprintf("    (void)W_norm;\n    const %s w_norm_const = static_cast<%s>(%sf);", elemType, elemType, strconv.FormatFloat(float64(cfg.wVal), 'g', -1, 32))
		normTileExpr = fmt.Sprintf("tile<%s, shape<TM, TK>>{(x.load(m, k) * rms_scale) * w_norm_const}", elemType)
		return epsSetup, weightView, normTileExpr
	}

	wComment := ""
	if cfg.wName != "" && cfg.wName != "w" && cfg.wName != "W_norm" {
		wComment = fmt.Sprintf(" // weight: %s", sanitizeTileIdent(cfg.wName))
	}
	weightView = fmt.Sprintf("    auto w_norm_span = tensor_span{assume_aligned<16>(W_norm), extents{1, K}};%s\n    auto w_norm = partition_view{w_norm_span, shape<1, TK>{}};", wComment)
	normTileExpr = fmt.Sprintf("tile<%s, shape<TM, TK>>{(x.load(m, k) * rms_scale) * w_norm.load(0, k)}", elemType)
	return epsSetup, weightView, normTileExpr
}

// GenerateDeclarativeTile generates C++ Tile source. Normalized projections
// require the shape and ABI contract documented by TileNormalizedProjection.
// Ordinary MMA and FFN kernels retain pointer arguments followed by runtime
// M,N,K. Buffers are disjoint, contiguous row-major matrices: MMA uses A[M,K],
// B[K,N], C[M,N]; FFN uses X[M,K], input weights[K,N], down weights[N,K],
// Out[M,K]. The reference launch uses CTA0 to visit every output tile; other
// CTAs do nothing. Nonpositive or overflowing byte extents do nothing before
// addresses are formed. Callers still own buffer sizes, lifetimes and launch
// admission. No tensor-core throughput or non-FP32 numerical parity is implied.
// Source generation does not establish compiler acceptance or device parity.
func GenerateDeclarativeTile(expr *Expr, opts CodegenOptions) (string, error) {
	if expr == nil {
		return "", fmt.Errorf("egraph: nil expr passed to GenerateDeclarativeTile")
	}
	if normalizedProjectionOp(expr.Op) {
		_, source, err := lowerNormalizedProjection(expr, opts)
		return source, err
	}
	if normalizedSwiGLUNode(expr) {
		return "", fmt.Errorf("egraph: %s requires full-row RMS normalization and composite output lowering", expr.Op)
	}

	if opts.KernelName == "" {
		opts.KernelName = "DeclarativeTileKernel"
	}
	if opts.Precision == "" {
		opts.Precision = "float"
	}

	if !strings.HasPrefix(expr.Op, "tile_") && len(expr.Children) > 0 {
		if fused := fuseNode(expr); fused != nil && strings.HasPrefix(fused.Op, "tile_") {
			expr = fused
		}
	}

	elemType := "float"
	switch opts.Precision {
	case "float", "float32":
		elemType = "float"
	case "half", "float16":
		elemType = "half"
	case "bf16", "bfloat16":
		elemType = "bfloat16"
	case "f8", "fp8", "f8e4m3", "f8E4M3FN":
		elemType = "cuda::fp8_e4m3"
	case "f4", "fp4", "f4e2m1", "f4E2M1FN":
		elemType = "cuda::fp4_e2m1"
	case "q8_0":
		elemType = "int8_t"
	default:
		elemType = "float"
	}

	var sb strings.Builder
	sb.WriteString("// Declarative C++ Tile IR generated by nvidia/x/egraph\n")
	sb.WriteString("#include \"cuda_tile.h\"\n")
	sb.WriteString("using namespace cuda::tiles;\n\n")

	canLowerRootNorm := expr.Op == "rmsnorm" && len(expr.Children) > 0
	if !IsLowerableTileIROp(expr.Op) && !canLowerRootNorm {
		return "", fmt.Errorf("egraph: no Tile IR lowering for operator %q", expr.Op)
	}

	cfg := extractRMSNormConfig(expr)

	switch expr.Op {
	case "tile_rmsnorm", "fused_rmsnorm", "rmsnorm":
		sb.WriteString(generateDeclarativeRMSNorm(opts.KernelName, elemType, cfg))
	case "tile_swiglu", "fused_swiglu":
		sb.WriteString(generateDeclarativeSwiGLU(opts.KernelName, elemType))
	case "tile_geglu", "fused_geglu":
		sb.WriteString(generateDeclarativeGeGLU(opts.KernelName, elemType))
	case "tile_gelu", "fused_gelu":
		sb.WriteString(generateDeclarativeGeLU(opts.KernelName, elemType))
	case "tile_mma", "tile_mma_aligned", "mma":
		sb.WriteString(generateDeclarativeMMA(opts.KernelName, elemType))
	case "tile_mma_fp8", "tile_fp8_gemm", "tile_fp8", "fp8_matmul":
		sb.WriteString(generateDeclarativeFP8(opts.KernelName))
	case "mmaf_scaled", "tile_fp4_gemm", "tile_fp4", "fp4_matmul":
		sb.WriteString(generateDeclarativeFP4(opts.KernelName))
	default:
		return "", fmt.Errorf("egraph: no Tile IR lowering for operator %q", expr.Op)
	}

	return sb.String(), nil
}

func generateDeclarativeRMSNorm(kernelName, elemType string, cfg rmsNormConfig) string {
	epsSetup, weightView, normTileExpr := declarativeRMSNormPrelude(elemType, cfg)
	return fmt.Sprintf(`extern "C" __tile_global__ void %s(
    const %s* __restrict__ X,
    const %s* __restrict__ W_norm,
    %s* __restrict__ Out,
    int M, int K,
    float eps
) {
    constexpr int TM = 16, TK = 16;
%s
    auto x_span = tensor_span{assume_aligned<16>(X), extents{M, K}};
%s
    auto out_span = tensor_span{assume_aligned<16>(Out), extents{M, K}};

    auto x = partition_view{x_span, shape<TM, TK>{}};
    auto out = partition_view{out_span, shape<TM, TK>{}};

    const int m_tiles = (M + TM - 1) / TM;
    const int k_tiles = (K + TK - 1) / TK;
    for (auto m : irange(0, m_tiles)) {
        auto sq_sum = full<tile<float, shape<TM, 1>>>(0.0f);
        for (auto k : irange(0, k_tiles)) {
            auto x_tile = x.load(m, k);
            sq_sum = sq_sum + row_sum(x_tile * x_tile);
        }
        auto rms_scale = rsqrt(sq_sum / static_cast<float>(K) + norm_eps);
        for (auto k : irange(0, k_tiles)) {
            auto x_norm = %s;
            out.store(x_norm, m, k);
        }
    }
}
`, kernelName, elemType, elemType, elemType, epsSetup, weightView, normTileExpr)
}

func generateDeclarativeSwiGLU(kernelName, elemType string) string {
	return generateDeclarativeFFN(kernelName, elemType, true, false)
}

func generateDeclarativeGeGLU(kernelName, elemType string) string {
	return generateDeclarativeFFN(kernelName, elemType, true, true)
}

func generateDeclarativeGeLU(kernelName, elemType string) string {
	return generateDeclarativeFFN(kernelName, elemType, false, true)
}

// The reference source uses masked pointer tiles rather than relying on a
// partition view's padding behavior. Reshape/masked-load syntax still requires
// qualification against the selected CUDA Tile header and compiler.
func declarativeMatrixSupport() string {
	return `#ifndef EGRAPH_DECLARATIVE_MATRIX_DEFINED
#define EGRAPH_DECLARATIVE_MATRIX_DEFINED
template<class T> inline bool egraph_tile_shape(int M, int N, int K, bool ffn) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    constexpr long long limit = 9223372036854775807LL / sizeof(T);
    return (long long)M <= limit / K && (long long)K <= limit / N &&
           (ffn || (long long)M <= limit / N);
}
inline auto egraph_tile_indices() {
    return reshape<shape<16, 16>>(iota<tile<long long, shape<256>>>());
}
template<class T> inline auto egraph_tile_load(const T* base, long long r, long long c, int rows, int cols) {
    auto indices = egraph_tile_indices();
    auto row = r + indices / 16LL;
    auto col = c + indices % 16LL;
    auto valid = (row < (long long)rows) & (col < (long long)cols);
    auto offset = select(valid, row * (long long)cols + col, full<tile<long long, shape<16, 16>>>(0LL));
    return load_masked(base + offset, valid, full<tile<T, shape<16, 16>>>(T(0)));
}
template<class T, class V> inline void egraph_tile_store(T* base, V value, long long r, long long c, int rows, int cols) {
    auto indices = egraph_tile_indices();
    auto row = r + indices / 16LL;
    auto col = c + indices % 16LL;
    auto valid = (row < (long long)rows) & (col < (long long)cols);
    auto offset = select(valid, row * (long long)cols + col, full<tile<long long, shape<16, 16>>>(0LL));
    store_masked(base + offset, value, valid);
}
#endif
`
}

func generateDeclarativeFFN(kernelName, elemType string, gated, gelu bool) string {
	inputWeight, downWeight := "W_in", "W_out"
	upParam, upInit, upMMA, upMultiply := "", "", "", ""
	if gated {
		inputWeight, downWeight = "W_gate", "W_down"
		upParam = fmt.Sprintf("    const %s* __restrict__ W_up,\n", elemType)
		upInit = fmt.Sprintf("                auto up = full<tile<%s, shape<16, 16>>>(0.0f);\n", elemType)
		upMMA = "                    up = mma(x, egraph_tile_load(W_up, (long long)k * 16, (long long)n * 16, K, N), up);\n"
		upMultiply = "                act = act * up;\n"
	}
	activation := "gate / (1.0f + exp(-gate))"
	if gelu {
		activation = "0.5f * gate * (1.0f + erf(gate * 0.7071067811865475f))"
	}
	return declarativeMatrixSupport() + fmt.Sprintf(`extern "C" __tile_global__ void %s(
    const %s* __restrict__ X,
    const %s* __restrict__ %s,
%s    const %s* __restrict__ %s,
    %s* __restrict__ Out,
    int M, int N, int K
) {
    if (!egraph_tile_shape<%s>(M, N, K, true)) return;
    if (bid().x != 0 || bid().y != 0 || bid().z != 0) return;
    const int row_tiles = (M - 1) / 16 + 1;
    const int hidden_tiles = (N - 1) / 16 + 1;
    const int output_tiles = (K - 1) / 16 + 1;
    for (auto m : irange(0, row_tiles)) {
        for (auto o : irange(0, output_tiles)) {
            auto down = full<tile<%s, shape<16, 16>>>(0.0f);
            for (auto n : irange(0, hidden_tiles)) {
                auto gate = full<tile<%s, shape<16, 16>>>(0.0f);
%s                for (auto k : irange(0, output_tiles)) {
                    auto x = egraph_tile_load(X, (long long)m * 16, (long long)k * 16, M, K);
                    gate = mma(x, egraph_tile_load(%s, (long long)k * 16, (long long)n * 16, K, N), gate);
%s                }
                auto act = %s;
%s                auto hidden = (long long)n * 16 + egraph_tile_indices() %% 16LL;
                act = select(hidden < (long long)N, act, full<decltype(act)>(0.0f));
                down = mma(act, egraph_tile_load(%s, (long long)n * 16, (long long)o * 16, N, K), down);
            }
            egraph_tile_store(Out, down, (long long)m * 16, (long long)o * 16, M, K);
        }
    }
}
`, kernelName, elemType, elemType, inputWeight, upParam, elemType, downWeight, elemType,
		elemType, elemType, elemType, upInit, inputWeight, upMMA, activation, upMultiply, downWeight)
}

func generateDeclarativeMMA(kernelName, elemType string) string {
	return declarativeMatrixSupport() + fmt.Sprintf(`extern "C" __tile_global__ void %s(
    const %s* __restrict__ A,
    const %s* __restrict__ B,
    %s* __restrict__ C,
    int M, int N, int K
) {
    if (!egraph_tile_shape<%s>(M, N, K, false)) return;
    if (bid().x != 0 || bid().y != 0 || bid().z != 0) return;
    const int row_tiles = (M - 1) / 16 + 1;
    const int column_tiles = (N - 1) / 16 + 1;
    const int reduction_tiles = (K - 1) / 16 + 1;
    for (auto m : irange(0, row_tiles)) {
        for (auto n : irange(0, column_tiles)) {
            auto sum = full<tile<%s, shape<16, 16>>>(0.0f);
            for (auto k : irange(0, reduction_tiles)) {
                sum = mma(egraph_tile_load(A, (long long)m * 16, (long long)k * 16, M, K),
                          egraph_tile_load(B, (long long)k * 16, (long long)n * 16, K, N), sum);
            }
            egraph_tile_store(C, sum, (long long)m * 16, (long long)n * 16, M, N);
        }
    }
}
`, kernelName, elemType, elemType, elemType, elemType, elemType)
}

func generateDeclarativeFP8(kernelName string) string {
	return fmt.Sprintf(`extern "C" __tile_global__ void %s(
    const float* __restrict__ A,
    const cuda::fp8_e4m3* __restrict__ B_fp8,
    const float* __restrict__ scale_b,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TM = 16, TN = 16, TK = 32;
    auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
    auto b = partition_view{tensor_span{assume_aligned<16>(B_fp8), extents{K, N}}, shape<TK, TN>{}};
    auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
    auto sum = full<tile<float, shape<TM, TN>>>(0.0f);
    for (auto k : irange(0, K / TK)) {
        sum = mma(a.load(0, k), b.load(k, 0), sum);
    }
    c.store(sum, 0, 0);
}
`, kernelName)
}

func generateDeclarativeFP4(kernelName string) string {
	return fmt.Sprintf(`extern "C" __tile_global__ void %s(
    const float* __restrict__ A,
    const cuda::fp4_e2m1* __restrict__ B_fp4,
    const float* __restrict__ scale_b,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TM = 16, TN = 16, TK = 64;
    auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
    auto b = partition_view{tensor_span{assume_aligned<16>(B_fp4), extents{K, N}}, shape<TK, TN>{}};
    auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
    auto sum = full<tile<float, shape<TM, TN>>>(0.0f);
    for (auto k : irange(0, K / TK)) {
        sum = mma(a.load(0, k), b.load(k, 0), sum);
    }
    c.store(sum, 0, 0);
}
`, kernelName)
}

// Lowerer translates egraph expressions into Tile IR modules and textual IR.
type Lowerer struct {
	opts CodegenOptions
}

// NewLowerer creates a new Lowerer configured with opts.
func NewLowerer(opts CodegenOptions) *Lowerer {
	if opts.KernelName == "" {
		opts.KernelName = "tile_kernel"
	}
	if opts.Precision == "" {
		opts.Precision = "float"
	}
	return &Lowerer{opts: opts}
}

// GenerateTileIR translates an egraph expression AST into valid Tile IR textual assembly syntax.
func GenerateTileIR(node *Node, opts ...CodegenOptions) (string, error) {
	var opt CodegenOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return NewLowerer(opt).Generate(node)
}

// ToTileIR lowers an extracted AST Expr into an intermediate Tile IR Module (tileir.Module).
//
// The resulting Module can be rendered as Tile IR text with Module.String.
// Normalized projections use the supported straight-line bytecode subset.
// Other lowerings may require operation or region encoding not yet supported.
func ToTileIR(expr *Expr, opts CodegenOptions) (*tileir.Module, error) {
	return NewLowerer(opts).Lower(expr)
}

// AssembleTileIR attempts to encode and assemble expr as a device CUBIN.
// Unsupported operation or region encoding returns an error wrapping
// [tileir.ErrUnsupportedASTEncoding] before invoking the assembler. Successful
// encoding or assembly does not establish numerical parity on a device.
func AssembleTileIR(ctx context.Context, expr *Expr, opts CodegenOptions, asm ...tileiras.Option) ([]byte, error) {
	mod, err := ToTileIR(expr, opts)
	if err != nil {
		return nil, fmt.Errorf("egraph: lower tile ir: %w", err)
	}
	bytecode, err := mod.EncodeBytecode()
	if err != nil {
		return nil, fmt.Errorf("egraph: encode tile ir bytecode: %w", err)
	}
	result, err := tileiras.Assemble(ctx, bytecode, asm...)
	if err != nil {
		return nil, fmt.Errorf("egraph: assemble tile ir: %w", err)
	}
	if len(result.CUBIN) == 0 {
		return nil, fmt.Errorf("egraph: assemble tile ir: assembler produced an empty cubin")
	}
	return result.CUBIN, nil
}

// Generate translates an egraph expression AST into valid Tile IR textual assembly syntax.
func (l *Lowerer) Generate(node *Node) (string, error) {
	mod, err := l.Lower(node)
	if err != nil {
		return "", err
	}
	return mod.String(), nil
}

func buildTileIRRMSNormOps(cfg rmsNormConfig, tile64, tile64f32 *tileir.TileType) (wParamName string, loadWOp *tileir.Operation, normOps []*tileir.Operation, normedSSA string) {
	epsSSA := "%cst_eps"
	if !cfg.hasEps && cfg.epsName != "" && cfg.epsName != "eps" {
		epsSSA = "%cst_" + sanitizeTileIdent(cfg.epsName)
	}
	epsAttr := fmt.Sprintf("<f32: %s>", formatTileFloat(cfg.epsVal))

	normOps = []*tileir.Operation{
		{OpCode: "cuda_tile.constant", Results: []string{"%cst_dim"}, Attributes: "<f32: 64.0>", Type: tile64f32.String(), TypeVal: tile64f32},
		{OpCode: "cuda_tile.constant", Results: []string{epsSSA}, Attributes: epsAttr, Type: tile64f32.String(), TypeVal: tile64f32},
		{OpCode: "cuda_tile.mulf", Results: []string{"%x_sq"}, Operands: []string{"%vX", "%vX"}, Type: tile64.String(), TypeVal: tile64},
		{OpCode: "cuda_tile.reduce_sum", Results: []string{"%x_sq_sum"}, Operands: []string{"%x_sq"}, Type: tile64.String(), TypeVal: tile64},
		{OpCode: "cuda_tile.divf", Results: []string{"%x_mean"}, Operands: []string{"%x_sq_sum", "%cst_dim"}, Type: tile64.String(), TypeVal: tile64},
		{OpCode: "cuda_tile.addf", Results: []string{"%x_var"}, Operands: []string{"%x_mean", epsSSA}, Type: tile64.String(), TypeVal: tile64},
		{OpCode: "cuda_tile.rsqrt", Results: []string{"%r_rms"}, Operands: []string{"%x_var"}, Type: tile64.String(), TypeVal: tile64},
		{OpCode: "cuda_tile.mulf", Results: []string{"%x_norm"}, Operands: []string{"%vX", "%r_rms"}, Type: tile64.String(), TypeVal: tile64},
	}

	if !cfg.hasWeight {
		return "", nil, normOps, "%x_norm"
	}

	wParamName = "%W_norm"
	wSSA := "%vWnorm"
	if !cfg.hasWVal && cfg.wName != "" && cfg.wName != "w" && cfg.wName != "W_norm" {
		ident := sanitizeTileIdent(cfg.wName)
		wParamName = "%W_norm_" + ident
		wSSA = "%vWnorm_" + ident
	}

	if cfg.hasWVal {
		wAttr := fmt.Sprintf("<f32: %s>", formatTileFloat(cfg.wVal))
		normOps = append(normOps,
			&tileir.Operation{OpCode: "cuda_tile.constant", Results: []string{"%cst_wnorm"}, Attributes: wAttr, Type: tile64.String(), TypeVal: tile64},
			&tileir.Operation{OpCode: "cuda_tile.mulf", Results: []string{"%x_scaled"}, Operands: []string{"%x_norm", "%cst_wnorm"}, Type: tile64.String(), TypeVal: tile64},
		)
		return wParamName, nil, normOps, "%x_scaled"
	}

	loadWOp = &tileir.Operation{
		OpCode:   "cuda_tile.load_view_tko",
		Results:  []string{wSSA},
		Operands: []string{wParamName, "%tok0"},
		Type:     tile64.String(),
		TypeVal:  tile64,
	}
	normOps = append(normOps,
		&tileir.Operation{OpCode: "cuda_tile.mulf", Results: []string{"%x_scaled"}, Operands: []string{"%x_norm", wSSA}, Type: tile64.String(), TypeVal: tile64},
	)
	return wParamName, loadWOp, normOps, "%x_scaled"
}

// Lower lowers an extracted AST Node into an intermediate Tile IR Module (tileir.Module).
func (l *Lowerer) Lower(node *Node) (*tileir.Module, error) {
	if node == nil {
		return nil, fmt.Errorf("egraph: nil node passed to Lower")
	}

	// If node is an unfused composite expression, attempt to fuse using TransformerTileRules.
	if !strings.HasPrefix(node.Op, "tile_") && len(node.Children) > 0 {
		if fused := fuseNode(node); fused != nil && strings.HasPrefix(fused.Op, "tile_") {
			// Shape specialization belongs to the original root, not a rewrite leaf.
			fused.Data = node.Data
			node = fused
		}
	}

	if normalizedProjectionOp(node.Op) {
		mod, _, err := lowerNormalizedProjection(node, l.opts)
		return mod, err
	}
	if normalizedSwiGLUNode(node) {
		return nil, fmt.Errorf("egraph: %s requires full-row RMS normalization and composite output lowering", node.Op)
	}

	var elemType tileir.Type = tileir.Float32
	switch l.opts.Precision {
	case "half", "float16", "f16":
		elemType = tileir.Float16
	case "bf16", "bfloat16":
		elemType = tileir.BFloat16
	case "f8", "fp8", "e4m3", "f8E4M3FN", "fp8_e4m3":
		elemType = tileir.Float8E4M3
	case "fp8_e5m2", "e5m2", "f8E5M2":
		elemType = tileir.Float8E5M2
	case "f4", "fp4", "e2m1", "f4E2M1FN":
		elemType = tileir.Float4E2M1
	}

	mod := &tileir.Module{
		Name:     "kernels",
		Producer: "egraph_compiler",
	}

	fn := &tileir.Function{
		Name:  l.opts.KernelName,
		Entry: true,
	}

	tilePtr := tileir.NewTile(nil, tileir.NewPointer(elemType))
	tileF32Ptr := tileir.NewTile(nil, tileir.NewPointer(tileir.Float32))
	tile64 := tileir.NewTile([]int64{64, 64}, elemType)
	tile64f32 := tileir.NewTile([]int64{64, 64}, tileir.Float32)

	canLowerRootNorm := node.Op == "rmsnorm" && len(node.Children) > 0
	if !IsLowerableTileIROp(node.Op) && !canLowerRootNorm {
		return nil, fmt.Errorf("egraph: no Tile IR lowering for operator %q", node.Op)
	}

	cfg := extractRMSNormConfig(node)

	switch node.Op {
	case "tile_rmsnorm", "fused_rmsnorm", "rmsnorm":
		wParamName, loadWOp, normOps, normedSSA := buildTileIRRMSNormOps(cfg, tile64, tile64f32)
		fn.Params = []*tileir.Param{
			{Name: "%X", Type: tilePtr.String(), TypeVal: tilePtr},
		}
		if wParamName != "" {
			fn.Params = append(fn.Params, &tileir.Param{Name: wParamName, Type: tilePtr.String(), TypeVal: tilePtr})
		}
		fn.Params = append(fn.Params, &tileir.Param{Name: "%Out", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr})

		ops := []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vX"}, Operands: []string{"%X", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
		}
		if loadWOp != nil {
			ops = append(ops, loadWOp)
		}
		ops = append(ops, normOps...)
		ops = append(ops,
			&tileir.Operation{OpCode: "cuda_tile.store_view_tko", Operands: []string{normedSSA, "%Out", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			&tileir.Operation{OpCode: "cuda_tile.return"},
		)
		fn.Operations = ops

	case "tile_swiglu", "fused_swiglu":
		fn.Params = []*tileir.Param{
			{Name: "%X", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_gate", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_up", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_down", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%Out", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vX"}, Operands: []string{"%X", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWg"}, Operands: []string{"%W_gate", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWu"}, Operands: []string{"%W_up", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWd"}, Operands: []string{"%W_down", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst1"}, Attributes: "<f32: 1.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%gate"}, Operands: []string{"%vX", "%vWg", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%up"}, Operands: []string{"%vX", "%vWu", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.negf", Results: []string{"%neg_gate"}, Operands: []string{"%gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.exp", Results: []string{"%exp_gate"}, Operands: []string{"%neg_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.addf", Results: []string{"%denom"}, Operands: []string{"%cst1", "%exp_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.divf", Results: []string{"%sig_gate"}, Operands: []string{"%cst1", "%denom"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%silu_gate"}, Operands: []string{"%gate", "%sig_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%act"}, Operands: []string{"%silu_gate", "%up"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%acc"}, Operands: []string{"%act", "%vWd", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%Out", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	case "tile_geglu", "fused_geglu":
		fn.Params = []*tileir.Param{
			{Name: "%X", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_gate", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_up", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_down", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%Out", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vX"}, Operands: []string{"%X", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWg"}, Operands: []string{"%W_gate", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWu"}, Operands: []string{"%W_up", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWd"}, Operands: []string{"%W_down", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst1"}, Attributes: "<f32: 1.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%gate"}, Operands: []string{"%vX", "%vWg", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%up"}, Operands: []string{"%vX", "%vWu", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.negf", Results: []string{"%neg_gate"}, Operands: []string{"%gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.exp", Results: []string{"%exp_gate"}, Operands: []string{"%neg_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.addf", Results: []string{"%denom"}, Operands: []string{"%cst1", "%exp_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.divf", Results: []string{"%sig_gate"}, Operands: []string{"%cst1", "%denom"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%gelu_gate"}, Operands: []string{"%gate", "%sig_gate"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%act"}, Operands: []string{"%gelu_gate", "%up"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%acc"}, Operands: []string{"%act", "%vWd", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%Out", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	case "tile_gelu", "fused_gelu":
		fn.Params = []*tileir.Param{
			{Name: "%X", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_in", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%W_out", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%Out", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vX"}, Operands: []string{"%X", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWin"}, Operands: []string{"%W_in", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vWout"}, Operands: []string{"%W_out", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst1"}, Attributes: "<f32: 1.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%mid"}, Operands: []string{"%vX", "%vWin", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.negf", Results: []string{"%neg_mid"}, Operands: []string{"%mid"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.exp", Results: []string{"%exp_mid"}, Operands: []string{"%neg_mid"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.addf", Results: []string{"%denom"}, Operands: []string{"%cst1", "%exp_mid"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.divf", Results: []string{"%sig"}, Operands: []string{"%cst1", "%denom"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%act"}, Operands: []string{"%mid", "%sig"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%acc"}, Operands: []string{"%act", "%vWout", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%Out", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	case "tile_mma_fp8", "tile_fp8_gemm", "tile_fp8", "fp8_matmul":
		tileFP8Ptr := tileir.NewTile(nil, tileir.NewPointer(tileir.Float8E4M3))
		tile64FP8 := tileir.NewTile([]int64{64, 64}, tileir.Float8E4M3)
		fn.Params = []*tileir.Param{
			{Name: "%A", Type: tileFP8Ptr.String(), TypeVal: tileFP8Ptr},
			{Name: "%B", Type: tileFP8Ptr.String(), TypeVal: tileFP8Ptr},
			{Name: "%ScaleA", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
			{Name: "%ScaleB", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
			{Name: "%C", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vA"}, Operands: []string{"%A", "%tok0"}, Type: tile64FP8.String(), TypeVal: tile64FP8},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vB"}, Operands: []string{"%B", "%tok0"}, Type: tile64FP8.String(), TypeVal: tile64FP8},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vSa"}, Operands: []string{"%ScaleA", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vSb"}, Operands: []string{"%ScaleB", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%raw"}, Operands: []string{"%vA", "%vB", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%scale"}, Operands: []string{"%vSa", "%vSb"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%acc"}, Operands: []string{"%raw", "%scale"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%C", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	case "mmaf_scaled", "tile_fp4_gemm", "tile_fp4", "fp4_matmul":
		tileFP4Ptr := tileir.NewTile(nil, tileir.NewPointer(tileir.Float4E2M1))
		tile64FP4 := tileir.NewTile([]int64{64, 64}, tileir.Float4E2M1)
		fn.Params = []*tileir.Param{
			{Name: "%A", Type: tileFP4Ptr.String(), TypeVal: tileFP4Ptr},
			{Name: "%B", Type: tileFP4Ptr.String(), TypeVal: tileFP4Ptr},
			{Name: "%ScaleA", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
			{Name: "%ScaleB", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
			{Name: "%C", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vA"}, Operands: []string{"%A", "%tok0"}, Type: tile64FP4.String(), TypeVal: tile64FP4},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vB"}, Operands: []string{"%B", "%tok0"}, Type: tile64FP4.String(), TypeVal: tile64FP4},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vSa"}, Operands: []string{"%ScaleA", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vSb"}, Operands: []string{"%ScaleB", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%raw"}, Operands: []string{"%vA", "%vB", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%scale"}, Operands: []string{"%vSa", "%vSb"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mulf", Results: []string{"%acc"}, Operands: []string{"%raw", "%scale"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%C", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	case "tile_mma", "tile_mma_aligned", "mma":
		fn.Params = []*tileir.Param{
			{Name: "%A", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%B", Type: tilePtr.String(), TypeVal: tilePtr},
			{Name: "%C", Type: tileF32Ptr.String(), TypeVal: tileF32Ptr},
		}
		fn.Operations = []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%tok0"}, Type: tileir.Token.String(), TypeVal: tileir.Token},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vA"}, Operands: []string{"%A", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.load_view_tko", Results: []string{"%vB"}, Operands: []string{"%B", "%tok0"}, Type: tile64.String(), TypeVal: tile64},
			{OpCode: "cuda_tile.constant", Results: []string{"%cst0"}, Attributes: "<f32: 0.0>", Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.mmaf", Results: []string{"%acc"}, Operands: []string{"%vA", "%vB", "%cst0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.store_view_tko", Operands: []string{"%acc", "%C", "%tok0"}, Type: tile64f32.String(), TypeVal: tile64f32},
			{OpCode: "cuda_tile.return"},
		}

	default:
		return nil, fmt.Errorf("egraph: no Tile IR lowering for operator %q", node.Op)
	}

	blk := &tileir.Block{
		Name:       "^bb0",
		Operations: fn.Operations,
	}
	fn.Blocks = []*tileir.Block{blk}
	mod.Functions = []*tileir.Function{fn}

	return mod, nil
}

func fuseNode(node *Node) *Node {
	if node == nil {
		return nil
	}
	g := New()
	var addExpr func(e *Expr) Id
	addExpr = func(e *Expr) Id {
		if len(e.Children) == 0 {
			if e.Data != nil {
				return g.Add(ENode{Op: e.Op, Data: e.Data})
			}
			return g.Add(NewLeaf(e.Op))
		}
		childIds := make([]Id, len(e.Children))
		for i, child := range e.Children {
			childIds[i] = addExpr(child)
		}
		return g.Add(NewNode(e.Op, childIds...))
	}
	rootId := addExpr(node)
	rules := TransformerTileRules()
	if _, err := SaturateChecked(g, rules, SaturationOptions{MaxIterations: 15}); err != nil {
		return node
	}
	best, _, err := ExtractBest(g, rootId, FusedMegakernelCost{})
	if err == nil && best != nil {
		return best
	}
	return node
}
